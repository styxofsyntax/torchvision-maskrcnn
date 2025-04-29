import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(
        window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(
            v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            try:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            except Exception as err:
                print('\n\n---------------- Exception ----------------')
                print(f'Error: {err}')

                import torchvision.transforms.functional as F
                import matplotlib.pyplot as plt
                import numpy as np
                import os

                # Print all filenames in the current batch
                print("\n--- Current batch filenames ---")
                for batch_idx, t in enumerate(targets):
                    filename = t.get("filename", None)
                    print(f"Batch index {batch_idx}: {filename}")
                print("--------------------------------\n")

                # Try to extract which index caused the problem
                problematic_idx = None
                err_str = str(err)
                if "at index" in err_str:
                    try:
                        problematic_idx = int(
                            err_str.split("at index")[-1].strip())
                        print(
                            f"Problematic sample is at batch index: {problematic_idx}")
                    except:
                        print("Couldn't parse index from error message.")

                # Visualize the problematic image and mask
                if problematic_idx is not None and problematic_idx < len(images):
                    img = images[problematic_idx].cpu()
                    img_pil = F.to_pil_image(img)

                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)
                    plt.imshow(np.array(img_pil))
                    plt.title(
                        f"Image Causing Error\n{targets[problematic_idx].get('filename', '')}")
                    plt.axis('off')

                    if "masks" in targets[problematic_idx] and len(targets[problematic_idx]["masks"]) > 0:
                        mask = targets[problematic_idx]["masks"][0].cpu(
                        ).numpy()

                        plt.subplot(1, 2, 2)
                        plt.imshow(mask, cmap="gray")
                        plt.title("Mask Causing Error")
                        plt.axis('off')
                    else:
                        print("No masks found for problematic target!")

                    plt.tight_layout()
                    plt.show()

                    # Save error samples
                    error_dir = "error_samples"
                    os.makedirs(error_dir, exist_ok=True)
                    img_pil.save(os.path.join(
                        error_dir, f"error_image_{targets[problematic_idx].get('filename', 'unknown')}.jpg"))
                    if "masks" in targets[problematic_idx] and len(targets[problematic_idx]["masks"]) > 0:
                        from PIL import Image
                        mask_to_save = Image.fromarray(
                            (mask * 255).astype(np.uint8))
                        mask_to_save.save(os.path.join(
                            error_dir, f"error_mask_{targets[problematic_idx].get('filename', 'unknown')}.jpg"))

                    print(f'Error samples saved in {error_dir}/')

                else:
                    print("Problematic image index not found or out of range.")

                raise err

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()}
                   for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target,
               output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time,
                             evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
