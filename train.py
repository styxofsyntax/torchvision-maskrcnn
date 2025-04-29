import torch
from torch.utils.data import DataLoader
import utils
from dataset import CustomDataset
from model import get_instance_segmentation_model
import engine
from torchvision.transforms import v2 as T


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def main():
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 3  # background + 2 classes

    dataset = CustomDataset(
        'data/train', transforms=get_transform(train=True))
    dataset_valid = CustomDataset(
        'data/valid', transforms=get_transform(train=False))

    data_loader = DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    data_loader_valid = DataLoader(
        dataset_valid, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    model = get_instance_segmentation_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.005, momentum=0.7, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1)

    num_epochs = 50

    best_val_loss = float('inf')
    patience = 5  # how many epochs to wait before early stop
    trigger_times = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        engine.train_one_epoch(
            model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()

        # --- Validation loss computation ---
        for images, targets in data_loader_valid:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if isinstance(
                v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            with torch.cuda.amp.autocast(enabled=None is not None):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            val_loss = losses_reduced.item()

        # val_loss /= len(data_loader_valid)
        print(f"Validation loss: {val_loss:.4f}")
        # ------------------------------------

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            # Save best model
            torch.save(model.state_dict(), "maskrcnn_best.pth")
            print("Saved Best Model")
        else:
            trigger_times += 1
            print(f'No improvement in {trigger_times} epochs.')

            if trigger_times >= patience:
                print('Early stopping triggered.')
                return

        coco_evaluator = engine.evaluate(
            model, data_loader_valid, device=device)
        print('\n\n-------------COCO Evaluator------------')
        print(type(coco_evaluator))
        print(coco_evaluator)
        print('\n\n')

    torch.save(model.state_dict(), "maskrcnn_last.pth")


if __name__ == "__main__":
    main()
