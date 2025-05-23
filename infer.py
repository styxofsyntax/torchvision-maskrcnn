import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from model import get_instance_segmentation_model
from train import get_transform

label_map = {
    1: "card",
    2: "damages",
}

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 3
model = get_instance_segmentation_model(num_classes)
model.to(device)

model.load_state_dict(torch.load("maskrcnn_model.pth", map_location=device))

image = read_image(
    "data/test/images/IMG_0013_JPG.rf.2337b56130b71363bd2d44947513d914.jpg")
eval_transform = get_transform(train=False)

model.eval()
with torch.no_grad():
    x = eval_transform(image)
    # convert RGBA -> RGB and move to device
    x = x[:3, ...].to(device)
    predictions = model([x, ])
    pred = predictions[0]

print(type(predictions))
print(predictions)

image = (255.0 * (image - image.min()) /
         (image.max() - image.min())).to(torch.uint8)
image = image[:3, ...]

score_threshold = 0.5
keep = pred["scores"] >= score_threshold

pred_boxes = pred["boxes"][keep].long()
pred_labels_raw = pred["labels"][keep]
pred_scores = pred["scores"][keep]
pred_masks = (pred["masks"][keep] > 0.7).squeeze(1)

pred_labels = [
    f"{label_map.get(label.item(), 'unknown')}: {score:.3f}"
    for label, score in zip(pred_labels_raw, pred_scores)
]

output_image = draw_bounding_boxes(
    image, pred_boxes, pred_labels, colors="red")
output_image = draw_segmentation_masks(
    output_image, pred_masks, alpha=0.5, colors="blue")

plt.figure(figsize=(12, 12))
plt.imshow(output_image.permute(1, 2, 0))
plt.axis('off')
plt.show()
