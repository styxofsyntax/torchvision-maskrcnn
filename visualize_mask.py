import matplotlib.pyplot as plt
from torchvision.io import read_image

img_name = 'IMG_8975_JPG.rf.ee8e9f7d1be67676454d2581438f0d91'
image = read_image(
    f"data/train/images/{img_name}.jpg")
mask = read_image(
    f"data/train/masks/{img_name}.png")

plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.title("Image")
plt.imshow(image.permute(1, 2, 0))
plt.subplot(122)
plt.title("Mask")
plt.imshow(mask.permute(1, 2, 0))
plt.show()
