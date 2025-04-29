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
        params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1)

    num_epochs = 10

    for epoch in range(num_epochs):
        engine.train_one_epoch(
            model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        engine.evaluate(model, data_loader_valid, device=device)

    torch.save(model.state_dict(), "maskrcnn_model.pth")


if __name__ == "__main__":
    main()
