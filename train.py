import torch.backends.cudnn as cudnn
import torch
from math import log10
from tqdm import tqdm
from torch import nn
from model import UNet
from dataset import TripletDataset
from utils import AverageMeter


# Data parameters
train_folder = './frames'
test_folder = './test_data'
crop_size = 128
scale = 2
in_channels = 9
out_channels = 3

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 16  # batch size
start_epoch = 0  # start at this epoch
iterations = 1e6  # number of training iterations
workers = 4 # TODO  # number of workers for loading data in the DataLoader
print_freq = 10  # print training status once every __ batches
lr = 1e-4  # learning rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, epoch, checkpoint

    # Initialize model or load checkpoint
    if checkpoint is None:
        model = UNet(in_channels, out_channels)
        # Initialize the optimizer
        optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr
        )
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    # Custom dataloaders
    train_loader = torch.utils.data.DataLoader(
        TripletDataset(train_folder, crop_size, scale),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        TripletDataset(test_folder, crop_size, scale),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True
    )

    # Total number of epochs to train for
    epochs = int(iterations // len(train_loader) + 1)

    # Epochs
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        train(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            epochs=epochs
        )
        test(
            test_loader=test_loader,
            model=model,
            criterion=criterion
        )

        # Save checkpoint
        torch.save({'epoch': epoch,
                    'model': model,
                    'optimizer': optimizer},
                    'checkpoint_unet.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch, epochs):
    model.train()

    losses = AverageMeter()
    psnrs = AverageMeter()

    pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}')
    for lr_imgs, hr_imgs in train_loader:
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        # Forward prop.
        sr_imgs = model(lr_imgs)

        # Loss
        loss = criterion(sr_imgs, hr_imgs)

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Update model
        optimizer.step()

        # Keep track of loss
        losses.update(loss.item())
        psnr = 10 * log10(1 / loss.item())
        psnrs.update(psnr)

        pbar.set_postfix_str(f'MSE {losses.val:.6f} PSNR {psnrs.val:.2f}')
        pbar.update(1)


def test(test_loader, model, criterion):
    psnrs = AverageMeter()
    losses = AverageMeter()

    pbar = tqdm(total=len(test_loader), desc=f'Test')
    with torch.no_grad():
        for lr_imgs, hr_imgs in test_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            sr_imgs = model(lr_imgs)

            loss = criterion(sr_imgs, hr_imgs)

            # Keep track of loss
            losses.update(loss.item())
            psnr = 10 * log10(1 / loss.item())
            psnrs.update(psnr)

            pbar.set_postfix_str(f'MSE {losses.val:.6f} PSNR {psnrs.val:.2f}')
            pbar.update(1)


if __name__ == '__main__':
    main()