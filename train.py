import os
import torch.nn as nn
import numpy as np
import torch
from byol_1channel import BYOL_1channel
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import utils
from backbone import HRNet

total_epochs = 2000
image_folder = "/data1/xuanhang_diao/uni_dicom/data"
WEIGHTS_PATH = "/data1/xuanhang_diao/uni_dicom/pts/alldata_precrop_sz144_window160_add2chbay_newAug"

device = "cuda"  # device = 'cuda'
first_crop = 256
IMAGE_SIZE = 144
BATCH_SIZE = 256
NUM_WORKERS = 16

BACKBONE = "HRNet"
assert BACKBONE in ["resnet18", "resnet50", "HRNet"]

if BACKBONE == "resnet18":
    backbone = models.resnet18(weights=None)
    backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
elif BACKBONE == "resnet50":
    backbone = models.resnet50(weights=None)
    backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
else:
    backbone = HRNet.hrnet_classification()
    backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

if not os.path.exists(WEIGHTS_PATH):
    os.makedirs(WEIGHTS_PATH)

aug_fn = torch.nn.Sequential(
    transforms.RandomErasing(p=0.5, scale=(0.1, 0.1)),
    transforms.RandomApply(
        [transforms.RandomCrop(192)], p=0.5
    ),
    transforms.RandomApply([transforms.ColorJitter(contrast=0.3, brightness=0.3)], p=0.5),
    transforms.RandomApply(
        [transforms.RandomVerticalFlip(p=1)], p=0.5
    ),
    transforms.RandomApply(
        [transforms.RandomHorizontalFlip(p=1)], p=0.5
    ),
    transforms.RandomApply(
        [transforms.RandomAffine(15, translate=(0.1, 0.1))], p=0.5
    ),
    transforms.RandomApply([transforms.GaussianBlur((3, 3), (1.0, 1.0))], p=0.5),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
)

learner = BYOL_1channel(
    backbone,
    image_size=IMAGE_SIZE,
    hidden_layer='avgpool',
    augment_fn=aug_fn,
)

learner = learner.to(device)
opt = torch.optim.Adam(learner.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(opt, T_max=25, eta_min=1e-6)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt, T_max=)


if __name__ == '__main__':
    dataset = utils.ImagesDataset_dicom_crop_first_with_window_1channel(image_folder, IMAGE_SIZE, crop_sz)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, pin_memory=True,
                              prefetch_factor=4, persistent_workers=True)
    Loss = []
    min_loss = 100
    for epoch in range(0, total_epochs + 1):
        print("start epoch %d" % epoch)

        learner.train()
        loss_curr = []
        for batch_idx, images in enumerate(train_loader, 0):
            images = images.to(device)
            loss = learner(images)
            running_loss = loss.detach().cpu().item()
            loss_curr.append(running_loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()

            if batch_idx % 5 == 0:
                print(loss.detach().cpu().numpy())
                Loss.append(running_loss)

                """
                                with torch.no_grad():
                    learner.eval()
                    projection, embedding = learner(images, return_embedding=True)
               """
        # scheduler.step()
        if sum(loss_curr) / len(loss_curr) < min_loss:
            min_loss = sum(loss_curr) / len(loss_curr)
            torch.save(learner, os.path.join(WEIGHTS_PATH, "models_{}.pth".format(epoch)))
        scheduler.step()
    np.save(os.path.basename(WEIGHTS_PATH) + ".npy", np.array(Loss))
