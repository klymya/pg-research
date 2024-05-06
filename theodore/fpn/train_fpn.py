import os
import shutil

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import cv2
import albumentations as A
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils as smp_utils

# FPN + "mobilenet_v2" # 4215425
# PSPNet + mobilenet_v2 = 2263353
# PAN + mobilenet_v2 = 2416552
# DeepLabV3Plus + mobilenet_v2 = 4378513


# ENCODER = 'timm-mobilenetv3_small_100' # 2721953 prams
ENCODER = 'mobilenet_v2' # 3328985 params
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'
EXP_NAME = f"fpn_{ENCODER}_schedule_25_upd_early_stopping"
LOG_DIR = f"runs/{EXP_NAME}"
NUM_EPOCHS = 1000
BATCH_SIZE = 64


class PGDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        split_names_pth: str,
        img_size: int = 128,
        augmentation=None,
        preprocessing=None,
    ):
        self.img_ext = ".jpg"
        self.mask_ext = ".png"

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.split_names_pth = split_names_pth

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.ids = [
            os.path.splitext(file)[0]
            for file in os.listdir(images_dir)
            if os.path.isfile(os.path.join(images_dir, file)) and file.endswith(self.img_ext)
        ]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        mask_ids = [
            os.path.splitext(file)[0]
            for file in os.listdir(masks_dir)
            if os.path.isfile(os.path.join(masks_dir, file)) and file.endswith(self.mask_ext)
        ]
        if not mask_ids:
            raise RuntimeError(f'No input file found in {masks_dir}, make sure you put your images there')

        if len(set(mask_ids).intersection(self.ids)) != len(self.ids):
            raise RuntimeError('Number of images and masks not equal')

        with open(self.split_names_pth, "rt") as f:
            split_idxs = f.readlines()
        split_idxs = [i.strip() for i in split_idxs]

        self.ids = [i for i in self.ids if i in split_idxs]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = os.path.join(self.masks_dir, name + self.mask_ext)
        img_file = os.path.join(self.images_dir, name + self.img_ext)

        # read data
        image = cv2.imread(img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, 0) // 255

        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        mask = mask[:, :, None]

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)


def save_checkpoint(model, checkpoint_dir, filename):
    checkpoint_path = os.path.join(checkpoint_dir, f'{filename}.pth')
    torch.save(model, checkpoint_path)


def main():
    shutil.rmtree(LOG_DIR, ignore_errors=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1,
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = PGDataset(
        "../theodore_unet/imgs/",
        "../theodore_unet/masks/",
        "../theodore_unet/train.txt",
        preprocessing=get_preprocessing(preprocessing_fn)
    )

    valid_dataset = PGDataset(
        "../theodore_unet/imgs/",
        "../theodore_unet/masks/",
        "../theodore_unet/val.txt",
        preprocessing=get_preprocessing(preprocessing_fn)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

    loss = smp_utils.losses.DiceLoss()
    metrics = [
        smp_utils.metrics.IoU(threshold=0.5),
    ]

    # optimizer = torch.optim.Adam([ 
    #     dict(params=model.parameters(), lr=0.0001),
    # ])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    # create epoch runners 
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp_utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp_utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # train model for 40 epochs
    writer = SummaryWriter(LOG_DIR)  # Create a SummaryWriter for logging
    best_val_loss = float('inf')
    last_val_loss = float('inf')
    no_improvement_count = 0

    max_score = 0
    for epoch in range(0, NUM_EPOCHS):
        print('\nEpoch: {}'.format(epoch))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        scheduler.step()

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')

        # Log metrics to TensorBoard
        writer.add_scalar('dice_loss/Train', train_logs["dice_loss"], epoch)
        writer.add_scalar('iou_score/Train', train_logs["iou_score"], epoch)
        writer.add_scalar('dice_loss/Val', valid_logs["dice_loss"], epoch)
        writer.add_scalar('iou_score/Val', valid_logs["iou_score"], epoch)

        # Save checkpoint if validation loss improves
        if valid_logs["dice_loss"] < best_val_loss:
            best_val_loss = valid_logs["dice_loss"]
            save_checkpoint(model, LOG_DIR, 'best')
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Save last checkpoint
        save_checkpoint(model, LOG_DIR, 'last')

        # Check for early stopping
        if epoch > 50 and no_improvement_count >= 50:
            print("No improvement for 50 epochs. Stopping training.")
            break

    writer.close()  # Close the SummaryWriter when done


if __name__ == "__main__":
    main()
