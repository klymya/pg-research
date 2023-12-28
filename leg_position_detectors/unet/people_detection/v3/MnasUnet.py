import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import math
import random
from datetime import datetime
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, models, utils
import torchvision.transforms.functional as F
import torch.nn.functional as F1
from torchvision.io import read_image
import torch.nn.init as init
from torchsummary import summary
from sklearn.model_selection import train_test_split
import cv2
import csv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
output_image_size = 224
bb_half_size = 8
# letter = 'mu'
# letter = '_drop02rad17_balanced_0.5'
letter = '_drop02rad17a'
pos_weight = 0.5
drop_out = 0.2
b_write = False
min_area = 1
reduction_for_SE = 16
valid_size1 = 0.05  # 0.02
learning_rate = 0.001
batch_train_valid1 = 4
local_machine = False
training1 = True
use_se = False
if use_se:
    letter = '_SE'
if not training1:
    import paramiko
    import time

if local_machine:
    if training1:
        # model_save_name1 = 'PTH_UNET_LOCAL_TRAINING\\MNSA_Unet' + letter + '.pth'
        # model_save_name1 = 'PTH_UNET_LOCAL_TRAINING_POS\\MNSA_Unet' + letter + '.pth'
        # model_save_name1 = 'PTH_MNSA_Unet/MNSA_UnetDecInvResKer3_Pos' + letter + '.pth'
        model_save_name1 = 'PTH_MNAS_Unet/MNSA_UnetT2' + letter + '.pth'
        csv_file_name1 = "c:/CamDatabase/CSV/1.csv"
    else:
        # model_save_name1 = 'PTH_UNET_LOCAL_TRAINING\\MNSA_Unet' + letter + '.pth'
        # model_save_name1 = 'PTH_MNSA_Unet/MNSA_UnetDecInvResKer3th' + letter + '.pth'
        # model_save_name0 = 'PTH_MNAS_Unet/MNAS_UnetDecInvResKer3_Pos' + letter + '.pth'
        # model_save_name0 = 'PTH_MNAS_Unet_EC2/MNAS_Unet' + letter + '.pth'
        model_save_name0 = 'PTH_MNAS_Unet_Pos/MNAS_Unet' + letter + '.pth'
        model_save_name1 = None
        # model_save_name0 = 'PTH_MNSA_Unet_EC2\MNSA_UnetDecInvResKer3_Pos.pth'
        # model_save_name1 = 'PTH_MNSA_Unet_EC2\MNSA_UnetDecInvResKer3_Pos_SE.pth'
        # csv_file_name1 = "c:/CamDatabase/Frames/ExperimentsOutputEC2_1/1.csv"
        csv_file_name1 = "c:/CamDatabase/Frames/ExperimentsOutputEC2_SE_1/5.csv"
        # csv_file_name1 = "c:/CamDatabase/Frames/ExperimentsOutput9/1.csv"
else:
    # model_save_name1 = 'PTH_MNSA_Unet/MNSA_Unet' + letter + '.pth'
    model_save_name1 = 'PTH_MNAS_Unet_Pos/MNAS_Unet' + letter + '.pth'
    csv_file_name1 = 'PTH_MNAS_Unet_Pos/MNAS_Unet' + letter + '.csv'
# directory_name1 = "c:/CamDatabase/Frames/UnetTest"
# directory_name1 = "c:/CamDatabase/Frames/UnetTraining"
# directory_name1 = "c:/CamDatabase/Frames/ExperimentsInput1"
# directory_name1 = "c:/CamDatabase/PositionsDB/Test"
# directory_name1 = "c:/Theodore/Test"
directory_name1 = "c:/CamDatabase/PositionsDB/TestThin"  #   "c:/CamDatabase/DBAnnotated/Test"
# directory_name1 = "c:/CamDatabase/PositionsDB/IntExp1"  #   "c:/CamDatabase/DBAnnotated/Test"
# directory_name1 = "c:\\CamDatabase\\Frames\\Exp9"

# output_masks_dir1 = "c:/CamDatabase/Frames/ExperimentsOutputEC2_1"
output_masks_dir1 = "c:/CamDatabase/Frames/ExperimentsOutputEC2_SE_1"
# output_masks_dir1 = "c:/CamDatabase/Frames/ExperimentsOutput9"
cam_filename1 = "cam_m"
host1 = "172.19.56.12"
# host1 = "172.22.56.14"  # meeting room
do_train_rotate1 = True
do_train_flip1 = True
do_train_dark1 = True
do_train_noise1 = True
do_noise_reduction1 = False


def visualize_feature_maps(model_unet, img):
    no_of_layers = 0
    conv_layers = []

    model_children = list(model_unet.children())

    for child in model_children:
        if type(child) == DoubleConv:
            #     no_of_layers += 1
            #     conv_layers.append(child)
            # elif type(child) == nn.Sequential:
            for layer in child.conv:
                if type(layer) == nn.Conv2d:
                    no_of_layers += 1
                    conv_layers.append(layer)
    print(no_of_layers)

    results = [conv_layers[0](img)]
    for i in range(1, len(conv_layers)):
        results.append(conv_layers[i](results[-1]))
    outputs = results

    for num_layer in range(len(outputs)):
        plt.figure(figsize=(320, 240))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        print("Layer ", num_layer + 1)
        if num_layer >= 0:
            for i, filter1 in enumerate(layer_viz):
                # if i == 16:
                #     break
                # plt.subplot(2, 8, i + 1)
                if i > -1:
                    plt.imshow(filter1, cmap='gray')
                    plt.axis("off")
                    plt.title('layer ' + str(num_layer + 1) + ' channel ' + str(i + 1), fontweight="bold")
                    plt.show()
                    cv2.waitKey(0)
        # cv2.destroyAllWindows()

        plt.close()


def take_image_from_camera(host, image_name):
    # time.sleep(5)
    image_full_name_host = image_name + ".jpeg"
    image_full_name_loc = image_name + ".jpg"
    username = "pg"
    password = "q1w2e3r4"
    client = paramiko.client.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=username, password=password)
    client.exec_command("sudo systemctl stop cp-core")
    client.exec_command("rm -f *.jpeg")
    image_capture_command = "sudo ffmpeg -f video4linux2 -i /dev/video0 -vframes 1 " + image_full_name_host
    client.exec_command(image_capture_command)
    time.sleep(1)
    scp = client.open_sftp()
    from_addr = '/home/pg/' + image_full_name_host
    to_addr = 'c:/Users/ilyar/PycharmProjects/pythonProject1/' + image_full_name_loc
    scp.get(from_addr, to_addr)
    scp.close()
    client.close()


# U-Net architecture
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class OneConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OneConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.dropout = nn.Dropout(p=drop_out)  # Dropout layer defined once
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up4 = nn.ConvTranspose2d(128, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down1(nn.MaxPool2d(2)(x))
        x1 = self.dropout(x1)  # Dropout applied
        x2 = self.down2(nn.MaxPool2d(2)(x1))
        x2 = self.dropout(x2)  # Dropout applied
        x3 = self.down3(nn.MaxPool2d(2)(x2))
        x3 = self.dropout(x3)  # Dropout applied
        x4 = self.down4(nn.MaxPool2d(2)(x3))
        x4 = self.dropout(x4)  # Dropout applied
        x = self.up1(x4)
        x = self.dropout(x)  # Dropout applied
        x = self.up2(torch.cat([x, x3], dim=1))
        x = self.dropout(x)  # Dropout applied
        x = self.up3(torch.cat([x, x2], dim=1))
        x = self.dropout(x)  # Dropout applied
        x = self.up4(torch.cat([x, x1], dim=1))
        # x = self.sigmoid(x)
        return x


def Conv_3x3(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def Conv_1x1(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def SepConv_3x3(inp, oup):  # input=32, output=16
    return nn.Sequential(
        # dw
        nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=reduction_for_SE):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Squeeze: Global average pooling
        squeeze = self.global_pool(x).view(x.size(0), -1)

        # Excitation: Fully connected layers with sigmoid activation
        excitation = self.fc(squeeze).view(x.size(0), -1, 1, 1)

        # Scale the original features
        x = x * excitation
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel, b_up):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup
        if b_up:
            self.use_res_connect = False

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel, stride, kernel // 2, groups=inp * expand_ratio,
                      bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidualWithSE(InvertedResidual):
    def __init__(self, inp, oup, stride, expand_ratio, kernel, b_up):
        super(InvertedResidualWithSE, self).__init__(inp, oup, stride, expand_ratio, kernel, b_up)
        # Insert SE block before ReLU6
        self.se_block = SEBlock(inp * expand_ratio, reduction=reduction_for_SE)
        self.conv.insert(-3, self.se_block)  # Insert SE block before the second ReLU6


class CustomMNasNetUnet(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes=1, input_size=224, width_mult=1.):
        super(CustomMNasNetUnet, self).__init__()

        # setting of inverted residual blocks
        self.inverted_residual_setting = [
            # t, c, n, s, k
            [3, 24, 3, 2, 3],  # -> 56x56
            [3, 40, 3, 2, 5],  # -> 28x28
            [6, 80, 3, 2, 5],  # -> 14x14
            [6, 96, 2, 1, 3],  # -> 14x14
            [6, 192, 4, 2, 5],  # -> 7x7
            [6, 320, 1, 1, 3],  # -> 7x7
        ]

        assert input_size % 32 == 0

        self.dropout = nn.Dropout(p=drop_out)  # Dropout layer defined once

        input_channel = 16
        self.num_of_blocks = len(self.inverted_residual_setting)
        index = 0
        self.down1 = [Conv_3x3(1, input_channel, 2), SepConv_3x3(input_channel, 16)]
        self.down2 = []
        self.down3 = []
        self.down4 = []
        self.down5 = []
        self.down6 = []
        self.down7 = []
        for t, c, n, s, k in self.inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if index == 0:
                    if i == 0:
                        if use_se:
                            self.down2.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k, False))
                        else:
                            self.down2.append(InvertedResidual(input_channel, output_channel, s, t, k, False))
                    else:
                        if use_se:
                            self.down2.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k, False))
                        else:
                            self.down2.append(InvertedResidual(input_channel, output_channel, 1, t, k, False))
                elif index == 1:
                    if i == 0:
                        if use_se:
                            self.down3.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k, False))
                        else:
                            self.down3.append(InvertedResidual(input_channel, output_channel, s, t, k, False))
                    else:
                        if use_se:
                            self.down3.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k, False))
                        else:
                            self.down3.append(InvertedResidual(input_channel, output_channel, 1, t, k, False))
                elif index == 2:
                    if i == 0:
                        if use_se:
                            self.down4.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k, False))
                        else:
                            self.down4.append(InvertedResidual(input_channel, output_channel, s, t, k, False))
                    else:
                        if use_se:
                            self.down4.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k, False))
                        else:
                            self.down4.append(InvertedResidual(input_channel, output_channel, 1, t, k, False))
                elif index == 3:
                    if i == 0:
                        if use_se:
                            self.down5.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k, False))
                        else:
                            self.down5.append(InvertedResidual(input_channel, output_channel, s, t, k, False))
                    else:
                        if use_se:
                            self.down5.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k, False))
                        else:
                            self.down5.append(InvertedResidual(input_channel, output_channel, 1, t, k, False))
                elif index == 4:
                    if i == 0:
                        if use_se:
                            self.down6.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k, False))
                        else:
                            self.down6.append(InvertedResidual(input_channel, output_channel, s, t, k, False))
                    else:
                        if use_se:
                            self.down6.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k, False))
                        else:
                            self.down6.append(InvertedResidual(input_channel, output_channel, 1, t, k, False))
                else:
                    if i == 0:
                        if use_se:
                            self.down7.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k, False))
                        else:
                            self.down7.append(InvertedResidual(input_channel, output_channel, s, t, k, False))
                    else:
                        if use_se:
                            self.down7.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k, False))
                        else:
                            self.down7.append(InvertedResidual(input_channel, output_channel, 1, t, k, False))

                input_channel = output_channel
            index += 1

        # building last several layers
        # self.features.append(Conv_1x1(input_channel, self.last_channel))
        # self.features.append(nn.AdaptiveAvgPool2d(1))

        # make it nn.Sequential
        # self.features = nn.Sequential(*self.features)
        self.down1 = nn.Sequential(*self.down1)
        self.down2 = nn.Sequential(*self.down2)
        self.down3 = nn.Sequential(*self.down3)
        self.down4 = nn.Sequential(*self.down4)
        self.down5 = nn.Sequential(*self.down5)
        self.down6 = nn.Sequential(*self.down6)
        self.down7 = nn.Sequential(*self.down7)

        self.up1 = nn.ConvTranspose2d(320, 96, kernel_size=2, stride=2)
        self.up1a = OneConv(96, 96)
        self.up2 = nn.ConvTranspose2d(192, 40, kernel_size=2, stride=2)
        self.up2a = OneConv(40, 40)
        self.up3 = nn.ConvTranspose2d(80, 24, kernel_size=2, stride=2)
        self.up3a = OneConv(24, 24)
        self.up4 = nn.ConvTranspose2d(48, 16, kernel_size=2, stride=2)
        self.up4a = OneConv(16, 16)
        self.up5 = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)

    def forward(self, x):
        # x_rgb = x.repeat(1, 3, 1, 1)
        # x = self.features(x_rgb)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x = self.down7(x6)
        x = self.up1(x)
        x = self.up1a(x)
        x = self.up2(torch.cat([x, x5], dim=1))
        x = self.up2a(x)
        x = self.up3(torch.cat([x, x3], dim=1))
        x = self.up3a(x)
        x = self.up4(torch.cat([x, x2], dim=1))
        x = self.up4a(x)
        x = self.up5(torch.cat([x, x1], dim=1))
        # x = self.sigmoid(x)
        return x

        # x = x.view(-1, self.last_channel)
        # x = self.classifier(x)
        # return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            # else:   #   up branch
            #     m.weight.data.normal_(0, math.sqrt(2. / n))


class CustomMNasNetUnetInvResDec(nn.Module):
    def __init__(self, in_channels, out_channels, input_size=224, width_mult=1.):

        super(CustomMNasNetUnetInvResDec, self).__init__()

        # setting of inverted residual blocks
        self.inverted_residual_setting = [
            # t, c, n, s, k
            [3, 24, 3, 2, 3],  # -> 56x56
            [3, 40, 3, 2, 5],  # -> 28x28
            [6, 80, 3, 2, 5],  # -> 14x14
            [6, 96, 2, 1, 3],  # -> 14x14
            [6, 192, 4, 2, 5],  # -> 7x7
            [6, 40, 1, 1, 3],  # -> 7x7
        ]

        assert input_size % 32 == 0

        self.dropout = nn.Dropout(p=drop_out)  # Dropout layer defined once

        input_channel = 16
        self.num_of_blocks = len(self.inverted_residual_setting)
        index = 0
        self.down1 = [Conv_3x3(1, input_channel, 2), SepConv_3x3(input_channel, 16)]
        self.down2 = []
        self.down3 = []
        self.down4 = []
        self.down5 = []
        self.down6 = []
        self.down7 = []
        for t, c, n, s, k in self.inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if index == 0:
                    if i == 0:
                        if use_se:
                            self.down2.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k, False))
                        else:
                            self.down2.append(InvertedResidual(input_channel, output_channel, s, t, k, False))
                    else:
                        if use_se:
                            self.down2.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k, False))
                        else:
                            self.down2.append(InvertedResidual(input_channel, output_channel, 1, t, k, False))
                elif index == 1:
                    if i == 0:
                        if use_se:
                            self.down3.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k, False))
                        else:
                            self.down3.append(InvertedResidual(input_channel, output_channel, s, t, k, False))
                    else:
                        if use_se:
                            self.down3.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k, False))
                        else:
                            self.down3.append(InvertedResidual(input_channel, output_channel, 1, t, k, False))
                elif index == 2:
                    if i == 0:
                        if use_se:
                            self.down4.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k, False))
                        else:
                            self.down4.append(InvertedResidual(input_channel, output_channel, s, t, k, False))
                    else:
                        if use_se:
                            self.down4.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k, False))
                        else:
                            self.down4.append(InvertedResidual(input_channel, output_channel, 1, t, k, False))
                elif index == 3:
                    if i == 0:
                        if use_se:
                            self.down5.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k, False))
                        else:
                            self.down5.append(InvertedResidual(input_channel, output_channel, s, t, k, False))
                    else:
                        if use_se:
                            self.down5.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k, False))
                        else:
                            self.down5.append(InvertedResidual(input_channel, output_channel, 1, t, k, False))
                elif index == 4:
                    if i == 0:
                        if use_se:
                            self.down6.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k, False))
                        else:
                            self.down6.append(InvertedResidual(input_channel, output_channel, s, t, k, False))
                    else:
                        if use_se:
                            self.down6.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k, False))
                        else:
                            self.down6.append(InvertedResidual(input_channel, output_channel, 1, t, k, False))
                else:
                    if i == 0:
                        if use_se:
                            self.down7.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k, False))
                        else:
                            self.down7.append(InvertedResidual(input_channel, output_channel, s, t, k, False))
                    else:
                        if use_se:
                            self.down7.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k, False))
                        else:
                            self.down7.append(InvertedResidual(input_channel, output_channel, 1, t, k, False))

                input_channel = output_channel
            index += 1

        self.down1 = nn.Sequential(*self.down1)
        self.down2 = nn.Sequential(*self.down2)
        self.down3 = nn.Sequential(*self.down3)
        self.down4 = nn.Sequential(*self.down4)
        self.down5 = nn.Sequential(*self.down5)
        self.down6 = nn.Sequential(*self.down6)
        self.down7 = nn.Sequential(*self.down7)

        self.up1 = nn.ConvTranspose2d(40, 24, kernel_size=3, stride=2, padding=1, output_padding=1)
        if use_se:
            self.up1a = InvertedResidualWithSE(120, 40, 1, 3, 3, True)
        else:
            self.up1a = InvertedResidual(120, 40, 1, 3, 3, True)
        self.up2 = nn.ConvTranspose2d(40, 24, kernel_size=3, stride=2, padding=1, output_padding=1)
        if use_se:
            self.up2a = InvertedResidualWithSE(64, 24, 1, 3, 3, True)
        else:
            self.up2a = InvertedResidual(64, 24, 1, 3, 3, True)
        self.up3 = nn.ConvTranspose2d(24, 12, kernel_size=3, stride=2, padding=1, output_padding=1)
        if use_se:
            self.up3a = InvertedResidualWithSE(36, 12, 1, 3, 3, True)
        else:
            self.up3a = InvertedResidual(36, 12, 1, 3, 3, True)
        self.up4 = nn.ConvTranspose2d(12, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        if use_se:
            self.up4a = InvertedResidualWithSE(24, 8, 1, 3, 3, True)
        else:
            self.up4a = InvertedResidual(24, 8, 1, 3, 3, True)
        self.up5 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # x_rgb = x.repeat(1, 3, 1, 1)
        # x = self.features(x_rgb)
        x1 = self.down1(x)
        x1 = self.dropout(x1)  # Dropout applied
        x2 = self.down2(x1)
        x2 = self.dropout(x2)  # Dropout applied
        x3 = self.down3(x2)
        x3 = self.dropout(x3)  # Dropout applied
        x4 = self.down4(x3)
        x4 = self.dropout(x4)  # Dropout applied
        x5 = self.down5(x4)
        x5 = self.dropout(x5)  # Dropout applied
        x6 = self.down6(x5)
        x6 = self.dropout(x6)  # Dropout applied
        x = self.down7(x6)
        x = self.dropout(x)  # Dropout applied
        x = self.up1(x)
        x = torch.cat([x, x5], dim=1)
        x = self.dropout(x)  # Dropout applied
        x = self.up1a(x)
        x = self.dropout(x)  # Dropout applied
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dropout(x)  # Dropout applied
        x = self.up2a(x)
        x = self.dropout(x)  # Dropout applied
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dropout(x)  # Dropout applied
        x = self.up3a(x)
        x = self.dropout(x)  # Dropout applied
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dropout(x)  # Dropout applied
        x = self.up4a(x)
        x = self.dropout(x)  # Dropout applied
        x = self.up5(x)
        return x


class CustomMNasNetUnetInvResDecArch3(nn.Module):
    def __init__(self, in_channels, out_channels, input_size=224, width_mult=1.):
        super(CustomMNasNetUnetInvResDecArch3, self).__init__()

        #   Original
        self.inverted_residual_setting = [
            # t, c, n, s, k
            [3, 12, 3, 2, 3],  # -> 56x56
            [3, 24, 3, 2, 5],  # -> 28x28
            [6, 80, 3, 2, 5],  # -> 14x14
            [6, 96, 2, 1, 3],  # -> 14x14
            [6, 192, 4, 2, 5],  # -> 7x7
            [6, 40, 1, 1, 3],  # -> 7x7
        ]
        # self.up1 = nn.ConvTranspose2d(40, 24, kernel_size=3, stride=2, padding=1, output_padding=1)
        #   End original

        # setting of inverted residual blocks
        #   less layers but greater width
        # self.inverted_residual_setting = [
        #     # t, c, n, s, k
        #     [3, 24, 3, 2, 3],  # -> 56x56
        #     [3, 40, 3, 2, 5],  # -> 28x28
        #     [4, 80, 3, 2, 5],  # -> 14x14
        #     [4, 96, 2, 1, 3],  # -> 14x14
        #     [5, 224, 4, 2, 5],  # -> 7x7
        #     [5, 80, 1, 1, 3],  # -> 7x7
        # ]
        # self.up1 = nn.ConvTranspose2d(80, 24, kernel_size=3, stride=2, padding=1, output_padding=1)
        #   more layers, less width
        # self.inverted_residual_setting = [
        #     # t, c, n, s, k
        #     [4, 24, 3, 2, 3],  # -> 56x56
        #     [4, 40, 3, 2, 5],  # -> 28x28
        #     [7, 60, 3, 2, 5],  # -> 14x14
        #     [7, 96, 2, 1, 3],  # -> 14x14
        #     [8, 168, 4, 2, 5],  # -> 7x7
        #     [8, 40, 1, 1, 3],  # -> 7x7
        # ]

        assert input_size % 32 == 0

        self.dropout = nn.Dropout(p=drop_out)  # Dropout layer defined once

        input_channel = 16
        self.num_of_blocks = len(self.inverted_residual_setting)
        index = 0
        self.down1 = [Conv_3x3(1, input_channel, 2), SepConv_3x3(input_channel, int(16 * width_mult))]
        self.down2 = []
        self.down3 = []
        self.down4 = []
        self.down5 = []
        self.down6 = []
        self.down7 = []
        for t, c, n, s, k in self.inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if index == 0:
                    if i == 0:
                        if use_se:
                            self.down2.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k, False))
                        else:
                            self.down2.append(InvertedResidual(input_channel, output_channel, s, t, k, False))
                    else:
                        if use_se:
                            self.down2.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k, False))
                        else:
                            self.down2.append(InvertedResidual(input_channel, output_channel, 1, t, k, False))
                elif index == 1:
                    if i == 0:
                        if use_se:
                            self.down3.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k, False))
                        else:
                            self.down3.append(InvertedResidual(input_channel, output_channel, s, t, k, False))
                    else:
                        if use_se:
                            self.down3.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k, False))
                        else:
                            self.down3.append(InvertedResidual(input_channel, output_channel, 1, t, k, False))
                elif index == 2:
                    if i == 0:
                        if use_se:
                            self.down4.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k, False))
                        else:
                            self.down4.append(InvertedResidual(input_channel, output_channel, s, t, k, False))
                    else:
                        if use_se:
                            self.down4.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k, False))
                        else:
                            self.down4.append(InvertedResidual(input_channel, output_channel, 1, t, k, False))
                elif index == 3:
                    if i == 0:
                        if use_se:
                            self.down5.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k, False))
                        else:
                            self.down5.append(InvertedResidual(input_channel, output_channel, s, t, k, False))
                    else:
                        if use_se:
                            self.down5.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k, False))
                        else:
                            self.down5.append(InvertedResidual(input_channel, output_channel, 1, t, k, False))
                elif index == 4:
                    if i == 0:
                        if use_se:
                            self.down6.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k, False))
                        else:
                            self.down6.append(InvertedResidual(input_channel, output_channel, s, t, k, False))
                    else:
                        if use_se:
                            self.down6.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k, False))
                        else:
                            self.down6.append(InvertedResidual(input_channel, output_channel, 1, t, k, False))
                else:
                    if i == 0:
                        if use_se:
                            self.down7.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k, False))
                        else:
                            self.down7.append(InvertedResidual(input_channel, output_channel, s, t, k, False))
                    else:
                        if use_se:
                            self.down7.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k, False))
                        else:
                            self.down7.append(InvertedResidual(input_channel, output_channel, 1, t, k, False))

                input_channel = output_channel
            index += 1

        self.down1 = nn.Sequential(*self.down1)
        self.down2 = nn.Sequential(*self.down2)
        self.down3 = nn.Sequential(*self.down3)
        self.down4 = nn.Sequential(*self.down4)
        self.down5 = nn.Sequential(*self.down5)
        self.down6 = nn.Sequential(*self.down6)
        self.down7 = nn.Sequential(*self.down7)

        self.up1 = nn.ConvTranspose2d(40, 40, kernel_size=3, stride=2, padding=1, output_padding=1)
        if use_se:
            self.up1a = InvertedResidualWithSE(136, 40, 1, 3, 3, True)
        else:
            self.up1a = InvertedResidual(136, 40, 1, 3, 3, True)
        self.up2 = nn.ConvTranspose2d(40, 56, kernel_size=3, stride=2, padding=1, output_padding=1)
        if use_se:
            self.up2a = InvertedResidualWithSE(80, 24, 1, 3, 3, True)
        else:
            self.up2a = InvertedResidual(80, 24, 1, 3, 3, True)
        self.up3 = nn.ConvTranspose2d(24, 40, kernel_size=3, stride=2, padding=1, output_padding=1)
        if use_se:
            self.up3a = InvertedResidualWithSE(52, 12, 1, 3, 3, True)
        else:
            self.up3a = InvertedResidual(52, 12, 1, 3, 3, True)
        self.up4 = nn.ConvTranspose2d(12, 24, kernel_size=3, stride=2, padding=1, output_padding=1)
        if use_se:
            self.up4a = InvertedResidualWithSE(40, 48, 1, 3, 3, True)
        else:
            self.up4a = InvertedResidual(40, 48, 1, 3, 3, True)
        self.up5 = nn.ConvTranspose2d(48, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

        # xavier_init(self.down1[0])
        # xavier_init(self.down1[1])
        # for layer in self.down2:
        #     xavier_init(layer)
        # for layer in self.down3:
        #     xavier_init(layer)
        # for layer in self.down4:
        #     xavier_init(layer)
        # for layer in self.down5:
        #     xavier_init(layer)
        # for layer in self.down6:
        #     xavier_init(layer)
        # for layer in self.down7:
        #     xavier_init(layer)
        # xavier_init(self.up1)
        # xavier_init(self.up1a)
        # xavier_init(self.up2)
        # xavier_init(self.up2a)
        # xavier_init(self.up3)
        # xavier_init(self.up3a)
        # xavier_init(self.up4)
        # xavier_init(self.up4a)
        # xavier_init(self.up5)

    def forward(self, x):
        # x_rgb = x.repeat(1, 3, 1, 1)
        # x = self.features(x_rgb)
        x1 = self.down1(x)
        x1 = self.dropout(x1)  # Dropout applied
        x2 = self.down2(x1)
        x2 = self.dropout(x2)  # Dropout applied
        x3 = self.down3(x2)
        x3 = self.dropout(x3)  # Dropout applied
        x4 = self.down4(x3)
        x4 = self.dropout(x4)  # Dropout applied
        x5 = self.down5(x4)
        x5 = self.dropout(x5)  # Dropout applied
        x6 = self.down6(x5)
        x6 = self.dropout(x6)  # Dropout applied
        x = self.down7(x6)
        x = self.dropout(x)  # Dropout applied
        x = self.up1(x)
        x = torch.cat([x, x5], dim=1)
        x = self.dropout(x)  # Dropout applied
        x = self.up1a(x)
        x = self.dropout(x)  # Dropout applied
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dropout(x)  # Dropout applied
        x = self.up2a(x)
        x = self.dropout(x)  # Dropout applied
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dropout(x)  # Dropout applied
        x = self.up3a(x)
        x = self.dropout(x)  # Dropout applied
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dropout(x)  # Dropout applied
        x = self.up4a(x)
        x = self.dropout(x)  # Dropout applied
        x = self.up5(x)
        return x


# def xavier_init(layer):
#     if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
#         init.xavier_normal_(layer.weight)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F1.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Intersection over Union (IoU) loss function
class IoULoss(nn.Module):
    def __init__(self, smooth=1):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, target):
        intersection = torch.sum(predicted * target, dim=(2, 3))
        predicted_sum = torch.sum(predicted, dim=(2, 3))
        target_sum = torch.sum(target, dim=(2, 3))
        if target_sum == 0:
            aa = 0
        union = torch.sum(predicted, dim=(2, 3)) + torch.sum(target, dim=(2, 3)) - intersection
        if union == 0:
            iou = 1
        else:
            iou = intersection / union  # Adding a small epsilon to avoid division by zero
        loss = 1 - iou
        mean_loss = loss.mean()
        # is_greater_than_one_p = torch.any(predicted > 1)
        # is_greater_than_one_m = torch.any(target > 1)
        # max_val = torch.max(target)
        # if mean_loss < 0:
        #     aa = 0

        return mean_loss


class CustomSubset(Subset):
    def __init__(self, dataset, indices, is_train=True):
        super().__init__(dataset, indices)
        self.is_train = is_train
        self.do_flip = dataset.do_flip
        self.do_rotate = dataset.do_rotate

    # def __getitem__(self, idx):
    #     image, mask = super().__getitem__(idx)
    #     if self.is_train:
    #         if self.do_flip:
    #             flipped_image = transforms.RandomHorizontalFlip(1)(image)
    #             flipped_mask = transforms.RandomHorizontalFlip(1)(mask)
    #
    #         if self.do_rotate:
    #             angles = [90, 180, -90]
    #             rand_ind = (random.randint(0, 1000)) % 3
    #             rand_angle = angles[rand_ind]
    #             coef = max(abs(math.sin(math.radians(rand_angle + 45))),
    #                        abs(math.sin(math.radians(90 - rand_angle - 45))))
    #             coef = math.sin(math.radians(45)) / coef
    #
    #             # rand_angle = 0
    #             # while rand_angle == 0:
    #             #     rand_angle = float(random.randint(0, 180) - 90)
    #             #     coef = max(abs(math.sin(math.radians(rand_angle + 45))), abs(math.sin(math.radians(90 - rand_angle - 45))))
    #             #     coef = math.sin(math.radians(45)) / coef
    #             downscale_size = (int(image.shape[1] * coef + 0.5), int(image.shape[2] * coef + 0.5))
    #             downscaled_image = F.resize(image, downscale_size, antialias=True)
    #             downscaled_mask = F.resize(mask, downscale_size, antialias=True)
    #             # Create an empty tensor of size 224x224
    #             output_size = (image.shape[1], image.shape[2])
    #             rotated_image = torch.zeros(1, output_size[0],
    #                                         output_size[1])  # Assuming RGB image, adjust channels if needed
    #             rotated_mask = torch.zeros(1, output_size[0],
    #                                        output_size[1])  # Assuming RGB image, adjust channels if needed
    #             # Compute the starting position for inserting the downscaled image
    #             start_x = (output_size[0] - downscaled_image.size(1)) // 2
    #             start_y = (output_size[1] - downscaled_image.size(2)) // 2
    #             # Insert the downscaled image into the rotated image tensor
    #             rotated_image[0, start_x: start_x + downscaled_image.size(1),
    #             start_y: start_y + downscaled_image.size(2)] = downscaled_image
    #             rotated_mask[0, start_x: start_x + downscaled_image.size(1),
    #             start_y: start_y + downscaled_image.size(2)] = downscaled_mask
    #
    #             # Rotate the image
    #             rotated_image = F.rotate(rotated_image, rand_angle)
    #             rotated_mask = F.rotate(rotated_mask, rand_angle)
    #             rotated_mask = torch.where(rotated_mask < 0.5, torch.tensor(0), torch.tensor(1))
    #
    #             # image1 = image * 255
    #             # image_s = image1.squeeze().detach().numpy()
    #             # rotated_image1 = rotated_image * 255
    #             # rotated_image_s = rotated_image1.squeeze().detach().numpy()
    #             #
    #             # mask1 = mask * 255
    #             # mask_s = mask1.squeeze().detach().numpy()
    #             # rotated_mask1 = rotated_mask * 255
    #             # rotated_mask_s = rotated_mask1.squeeze().detach().numpy()
    #             #
    #             # cv2.imshow("i", image_s.astype(np.uint8))
    #             # cv2.imshow("ri", rotated_image_s.astype(np.uint8))
    #             # cv2.imshow("m", mask_s.astype(np.uint8))
    #             # cv2.imshow("rm", rotated_mask_s.astype(np.uint8))
    #             # cv2.waitKey(0)
    #             # cv2.destroyAllWindows()
    #             # aa = 0
    #
    #         if self.do_flip and self.do_rotate:
    #             concatenated_images = torch.cat((image, flipped_image, rotated_image), dim=0)
    #             concatenated_masks = torch.cat((mask, flipped_mask, rotated_mask), dim=0)
    #             return concatenated_images, concatenated_masks
    #         elif self.do_flip:
    #             concatenated_images = torch.cat((image, flipped_image), dim=0)
    #             concatenated_masks = torch.cat((mask, flipped_mask), dim=0)
    #             return concatenated_images, concatenated_masks
    #         elif self.do_rotate:
    #             concatenated_images = torch.cat((image, rotated_image), dim=0)
    #             concatenated_masks = torch.cat((mask, rotated_mask), dim=0)
    #             return concatenated_images, concatenated_masks
    #         else:
    #             return image, mask
    #     else:
    #         return image, mask


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, do_flip, do_rotate, do_noise_reduction, transform_c=None, mask_transform=None):
        self.img_masks = create_directory_dictionary(img_dir)  # dictionary
        self.img_dir = img_dir
        self.transform = transform_c
        self.mask_transform = mask_transform
        self.do_flip = do_flip
        self.do_rotate = do_rotate
        self.do_noise_reduction = do_noise_reduction

    def __len__(self):
        return len(self.img_masks)

    def __getitem__(self, idx):
        img_path, mask_path = list(self.img_masks.items())[idx]
        img_path = os.path.join(self.img_dir, img_path)
        mask_path = os.path.join(self.img_dir, mask_path)
        image = read_image(img_path)
        if self.do_noise_reduction:
            image_np = image.numpy()
            # Transpose channels to match the expected format (height, width, channels)
            image_np = np.transpose(image_np, (1, 2, 0))
            # Apply Gaussian noise reduction using OpenCV
            kernel_size = (5, 5)
            blurred_image = cv2.GaussianBlur(image_np, kernel_size, 0)
            # Transpose back to the original tensor format (channels, height, width)
            blurred_image = np.transpose(blurred_image, (2, 0, 1))
            # Convert NumPy array back to torch tensor
            image = torch.from_numpy(blurred_image)

        mask = read_image(mask_path)
        image = image.float()
        mask = mask.float()
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask /= 255
        mask = torch.where(mask < 0.5, torch.tensor(0), torch.tensor(1))
        return image, mask


class FlippedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        images, labels = self.dataset[index]
        # flipped_images = torch.flip(images, dims=[3])  # Flip along the horizontal axis
        flipped_images = transforms.RandomHorizontalFlip(1)(images)
        # return images, flipped_images, labels
        return flipped_images, labels

    def __len__(self):
        return len(self.dataset)


def custom_augmentation(image_mask, flip, rotate, dark, b_noise, transforms1):
    images = image_mask[0]
    masks = image_mask[1]
    output_images = []
    output_masks = []
    for ind in range(images.shape[0]):
        image = images[ind]
        mask = masks[ind]
        images_list = []
        masks_list = []
        images_list.append(image)
        masks_list.append(mask)
        if dark:
            image_dark1 = image * 0.5
            images_list.append(image_dark1)
            masks_list.append(mask)
            image_dark2 = image * 0.3
            images_list.append(image_dark2)
            masks_list.append(mask)
            # image_dark3 = image * 1.4
            # image_dark3 = torch.clamp(image_dark3, 0, 1)
            # images_list.append(image_dark3)
            # masks_list.append(mask)
        if b_noise:
            mean = 0.0
            std_dev = 0.07
            # Generate Gaussian noise with the same shape as the original tensor
            noise = torch.randn_like(image) * std_dev + mean
            noisy_tensor = image + noise
            noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
            images_list.append(noisy_tensor)
            masks_list.append(mask)
        if flip:
            flipped_image = transforms1.RandomHorizontalFlip(1)(image)
            flipped_mask = transforms1.RandomHorizontalFlip(1)(mask)
            images_list.append(flipped_image)
            masks_list.append(flipped_mask)
        if rotate:
            angles = [90, 180, -90, 45, -45]
            # rand_angle1 = (random.randint(1, 179)) - 90
            # angles.append(rand_angle1)
            # rand_angle2 = (random.randint(1, 179)) - 90
            # angles.append(rand_angle2)
            for rand_angle in angles:
                # rand_ind = (random.randint(0, 1000)) % 3
                # rand_angle = angles[rand_ind]
                coef = max(abs(math.sin(math.radians(rand_angle + 45))),
                           abs(math.sin(math.radians(90 - rand_angle - 45))))
                coef = math.sin(math.radians(45)) / coef

                downscale_size = (int(image.shape[1] * coef + 0.5), int(image.shape[2] * coef + 0.5))
                downscaled_image = F.resize(image, downscale_size, antialias=True)
                downscaled_mask = F.resize(mask, downscale_size, antialias=True)
                # Create an empty tensor of size 224x224
                output_size = (image.shape[1], image.shape[2])
                rotated_image = torch.zeros(1, output_size[0],
                                            output_size[1])  # Assuming RGB image, adjust channels if needed
                rotated_mask = torch.zeros(1, output_size[0],
                                           output_size[1])  # Assuming RGB image, adjust channels if needed
                # Compute the starting position for inserting the downscaled image
                start_x = (output_size[0] - downscaled_image.size(1)) // 2
                start_y = (output_size[1] - downscaled_image.size(2)) // 2
                # Insert the downscaled image into the rotated image tensor
                rotated_image[0, start_x: start_x + downscaled_image.size(1),
                start_y: start_y + downscaled_image.size(2)] = downscaled_image
                rotated_mask[0, start_x: start_x + downscaled_image.size(1),
                start_y: start_y + downscaled_image.size(2)] = downscaled_mask

                # Rotate the image
                rotated_image = F.rotate(rotated_image, rand_angle)
                rotated_mask = F.rotate(rotated_mask, rand_angle)
                rotated_mask = torch.where(rotated_mask < 0.5, torch.tensor(0), torch.tensor(1))
                images_list.append(rotated_image)
                masks_list.append(rotated_mask)

        concatenated_images = torch.cat(images_list, dim=0)
        concatenated_masks = torch.cat(masks_list, dim=0)
        output_images.append(concatenated_images)
        output_masks.append(concatenated_masks)

    output_images1 = torch.stack(output_images, dim=0)
    output_masks1 = torch.stack(output_masks, dim=0)
    return output_images1, output_masks1


def chat_gpt_training(device_c, train_dataloader_c, valid_dataloader_c, num_epochs, path_to_save_model, csv_file_name):
    # Assuming your input images have 1 channels (RGB) and size 640x480
    # Define the input channels and output channels (classes)
    in_channels = 1
    out_channels = 1  # Assuming binary segmentation
    # model_c = CustomMNasNetUnet(in_channels, out_channels)
    model_c = CustomMNasNetUnetInvResDec(in_channels, out_channels, 224, 1)
    # model_c = UNet(in_channels, out_channels)
    # model_c = UNetWithMnasNetEncoder(num_classes=1)  # Adjust num_classes according to your task
    if os.path.exists(path_to_save_model):
        model_c.load_state_dict(torch.load(path_to_save_model, map_location='cpu'))
        model_c.to(device_c)
        # model_c.eval()
        print("Used previously trained model ", path_to_save_model)

    # Print the model architecture
    # print(model_c)

    # parameters = list(model_c.parameters())
    total_params = sum(p.numel() for p in model_c.parameters())
    print(f"Number of parameters: {total_params}")
    # Define your loss function and optimizer
    # Create an instance of the IoU loss
    # criterion = IoULoss()
    criterion = nn.BCELoss()
    # criterion = FocalLoss(gamma=2)
    # global pos_weight
    # pos_weight1 = torch.tensor(pos_weight)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight1)

    global learning_rate
    optimizer_c = optim.Adam(model_c.parameters(), lr=learning_rate)    #   , weight_decay=0.0002)
    # optimizer_c = optim.SGD(model_c.parameters(), lr=learning_rate)
    # optimizer_c = optim.RMSprop(model_c.parameters(), lr=learning_rate)

    print('learning_rate = ', learning_rate)

    model_c.to(device_c)

    # Training loop
    best_loss = float('inf')  # Initialize with a large value
    # best_loss = 0.3687  # Initialize with a large value
    # best_accuracy = 0.
    best_loss_epoch = -1
    # best_accuracy_epoch = -1
    patience = 10  # Number of epochs to wait before early stopping
    counter = 0  # Counter for early stopping

    with open(csv_file_name, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["Epoch", "Training loss", "Validation loss"])
        for epoch in range(num_epochs):
            model_c.train()
            train_loss = 0.0
            number_of_train_data = 0
            steps = 0
            for i, images_and_masks in enumerate(train_dataloader_c):
                steps += 1
                images, masks = custom_augmentation(images_and_masks, do_train_flip1, do_train_rotate1, do_train_dark1, do_train_noise1, transforms)
                # images = images_and_masks[0]
                images = images.view(-1, 1, images.shape[2], images.shape[3])
                # masks = images_and_masks[1]
                masks = masks.view(-1, 1, masks.shape[2], masks.shape[3])
                b_show_t = False
                if b_show_t:
                    or_image = images[10].squeeze(1).cpu().numpy().squeeze(0) * 255
                    dark_image1 = images[11].squeeze(1).cpu().numpy().squeeze(0) * 255
                    dark_image2 = images[12].squeeze(1).cpu().numpy().squeeze(0) * 255
                    dark_image3 = images[13].squeeze(1).cpu().numpy().squeeze(0) * 255
                    noise_image = images[14].squeeze(1).cpu().numpy().squeeze(0) * 255
                    flip_image = images[15].squeeze(1).cpu().numpy().squeeze(0) * 255
                    rot_image1 = images[16].squeeze(1).cpu().numpy().squeeze(0) * 255
                    rot_image2 = images[17].squeeze(1).cpu().numpy().squeeze(0) * 255
                    rot_image3 = images[18].squeeze(1).cpu().numpy().squeeze(0) * 255
                    rot_image4 = images[19].squeeze(1).cpu().numpy().squeeze(0) * 255
                    or_mask = masks[10].squeeze(1).cpu().numpy().squeeze(0) * 255
                    dark_mask1 = masks[11].squeeze(1).cpu().numpy().squeeze(0) * 255
                    dark_mask2 = masks[12].squeeze(1).cpu().numpy().squeeze(0) * 255
                    dark_mask3 = masks[13].squeeze(1).cpu().numpy().squeeze(0) * 255
                    noise_mask = masks[14].squeeze(1).cpu().numpy().squeeze(0) * 255
                    flip_mask = masks[15].squeeze(1).cpu().numpy().squeeze(0) * 255
                    rot_mask1 = masks[16].squeeze(1).cpu().numpy().squeeze(0) * 255
                    rot_mask2 = masks[17].squeeze(1).cpu().numpy().squeeze(0) * 255
                    rot_mask3 = masks[18].squeeze(1).cpu().numpy().squeeze(0) * 255
                    rot_mask4 = masks[19].squeeze(1).cpu().numpy().squeeze(0) * 255
                    cv2.imshow("or", or_image.astype(np.uint8))
                    cv2.imshow("dark1", dark_image1.astype(np.uint8))
                    cv2.imshow("dark2", dark_image2.astype(np.uint8))
                    cv2.imshow("dark3", dark_image3.astype(np.uint8))
                    cv2.imshow("noise", noise_image.astype(np.uint8))
                    cv2.imshow("flip", flip_image.astype(np.uint8))
                    cv2.imshow("rot1", rot_image1.astype(np.uint8))
                    cv2.imshow("rot2", rot_image2.astype(np.uint8))
                    cv2.imshow("rot3", rot_image3.astype(np.uint8))
                    cv2.imshow("rot4", rot_image4.astype(np.uint8))
                    cv2.imshow("or mask", or_mask.astype(np.uint8))
                    cv2.imshow("dark mask1", dark_mask1.astype(np.uint8))
                    cv2.imshow("dark mask2", dark_mask2.astype(np.uint8))
                    cv2.imshow("dark mask3", dark_mask3.astype(np.uint8))
                    cv2.imshow("noise mask", noise_mask.astype(np.uint8))
                    cv2.imshow("flip  mask", flip_mask.astype(np.uint8))
                    cv2.imshow("rot mask1", rot_mask1.astype(np.uint8))
                    cv2.imshow("rot mask2", rot_mask2.astype(np.uint8))
                    cv2.imshow("rot mask3", rot_mask3.astype(np.uint8))
                    cv2.imshow("rot mask4", rot_mask4.astype(np.uint8))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                # Forward pass
                images = images.to(device_c)
                masks = masks.to(device_c)
                masks = masks.float()
                outputs1 = model_c(images)
                outputs = torch.sigmoid(outputs1)
                # pos_weight2 = torch.where(masks == 1, torch.tensor(pos_weight), torch.tensor(1.0))
                # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight2)
                loss = criterion(outputs, masks)

                # Backward pass and optimization
                optimizer_c.zero_grad()
                # loss.requires_grad = True
                loss.backward()
                optimizer_c.step()
                # Compute the total loss
                train_loss += loss.item() * images.shape[0]
                # Compute the accuracy
                # predicted = torch.round(outputs.data)
                # train_correct += (predicted == masks).sum().item()
                number_of_train_data += images.shape[0]  # batch size

            # Evaluate on the validation set
            model_c.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_correct = 0
                number_of_valid_data = 0
                for inputs_v, masks_v in valid_dataloader_c:
                    masks_v = masks_v.to(device_c)
                    masks_v = masks_v.float()

                    # Forward pass
                    inputs_v = inputs_v.to(device_c)
                    outputs1 = model_c(inputs_v)
                    outputs = torch.sigmoid(outputs1)

                    # pos_weight2 = torch.where(masks_v == 1, torch.tensor(pos_weight), torch.tensor(1.0))
                    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight2)
                    loss_v = criterion(outputs, masks_v)
                    val_loss += loss_v.item() * inputs_v.shape[0]
                    # Compute the accuracy
                    # predicted = torch.round(outputs.data)
                    # val_correct += (predicted == labels_v).sum().item()
                    number_of_valid_data += inputs_v.shape[0]  # batch size

            if (epoch % 10) == 0:
                print(number_of_train_data, number_of_valid_data)

            train_loss /= number_of_train_data  # Because of flipping and rotation  * 3
            # train_accuracy = train_correct / number_of_train_data  # Because of flipping and rotation  * 3
            val_loss /= number_of_valid_data
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
            # val_accuracy = val_correct / number_of_valid_data  # Because of flipping and rotation  * 3
            print(f"Epoch {epoch}/{num_epochs} steps {steps} - Training Loss: {train_loss:.6f} - "
                  f"Validation Loss: {val_loss:.6f} - "
                  f"{formatted_datetime}"
                  )
            data = [epoch, train_loss, val_loss]
            csv_writer.writerow(data)

            # Save the best model based on validation loss
            if val_loss < best_loss:
                best_loss = val_loss
                best_loss_epoch = epoch
                torch.save(model_c.state_dict(), path_to_save_model)
                # torch.save(model_c, path_to_save_model)
                counter = 0
            else:
                counter += 1
            # if val_accuracy > best_accuracy:
            #     best_accuracy = val_accuracy
            #     best_accuracy_epoch = epoch
            # Check for early stopping
            if counter >= patience:
                if learning_rate < 0.0001:
                    print("Validation loss has not improved significantly. Stopping training.")
                    break
                else:
                    learning_rate = learning_rate / 4
                    model_c.load_state_dict(torch.load(path_to_save_model, map_location='cpu'))
                    model_c.to(device_c)
                    model_c.eval()
                    optimizer_c = optim.Adam(model_c.parameters(), lr=learning_rate)  # , weight_decay=0.001)
                    counter = 0
                    print('learning_rate = ', learning_rate)
            # Save the trained model
            # name, extension = os.path.splitext(path_to_save_model)
            # name_per_epoch = name + '_' + str(epoch) + extension
            # torch.save(model_c.state_dict(), name_per_epoch)
            model_c.train()
    print(f"The best loss was in epoch {best_loss_epoch} and it is {best_loss:.4f}")
    # print(f"The best accuracy was in epoch {best_accuracy_epoch} and it is {best_accuracy:.4f}")
    return model_c


def chat_gpt_test(device_c, test_dataloader, model_t):
    # Evaluate on the test set
    # model_t.eval()
    # criterion = nn.MSELoss()
    # criterion = IoULoss()
    criterion = nn.BCELoss()

    with torch.no_grad():
        test_loss = 0.0
        for inputs_t, labels_t in test_dataloader:
            inputs_t = inputs_t.to(device_c)
            labels_t = labels_t.to(device_c)
            # labels_t = labels_t.unsqueeze(1)

            outputs = model_t(inputs_t)
            output = torch.sigmoid(outputs)
            loss_t = criterion(output, labels_t.float())
            test_loss += loss_t.item()

    # Print training and validation losses for the epoch
    print(f"Test Loss: {test_loss / len(test_dataloader):.4f}")
    aa = 0


def extract_parameter(filename):
    # Extract the parameter from the filename
    return int(filename.split('-')[1].split('_')[0])


def create_directory_dictionary(directory):
    # Create an empty dictionary
    directory_dict = {}

    # Iterate through the files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a JPG file
        # if filename.endswith('.jpg') and 'mask' not in filename:
        if filename.endswith('.jpg'):
            # Extract the parameter from the filename
            # name, extension = os.path.splitext(filename)
            # mask_filename = name + '_mask' + extension
            filename1 = filename
            mask_filename = filename1.replace('.jpg', '.png')

            # Add the filename and parameter to the dictionary
            directory_dict[filename] = mask_filename

    return directory_dict


# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    # images_batch, landmarks_batch = \
    #         sample_batched['image'], sample_batched['landmarks']
    images_batch, landmarks_batch = sample_batched[0], sample_batched[1]
    # batch_size = len(images_batch)
    # im_size = images_batch.size(2) * images_batch.size(3)
    # grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    # for i in range(batch_size):
    #     plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
    #                 landmarks_batch[i, :, 1].numpy() + grid_border_size,
    #                 s=10, marker='.', c='r')
    #
    #     plt.title('Batch from dataloader')


def load_split_train_test(train_data_dir, valid_data_dir, test_data_dir, train_valid_test_composed_transform,
                          mask_transform, batch_train_valid, batch_test, valid_size=.15):
    test_dataset = CustomImageDataset(test_data_dir, False, False, False, train_valid_test_composed_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_test)
    if valid_size > 0:  # take validation from training if valid_size > 0
        train_dataset_full = CustomImageDataset(train_data_dir, do_train_flip1, do_train_rotate1, do_noise_reduction1,
                                                train_valid_test_composed_transform, mask_transform)
        # Split the dataset into training and validation sets
        train_samples = len(train_dataset_full)
        train_indices, valid_indices = train_test_split(np.arange(train_samples), test_size=valid_size, random_state=42)

        # Create the training dataset and dataloader
        train_dataset = CustomSubset(train_dataset_full, train_indices, True)
        train_loader = DataLoader(train_dataset, batch_size=batch_train_valid, shuffle=True)

        # Create the validation dataset and dataloader
        valid_dataset = CustomSubset(train_dataset_full, valid_indices, False)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_train_valid, shuffle=False)
    else:  # separate validation dir
        train_dataset_full = CustomImageDataset(train_data_dir, do_train_flip1, do_train_rotate1, do_noise_reduction1,
                                                train_valid_test_composed_transform, mask_transform)
        train_indices = range(len(train_dataset_full))
        train_dataset = CustomSubset(train_dataset_full, train_indices, True)
        num_train = len(train_dataset_full)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_train_valid)
        valid_dataset_full = CustomImageDataset(valid_data_dir, False, False, do_noise_reduction1, train_valid_test_composed_transform,
                                                mask_transform)
        valid_indices = range(len(valid_dataset_full))
        valid_dataset = CustomSubset(valid_dataset_full, valid_indices, False)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_train_valid)

    return train_loader, valid_loader, test_loader


def filter_and_draw_bounding_boxes(predicted_mask_with_bb_np, image_original1, min_area=None, max_boxes=None):
    # Find connected components
    image_original = np.copy(image_original1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(predicted_mask_with_bb_np.astype(np.uint8))

    # Create a new mask to store the filtered components
    filtered_mask = np.zeros_like(predicted_mask_with_bb_np).astype(np.uint8)
    filtered_mask_with_bb = np.zeros_like(predicted_mask_with_bb_np).astype(np.uint8)

    # Create a list to store bounding boxes and areas
    bounding_boxes = []
    foot_points = []

    # Iterate through each component (excluding the background component)
    for label in range(1, num_labels):
        # Get the statistics of the component
        x, y, w, h, area = stats[label]

        # Check maximum number of boxes condition
        if max_boxes is not None and len(bounding_boxes) >= max_boxes:
            break

        # Check minimum area condition
        if min_area is not None and area < min_area:
            continue

        # Copy the component to the filtered mask
        component_mask = (labels == label).astype(np.uint8) * 255
        filtered_mask = cv2.bitwise_or(filtered_mask, component_mask)

        # Add the bounding box and area to the list
        bounding_boxes.append((x, y, w, h))
        foot_points.append((x + w // 2, y + h // 2))

    # Draw bounding boxes on the original image
    # result_image = filtered_mask.copy()
    for box in bounding_boxes:
        x, y, w, h = box
        cv2.rectangle(filtered_mask_with_bb, (x, y), (x + w, y + h), (255, 255, 255), -1)    #   -1 fills
        cv2.rectangle(image_original, (x, y), (x + w, y + h), (255, 255, 255), 1)

    return foot_points, bounding_boxes, filtered_mask_with_bb, filtered_mask, image_original


def find_binarization_threshold(probs):
    epsilon = 1e-6
    epsilon_for_equality = 1e-2
    lower_bound = 0.0
    upper_bound = 1.0
    target_sum = probs.sum()
    count = 0

    while upper_bound - lower_bound > epsilon and count < 100:
        threshold = (lower_bound + upper_bound) / 2
        binarized_image = (probs >= threshold).float()
        current_sum = binarized_image.sum()

        # if current_sum >= target_sum - epsilon_for_equality:
        if current_sum >= target_sum * 0.995:
            lower_bound = threshold
        else:
            upper_bound = threshold
        count += 1

    return (lower_bound + upper_bound) / 2


def find_binarization_threshold_min(probs):
    epsilon = 1e-6
    epsilon_for_equality = 1e-2
    lower_bound = 0.0
    upper_bound = 1.0
    target_sum = probs.sum()
    count = 0

    while upper_bound - lower_bound > epsilon and count < 100:
        threshold = (lower_bound + upper_bound) / 2
        binarized_image = (probs >= threshold).float()
        current_sum = binarized_image.sum()

        if current_sum <= target_sum + epsilon_for_equality:
            upper_bound = threshold
        else:
            lower_bound = threshold
        count += 1

    return (lower_bound + upper_bound) / 2


def find_average_prob_below_threshold(probs, threshold=0.5):
    # Convert probabilities to a boolean mask where True corresponds to probabilities less than 0.5
    mask_for_small_prob = probs < threshold
    # Calculate the sum and count of probabilities less than threshold
    sum_probabilities = torch.sum(probs[mask_for_small_prob])
    count_probabilities = torch.sum(mask_for_small_prob)
    # Calculate the average if there are probabilities less than 0.5
    average_small_probability = sum_probabilities / count_probabilities if count_probabilities > 0 else 0.0
    return average_small_probability.item()


def rotate_image(image, rand_angle=None):
    if rand_angle is None:
        rand_angle = 0
        while rand_angle == 0:
            rand_angle = float(random.randint(1, 179) - 90)
    # coef = max(abs(math.sin(math.radians(rand_angle))), abs(math.sin(math.radians(90 - rand_angle))))
    coef = max(abs(math.sin(math.radians(rand_angle + 45))), abs(math.sin(math.radians(90 - rand_angle - 45))))
    coef = math.sin(math.radians(45)) / coef

    downscale_size = (int(image.shape[2] * coef + 0.5), int(image.shape[3] * coef + 0.5))
    downscaled_image = F.resize(image, downscale_size, antialias=True)
    # Create an empty tensor of size 224x224
    output_size = (image.shape[2], image.shape[3])
    rotated_image = torch.zeros(1, output_size[0], output_size[1])  # Assuming RGB image, adjust channels if needed
    # Compute the starting position for inserting the downscaled image
    start_x = (output_size[0] - downscaled_image.size(2)) // 2
    start_y = (output_size[1] - downscaled_image.size(3)) // 2
    # Insert the downscaled image into the rotated image tensor
    rotated_image[0, start_x: start_x + downscaled_image.size(2),
    start_y: start_y + downscaled_image.size(3)] = downscaled_image[0, 0, :, :]

    # Rotate the image
    rotated_image = F.rotate(rotated_image, rand_angle)
    return rotated_image, rand_angle, coef


def rotate_image_back(image, angle, coef):
    rotated_back_ds_image = F.rotate(image, -angle)
    # coef = max(abs(math.sin(math.radians(angle))), abs(math.sin(math.radians(90 - angle))))
    downscale_size = (int(image.shape[2] * coef + 0.5), int(image.shape[3] * coef + 0.5))
    upscale_size = (int(image.shape[2]), int(image.shape[3] + 0.5))
    output_image = torch.zeros(1, 1, downscale_size[0],
                               downscale_size[1])  # Assuming RGB image, adjust channels if needed
    # Compute the starting position for inserting the downscaled image
    start_x = (upscale_size[0] - downscale_size[0]) // 2
    start_y = (upscale_size[1] - downscale_size[1]) // 2
    # Insert the downscaled image into the rotated image tensor
    output_image[0][0] = rotated_back_ds_image[0, 0, start_x: start_x + downscale_size[0],
                         start_y: start_y + downscale_size[1]]
    output_image = F.resize(output_image, upscale_size, antialias=True)
    return output_image


def iou_loss(predicted, target):
    intersection = np.sum(np.array(predicted) * np.array(target))
    predicted_sum = np.sum(predicted)
    target_sum = np.sum(target)
    if target_sum == 0:
        aa = 0
    union = np.sum(predicted) + np.sum(target) - intersection
    if union == 0:
        iou = 1
    else:
        iou = intersection / union  # Adding a small epsilon to avoid division by zero
    loss = 1 - iou

    return loss


def infer_one_frame1(filename_without_ext, directory_name, output_masks_dir, test_composed_transform, mask_transform,
                    model, device_i, count,
                    loss, loss_bin, csv_writer):
    b_write = False
    if not directory_name1:
        filename = filename_without_ext + '.jpg'
        full_frame_path = filename
    else:
        filename = filename_without_ext
        full_frame_path = os.path.join(directory_name, filename)
        name, extension = os.path.splitext(filename)
        filename_predicted_mask = name + '_predicted_mask' + extension
        filename_best_predicted_mask = name + '_best_predicted_mask' + extension
        full_output_predicted_mask_path = os.path.join(output_masks_dir, filename_predicted_mask)
        full_output_predicted_best_mask_path = os.path.join(output_masks_dir, filename_best_predicted_mask)
        # full_output_mask_path = os.path.join(output_masks_dir, filename)
        mask_filename = name + ".png"
        full_mask_path = os.path.join(directory_name, mask_filename)
        full_output_original_mask_path = os.path.join(output_masks_dir, mask_filename)
        mask_with_bb_filename = name + '_mask_bb' + extension
        full_mask_with_bb_path = os.path.join(output_masks_dir, mask_with_bb_filename)
        original_with_bb_filename = name + '_bb' + extension
        full_original_with_bb_path = os.path.join(output_masks_dir, original_with_bb_filename)
        original_filename = filename[:filename.find('-')] + filename[filename.find('-'):]
        full_original_path = os.path.join(output_masks_dir, original_filename)
    if os.path.exists(full_frame_path):
        count += 1
        if not directory_name:
            img = cv2.imread(full_frame_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(full_frame_path, img)  # save frame as png file

        frame = read_image(full_frame_path)
        frame = frame.float()

        mask_or = cv2.imread(full_mask_path)
        mask_or = cv2.cvtColor(mask_or, cv2.COLOR_BGR2GRAY)
        if b_write:
            cv2.imwrite(full_output_original_mask_path, mask_or)  # save frame as png file

        mask = read_image(full_mask_path)
        mask = mask.float()

        # criterion = IoULoss()
        criterion = nn.BCELoss()
        # mask = mask.unsqueeze(0).to(device_i)
        input_tensor = test_composed_transform(frame).unsqueeze(0).to(device_i)
        mask_tensor = test_composed_transform(mask).unsqueeze(0).to(device_i)
        mask_tensor = torch.where(mask_tensor < 0.5, torch.tensor(0), torch.tensor(1))
        mask_tensor = mask_tensor.float()

        # output1 = model(input_tensor)
        # output = torch.sigmoid(output1)
        # bin_thresh = find_binarization_threshold(output)

        # predicted_mask_0 = torch.where(output < bin_thresh, torch.tensor(0), torch.tensor(255))
        # predicted_mask_0s = predicted_mask_0.squeeze().numpy()
        # cv2.imshow("predicted_mask_0", predicted_mask_0s.astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # frame_loss = criterion(output, mask_tensor)
        bin_thresh_rot = -1
        frame_loss_rot = 1
        min_loss = 1000
        max_th = -1
        min_ave_prob = 1
        min_ave_prob_ind = -1
        min_loss_ind = -1
        loss_for_best_threshold = 1
        b_show = False
        min_area = 10
        rotated_back_list = []
        data = [filename]
        # angles = [15, 30, 45, 60, 75, 90]
        # angles = [0, 15, 30, 45, 60, 75, 90, 180, -90, 'f']
        angles = [0, 90, 180, -90, 'f']
        for ind in range(len(angles)):
            ang = angles[ind]
            if isinstance(ang, (int, float)):
                rotated_image, angle, ds_coef = rotate_image(input_tensor, ang)
                rotated_image = rotated_image.unsqueeze(0).to(device_i)
            elif ang == 'f':
                # Perform a horizontal flip
                rotated_image = torch.flip(input_tensor, [3])
            else:
                break
            output_rotated = model(rotated_image)
            output_rotated_sig = torch.sigmoid(output_rotated)
            bin_thresh_rot = find_binarization_threshold(output_rotated_sig)
            # Calculate the average of the probabilities less than threshold
            average_small_probability_0_5 = find_average_prob_below_threshold(output_rotated_sig, 0.5)

            if ang == 'f':  #   back flipping
                rotated_back = torch.flip(output_rotated_sig, [3])
            else:
                rotated_back = rotate_image_back(output_rotated_sig, ang, ds_coef)
            frame_loss_rot = criterion(rotated_back, mask_tensor)
            if frame_loss_rot < min_loss:
                min_loss = frame_loss_rot
                min_loss_ind = ind
            if average_small_probability_0_5 < min_ave_prob:
                min_ave_prob = average_small_probability_0_5
                min_ave_prob_ind = ind
                loss_for_best_threshold = frame_loss_rot
            data.append(frame_loss_rot.item())
            # data.append(bin_thresh_rot)
            data.append(average_small_probability_0_5)
            rotated_back = torch.where(rotated_back < bin_thresh_rot, torch.tensor(0), torch.tensor(255))
            # rotated_back_s = rotated_back.squeeze().numpy()
            predicted_mask_np = rotated_back.squeeze(0).cpu().numpy().squeeze(0)
            # predicted_mask_with_bb_np = np.copy(predicted_mask_np)
            # original_with_bb_np = input_tensor * 255
            # predicted_mask_with_bb_np, predicted_mask_np, original_with_bb_np = filter_and_draw_bounding_boxes(
            #     predicted_mask_with_bb_np, original_with_bb_np, min_area)
            # cv2.imwrite(full_mask_with_bb_path, predicted_mask_with_bb_np)
            # cv2.imwrite(full_original_with_bb_path, original_with_bb_np)
            predicted_mask_name = name + '_mask' + str(ang) + '.jpg'
            full_predicted_mask_name = os.path.join(output_masks_dir, predicted_mask_name)
            cv2.imwrite(full_predicted_mask_name, predicted_mask_np)
            original = input_tensor * 255
            original = original.squeeze(0).cpu().numpy().squeeze(0)
            cv2.imwrite(full_original_path, original)

            rotated_back_list.append(predicted_mask_np)
            if b_show:
                output_rotated1 = torch.where(output_rotated_sig < bin_thresh_rot, torch.tensor(0), torch.tensor(255))
                output_rotated_s = output_rotated1.squeeze().detach().numpy()
                rotated_image1 = rotated_image * 255
                rotated_image_s = rotated_image1.squeeze().numpy()
                input_tensor1 = input_tensor * 255
                input_tensor_s = input_tensor1.squeeze().numpy()
                cv2.imshow("it", input_tensor_s.astype(np.uint8))
                cv2.imshow("ri", rotated_image_s.astype(np.uint8))
                cv2.imshow("output_rotated", output_rotated_s.astype(np.uint8))
                cv2.imshow("rb", predicted_mask_np.astype(np.uint8))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                aa = 0

        data.append(angles[min_loss_ind])
        data.append(angles[min_ave_prob_ind])
        data.append(min_loss.item())
        data.append(loss_for_best_threshold.item())
        data.append(min_ave_prob)
        csv_writer.writerow(data)
        predicted_mask_np = rotated_back_list[0]
        cv2.imwrite(full_output_predicted_mask_path, predicted_mask_np)
        predicted_mask_best = rotated_back_list[min_ave_prob_ind]
        cv2.imwrite(full_output_predicted_best_mask_path, predicted_mask_best)
        predicted_mask_with_bb_np = np.copy(predicted_mask_np)
        original_with_bb_np = input_tensor * 255
        original_with_bb_np = original_with_bb_np.squeeze(0).cpu().numpy().squeeze(0)
        predicted_mask_with_bb_np, predicted_mask_np, original_with_bb_np = filter_and_draw_bounding_boxes(
            predicted_mask_with_bb_np, original_with_bb_np, min_area)
        # cv2.imshow("mask_bb", predicted_mask_with_bb_np.astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite(full_mask_with_bb_path, predicted_mask_with_bb_np)
        cv2.imwrite(full_original_with_bb_path, original_with_bb_np)
        print(f"{filename} - "
              f"bin_thresh: {bin_thresh_rot:.4f} - "
              f"frame_loss: {frame_loss_rot.item():.4f}"
              )

        # cv2.imshow("Predicted mask", predicted_mask)
        # cv2.imshow("Mask", mask_tensor)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # if count:
    #     loss /= count
    #     loss_bin /= count

    # return count, loss, loss_bin


def calculate_intersection_ratio(box1, box2):
    if not box1 and not box2:
        return 1  # Both arrays are empty, ratio is 1

    if not box1:
        return 0  # First array is empty, ratio is 0

    intersection_counter = 0

    for bbox1 in box1:
        for bbox2 in box2:
            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2
            x1 = max(x1 - bb_half_size, 0)
            y1 = max(y1 - bb_half_size, 0)
            w1 = min(w1 + bb_half_size * 2, output_image_size - x1)
            h1 = min(h1 + bb_half_size * 2, output_image_size - y1)
            x2 = max(x2 - bb_half_size, 0)
            y2 = max(y2 - bb_half_size, 0)
            w2 = min(w2 + bb_half_size * 2, output_image_size - x2)
            h2 = min(h2 + bb_half_size * 2, output_image_size - y2)

            # Check for intersection
            if (
                x1 < x2 + w2
                and x1 + w1 > x2
                and y1 < y2 + h2
                and y1 + h1 > y2
            ):
                intersection_counter += 1
                break  # Break once an intersection is found for the current bbox1

    return intersection_counter / len(box1)


def calculate_metrics(ground_truth, predicted):
    true_positive = 0
    false_positive = 0

    if not ground_truth and not predicted:
        # Both arrays are empty
        # return {'accuracy': 1, 'precision': 1, 'recall': 1}
        return 1, 1, 1

    for pred_foot in predicted:
        pred_x, pred_y = pred_foot
        found_intersection = False

        for gt_foot in ground_truth:
            gt_x, gt_y = gt_foot

            # Check for intersection
            x_overlap = max(0, min(pred_x + bb_half_size, gt_x + bb_half_size) - max(pred_x - bb_half_size, gt_x - bb_half_size))
            y_overlap = max(0, min(pred_y + bb_half_size, gt_y + bb_half_size) - max(pred_y - bb_half_size, gt_y - bb_half_size))
            intersection_area = x_overlap * y_overlap

            if intersection_area > 0:
                # Intersection found
                true_positive += 1
                found_intersection = True
                break

        if not found_intersection:
            # No intersection found, increment false positive
            false_positive += 1

    # Calculate false negatives
    false_negative = len(ground_truth) - true_positive

    # Calculate accuracy, precision, and recall
    total_gt_boxes = len(ground_truth)
    total_pred_boxes = len(predicted)

    accuracy = true_positive / max(1, (true_positive + false_positive + false_negative))
    precision = true_positive / max(1, total_pred_boxes)
    recall = true_positive / max(1, total_gt_boxes)

    return accuracy, precision, recall


def infer_one_frame(filename_without_ext, directory_name, output_masks_dir, test_composed_transform, mask_transform,
                    models_inf, device_i, count,
                    loss, loss_bin, csv_writer):
    if not directory_name1:
        filename = filename_without_ext + '.jpg'
        full_frame_path = filename
    else:
        filename = filename_without_ext
        full_frame_path = os.path.join(directory_name, filename)
        name, extension = os.path.splitext(filename)
        mask_filename = name + ".png"
        output_original_mask_name = name + "_original_mask.jpg"

        full_mask_path = os.path.join(directory_name, mask_filename)
        full_output_original_mask_path = os.path.join(output_masks_dir, output_original_mask_name)
        original_filename = filename[:filename.find('-')] + filename[filename.find('-'):]
        full_original_path = os.path.join(output_masks_dir, original_filename)
    if os.path.exists(full_frame_path):
        count += 1
        if not directory_name:
            img = cv2.imread(full_frame_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(full_frame_path, img)  # save frame as png file

        frame = read_image(full_frame_path)
        frame = frame.float()

        mask = read_image(full_mask_path)
        mask = mask.float()

        criterion = nn.BCELoss()
        input_tensor = test_composed_transform(frame).unsqueeze(0).to(device_i)
        mask_tensor = test_composed_transform(mask).unsqueeze(0).to(device_i)
        mask_tensor = torch.where(mask_tensor < 0.5, torch.tensor(0), torch.tensor(1))
        mask_tensor = mask_tensor.float()
        binarized_original_mask = torch.where(mask_tensor < 0.5, torch.tensor(0), torch.tensor(255))
        binarized_original_mask_np = binarized_original_mask.squeeze(0).cpu().numpy().squeeze(0)

        data = [filename]
        frame_loss = []
        bin_thresh = []
        average_small_probability_0_5 = []
        for ind in range(len(models_inf)):
            model = models_inf[ind]
            output = model(input_tensor)
            output_sig = torch.sigmoid(output)
            bin_thresh_t = find_binarization_threshold(output_sig)
            bin_thresh.append(bin_thresh_t)
            # Calculate the average of the probabilities less than threshold
            average_small_probability_0_5_t = find_average_prob_below_threshold(output_sig, 0.5)
            frame_loss_t = criterion(output_sig, mask_tensor)
            frame_loss.append(frame_loss_t.item())
            average_small_probability_0_5.append(average_small_probability_0_5_t)
            binarized = torch.where(output_sig < bin_thresh_t, torch.tensor(0), torch.tensor(255))
            predicted_mask_np = binarized.squeeze(0).cpu().numpy().squeeze(0)
            predicted_mask_name = name + '_mask' + str(ind) + '.jpg'
            full_predicted_mask_name = os.path.join(output_masks_dir, predicted_mask_name)
            if b_write:
                cv2.imwrite(full_predicted_mask_name, predicted_mask_np)

            predicted_mask_with_bb_np = np.copy(predicted_mask_np)
            binarized_original_mask_with_bb_np = np.copy(binarized_original_mask_np)
            original_with_bb_np = input_tensor * 255
            original_with_bb_np = original_with_bb_np.squeeze(0).cpu().numpy().squeeze(0)
            binarized_0999 = torch.where(output_sig < 0.999, torch.tensor(0), torch.tensor(255))
            predicted_mask_0999_np = binarized_0999.squeeze(0).cpu().numpy().squeeze(0)
            binarized_099 = torch.where(output_sig < 0.99, torch.tensor(0), torch.tensor(255))
            predicted_mask_099_np = binarized_099.squeeze(0).cpu().numpy().squeeze(0)
            binarized_090 = torch.where(output_sig < 0.9, torch.tensor(0), torch.tensor(255))
            predicted_mask_090_np = binarized_090.squeeze(0).cpu().numpy().squeeze(0)
            binarized_070 = torch.where(output_sig < 0.70, torch.tensor(0), torch.tensor(255))
            predicted_mask_070_np = binarized_070.squeeze(0).cpu().numpy().squeeze(0)
            binarized_050 = torch.where(output_sig < 0.50, torch.tensor(0), torch.tensor(255))
            predicted_mask_050_np = binarized_050.squeeze(0).cpu().numpy().squeeze(0)
            binarized_040 = torch.where(output_sig < 0.40, torch.tensor(0), torch.tensor(255))
            predicted_mask_040_np = binarized_040.squeeze(0).cpu().numpy().squeeze(0)
            binarized_030 = torch.where(output_sig < 0.30, torch.tensor(0), torch.tensor(255))
            predicted_mask_030_np = binarized_030.squeeze(0).cpu().numpy().squeeze(0)
            binarized_025 = torch.where(output_sig < 0.25, torch.tensor(0), torch.tensor(255))
            predicted_mask_025_np = binarized_025.squeeze(0).cpu().numpy().squeeze(0)
            binarized_020 = torch.where(output_sig < 0.20, torch.tensor(0), torch.tensor(255))
            predicted_mask_020_np = binarized_020.squeeze(0).cpu().numpy().squeeze(0)
            binarized_015 = torch.where(output_sig < 0.15, torch.tensor(0), torch.tensor(255))
            predicted_mask_015_np = binarized_015.squeeze(0).cpu().numpy().squeeze(0)

            foot_points_predicted, bounding_boxes_predicted, predicted_mask_with_bb_np, predicted_mask_np, original_with_bb_np = filter_and_draw_bounding_boxes(
                predicted_mask_with_bb_np, original_with_bb_np, min_area)
            foot_points_original, bounding_boxes_original, binarized_original_mask_with_bb_np, binarized_original_mask_np, t = filter_and_draw_bounding_boxes(
                binarized_original_mask_with_bb_np, original_with_bb_np, min_area)
            bb_accuracy = calculate_intersection_ratio(bounding_boxes_original, bounding_boxes_predicted)
            bb_accuracy1 = calculate_intersection_ratio(bounding_boxes_predicted, bounding_boxes_original)
            accuracy, precision, recall = calculate_metrics(foot_points_original, foot_points_predicted)

            if bb_accuracy1 != bb_accuracy:
                aa = 0
            bb_accuracy = min(bb_accuracy, bb_accuracy1)
            IoUt = iou_loss(predicted_mask_with_bb_np / 255, binarized_original_mask_with_bb_np / 255)

            # bounding_boxes_predicted_0999, _, _, _ = filter_and_draw_bounding_boxes(
            #     predicted_mask_0999_np, original_with_bb_np, min_area)
            # bb_accuracy_0999 = calculate_intersection_ratio(bounding_boxes_original, bounding_boxes_predicted_0999)
            # bb_accuracy_0999_1 = calculate_intersection_ratio(bounding_boxes_predicted_0999, bounding_boxes_original)
            # bb_accuracy_0999 = min(bb_accuracy_0999, bb_accuracy_0999_1)
            # bounding_boxes_predicted_099, _, _, _ = filter_and_draw_bounding_boxes(
            #     predicted_mask_099_np, original_with_bb_np, min_area)
            # bb_accuracy_099 = calculate_intersection_ratio(bounding_boxes_original, bounding_boxes_predicted_099)
            # bb_accuracy_099_1 = calculate_intersection_ratio(bounding_boxes_predicted_099, bounding_boxes_original)
            # bb_accuracy_099 = min(bb_accuracy_099, bb_accuracy_099_1)
            # bounding_boxes_predicted_090, _, _, _ = filter_and_draw_bounding_boxes(
            #     predicted_mask_090_np, original_with_bb_np, min_area)
            # bb_accuracy_090 = calculate_intersection_ratio(bounding_boxes_original, bounding_boxes_predicted_090)
            # bb_accuracy_090_1 = calculate_intersection_ratio(bounding_boxes_predicted_090, bounding_boxes_original)
            # bb_accuracy_090 = min(bb_accuracy_090, bb_accuracy_090_1)
            # bounding_boxes_predicted_070, _, _, _ = filter_and_draw_bounding_boxes(
            #     predicted_mask_070_np, original_with_bb_np, min_area)
            # bb_accuracy_070 = calculate_intersection_ratio(bounding_boxes_original, bounding_boxes_predicted_070)
            # bb_accuracy_070_1 = calculate_intersection_ratio(bounding_boxes_predicted_070, bounding_boxes_original)
            # bb_accuracy_070 = min(bb_accuracy_070, bb_accuracy_070_1)
            # bounding_boxes_predicted_050, _, _, _ = filter_and_draw_bounding_boxes(
            #     predicted_mask_050_np, original_with_bb_np, min_area)
            # bb_accuracy_050 = calculate_intersection_ratio(bounding_boxes_original, bounding_boxes_predicted_050)
            # bb_accuracy_050_1 = calculate_intersection_ratio(bounding_boxes_predicted_050, bounding_boxes_original)
            # bb_accuracy_050 = min(bb_accuracy_050, bb_accuracy_050_1)
            # bounding_boxes_predicted_040, _, _, _ = filter_and_draw_bounding_boxes(
            #     predicted_mask_040_np, original_with_bb_np, min_area)
            # bb_accuracy_040 = calculate_intersection_ratio(bounding_boxes_original, bounding_boxes_predicted_040)
            # bb_accuracy_040_1 = calculate_intersection_ratio(bounding_boxes_predicted_040, bounding_boxes_original)
            # bb_accuracy_040 = min(bb_accuracy_040, bb_accuracy_040_1)
            # bounding_boxes_predicted_030, _, _, _ = filter_and_draw_bounding_boxes(
            #     predicted_mask_030_np, original_with_bb_np, min_area)
            # bb_accuracy_030 = calculate_intersection_ratio(bounding_boxes_original, bounding_boxes_predicted_030)
            # bb_accuracy_030_1 = calculate_intersection_ratio(bounding_boxes_predicted_030, bounding_boxes_original)
            # bb_accuracy_030 = min(bb_accuracy_030, bb_accuracy_030_1)
            # bounding_boxes_predicted_025, _, _, _ = filter_and_draw_bounding_boxes(
            #     predicted_mask_025_np, original_with_bb_np, min_area)
            # bb_accuracy_025 = calculate_intersection_ratio(bounding_boxes_original, bounding_boxes_predicted_025)
            # bb_accuracy_025_1 = calculate_intersection_ratio(bounding_boxes_predicted_025, bounding_boxes_original)
            # bb_accuracy_025 = min(bb_accuracy_025, bb_accuracy_025_1)
            # bounding_boxes_predicted_020, _, _, _ = filter_and_draw_bounding_boxes(
            #     predicted_mask_020_np, original_with_bb_np, min_area)
            # bb_accuracy_020 = calculate_intersection_ratio(bounding_boxes_original, bounding_boxes_predicted_020)
            # bb_accuracy_020_1 = calculate_intersection_ratio(bounding_boxes_predicted_020, bounding_boxes_original)
            # bb_accuracy_020 = min(bb_accuracy_020, bb_accuracy_020_1)
            # bounding_boxes_predicted_015, _, _, _ = filter_and_draw_bounding_boxes(
            #     predicted_mask_015_np, original_with_bb_np, min_area)
            # bb_accuracy_015 = calculate_intersection_ratio(bounding_boxes_original, bounding_boxes_predicted_015)
            # bb_accuracy_015_1 = calculate_intersection_ratio(bounding_boxes_predicted_015, bounding_boxes_original)
            # bb_accuracy_015 = min(bb_accuracy_015, bb_accuracy_015_1)

            data.append(frame_loss_t.item())
            data.append(IoUt)
            data.append(bin_thresh_t)
            data.append(average_small_probability_0_5_t)
            data.append(len(foot_points_original))
            data.append(len(foot_points_predicted))
            if len(foot_points_original) == len(foot_points_predicted):
                data.append(1)
            else:
                data.append(0)
            if abs(len(foot_points_original) - len(foot_points_predicted)) < 2:
                data.append(1)
            else:
                data.append(0)
            data.append(accuracy)
            data.append(precision)
            data.append(recall)
            # data.append(bb_accuracy)
            # data.append(bb_accuracy_0999)
            # data.append(bb_accuracy_099)
            # data.append(bb_accuracy_090)
            # data.append(bb_accuracy_070)
            # data.append(bb_accuracy_050)
            # data.append(bb_accuracy_040)
            # data.append(bb_accuracy_030)
            # data.append(bb_accuracy_025)
            # data.append(bb_accuracy_020)
            # data.append(bb_accuracy_015)

            mask_with_bb_filename = name + '_mask_bb_' + str(ind) + extension
            full_mask_with_bb_path = os.path.join(output_masks_dir, mask_with_bb_filename)
            original_with_bb_filename = name + '_bb_' + str(ind) + extension
            full_original_with_bb_path = os.path.join(output_masks_dir, original_with_bb_filename)

            # cv2.imshow("mask_bb", binarized_original_mask_with_bb_np.astype(np.uint8))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            #
            # if b_write:
            #     cv2.imwrite(full_output_original_mask_path, binarized_original_mask_np)  # save original mask

            if b_write:
                cv2.imwrite(full_mask_with_bb_path, predicted_mask_with_bb_np)
                cv2.imwrite(full_original_with_bb_path, original_with_bb_np)
                cv2.imwrite(full_output_original_mask_path, binarized_original_mask_with_bb_np)  # save original mask

        if len(models_inf) == 2:
            if frame_loss[0] < frame_loss[1]:
                min_loss = frame_loss[0]
                min_loss_ind = 0
            else:
                min_loss = frame_loss[1]
                min_loss_ind = 1

            accuracy_bin_th = 0
            accuracy_small_prob = 0
            if (frame_loss[0] < frame_loss[1] and bin_thresh[0] > bin_thresh[1]) or (frame_loss[1] < frame_loss[0] and bin_thresh[1] > bin_thresh[0]):
                accuracy_bin_th = 1
            if (frame_loss[0] < frame_loss[1] and average_small_probability_0_5[0] < average_small_probability_0_5[1]) or (frame_loss[1] < frame_loss[0] and average_small_probability_0_5[1] < average_small_probability_0_5[0]):
                accuracy_small_prob = 1
            data.append(accuracy_bin_th)
            data.append(accuracy_small_prob)

        original = input_tensor * 255
        original = original.squeeze(0).cpu().numpy().squeeze(0)
        if b_write:
            cv2.imwrite(full_original_path, original)

        csv_writer.writerow(data)


def infer_unet(frame_filename, directory_name, output_masks_dir, model0_path, model1_path, test_composed_transform, mask_transform, batch_test, csv_file_name):
    # Load the saved model
    device_i = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model0 = CustomMNasNetUnetInvResDec(1, 1)
    model1 = None
    # print(model)
    # summary(model0, (1, 224, 224))
    total_params0 = sum(p.numel() for p in model0.parameters())
    print(f"Number of parameters 0: {total_params0}")
    model0.load_state_dict(torch.load(model0_path, map_location='cpu'))
    model0.to(device_i)
    model0.eval()
    if model1_path is not None:
        global use_se
        use_se = True
        model1 = CustomMNasNetUnetInvResDec(1, 1)
        # summary(model1, (1, 224, 224))
        total_params1 = sum(p.numel() for p in model1.parameters())
        print(f"Number of parameters 1: {total_params1}")
        model1.load_state_dict(torch.load(model1_path, map_location='cpu'))
        model1.to(device_i)
        model1.eval()
        models_i = [model0, model1]
    else:
        models_i = [model0]
    loss = 0
    loss_bin = 0
    count = 0

    if directory_name is not None:
        if os.path.exists(directory_name):
            jpg_files = [f for f in os.listdir(directory_name) if f.endswith('.jpg')]
            with open(csv_file_name, 'w', newline='') as file:
                csv_writer = csv.writer(file)
                if len(models_i) == 2:
                    # csv_writer.writerow(["jpg file name", "loss 0", "IoU loss 0", "bin_th 0", "prob < 0.5 for 0", "bb accuracy 0", "bb accuracy999 0", "bb accuracy99 0", "bb accuracy90 0", "bb accuracy70 0", "bb accuracy50 0", "bb accuracy40 0", "bb accuracy30 0", "bb accuracy25 0", "bb accuracy20 0", "bb accuracy15 0", "loss 1", "IoU loss 1", "bin_th 1", "prob < 0.5 for 1", "bb accuracy 1", "bb accuracy999 1", "bb accuracy99 1","bb accuracy90 1","bb accuracy70 1","bb accuracy50 1", "bb accuracy40 1", "bb accuracy30 1", "bb accuracy25 1", "bb accuracy20 1", "bb accuracy15 1", "accuracy bin thresh", "accuracy small prob"])
                    csv_writer.writerow(["jpg file name", "loss 0", "IoU loss 0", "bin_th 0", "prob < 0.5 for 0", "bb accuracy 0", "bb accuracy999 0", "bb accuracy99 0", "bb accuracy90 0", "bb accuracy70 0", "bb accuracy50 0", "bb accuracy40 0", "bb accuracy30 0", "bb accuracy25 0", "bb accuracy20 0", "bb accuracy15 0", "loss 1", "IoU loss 1", "bin_th 1", "prob < 0.5 for 1", "bb accuracy 1", "accuracy bin thresh", "accuracy small prob"])
                else:
                    csv_writer.writerow(["jpg file name", "loss", "IoU loss", "bin_th", "prob < 0.5 for", "gt number of people", "predicted number of people", "equal", "less than 2", "accuracy", "precision", "recall"])
                    # csv_writer.writerow(["jpg file name", "loss", "IoU loss", "bin_th", "prob < 0.5 for", "accuracy", "precision", "recall", "bb accuracy", "bb accuracy999", "bb accuracy99", "bb accuracy90", "bb accuracy70", "bb accuracy50", "bb accuracy40", "bb accuracy30", "bb accuracy25", "bb accuracy20","bb accuracy15"])

                # jpg_files1 = [jpg_files[0], jpg_files[1]]
                for filename in jpg_files:
                    # infer_one_frame1(filename, directory_name, output_masks_dir,
                    #                                         test_composed_transform, mask_transform, model,
                    #                                         device_i, count, loss, loss_bin, csv_writer)
                    infer_one_frame(filename, directory_name, output_masks_dir,
                                                            test_composed_transform, mask_transform, models_i,
                                                            device_i, count, loss, loss_bin, csv_writer)

            # if count:
            #     loss /= count
            #     loss_bin /= count
            #     print(f"Loss: {loss:.4f} - "
            #           f"loss_round: {loss_bin:.4f} - "
            #           )
            # test_dataset = CustomImageDataset(directory_name, False, False, test_composed_transform, mask_transform)
            # test_loader = DataLoader(test_dataset, batch_size=batch_test)
            # chat_gpt_test(device_i, test_loader, models_i[0])
    else:
        infer_one_frame(frame_filename, directory_name, output_masks_dir, test_composed_transform, mask_transform,
                        models_i, device_i, count,
                        loss, loss_bin)

    return count, loss, loss_bin


def main(batch_train_valid, batch_test, training, model_save_name, train_valid_test_composed_transform,
         test_composed_transform, mask_transform, valid_size):
    if training:
        if local_machine:
            # training_path = "c:\\CamDatabase\\Frames\\UnetTest"
            # training_path = "c:\\CamDatabase\\Frames\\UnetTraining"
            # training_path = "c:\\CamDatabase\\DBAnnotated\\Training"
            training_path = "c:\\CamDatabase\\PositionsDB\\Test"
            # training_path = "c:/Theodore/Test"
            # valid_path = "c:\\CamDatabase\\Frames\\UnetTest"
            valid_path = "c:/Theodore/Test"
            # test_path = "c:/Theodore/Test"
            test_path = "c:\\CamDatabase\\DBAnnotated\\Test"
        else:  # cloud
            # training_path = "UnetTraining"
            # training_path = "theodore/training"
            # valid_path = "theodore/test"
            # test_path = "theodore/test"
            # training_path = "PositionsDB/Training5points"
            # training_path = "PositionsDB/Training1"
            training_path = "PositionsDB/TrainingThin"
            valid_path = "PositionsDB/Test"
            test_path = "PositionsDB/Test"
        train_dataloader1, valid_dataloader1, test_dataloader1 = load_split_train_test(training_path, valid_path,
                                                                                       test_path,
                                                                                       train_valid_test_composed_transform,
                                                                                       mask_transform,
                                                                                       batch_train_valid, batch_test,
                                                                                       valid_size)
        device1 = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {device1} device")
        path_to_save_model1 = model_save_name
        model1 = chat_gpt_training(device1, train_dataloader1, valid_dataloader1, num_epochs1, path_to_save_model1, csv_file_name1)
        # chat_gpt_test(device1, test_dataloader1, model1)
        aa = 0
    else:  # inference
        frame_filename1 = None
        # frame_filename1 = "c:/CamDatabase/Test/Images/PG1_dev_H250_HD_designed_No_Occlusion_1509275690-4_1596.jpg"
        # directory_name1 = "c:/CamDatabase/Test1/Tests"
        # directory_name1 = "c:/CamDatabase/FramesWithoutMasks/TestHomogeneous"
        model_path0 = model_save_name0
        model_path1 = model_save_name1
        if not directory_name1 and not frame_filename1:
            for ind in range(10):
                take_image_from_camera(host1, cam_filename1 + str(ind))
        infer_unet(cam_filename1, directory_name1, output_masks_dir1, model_path0, model_path1, test_composed_transform,
                   mask_transform, batch_test, csv_file_name1)


if __name__ == '__main__':
    batch_test1 = 1
    num_epochs1 = 2000
###################################
    # outputs = torch.randn(15, 1, 24, 24)
    # masks = torch.randint(0, 2, (15, 1, 24, 24)).float()
    #
    # # Create pos_weight tensor based on mask values
    # pos_weight = torch.where(masks == 1, torch.tensor(10.0), torch.tensor(1.0))
    #
    # # Instantiate BCE loss with balanced class weights
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    #
    # # Calculate the loss
    # loss = criterion(outputs, masks)
    # targets = torch.tensor([[0, 0, 1, 1]])
    # inputs = torch.tensor([[1.0, -1.0, 2.0, -2.0]])
    # targets = torch.tensor([[0]])
    # inputs = torch.tensor([[0.0000001]])
    # targets = torch.tensor([[1]])
    # inputs = torch.tensor([[0.9]])

    # Instantiate Focal Loss with gamma=2
    # focal_loss = FocalLoss(gamma=-2)

    # Calculate Focal Loss
    # lossBCE_crit = nn.BCELoss(inputs, targets.float())
    # BCEloss = lossBCE_crit(inputs, targets.float())
    # loss = focal_loss(inputs, targets.float())
    #
    # print(f"Focal Loss: {loss.item()}")
###################################
    mean1 = 0
    std1 = 255
    height = 224
    width = 224

    train_valid_test_composed_transform1 = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((height, width), antialias=True),
        transforms.Normalize(mean=[mean1], std=[std1])
    ])
    mask_transform1 = transforms.Compose([
        transforms.Resize((height, width), antialias=True),
        transforms.Normalize(mean=[mean1], std=[std1])
        # ToTensor
    ])
    test_composed_transform1 = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((height, width), antialias=True),
        transforms.Normalize(mean=[mean1], std=[std1])
    ])
    main(batch_train_valid1, batch_test1, training1, model_save_name1, train_valid_test_composed_transform1,
         test_composed_transform1, mask_transform1, valid_size1)
