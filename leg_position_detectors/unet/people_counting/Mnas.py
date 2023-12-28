import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import math
from datetime import datetime
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, models, utils
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.models import mnasnet0_5
from sklearn.model_selection import train_test_split
import csv
import shutil
from torchsummary import summary
from torch.cuda.amp import autocast

autocast(enabled=False)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
network_type = 'MnasNet'
letter = '015_l1'
drop_out = 0.15
divide_normalization_number_of_people = 25
reduction_for_SE = 16
valid_size1 = 0.05
learning_rate = 0.001
local_machine = True
training1 = False
use_se = False
copy_outliers = False
outliers_dir = 'Outliers'
outliers_threshold = 2
if not training1:
    import cv2
    import paramiko
    import time

if use_se:
    letter = '_SE'
if local_machine:
    if training1:
        # model_save_name1 = 'PTH2\\reg_model_18_DS2_255' + letter + '.pth'
        model_save_name1 = 'PTH_Temp\\MNAS' + letter + '.pth'
        csv_file_name1 = "c:/CamDatabase/CSV/2.csv"
    else:
        # model_save_name1 = 'PTH1\\reg_model_18_DS2_255' + letter + '.pth'
        # model_save_name1 = 'PTH_MNS_MY\\reg_model_' + letter + '.pth'
        model_save_name1 = 'PTH_MNAS_POS\\MNAS' + letter + '.pth'
        # model_save_name1 = 'PTH_MNAS_EC2\\MNAS' + letter + '.pth'
        csv_file_name1 = "c:\\Users\\ilyar\\PycharmProjects\\pythonProject1\\CSV_POS1\\MNAS" + letter + "a.csv"
        # csv_file_name1 = "c:\\Users\\ilyar\\PycharmProjects\\pythonProject1\\CSV_MNSA\\stat91_" + letter + ".csv"
else:
    model_save_name1 = 'PTH_MNAS_Pos/MNAS' + letter + '.pth'
    csv_file_name1 = 'PTH_MNAS_Pos/MNAS' + letter + '.csv'
    # model_save_name1 = 'PTH_MNSA/reg_model_' + letter + '.pth'
# directory_name1 = "c:/CamDatabase/Test1/Tests"
directory_name1 = "c:/CamDatabase/PositionsDB/TestThin"  # "c:/CamDatabase/DBAnnotated/Test"
# directory_name1 = "c:/CamDatabase/Frames/ExperimentsInput1"

# directory_name1 = "c:/CamDatabase/FramesWithoutMasks/TestHomogeneousVis2"
# directory_name1 = "c:/CamDatabase/FramesWithoutMasks/TestHomogeneous"
# directory_name1 = None
cam_filename1 = "cam_m"
host1 = "172.19.56.12"
# host1 = "172.22.56.14"  # meeting room
do_train_flip1 = True
do_train_rotate1 = True
do_train_dark1 = True


def visualize_feature_maps_mnas(model_unet, img):
    conv_layers = []
    conv_layers1 = []
    # model_children = list(model_unet.children())
    model_children = list(model_unet.model.mnasnet.children())
    numb_p = model_unet.model(img)
    # l1 = model_children[0]
    l1 = model_unet.model.mnasnet.layers
    for e in l1:
        conv_layers.append(e)
    # l2 = model_children[1]
    l2 = model_unet.model.mnasnet.classifier
    for e in l2:
        conv_layers1.append(e)

    # for child in model_children[0]:
    if True:
        # str_type = str(type(child))
        # layer_name = re.search(r"'([^']*)'", str_type.split()[1]).group(1)
        # layer_name = layer_name.split('.')[-1]
        # print(layer_name)
        # if layer_name == 'Conv2d':    #   type(child) == CustomInception:
        if True:  # type(child) == CustomInception:
            # filter = child.inception.Conv2d_1a_3x3
            no_of_layers = 16
            print(no_of_layers)

            # img = img.repeat(1, 3, 1, 1)  # Repeat the single-channel image to make it 3-channel

            # results = [conv_layers[0](img)]
            results = [l1[0](img)]
            # res = results
            for i in range(1, len(l1)):
                results.append(l1[i](results[-1]))
            # results1 = results
            # for i in range(len(l2)):
            #     results1.append(l2[i](results1[-1]))

            outputs = results
            dir_save = 'c:/PicForPresentationMnas/'

            for num_layer in range(len(outputs)):
                layer_viz = outputs[num_layer][0, :, :, :]
                layer_viz = layer_viz.data
                print("Layer ", num_layer + 1)
                if num_layer >= 0:  # 11:  # or num_layer < 3:
                    for i, filter1 in enumerate(layer_viz):
                        if i >= 0:  # 10 > i > -1 or i > len(layer_viz) - 10:
                            file_save = dir_save + 'layer_' + str(num_layer + 1) + '_img' + '_' + str(i + 1) + '.jpg'
                            filter1_stretched = filter1 * 255
                            numpy_array = filter1_stretched.cpu().numpy()
                            opencv_image = cv2.cvtColor(numpy_array, cv2.COLOR_GRAY2BGR)
                            output_width = 320
                            output_height = 240

                            # Resize the image to the desired output size
                            opencv_image = cv2.resize(opencv_image, (output_width, output_height))
                            # opencv_image = cv2.transpose(opencv_image)
                            # Flip the image horizontally (around the y-axis)
                            # opencv_image = cv2.flip(opencv_image, 1)
                            cv2.imwrite(file_save, opencv_image)
                            # plt.imshow(opencv_image, cmap='gray')
                            # plt.axis("off")
                            # plt.title('layer ' + str(num_layer + 1) + ' channel ' + str(i + 1), fontweight="bold")
                            # plt.show()
                            # cv2.waitKey(0)

                # plt.close()
    aa = 0


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
    # ssh_client = paramiko.SSHClient()
    # command = 'ssh pg@[172.19.56.12]'
    # os.system(command)
    aa = 0


# def read_image_from_camera():
#     # Create a VideoCapture object
#     cap = cv2.VideoCapture(0)
#
#     # Check if camera opened successfully
#     if cap.isOpened() is False:
#         print("Unable to read camera feed")
#
#     # Default resolutions of the frame are obtained.The default resolutions are system dependent.
#     # We convert the resolutions from float to integer.
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
#     print(int(cap.get(cv2.CAP_PROP_FPS)), frame_width, frame_height)
#     success, img = cap.read()
#     if success:
#         return img


def write_csv(file_name_csv1, data0):
    with open(file_name_csv1, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data0)


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


class CustomMNasNet(nn.Module):
    def __init__(self, num_classes=1, in_channels=1):
        super(CustomMNasNet, self).__init__()

        # Load the MobileNet architecture
        self.mnasnet = mnasnet0_5(pretrained=True)
        # self.mnasnet.layers[0].in_channels = 1

        # shorter_model = []
        # l1 = list(self.mnasnet.children())
        #
        # for e in l1[0]:
        #     shorter_model.append(e)
        # newmodel = torch.nn.Sequential(*(list(self.mnasnet.children())[:-1]))
        # print(self.mnasnet)
        # print(newmodel)

        # Modify the last fully connected layer for regression
        num_features = self.mnasnet.classifier[1].in_features
        self.mnasnet.classifier[1] = nn.Linear(num_features, 1)
        # self.mnasnet.classifier = nn.Sequential(
        #     nn.Linear(num_features, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1))  # Regression output

    def forward(self, x):
        # x = x.flatten(start_dim=1)
        # x = self.regressor(x)
        x_rgb = x.repeat(1, 3, 1, 1)
        return self.mnasnet(x_rgb)


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


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

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
    def __init__(self, inp, oup, stride, expand_ratio, kernel):
        super(InvertedResidualWithSE, self).__init__(inp, oup, stride, expand_ratio, kernel)
        # Insert SE block before ReLU6
        self.se_block = SEBlock(inp * expand_ratio, reduction=reduction_for_SE)
        self.conv.insert(-3, self.se_block)  # Insert SE block before the second ReLU6


class CustomMNasNetMy(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, width_mult=1.):
        super(CustomMNasNetMy, self).__init__()

        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s, k
            # [3, 24,  3, 2, 3],  # -> 56x56
            # [3, 40,  3, 2, 5],  # -> 28x28
            # [6, 80,  3, 2, 5],  # -> 14x14
            # [6, 96,  2, 1, 3],  # -> 14x14
            # [6, 192, 4, 2, 5],  # -> 7x7
            # [6, 320, 1, 1, 3],  # -> 7x7

            [3, 12, 3, 2, 3],  # -> 56x56
            [3, 20, 3, 2, 5],  # -> 28x28
            [6, 40, 3, 2, 5],  # -> 14x14
            [6, 48, 2, 1, 3],  # -> 14x14
            [6, 96, 4, 2, 5],  # -> 7x7
            [6, 160, 1, 1, 3],  # -> 7x7

        ]

        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280

        self.dropout = nn.Dropout(p=drop_out)  # Dropout layer defined once

        # building first two layer
        self.features = [Conv_3x3(1, input_channel, 2), SepConv_3x3(input_channel, 16)]
        input_channel = 16

        # building inverted residual blocks (MBConv)
        for t, c, n, s, k in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    if use_se:
                        self.features.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k))
                    else:
                        self.features.append(InvertedResidual(input_channel, output_channel, s, t, k))
                else:
                    if use_se:
                        self.features.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k))
                    else:
                        self.features.append(InvertedResidual(input_channel, output_channel, 1, t, k))
                input_channel = output_channel

        # building last several layers
        self.features.append(Conv_1x1(input_channel, self.last_channel))
        self.features.append(nn.AdaptiveAvgPool2d(1))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(self.last_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        # x_rgb = x.repeat(1, 3, 1, 1)
        x = self.features(x)
        # x = self.se_block(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

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
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class CustomMNasNetMy1(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, width_mult=1.):
        super(CustomMNasNetMy1, self).__init__()

        # setting of inverted residual blocks
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
        # input_channel = int(32 * width_mult)
        self.last_channel = int(1960 * width_mult) if width_mult > 1.0 else 1960

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
                            self.down2.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k))
                        else:
                            self.down2.append(InvertedResidual(input_channel, output_channel, s, t, k))
                    else:
                        if use_se:
                            self.down2.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k))
                        else:
                            self.down2.append(InvertedResidual(input_channel, output_channel, 1, t, k))
                elif index == 1:
                    if i == 0:
                        if use_se:
                            self.down3.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k))
                        else:
                            self.down3.append(InvertedResidual(input_channel, output_channel, s, t, k))
                    else:
                        if use_se:
                            self.down3.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k))
                        else:
                            self.down3.append(InvertedResidual(input_channel, output_channel, 1, t, k))
                elif index == 2:
                    if i == 0:
                        if use_se:
                            self.down4.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k))
                        else:
                            self.down4.append(InvertedResidual(input_channel, output_channel, s, t, k))
                    else:
                        if use_se:
                            self.down4.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k))
                        else:
                            self.down4.append(InvertedResidual(input_channel, output_channel, 1, t, k))
                elif index == 3:
                    if i == 0:
                        if use_se:
                            self.down5.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k))
                        else:
                            self.down5.append(InvertedResidual(input_channel, output_channel, s, t, k))
                    else:
                        if use_se:
                            self.down5.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k))
                        else:
                            self.down5.append(InvertedResidual(input_channel, output_channel, 1, t, k))
                elif index == 4:
                    if i == 0:
                        if use_se:
                            self.down6.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k))
                        else:
                            self.down6.append(InvertedResidual(input_channel, output_channel, s, t, k))
                    else:
                        if use_se:
                            self.down6.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k))
                        else:
                            self.down6.append(InvertedResidual(input_channel, output_channel, 1, t, k))
                else:
                    if i == 0:
                        if use_se:
                            self.down7.append(InvertedResidualWithSE(input_channel, output_channel, s, t, k))
                        else:
                            self.down7.append(InvertedResidual(input_channel, output_channel, s, t, k))
                    else:
                        if use_se:
                            self.down7.append(InvertedResidualWithSE(input_channel, output_channel, 1, t, k))
                        else:
                            self.down7.append(InvertedResidual(input_channel, output_channel, 1, t, k))

                input_channel = output_channel
            index += 1

        self.down1 = nn.Sequential(*self.down1)
        self.down2 = nn.Sequential(*self.down2)
        self.down3 = nn.Sequential(*self.down3)
        self.down4 = nn.Sequential(*self.down4)
        self.down5 = nn.Sequential(*self.down5)
        self.down6 = nn.Sequential(*self.down6)
        self.down7 = nn.Sequential(*self.down7)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(self.last_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.down1(x)
        x = self.dropout(x)  # Dropout applied
        x = self.down2(x)
        x = self.dropout(x)  # Dropout applied
        x = self.down3(x)
        x = self.dropout(x)  # Dropout applied
        x = self.down4(x)
        x = self.dropout(x)  # Dropout applied
        x = self.down5(x)
        x = self.dropout(x)  # Dropout applied
        x = self.down6(x)
        x = self.dropout(x)  # Dropout applied
        x = self.down7(x)
        x = self.dropout(x)  # Dropout applied
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

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
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class PeopleCounter(nn.Module):
    def __init__(self):
        super(PeopleCounter, self).__init__()
        self.model = CustomMNasNetMy1(num_classes=1)

    def forward(self, x):
        x = self.model(x)
        return x


def custom_augmentation(image_labels, flip, rotate, dark, transforms1):
    images = image_labels[0]
    labels = image_labels[1] / divide_normalization_number_of_people
    output_images = []
    output_labels = []
    angles = [90, 180, -90, 45, -45, -180]
    if flip and rotate and dark:
        count = 4 + len(angles)
    elif flip and rotate:
        count = 2 + len(angles)
    elif flip and dark:
        count = 4
    elif dark and rotate:
        count = 3 + len(angles)
    elif flip:
        count = 2
    elif rotate:
        count = 1 + len(angles)
    elif dark:
        count = 3
    else:
        count = 1

    for ind in range(images.shape[0]):
        image = images[ind]
        label = labels[ind]
        images_list = []
        images_list.append(image)
        if dark:
            image_dark1 = image * 0.5
            images_list.append(image_dark1)
            image_dark2 = image * 0.3
            images_list.append(image_dark2)
        if flip:
            flipped_image = transforms.RandomHorizontalFlip(1)(image)
            images_list.append(flipped_image)
        if rotate:
            # rand_ind = (random.randint(0, 1000)) % 3
            # rand_angle = angles[rand_ind]
            for rand_angle in angles:
                coef = max(abs(math.sin(math.radians(rand_angle + 45))),
                           abs(math.sin(math.radians(90 - rand_angle - 45))))
                coef = math.sin(math.radians(45)) / coef
                downscale_size = (int(image.shape[1] * coef + 0.5), int(image.shape[2] * coef + 0.5))
                downscaled_image = F.resize(image, downscale_size, antialias=True)
                # Create an empty tensor of size 224x224
                output_size = (image.shape[1], image.shape[2])
                rotated_image = torch.zeros(1, output_size[0],
                                            output_size[1])  # Assuming RGB image, adjust channels if needed
                # Compute the starting position for inserting the downscaled image
                start_x = (output_size[0] - downscaled_image.size(1)) // 2
                start_y = (output_size[1] - downscaled_image.size(2)) // 2
                # Insert the downscaled image into the rotated image tensor
                rotated_image[0, start_x: start_x + downscaled_image.size(1),
                start_y: start_y + downscaled_image.size(2)] = downscaled_image

                # Rotate the image
                rotated_image = F.rotate(rotated_image, rand_angle)
                images_list.append(rotated_image)
                #   show
                # rotated_pil = F.to_pil_image(rotated_image[0, :, :])
                # plt.imshow(rotated_pil)
                # plt.axis('off')
                # plt.show()
        concatenated_images = torch.cat(images_list, dim=0)
        output_images.append(concatenated_images)
        output_labels.append(torch.tensor([label] * count))

    output_images1 = torch.stack(output_images, dim=0)
    output_labels1 = torch.stack(output_labels, dim=0)

    return output_images1, output_labels1


def chat_gpt_training(device_c, train_dataloader_c, valid_dataloader_c, num_epochs, path_to_save_model, csv_file_name):
    # Assuming your input images have 1 channels (RGB) and size 640x480
    model_c = PeopleCounter()
    if os.path.exists(path_to_save_model):
        model_c.load_state_dict(torch.load(path_to_save_model, map_location='cpu'))
        model_c.to(device_c)
        # model_c.eval()
        print("Used previously trained model ", path_to_save_model)

    total_params = sum(p.numel() for p in model_c.parameters())
    print(f"Number of parameters: {total_params}")
    # Define your loss function and optimizer
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    global learning_rate
    optimizer_c = optim.Adam(model_c.parameters(), lr=learning_rate)
    print('learning_rate = ', learning_rate)
    # optimizer_c = bnb.optim.Adam(model_c.parameters(), lr=0.001)

    model_c.to(device_c)

    # Training loop
    best_loss = float('inf')  # Initialize with a large value
    best_accuracy = 0.
    best_loss_epoch = -1
    best_accuracy_epoch = -1
    patience = 10  # Number of epochs to wait before early stopping
    counter = 0  # Counter for early stopping

    with open(csv_file_name, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["Epoch", "Training loss", "Training accuracy", "Validation loss", "Validation accuracy"])
        for epoch in range(num_epochs):
            model_c.train()
            train_loss = 0.0
            train_correct = 0
            number_of_train_data = 0
            steps = 0
            # for i, (original_images, flipped_images, rotated_images, labels) in enumerate(train_dataloader_c):
            for i, original_images_and_labels in enumerate(train_dataloader_c):
                steps += 1
                images, labels = custom_augmentation(original_images_and_labels, do_train_flip1, do_train_rotate1,
                                                     do_train_dark1, transforms)
                images = images.view(-1, 1, images.shape[2], images.shape[3])
                labels = labels.view(-1, 1)
                labels = labels.to(device_c)

                # Forward pass
                image = images.to(device_c)
                outputs = model_c(image)
                loss = criterion(outputs, labels.float())

                # Backward pass and optimization
                optimizer_c.zero_grad()
                loss.backward()
                optimizer_c.step()
                # Compute the total loss
                train_loss += loss.item() * image.shape[0]
                # Compute the accuracy
                predicted = torch.round(outputs.data * divide_normalization_number_of_people)
                gt_people = torch.round(labels * divide_normalization_number_of_people)
                train_correct += (predicted == gt_people).sum().item()
                number_of_train_data += image.shape[0]  # batch size

            # Evaluate on the validation set
            model_c.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_correct = 0
                number_of_valid_data = 0
                for inputs_v, labels_v in valid_dataloader_c:
                    labels_v = labels_v.to(device_c)
                    labels_v = labels_v / divide_normalization_number_of_people
                    labels_v = labels_v.unsqueeze(1)

                    # Forward pass
                    image = inputs_v.to(device_c)
                    outputs = model_c(image)
                    loss_v = criterion(outputs, labels_v.float())
                    val_loss += loss_v.item() * image.shape[0]
                    # Compute the accuracy
                    predicted = torch.round(outputs.data * divide_normalization_number_of_people)
                    gt_people = torch.round(labels_v * divide_normalization_number_of_people)
                    val_correct += (predicted == gt_people).sum().item()
                    number_of_valid_data += image.shape[0]  # batch size

            if (epoch % 10) == 0:
                print(number_of_train_data, number_of_valid_data)

            train_loss /= number_of_train_data  # Because of flipping and rotation  * 3
            train_accuracy = train_correct / number_of_train_data  # Because of flipping and rotation  * 3
            val_loss /= number_of_valid_data
            val_accuracy = val_correct / number_of_valid_data  # Because of flipping and rotation  * 3
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
            print(f"Epoch {epoch}/{num_epochs} steps {steps} - Training Loss: {train_loss:.4f} - "
                  f"Train accuracy: {train_accuracy:.4f} - "
                  f"Validation Loss: {val_loss:.4f} - "
                  f"Validation accuracy: {val_accuracy:.4f} - "
                  f"{formatted_datetime}"
                  )
            data = [epoch, train_loss, train_accuracy, val_loss, val_accuracy]
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
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_accuracy_epoch = epoch
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
    print(f"The best loss was in epoch {best_loss_epoch} and it is {best_loss:.4f}")
    print(f"The best accuracy was in epoch {best_accuracy_epoch} and it is {best_accuracy:.4f}")
    return model_c


def chat_gpt_test(device_c, test_dataloader, model_t):
    # Evaluate on the test set
    # model_t.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        test_loss = 0.0
        for inputs_t, labels_t in test_dataloader:
            inputs_t = inputs_t.to(device_c)
            labels_t = labels_t.to(device_c)
            labels_t = labels_t / divide_normalization_number_of_people
            labels_t = labels_t.unsqueeze(1)

            outputs = model_t(inputs_t)
            loss_t = criterion(outputs, labels_t.float())
            test_loss += loss_t.item()

    # Print training and validation losses for the epoch
    print(f"Test Loss: {test_loss / len(test_dataloader):.4f}")


def extract_parameter(filename):
    # Extract the parameter from the filename
    return int(filename.split('-')[1].split('_')[0])


def create_directory_dictionary(directory):
    # Create an empty dictionary
    directory_dict = {}

    # Iterate through the files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a JPG file
        if filename.endswith('.jpg'):
            # Extract the parameter from the filename
            parameter = extract_parameter(filename)

            # Add the filename and parameter to the dictionary
            directory_dict[filename] = parameter

    return directory_dict


class CustomSubset(Subset):
    def __init__(self, dataset, indices, is_train=True):
        super().__init__(dataset, indices)
        self.is_train = is_train
        self.do_flip = dataset.do_flip
        self.do_rotate = dataset.do_rotate

    # def __getitem__(self, idx):
    #     # img_path, label = list(self.img_labels.items())[idx]
    #     # img_path = os.path.join(self.img_dir, img_path)
    #     # image = read_image(img_path)
    #     # image = image.float()
    #     # if self.transform:
    #     #     image = self.transform(image)
    #     image, label = super().__getitem__(idx)
    #     if self.is_train:
    #         if self.do_flip:
    #             flipped_image = transforms.RandomHorizontalFlip(1)(image)
    #         if self.do_rotate:
    #             angles = [90, 180, -90]
    #             rand_ind = (random.randint(0, 1000)) % 3
    #             rand_angle = angles[rand_ind]
    #             coef = max(abs(math.sin(math.radians(rand_angle + 45))),
    #                        abs(math.sin(math.radians(90 - rand_angle - 45))))
    #             coef = math.sin(math.radians(45)) / coef
    #             # rand_angle = 0
    #             # while rand_angle == 0:
    #             #     rand_angle = float(random.randint(0, 180) - 90)
    #             #     coef = max(abs(math.sin(math.radians(rand_angle))), abs(math.sin(math.radians(90 - rand_angle))))
    #             downscale_size = (int(image.shape[1] * coef + 0.5), int(image.shape[2] * coef + 0.5))
    #             downscaled_image = F.resize(image, downscale_size, antialias=True)
    #             # Create an empty tensor of size 224x224
    #             output_size = (image.shape[1], image.shape[2])
    #             rotated_image = torch.zeros(1, output_size[0],
    #                                         output_size[1])  # Assuming RGB image, adjust channels if needed
    #             # Compute the starting position for inserting the downscaled image
    #             start_x = (output_size[0] - downscaled_image.size(1)) // 2
    #             start_y = (output_size[1] - downscaled_image.size(2)) // 2
    #             # Insert the downscaled image into the rotated image tensor
    #             rotated_image[0, start_x: start_x + downscaled_image.size(1),
    #             start_y: start_y + downscaled_image.size(2)] = downscaled_image
    #
    #             # Rotate the image
    #             rotated_image = F.rotate(rotated_image, rand_angle)
    #             #   show
    #             # rotated_pil = F.to_pil_image(rotated_image[0, :, :])
    #             # plt.imshow(rotated_pil)
    #             # plt.axis('off')
    #             # plt.show()
    #         if self.do_flip and self.do_rotate:
    #             concatenated_images = torch.cat((image, flipped_image, rotated_image), dim=0)
    #             repeated_label = torch.tensor([label] * 3)
    #             return concatenated_images, repeated_label
    #         elif self.do_flip:
    #             concatenated_images = torch.cat((image, flipped_image), dim=0)
    #             repeated_label = torch.tensor([label] * 2)
    #             return concatenated_images, repeated_label
    #         elif self.do_rotate:
    #             concatenated_images = torch.cat((image, rotated_image), dim=0)
    #             repeated_label = torch.tensor([label] * 2)
    #             return concatenated_images, repeated_label
    #         else:
    #             return image, label
    #     else:
    #         return image, label


class CustomImageDataset(Dataset):
    # def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    def __init__(self, img_dir, do_flip, do_rotate, transform=None, target_transform=None):
        self.img_labels = create_directory_dictionary(img_dir)  # dictionary
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.do_flip = do_flip
        self.do_rotate = do_rotate

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = list(self.img_labels.items())[idx]
        img_path = os.path.join(self.img_dir, img_path)
        image = read_image(img_path)
        image = image.float()
        if self.transform:
            image = self.transform(image)
        return image, label


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
                          batch_train_valid, batch_test, valid_size=.15):
    test_dataset = CustomImageDataset(test_data_dir, False, False, train_valid_test_composed_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_test)
    if valid_size > 0:  # take validation from training ifvalid_size > 0
        train_dataset_full = CustomImageDataset(train_data_dir, do_train_flip1, do_train_rotate1,
                                                train_valid_test_composed_transform)
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
        train_dataset_full = CustomImageDataset(train_data_dir, do_train_flip1, do_train_rotate1,
                                                train_valid_test_composed_transform)
        train_indices = range(len(train_dataset_full))
        train_dataset = CustomSubset(train_dataset_full, train_indices, True)
        num_train = len(train_dataset_full)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_train_valid)
        valid_dataset_full = CustomImageDataset(valid_data_dir, False, False, train_valid_test_composed_transform)
        valid_indices = range(len(valid_dataset_full))
        valid_dataset = CustomSubset(valid_dataset_full, valid_indices, False)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_train_valid)

    return train_loader, valid_loader, test_loader


def infer_one_frame(filename_without_ext, directory_name, test_composed_transform, model, csv_writer, device_i, count,
                    loss, loss_round):
    if not directory_name1:
        filename = filename_without_ext + '.jpg'
        full_frame_path = filename
    else:
        filename = filename_without_ext
        full_frame_path = os.path.join(directory_name, filename)
    if os.path.exists(full_frame_path):
        # cap = cv2.VideoCapture(filename)
        # _, frame = cap.read()
        # bit_depth = frame.dtype.itemsize * 8
        # frame_width = int(cap.get(3))
        # frame_height = int(cap.get(4))
        # print(int(cap.get(cv2.CAP_PROP_FPS)), frame_width, frame_height)
        # success, img = cap.read()
        # aa = 0

        count += 1
        # Extract the number from the filename
        if directory_name:
            number = extract_parameter(filename)
        else:
            number = 0
        # Read the frame
        # frame = cv2.imread(full_frame_path)
        if not directory_name:
            img = cv2.imread(full_frame_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(full_frame_path, img)  # save frame as png file

        frame = read_image(full_frame_path)
        frame = frame.float()

        # Convert the frame to a PIL Image
        # pil_image = Image.fromarray(frame)

        # Downscale the frame, do gray scale
        # Convert to tensor and normalize

        input_tensor = test_composed_transform(frame).unsqueeze(0).to(device_i)
        # if network_type == 'MnasNet':
        #     input_tensor = input_tensor.repeat(1, 3, 1, 1)  # Repeat the single-channel image to make it 3-channel
        # model.model.mnasnet.layers[0].in_channels = 1

        # summary(model, (input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
        # total_params = sum(p.numel() for p in model.parameters())
        # print(f"Number of parameters: {total_params}")

        # Perform inference
        with torch.no_grad():
            # visualize_feature_maps_inception(model, input_tensor)
            # visualize_feature_maps_mnas(model, input_tensor)
            output = model(input_tensor)

        # Extract the people count from the output
        people_count1f = output.item() * divide_normalization_number_of_people
        people_count1 = int(round(people_count1f))
        loss += (people_count1f - number) * (people_count1f - number)
        loss_round += (people_count1 - number) * (people_count1 - number)

        if directory_name:
            # if 0 < number < 6 and number != people_count1:
            #     aa = 0
            print(f"Number from file '{filename}': {number} : {people_count1}: {people_count1f}")
            g1 = 0
            g2 = 1
            if number == people_count1:
                g1 = 1
            if abs(number - people_count1) > 1:
                g2 = 0
            if directory_name:
                global outliers_dir, outliers_threshold
                if copy_outliers and abs(people_count1f - number) > outliers_threshold:
                    file_name_without_extension1 = os.path.basename(filename_without_ext).split('.')[0]
                    outliers_dir = os.path.join(os.path.dirname(directory_name), outliers_dir)
                    shutil.copy(os.path.join(directory_name, filename_without_ext),
                                os.path.join(outliers_dir, filename_without_ext))
                    shutil.copy(os.path.join(directory_name, file_name_without_extension1 + '.png'),
                                os.path.join(outliers_dir, file_name_without_extension1 + '.png'))
                    if os.path.exists(os.path.join(directory_name, file_name_without_extension1 + '.txt')):
                        shutil.copy(os.path.join(directory_name, file_name_without_extension1 + '.txt'),
                                    os.path.join(outliers_dir, file_name_without_extension1 + '.txt'))

                csv_writer.writerow([filename, number, people_count1, people_count1f, g1, g2])
        else:
            cv2.imshow(str(people_count1), img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # if abs(number - people_count1) > 1:
        # if abs(number - people_count1) > 0:
        #     shutil.move(full_frame_path, outliers_dir)

    return count, loss, loss_round


def infer_people_count(frame_filename, directory_name, model_path, test_composed_transform, csv_file_name,
                       outliers_dir):
    # Load the saved model
    device_i = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PeopleCounter()
    # summary(model, (1, 320, 240))
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Number of parameters: {total_params}")

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device_i)
    model.eval()
    loss = 0
    loss_round = 0
    count = 0

    with open(csv_file_name, 'w', newline='') as csvfile:
        if directory_name is not None:
            if os.path.exists(directory_name):
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(
                    ["jpg file name", "gt number of people", "rounded predicted", "float predicted", "equal", "less 2"])
                for filename in os.listdir(directory_name):
                    if filename.endswith('.jpg'):
                        count, loss, loss_round = infer_one_frame(filename, directory_name, test_composed_transform,
                                                                  model,
                                                                  csv_writer, device_i, count, loss, loss_round)
                if count:
                    loss /= count
                    loss_round /= count
                    print(f"Loss: {loss:.4f} - "
                          f"loss_round: {loss_round:.4f} - "
                          )
        # else:
        #     infer_one_frame(frame_filename, directory_name, test_composed_transform, model, csvfile, device_i, count,
        #                     loss, loss_round)

    return count, loss, loss_round


def main(batch_train_valid, batch_test, training, model_save_name, train_valid_test_composed_transform,
         test_composed_transform, csv_file_name,
         outliers_dir, valid_size):
    if training:
        if local_machine:
            # training_path = "c:\\CamDatabase\\Training1\\Images"
            # training_path = "c:\\CamDatabase\\FramesWithoutMasks\\ValidHomogeneous"
            training_path = "c:\\CamDatabase\\PositionsDB\\TestThin"
            # training_path = "c:\\CamDatabase\\FramesWithoutMasks\\TestHomogeneous"
            valid_path = "c:\\CamDatabase\\FramesWithoutMasks\\TestHomogeneous"
            test_path = "c:\\CamDatabase\\FramesWithoutMasks\\TestHomogeneous"
        else:  # cloud
            training_path = "PositionsDB/TrainingThin"
            valid_path = "PositionsDB/Test"
            test_path = "PositionsDB/Test"
            # training_path = "Images"
            # valid_path = "ValidHomogeneous"
            # test_path = "TestHomogeneous"
        train_dataloader1, valid_dataloader1, test_dataloader1 = load_split_train_test(training_path, valid_path,
                                                                                       test_path,
                                                                                       train_valid_test_composed_transform,
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
        model1 = chat_gpt_training(device1, train_dataloader1, valid_dataloader1, num_epochs1, path_to_save_model1,
                                   csv_file_name)
        chat_gpt_test(device1, test_dataloader1, model1)
        aa = 0
    else:  # inference
        # frame_filename1 = "c:/CamDatabase/Test/Images/PG1_dev_H250_HD_designed_No_Occlusion_1509275690-4_1596.jpg"
        # frame_filename1 = "c:/CamDatabase/Test/Images/PG1_dev_H250_HD_designed_No_Occlusion_1509275690-4_1596.jpg"
        # directory_name1 = "c:/CamDatabase/Test1/Tests"
        # directory_name1 = "c:/CamDatabase/Frames/ExperimentsInput3"
        # directory_name1 = "c:/CamDatabase/FramesWithoutMasks/TestHomogeneous"
        model_path1 = model_save_name
        if not directory_name1:
            for ind in range(10):
                take_image_from_camera(host1, cam_filename1 + str(ind))
        infer_people_count(cam_filename1, directory_name1, model_path1, test_composed_transform,
                           csv_file_name, outliers_dir)


if __name__ == '__main__':
    batch_train_valid1 = 6
    batch_test1 = 1
    num_epochs1 = 2000

    mean1 = 0
    std1 = 255
    height = 0
    width = 0  # for rescaling
    height = 224
    width = 224

    train_valid_test_composed_transform1 = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((height, width), antialias=True),
        transforms.Normalize(mean=[mean1], std=[std1])
    ])
    test_composed_transform1 = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((height, width), antialias=True),
        transforms.Normalize(mean=[mean1], std=[std1])
    ])
    outliers_dir1 = "c:\\CamDatabase\\FramesWithoutMasks\\Outliers"
    main(batch_train_valid1, batch_test1, training1, model_save_name1, train_valid_test_composed_transform1,
         test_composed_transform1, csv_file_name1, outliers_dir1, valid_size1)
