import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import torch
from torch import nn
from torchvision import transforms, models, utils
from torchvision.io import read_image
import cv2
import csv
import shutil

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
output_image_size = 224
bb_half_size = 6
# letter = 'mu'
letter = ''
b_there_is_ref = True
b_write = True
min_area = 15
bin_threshold = 0.06
reduction_for_SE = 16
local_machine = True
use_se = False
if use_se:
    letter = '_SE'
copy_outliers = False
outliers_threshold = 0.01
import time

mean1 = 0
std1 = 255
height = 224
width = 224

if local_machine:
    # model_save_name1 = 'PTH_UNET_LOCAL_TRAINING\\MNSA_Unet' + letter + '.pth'
    # model_save_name1 = 'PTH_MNSA_Unet/MNSA_UnetDecInvResKer3th' + letter + '.pth'
    # model_save_name0 = 'PTH_MNAS_Unet/MNAS_UnetDecInvResKer3_Pos' + letter + '.pth'
    model_save_name0 = 'MNAS_Unet' + letter + '.pth'
    model_save_name1 = None
    # model_save_name0 = 'PTH_MNSA_Unet_EC2\MNSA_UnetDecInvResKer3_Pos.pth'
    # model_save_name1 = 'PTH_MNSA_Unet_EC2\MNSA_UnetDecInvResKer3_Pos_SE.pth'
    # csv_file_name1 = "c:/CamDatabase/Frames/ExperimentsOutputEC2_1/1.csv"
    # output_masks_dir1 = "c:/CamDatabase/Frames/ExperimentsOutputEC2_2"
    output_masks_dir1 = "c:/CamDatabase/Frames/ExperimentsOutputEC2_2"
    csv_file_name1 = output_masks_dir1 + "/0.06s" + letter + ".csv"
    # csv_file_name1 = "c:/CamDatabase/Frames/ExperimentsOutput9/1.csv"
    directory_name1 = "c:/CamDatabase/PositionsDB/TestThin"
    outliers_dir = 'Outliers'
else:
    # model_save_name1 = 'PTH_MNSA_Unet/MNSA_Unet' + letter + '.pth'
    model_save_name0 = 'PTH_MNAS_Unet_Pos/MNAS_Unet' + letter + '.pth'
    model_save_name1 = None
    csv_file_name1 = 'PTH_MNAS_Unet_Pos/MNAS_Unet' + letter + '.csv'
    directory_name1 = 'PositionsDB/Training2'
    output_masks_dir1 = 'Temp'
    outliers_dir = 'PositionsDB/OutliersTraining2'


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


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
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
        x2 = self.down2(nn.MaxPool2d(2)(x1))
        x3 = self.down3(nn.MaxPool2d(2)(x2))
        x4 = self.down4(nn.MaxPool2d(2)(x3))
        x = self.up1(x4)
        x = self.up2(torch.cat([x, x3], dim=1))
        x = self.up3(torch.cat([x, x2], dim=1))
        x = self.up4(torch.cat([x, x1], dim=1))
        # x = self.sigmoid(x)
        return x


def Conv_3x3(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
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


class CustomMNasNetUnetInvResDec(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes=1, input_size=224, width_mult=1.):
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
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x = self.down7(x6)
        x = self.up1(x)
        x = torch.cat([x, x5], dim=1)
        x = self.up1a(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up2a(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up3a(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up4a(x)
        x = self.up5(x)
        return x


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
        h = bb_half_size * 2
        w = bb_half_size * 2

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


def find_average_prob_below_threshold(probs, threshold=0.5):
    # Convert probabilities to a boolean mask where True corresponds to probabilities less than 0.5
    mask_for_small_prob = probs < threshold
    # Calculate the sum and count of probabilities less than threshold
    sum_probabilities = torch.sum(probs[mask_for_small_prob])
    count_probabilities = torch.sum(mask_for_small_prob)
    # Calculate the average if there are probabilities less than 0.5
    average_small_probability = sum_probabilities / count_probabilities if count_probabilities > 0 else 0.0
    return average_small_probability.item()


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


def calculate_metrics(ground_truth1, predicted):
    ground_truth = ground_truth1.copy()
    true_positive = 0
    false_positive = 0

    if not ground_truth and not predicted:
        # Both arrays are empty
        # return {'accuracy': 1, 'precision': 1, 'recall': 1}
        return 1, 1, 1

    for pred_foot in predicted:
        pred_x, pred_y = pred_foot
        found_intersection = False

        for ind in range(len(ground_truth)):
            gt_foot = ground_truth[ind]
            gt_x, gt_y = gt_foot

            # Check for intersection
            x_overlap = max(0, min(pred_x + bb_half_size, gt_x + bb_half_size) - max(pred_x - bb_half_size, gt_x - bb_half_size))
            y_overlap = max(0, min(pred_y + bb_half_size, gt_y + bb_half_size) - max(pred_y - bb_half_size, gt_y - bb_half_size))
            intersection_area = x_overlap * y_overlap

            if intersection_area > 0:
                # Intersection found
                true_positive += 1
                found_intersection = True
                ground_truth.pop(ind)
                break

        if not found_intersection:
            # No intersection found, increment false positive
            false_positive += 1

    ground_truth = ground_truth1.copy()
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
    # if filename_without_ext == 'PG1_dev_H250_HD_designed_No_Occlusion_1513593256-3_1626.jpg':
    #     aa = 0
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

    global outliers_dir, outliers_threshold
    if os.path.exists(full_frame_path):
        count += 1
        if not directory_name:
            img = cv2.imread(full_frame_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(full_frame_path, img)  # save frame as png file

        frame = read_image(full_frame_path)
        frame = frame.float()

        if b_there_is_ref:
            mask = read_image(full_mask_path)
            mask = mask.float()

        criterion = nn.BCELoss()
        input_tensor = test_composed_transform(frame).unsqueeze(0).to(device_i)
        if b_there_is_ref:
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
            # Calculate the average of the probabilities less than threshold
            average_small_probability_0_5_t = find_average_prob_below_threshold(output_sig, 0.5)
            if b_there_is_ref:
                frame_loss_t = criterion(output_sig, mask_tensor)
                frame_loss.append(frame_loss_t.item())
            else:
                frame_loss.append(0)
            average_small_probability_0_5.append(average_small_probability_0_5_t)
            binarized = torch.where(output_sig < bin_threshold, torch.tensor(0), torch.tensor(255))
            predicted_mask_np = binarized.squeeze(0).cpu().numpy().squeeze(0)
            predicted_mask_name = name + '_mask' + str(ind) + '.jpg'
            full_predicted_mask_name = os.path.join(output_masks_dir, predicted_mask_name)
            if b_write:
                cv2.imwrite(full_predicted_mask_name, predicted_mask_np)
            predicted_mask_with_bb_np = np.copy(predicted_mask_np)
            if b_there_is_ref:
                binarized_original_mask_with_bb_np = np.copy(binarized_original_mask_np)
            original_with_bb_np = input_tensor * 255
            original_with_bb_np = original_with_bb_np.squeeze(0).cpu().numpy().squeeze(0)

            foot_points_predicted, bounding_boxes_predicted, predicted_mask_with_bb_np, predicted_mask_np, original_with_bb_np = filter_and_draw_bounding_boxes(
                predicted_mask_with_bb_np, original_with_bb_np, min_area)
            if b_there_is_ref:
                foot_points_original, bounding_boxes_original, binarized_original_mask_with_bb_np, binarized_original_mask_np, t = filter_and_draw_bounding_boxes(
                    binarized_original_mask_with_bb_np, original_with_bb_np, 0)
                accuracy, precision, recall = calculate_metrics(foot_points_original, foot_points_predicted)
                IoUt = iou_loss(predicted_mask_with_bb_np / 255, binarized_original_mask_with_bb_np / 255)
            else:
                IoUt = 0
            if b_there_is_ref:
                data.append(frame_loss_t.item())
            else:
                data.append(0)
            if b_there_is_ref:
                data.append(accuracy)
                data.append(precision)
                data.append(recall)
                if copy_outliers and precision < outliers_threshold and recall < outliers_threshold and abs(len(foot_points_original) - len(foot_points_predicted)) > 3:
                    file_name_without_extension1 = os.path.basename(filename_without_ext).split('.')[0]
                    if local_machine:
                        outliers_dir = os.path.join(os.path.dirname(directory_name), outliers_dir)
                    # print(os.path.join(directory_name, filename_without_ext), os.path.join(outliers_dir, filename_without_ext))
                    shutil.move(os.path.join(directory_name, filename_without_ext), os.path.join(outliers_dir, filename_without_ext))
                    shutil.move(os.path.join(directory_name, file_name_without_extension1 + '.png'), os.path.join(outliers_dir, file_name_without_extension1 + '.png'))
                    if os.path.exists(os.path.join(directory_name, file_name_without_extension1 + '.txt')):
                        shutil.move(os.path.join(directory_name, file_name_without_extension1 + '.txt'), os.path.join(outliers_dir, file_name_without_extension1 + '.txt'))

            else:
                data.append(0)
                data.append(0)
                data.append(0)
                data.append(0)
                data.append(0)

            mask_with_bb_filename = name + '_mask_bb_' + str(ind) + extension
            full_mask_with_bb_path = os.path.join(output_masks_dir, mask_with_bb_filename)
            original_with_bb_filename = name + '_bb_' + str(ind) + extension
            full_original_with_bb_path = os.path.join(output_masks_dir, original_with_bb_filename)

            if b_write:
                cv2.imwrite(full_mask_with_bb_path, predicted_mask_with_bb_np)
                cv2.imwrite(full_original_with_bb_path, original_with_bb_np)
                if b_there_is_ref:
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
            if b_there_is_ref:
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


def infer_unet(directory_name, output_masks_dir, model0_path, model1_path, test_composed_transform, mask_transform, csv_file_name):
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

    if os.path.exists(directory_name):
        jpg_files = [f for f in os.listdir(directory_name) if f.endswith('.jpg')]
        with open(csv_file_name, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["jpg file name", "loss", "accuracy", "precision", "recall"])
                # csv_writer.writerow(["jpg file name", "loss", "IoU loss", "bin_th", "prob < 0.5 for", "accuracy", "precision", "recall", "bb accuracy", "bb accuracy999", "bb accuracy99", "bb accuracy90", "bb accuracy70", "bb accuracy50", "bb accuracy40", "bb accuracy30", "bb accuracy25", "bb accuracy20","bb accuracy15"])

            # jpg_files1 = [jpg_files[0], jpg_files[1]]
            for filename in jpg_files:
                # infer_one_frame1(filename, directory_name, output_masks_dir,
                #                                         test_composed_transform, mask_transform, model,
                #                                         device_i, count, loss, loss_bin, csv_writer)
                infer_one_frame(filename, directory_name, output_masks_dir,
                                                        test_composed_transform, mask_transform, models_i,
                                                        device_i, count, loss, loss_bin, csv_writer)

    return count, loss, loss_bin


def main(model_save_name, test_composed_transform, mask_transform):

    device1 = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device1} device")
    path_to_save_model1 = model_save_name
    frame_filename1 = None
    # frame_filename1 = "c:/CamDatabase/Test/Images/PG1_dev_H250_HD_designed_No_Occlusion_1509275690-4_1596.jpg"
    # directory_name1 = "c:/CamDatabase/Test1/Tests"
    # directory_name1 = "c:/CamDatabase/FramesWithoutMasks/TestHomogeneous"
    model_path0 = model_save_name0
    model_path1 = model_save_name1
    # if not directory_name1 and not frame_filename1:
    #     for ind in range(10):
    #         take_image_from_camera(host1, cam_filename1 + str(ind))
    infer_unet(directory_name1, output_masks_dir1, model_path0, model_path1, test_composed_transform,
               mask_transform, csv_file_name1)


if __name__ == '__main__':

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
    main(model_save_name1, test_composed_transform1, mask_transform1)
