import os
import numpy as np
from scipy.spatial.distance import cdist
import torch
from torch import nn
from torchvision import transforms, models, utils
from torchvision.io import read_image
import onnx
import onnxruntime
import cv2
import csv
import shutil
import math

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
output_image_size = 224
distance_threshold = 22
# letter = 'mu'
letter = '_drop'
b_there_is_ref = True
b_write = True
min_area = 12
max_area_to_divide = 90
max_area_to_divide_component = 10000
bin_threshold_hard = 0.13
reduction_for_SE = 16
local_machine = True
use_se = False
if use_se:
    letter = '_SE'
copy_outliers = False
outliers_threshold = 0.01

# model_save_name0 = 'PTH_MNAS_Unet_EC2_Drop1/MNAS_Unet_drop02rad15.pth'
model_save_name0 = 'PTH_MNAS_Unet/MNAS_Unet_drop02rad17.pth'
model_save_name1 = None
output_masks_dir1 = "c:/CamDatabase/Frames/ExperimentsOutputEC2_dist22b"
csv_file_name1 = output_masks_dir1 + "/rad17_dist22_0.13_12_90s3" + letter + ".csv"
directory_name1 = "c:/CamDatabase/PositionsDB/TestThin"
outliers_dir = 'Outliers'


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


def filter_and_draw_bounding_boxes(predicted_mask_with_bb_np, image_original1, color, foot_points_if_there_is, min_area=None, max_boxes=None):
    # Find connected components
    # image_original = np.copy(image_original1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(predicted_mask_with_bb_np.astype(np.uint8))

    # Create a new mask to store the filtered components
    filtered_mask = np.zeros_like(predicted_mask_with_bb_np).astype(np.uint8)
    filtered_mask_with_bb = np.zeros_like(predicted_mask_with_bb_np).astype(np.uint8)

    # Create a list to store bounding boxes and areas
    bounding_boxes = []
    foot_points = []

    max_area = 0
    for label in range(1, num_labels):
        # Get the statistics of the component
        x, y, w, h, area = stats[label]
        if area > max_area:
            max_area = area
        if min_area is not None and area < min_area:
            black_box = filtered_mask_with_bb[y:y + h, x:x + w]
            predicted_mask_with_bb_np[y:y+h, x:x+w] = black_box

    # Iterate through each component (excluding the background component)
    b_there_is_big_component = True
    image_original = np.copy(image_original1)
    while b_there_is_big_component:
        b_there_is_big_component = False
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(predicted_mask_with_bb_np.astype(np.uint8))
        for label in range(1, num_labels):
            # Get the statistics of the component
            x, y, w, h, area = stats[label]

            if area > max_area_to_divide_component:
                kernel = np.ones((5, 5), np.uint8)
                eroded_box = cv2.erode(predicted_mask_with_bb_np.astype(np.uint8)[y:y + h, x:x + w], kernel, iterations=1)
                num_labels_b, labels_b, stats_b, centroids_b = cv2.connectedComponentsWithStats(eroded_box)
                predicted_mask_with_bb_np[y:y+h, x:x+w] = eroded_box
                if num_labels_b > 2:
                    b_there_is_big_component = True
                    break

    for label in range(1, num_labels):
        # Get the statistics of the comp`onent
        x, y, w, h, area = stats[label]
        x = round(centroids[label][0])
        y = round(centroids[label][1])

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
        # foot_points.append((x + w // 2, y + h // 2))
        foot_points.append((x, y))

    filtered_mask_with_bb1 = np.zeros((filtered_mask_with_bb.shape[1], filtered_mask_with_bb.shape[0], 3), dtype=np.uint8)
    if foot_points_if_there_is is not None:
        foot_points = foot_points_if_there_is
    for point in foot_points:
        cv2.circle(filtered_mask_with_bb1, point, distance_threshold // 2, color, thickness=1)
        cv2.circle(image_original, point, distance_threshold // 2, (255, 255, 255), thickness=1)

    # Draw bounding boxes on the original image
    # for box in bounding_boxes:
    #     x, y, w, h = box
    #     cv2.rectangle(filtered_mask_with_bb, (x, y), (x + w, y + h), (255, 255, 255), -1)    #   -1 fills
    #     cv2.rectangle(image_original, (x, y), (x + w, y + h), (255, 255, 255), 1)

    return foot_points, bounding_boxes, filtered_mask_with_bb1, filtered_mask, image_original, max_area


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


def find_average_prob_over_threshold(probs, threshold=0.5):
    # Convert probabilities to a boolean mask where True corresponds to probabilities less than 0.5
    mask_for_small_prob = probs > threshold
    # Calculate the sum and count of probabilities less than threshold
    sum_probabilities = torch.sum(probs[mask_for_small_prob])
    count_probabilities = torch.sum(mask_for_small_prob)
    # Calculate the average if there are probabilities less than 0.5
    average_small_probability = (sum_probabilities / count_probabilities).item() if count_probabilities > 0 else 0.0
    return average_small_probability


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


def find_bin_threshold_people_count(probs, actual_number_of_people):
    epsilon = 1e-6
    lower_bound = 0.0
    upper_bound = 1.0
    count = 0
    # binarized = torch.where(output_sig < bin_threshold, torch.tensor(0), torch.tensor(255))

    while upper_bound - lower_bound > epsilon and count < 100:
        threshold = (lower_bound + upper_bound) / 2
        binarized_image = torch.where(probs < threshold, torch.tensor(0), torch.tensor(255))
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binarized_image.squeeze(0).cpu().numpy().squeeze(0).astype(np.uint8))

        if num_labels - 1 > actual_number_of_people:
            lower_bound = threshold
        else:
            if num_labels - 1 < actual_number_of_people:
                upper_bound = threshold
            else:
                break
        count += 1
    if threshold != (lower_bound + upper_bound) / 2:
        aa = 0
    if actual_number_of_people == 0:
        aa = 0
    return (lower_bound + upper_bound) / 2


def calculate_metrics(ground_truth_points, predicted_points, threshold):
    true_positive = 0
    false_positive = 0

    if len(ground_truth_points) == 0 and len(predicted_points) == 0:
        # Both arrays are empty, set metrics to 1
        return 1, 1, 1

    if len(ground_truth_points) == 0:
        # No ground truth points, set precision to 0, recall to 1, accuracy to 0
        return 0, 0, 1

    if len(predicted_points) == 0:
        # No predicted points, set precision to 0, recall to 0, accuracy to 0
        return 0, 0, 0

    matched_gt_indices = set()
    matched_predicted_indices = set()

    distances = cdist(predicted_points, ground_truth_points)

    for i in range(len(ground_truth_points)):
        min_distance = 1000
        min_index = -1
        for j in range(len(predicted_points)):
            if j not in matched_predicted_indices:
                if distances[j, i] < min_distance:
                    min_distance = distances[j, i]
                    min_index = j
        # min_distance = min(distances[i])
        # min_index = distances[i].argmin()

        if min_distance <= threshold:
            if i not in matched_gt_indices:
                true_positive += 1
                matched_gt_indices.add(i)
                matched_predicted_indices.add(min_index)
            else:
                false_positive += 1
        elif min_distance < 1000:
            false_positive += 1

    false_negative = len(ground_truth_points) - len(matched_gt_indices)
    false_positive = len(predicted_points) - len(matched_predicted_indices)

    accuracy = true_positive / max(1, (true_positive + false_positive + false_negative))
    precision = true_positive / max(1, len(predicted_points))
    recall = true_positive / max(1, len(ground_truth_points))

    return accuracy, precision, recall


def calculate_metrics2(ground_truth_points, predicted_points, threshold):
    true_positive = 0
    false_positive = 0

    if len(ground_truth_points) == 0 and len(predicted_points) == 0:
        # Both arrays are empty, set metrics to 1
        return 1, 1, 1

    if len(ground_truth_points) == 0:
        # No ground truth points, set precision to 0, recall to 1, accuracy to 0
        return 0, 0, 1

    if len(predicted_points) == 0:
        # No predicted points, set precision to 0, recall to 0, accuracy to 0
        return 0, 0, 0

    matched_gt_indices = set()

    distances = cdist(predicted_points, ground_truth_points)

    for i in range(len(predicted_points)):
        min_distance = min(distances[i])
        min_index = distances[i].argmin()

        if min_distance <= threshold:
            if min_index not in matched_gt_indices:
                true_positive += 1
                matched_gt_indices.add(min_index)
            else:
                false_positive += 1
        else:
            false_positive += 1

    false_negative = len(ground_truth_points) - len(matched_gt_indices)

    accuracy = true_positive / max(1, (true_positive + false_positive + false_negative))
    precision = true_positive / max(1, len(predicted_points))
    recall = true_positive / max(1, len(ground_truth_points))

    return accuracy, precision, recall


def scale_coordinates(txt_file, scale_x, scale_y):
    scaled_points = []
    if os.path.exists(txt_file):
        with open(txt_file, 'r') as file:
            for line in file:
                # Split the line into numbers
                numbers = list(map(float, line.split()))

                # Scale the last two coordinates
                scaled_x = numbers[-2] * scale_x
                scaled_y = numbers[-1] * scale_y

                # Append the scaled coordinates to the result list
                scaled_points.append((scaled_x, scaled_y))

    return scaled_points


def onnx_import(input_tensor, output, model0):
    torch.onnx.export(model0,  # model being run
                      input_tensor,  # model input (or a tuple for multiple inputs)
                      "super_resolution.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    onnx_model = onnx.load("super_resolution.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession("super_resolution.onnx", providers=["CPUExecutionProvider"])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_tensor)}
    ort_outs = ort_session.run(None, ort_inputs)
    tensor_data = torch.tensor(ort_outs[0])
    output_sig = torch.sigmoid(tensor_data)
    output_sig_numpy = output_sig.numpy()
    output = output.detach().numpy()
    abs_diff = np.abs(output - output_sig_numpy)
    max_difference = np.max(abs_diff)
    average_difference = np.mean(abs_diff)
    # compare ONNX Runtime and PyTorch results
    # np.testing.assert_allclose(output, output_sig_numpy, rtol=1e-02, atol=1e-03)
    # np.testing.assert_allclose(to_numpy(output), output_sig, rtol=1e-03, atol=1e-05)

    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    return output_sig


def infer_one_frame(filename_without_ext, directory_name, output_masks_dir, test_composed_transform, mask_transform,
                    models_inf, device_i, count,
                    loss, loss_bin, csv_writer):
    # if filename_without_ext == 'PG1_dev_H250_HD_designed_No_Occlusion_1513593256-3_1626.jpg':
    #     aa = 0
    actual_number_of_people = int(filename_without_ext.split('-')[1].split('_')[0])
    if not directory_name1:
        filename = filename_without_ext + '.jpg'
        full_frame_path = filename
    else:
        filename = filename_without_ext
        full_frame_path = os.path.join(directory_name, filename)
        name, extension = os.path.splitext(filename)
        mask_filename = name + ".png"
        txt_filename = name + ".txt"
        txt_filename = os.path.join(directory_name, txt_filename)
        output_original_mask_name = name + "_original_mask.jpg"
        output_predicted_full_range_mask_name = name + "_full_range.jpg"
        output_both_masks_name = name + "_both_masks.jpg"

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
        scale_x = output_image_size / frame.shape[2]
        scale_y = output_image_size / frame.shape[1]
        foot_points_original_txt = scale_coordinates(txt_filename, scale_x, scale_y)
        # Assuming you already have the scaled_data from the previous function call
        rounded_foot_points_original = [(round(x), round(y)) for x, y in foot_points_original_txt]

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
        bin_thresh_people = []
        average_small_probability_0_5 = []
        average_big_probability_0_5 = []
        for ind in range(len(models_inf)):
            model = models_inf[ind]
            output = model(input_tensor)
            output_sig = torch.sigmoid(output)
            # output_sig = onnx_import(input_tensor, output_sig, model)
            bin_thresh_t = find_binarization_threshold(output_sig)
            bin_thresh.append(bin_thresh_t)
            # Calculate the average of the probabilities less than threshold
            average_small_probability_0_5_t = find_average_prob_below_threshold(output_sig, bin_threshold_hard)
            average_big_probability_0_5_t = find_average_prob_over_threshold(output_sig, bin_threshold_hard)
            frame_loss_t = criterion(output_sig, mask_tensor)
            frame_loss.append(frame_loss_t.item())
            average_small_probability_0_5.append(average_small_probability_0_5_t)
            average_big_probability_0_5.append(average_big_probability_0_5_t)
            bin_threshold = find_bin_threshold_people_count(output_sig, actual_number_of_people)
            bin_thresh_people.append(bin_threshold)
            max_area = 100
            local_bin_threshold = bin_threshold_hard
            while max_area > max_area_to_divide:
                binarized = torch.where(output_sig < local_bin_threshold, torch.tensor(0), torch.tensor(255))
                predicted_mask_np = binarized.squeeze(0).cpu().numpy().squeeze(0)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    predicted_mask_np.astype(np.uint8))
                max_area = 0
                for label in range(1, num_labels):
                    # Get the statistics of the component
                    x, y, w, h, area = stats[label]
                    if area > max_area:
                        max_area = area
                if max_area > max_area_to_divide:
                    local_bin_threshold += 0.0025

            binarized = torch.where(output_sig < local_bin_threshold, torch.tensor(0), torch.tensor(255))
            full_range_mask = output_sig * 255
            full_range_mask = full_range_mask.detach().numpy()
            full_range_mask = full_range_mask.astype(np.uint8)
            full_range_mask = full_range_mask[0, 0]
            # ret2, predicted_mask_np = cv2.threshold(full_range_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # blur = cv2.GaussianBlur(full_range_mask, (5, 5), 0)
            # ret2, predicted_mask_np = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            predicted_mask_np = binarized.squeeze(0).cpu().numpy().squeeze(0)
            predicted_mask_name = name + '_mask' + str(ind) + '.jpg'
            full_predicted_mask_name = os.path.join(output_masks_dir, predicted_mask_name)
            if b_write:
                cv2.imwrite(full_predicted_mask_name, predicted_mask_np)
                output_predicted_full_range_mask_name = os.path.join(output_masks_dir, output_predicted_full_range_mask_name)
                cv2.imwrite(output_predicted_full_range_mask_name, full_range_mask)

            predicted_mask_with_bb_np = np.copy(predicted_mask_np)
            binarized_original_mask_with_bb_np = np.copy(binarized_original_mask_np)
            original_with_bb_np = input_tensor * 255
            original_with_bb_np = original_with_bb_np.squeeze(0).cpu().numpy().squeeze(0)

            foot_points_predicted, bounding_boxes_predicted, predicted_mask_with_bb_np, predicted_mask_np, original_with_bb_np, max_area = filter_and_draw_bounding_boxes(
                predicted_mask_with_bb_np, original_with_bb_np, (0, 255, 0), None, min_area)
            foot_points_original, bounding_boxes_original, binarized_original_mask_with_bb_np, binarized_original_mask_np, t, t1 = filter_and_draw_bounding_boxes(
                binarized_original_mask_with_bb_np, original_with_bb_np, (0, 0, 255), rounded_foot_points_original)
            accuracy, precision, recall = calculate_metrics(foot_points_original_txt, foot_points_predicted, distance_threshold)
            mask1 = np.zeros((predicted_mask_with_bb_np.shape[1], predicted_mask_with_bb_np.shape[0], 3), dtype=np.uint8)
            mask2 = np.zeros((predicted_mask_with_bb_np.shape[1], predicted_mask_with_bb_np.shape[0], 3), dtype=np.uint8)
            for point in foot_points_predicted:
                cv2.circle(mask1, point, distance_threshold // 2, (0, 255, 0), thickness=-1)
            for point in foot_points_original:
                cv2.circle(mask2, point, distance_threshold // 2, (0, 0, 255), thickness=-1)
            both_masks = np.maximum(mask1, mask2)
            if b_write:
                cv2.imwrite(os.path.join(output_masks_dir, output_both_masks_name), both_masks)
            data.append(frame_loss_t.item())
            data.append(actual_number_of_people)
            data.append(accuracy)
            data.append(precision)
            data.append(recall)

            mask_with_bb_filename = name + '_mask_bb_' + str(ind) + extension
            full_mask_with_bb_path = os.path.join(output_masks_dir, mask_with_bb_filename)
            original_with_bb_filename = name + '_bb_' + str(ind) + extension
            full_original_with_bb_path = os.path.join(output_masks_dir, original_with_bb_filename)

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


def infer_unet(directory_name, output_masks_dir, model0_path, model1_path, test_composed_transform, mask_transform, csv_file_name):
    # Load the saved model
    device_i = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model0 = CustomMNasNetUnetInvResDec(1, 1)
    # model0 = CustomMNasNetUnetInvResDecArch3(1, 1)
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
            csv_writer.writerow(["jpg file name", "loss", "gt number of people", "accuracy", "precision", "recall"])
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
    mean1 = 0
    std1 = 255
    height = 224
    width = 224

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
