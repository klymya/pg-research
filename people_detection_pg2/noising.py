import os
import glob
from functools import partial
import cv2
import numpy as np
import shutil
from tqdm.contrib.concurrent import process_map


BASE_DATASET_PTHS = [
    "/home/klymy/data/people_detection/filtered_v3_margin_15_gaussian_noise",
    "/home/klymy/data/people_detection/filtered_v3_margin_15_uniform_noise",
    "/home/klymy/data/people_detection/filtered_v3_margin_15_salt_peper_noise",
]


def gasuss_noise(image, mu=0.0, sigma=0.05):
    """
     添加高斯噪声
    :param image: 输入的图像
    :param mu: 均值
    :param sigma: 标准差
    :return: 含有高斯噪声的图像
    """
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mu, sigma, image.shape)
    gauss_noise = image + noise
    if gauss_noise.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    gauss_noise = np.clip(gauss_noise, low_clip, 1.0)
    gauss_noise = np.uint8(gauss_noise * 255)
    return gauss_noise


def salt_pepper_noise(img, prob = 0.01):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = img.copy()
    sp_noise = np.zeros_like(img)
    if len(img.shape) == 2:
        black = 0
        white = 255            
    else:
        colorspace = img.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    sp_noise[probs < (prob / 2)] = black
    sp_noise[probs > 1 - (prob / 2)] = white
    return output


def uniform_noise(img, gamma=0.1):
    uni_noise = np.zeros(img.shape[:2])
    cv2.randu(uni_noise, 0, 256)
    uni_noise = (uni_noise * gamma).astype(np.uint8)
    
    if len(img.shape) == 2:
        output = cv2.add(img, uni_noise)
    elif len(img.shape) == 3:
        merged = cv2.merge([uni_noise, uni_noise, uni_noise])
        output = cv2.add(img, merged)
    return output


def processor(f, img_pth):
    img = cv2.imread(img_pth)

    img = f(img)

    cv2.imwrite(img_pth, img, [cv2.IMWRITE_JPEG_QUALITY, 80])


processor_gaussian = partial(processor, gasuss_noise)
processor_uniform = partial(processor, uniform_noise)
processor_salt_pepper = partial(processor, salt_pepper_noise)


for dataset_pth, img_processor, desc in zip(
    BASE_DATASET_PTHS,
    (processor_gaussian, processor_uniform, processor_salt_pepper),
    ("Gaussian noise", "Uniform noise", "Salt and pepper noise"),
):

    base_pth = os.path.join(dataset_pth, "images")

    shutil.rmtree(dataset_pth, ignore_errors=True)
    shutil.copytree(
        "/home/klymy/data/people_detection/filtered_v3_margin_15",
        dataset_pth
    )

    # Remove cache files
    cache_pths = glob.glob(os.path.join(dataset_pth, "labels", "*.cache"))
    print(cache_pths)
    for cache_pth in cache_pths:
        os.remove(cache_pth)

    for i in ["project.yaml", "project_algotest.yaml", "project_holdout.yaml"]:
        with open(os.path.join(dataset_pth, i), "r") as f:
            lines = f.readlines()
        lines[0] = f"path: {dataset_pth}\n"

        with open(os.path.join(dataset_pth, i), "w") as f:
            f.writelines(lines)

    img_paths = []
    for i in ["train", "val", "tst", "holdout", "algotest"]:
        pth = os.path.join(base_pth, i)
        img_paths += [os.path.join(pth, j) for j in glob.glob(os.path.join(pth, "*.jpg"))]
    print(len(img_paths))

    r = process_map(img_processor, img_paths, max_workers=10, desc=desc)
