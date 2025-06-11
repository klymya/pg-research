import argparse
import os
import glob

from tqdm.contrib.concurrent import thread_map# or thread_map
import cv2

from alif_awb import white_balance_vectorized


def f(args):
    input_path, output_path = args
    img = cv2.imread(input_path)
    if img is None:
        print(f"Could not read image {input_path}")
        return 0

    # Apply white balance
    img_wb = white_balance_vectorized(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    cv2.imwrite(output_path, cv2.cvtColor(img_wb, cv2.COLOR_RGB2BGR))
    return 1


def main(input_dir):
    output_dir = input_dir + '_wb'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_files = glob.glob(os.path.join(input_dir, '*.tif'))
    output_files = [os.path.join(output_dir, os.path.basename(f).replace('.tif', '.jpg')) for f in input_files]

    r = thread_map(f, zip(input_files, output_files), max_workers=8)
    count = 0
    for i in r:
        count += i

    print(count == len(input_files))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply white balance to a batch of images.')
    parser.add_argument('input_dir', type=str, help='Directory containing input images')
    args = parser.parse_args()

    main(args.input_dir)
