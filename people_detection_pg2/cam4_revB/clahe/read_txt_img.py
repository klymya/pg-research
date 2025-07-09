import numpy as np
import matplotlib.pyplot as plt

def load_image_from_txt(filename, width, height):
    # Read all numbers from the file
    with open(filename, 'r') as f:
        data_str = f.read()

    # Convert string to list of integers
    data_list = list(map(int, data_str.split()))

    # Convert to numpy array
    img_array = np.array(data_list, dtype=np.uint8)

    # Reshape to (height, width) or (height, width, channels) as needed
    img_array = img_array.reshape((height, width, -1))

    return img_array

if __name__ == "__main__":
    # Example image size, update as per your image
    img_width = 320
    img_height = 240

    img = load_image_from_txt('processed_image_data.txt', img_width, img_height)

    plt.imshow(img)  # Use cmap='gray' for grayscale, remove for RGB
    plt.axis('off')
    plt.show()
