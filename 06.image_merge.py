import os
import cv2
import numpy as np

# Baseline merge

path = 'sliced_images/'

images_list = []
for address, dirs, files in os.walk(path):
    for name in files:
        images_list.append(os.path.join(address, name))


def image_validation(original_image_as_array, merged_image_as_array):
    return np.array_equal(original_image_as_array, merged_image_as_array)


row_block = []
for x in range(0, 1920, 960):
    col_blocks = []
    for y in range(0, 1080, 540):
        namefile = f"sliced_images/IMAGAA-from({x},{y})to({x+960},{y+540}).png"
        block = cv2.imread(namefile)
        col_blocks.append(block)
    row_block.append(np.concatenate(col_blocks, axis=0))
full_image = np.concatenate(row_block, axis=1)
cv2.imwrite("to_fullLLL.png", full_image)
print("Validation ", image_validation(cv2.imread("IMAGAA.jpg"), full_image))
