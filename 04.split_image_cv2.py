import cv2
import numpy as np
from PIL import Image
import os

# Wrong way
# name = 'IMAGAA'
# exten = ".jpg"
# from_image = name + exten
# img = cv2.imread(from_image)
# yy = 0
# crop_img_height = 180
# crop_img_width = 320
# row_counter = 0
# for row in range(int(img.shape[0] / crop_img_height)):
#     column_counter = 0
#     xx = 0
#     for column in range(int(img.shape[1] / crop_img_width)):
#         crop_img = img[yy: yy + crop_img_height, xx: xx + crop_img_width]
#         img_name = name + '(' + str(crop_img_width * (column_counter)) + '-' + \
#                    str(crop_img_width * (column_counter + 1)) + '-' + str(crop_img_height * (row_counter)) + '-' + \
#                    str(crop_img_height * (row_counter + 1)) + exten
#         if not os.path.exists("sliced_images"):
#             os.makedirs("sliced_images")
#         cv2.imwrite(f"sliced_images/{img_name}", crop_img)
#         column_counter += 1
#         xx += crop_img_width
#         cv2.waitKey(0)
#     row_counter += 1
#     yy += crop_img_height

# stride_tricks way
name = 'IMAGAA(0-320-0-180'
exten = ".jpg"
from_image = name + exten
img = Image.open(from_image)
numpydata = np.asarray(img)

def roll(a,      # ND array
         b,      # rolling 2D window array
         dx=1,   # horizontal step, abscissa, number of columns
         dy=32,   # vertical step, ordinate, number of rows
         dz=45):  # transverse step, applicate, number of layers
    shape = a.shape[:-3] + \
            ((a.shape[-3] - b.shape[-3]) // dz + 1,) + \
            ((a.shape[-2] - b.shape[-2]) // dy + 1,) + \
            ((a.shape[-1] - b.shape[-1]) // dx + 1,) + \
            b.shape  # multidimensional "sausage" with 3D cross-section
    strides = a.strides[:-3] + \
              (a.strides[-3] * dz,) + \
              (a.strides[-2] * dy,) + \
              (a.strides[-1] * dx,) + \
              a.strides[-3:]
    print('shape =', shape, " strides =", strides)  # for debugging
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

count = 0
for i in roll(numpydata, np.ndarray((90,160,3))):
    for el in i:
        for imaga in el:
            print(imaga)
            count +=1
            pilImage = Image.fromarray(imaga)
            print(pilImage.mode)
            print(pilImage.size)
            namefile = "geeks" + str(count) + ".jpg"
            if not os.path.exists("sliced_images"):
                os.makedirs("sliced_images")
            pilImage.save(f"sliced_images/{namefile}")