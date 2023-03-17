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
name = 'IMAGAA'
exten = ".jpg"
from_image = name + exten
img = Image.open(from_image)
numpydata = np.asarray(img)

# def roll(a,      # ND array
#          b,      # rolling 2D window array
#          dx=1,   # horizontal step, abscissa, number of columns
#          dy=32,   # vertical step, ordinate, number of rows
#          dz=45):  # transverse step, applicate, number of layers
#     shape = a.shape[:-3] + \
#             ((a.shape[-3] - b.shape[-3]) // dz + 1,) + \
#             ((a.shape[-2] - b.shape[-2]) // dy + 1,) + \
#             ((a.shape[-1] - b.shape[-1]) // dx + 1,) + \
#             b.shape  # multidimensional "sausage" with 3D cross-section
#     strides = a.strides[:-3] + \
#               (a.strides[-3] * dz,) + \
#               (a.strides[-2] * dy,) + \
#               (a.strides[-1] * dx,) + \
#               a.strides[-3:]
#     print('shape =', shape, " strides =", strides)  # for debugging
#     return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
#
# count = 0
# for i in roll(numpydata, np.ndarray((90,160,3))):
#     for el in i:
#         for imaga in el:
#             print(imaga)
#             count +=1
#             pilImage = Image.fromarray(imaga)
#             print(pilImage.mode)
#             print(pilImage.size)
#             namefile = "geeks" + str(count) + ".jpg"
#             if not os.path.exists("sliced_images"):
#                 os.makedirs("sliced_images")
#             pilImage.save(f"sliced_images/{namefile}")


# This is a function for a slicing image with selected steps
# This function takes as input a link to an image, width and height of sliding window, step on side and down (pixels)
# Important! It works only with 24-bit images. 32-bit images will convert into 24-bit and cause color distortion.
#
# Функция для нарезки изображений скользящим окном с выбранным шагом.
# На вход принимает ссылку на изображение, ширину и высоту окна, шаг по горизонтали и по
# вертикали(в пикселях).
# ВАЖНО! Работает только на 24-х битных изображениях, 32 битные обрезает до 24 бит.
def slicing_image_with_sliding_window(image_name, wind_width, wind_height, step_x, step_y):
    # Validation on image link
    # Валидация ссылки на изображение
    try:
        load_image = cv2.imread(image_name)
    except:
        return print(f"Check Image name {image_name}")
    np_array = np.asarray(load_image)
    # Here is some check:
    # 1. Image shapes should be larger than window shapes.
    # Заранее происходит ряд сравнений:
    # 1. Размеры окна с размером изображения (требуется, чтобы окно было меньше изображения), иначе выводится сообщение.
    if wind_height > load_image.shape[0] or wind_width > load_image.shape[1]:
        return print(f"Sliding window size is larger than image size\nImage name - {image_name}\n"
                     f"Image size - {load_image.shape}")
    # 2. Image shapes should be larger than window steps.
    # 2. Размеры шагов с соответствующими размерами изображения, иначе выводится сообщение.
    elif step_x > load_image.shape[1] or step_y > load_image.shape[0]:
        return print(f"Step size is larger than image size\nImage name - {image_name}\n"
                     f"Image size - {load_image.shape}, horizontal step - {step_x}, vertical step - {step_y}")
    # 3. Window shapes with sum of steps should be equal to image shapes or we lose data
    # 3. Размеры окна с учетом шагов в обоих направлениях сравниваются с нулем для выявления потерянных частей
    # изображения, в случае обнаружения выводится сообщение с полной информацией.
    elif (load_image.shape[1] - wind_width) % step_x != 0 or (load_image.shape[0] - wind_height) % step_y != 0:
        print(f"Unable to slide across the entire width or height with this steps!\nImage name - {image_name}\n"
              f"Image size - {load_image.shape}, Window size - ({wind_width},{wind_height})\n"
              f"horizontal step - {step_x}, vertical step - {step_y}\n"
              f"Lost pixels: Horizontal - {((load_image.shape[1] - wind_width) % step_x)}, "
              f"Vertical - {((load_image.shape[0] - wind_height) % step_y)}")
    if not os.path.exists("sliced_images"):
        os.makedirs("sliced_images")
    # On an input data we convert some array metadata for a stride_tricks.as_strided function
    # На основе поданных данных подготавливается размер массива и шаги отступа для передачи в stride_tricks.as_strided
    shape = ((np_array.shape[-3] - wind_height) // step_y + 1, ) + ((np_array.shape[-2] - wind_width) // step_x + 1, ) + \
            (1, ) + (wind_height, wind_width, 3)
    strides = (np_array.strides[0] * step_y, ) + (np_array.strides[1] * step_x, ) + (np_array.strides[2], ) + \
              np_array.strides
    # This loop saves each slice to an image with a changed name containing start and end point of an original image
    # Цикл по сохранению нарезанных изображений с указанием точки начала и конца от исходного
    count_col = 0
    for column_roll in np.lib.stride_tricks.as_strided(np_array, shape=shape, strides=strides):
        count_row = 0
        for row_roll in column_roll:
            for sliced_image in row_roll:
                sliced_image_name = f"{image_name[:-4]}-from({(count_row * step_x)},{(count_col * step_y)})to(" \
                                    f"{(count_row * step_x) + wind_width},{(count_col * step_y) + wind_height}" \
                                    f").jpg"
                cv2.imwrite(f"sliced_images/{sliced_image_name}", sliced_image)
            count_row += 1
        count_col += 1


slicing_image_with_sliding_window(from_image, 1280, 720, 160, 180)