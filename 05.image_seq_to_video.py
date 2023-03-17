import cv2
import os

# This function convert image sequence to video
# Эта функция преобразует последовательность изображений в видео
# Path to image folders
# Путь к изображениям
path = 'F:/DATASETS/DAVIS/DAVIS/JPEGImages/480p/bear/'


def image_to_video(image_folder, fps):
    # Create a list of images
    # Создание списка изображений из указанной папки
    filelist = []
    for address, dirs, files in os.walk(image_folder):
        for name in files:
            filelist.append(os.path.join(address, name))
    # Reading the first image to specify dimensions of a video
    # Чтение первого изображения для указания размеров видео
    frame = cv2.imread(filelist[0])
    writer = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]),
                             isColor=len(frame.shape) > 2)
    for frame in map(cv2.imread, filelist):
        writer.write(frame)
    writer.release()


image_to_video(path, 25)
