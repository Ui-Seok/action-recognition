import cv2
import numpy as np
import glob


def img2video(img_folder, top_list, index_list):
    img_array = []
    green = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for filename in glob.glob(img_folder + '/*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        for num in range(len(top_list)):
            idx = top_list[num]
            num += 1
            img = cv2.putText(img, f"TOP {num}: {index_list[idx]}", (10, 15 + num * 10), font, 0.3, green, 1)
        img_array.append(img)

    top_class = index_list[top_list[0]]    
    out = cv2.VideoWriter(f'{top_class}_visualization_test.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
    
def video2img(video_path, image_path):
    for video in glob.glob(video_path + '/*.mp4'):
        vidcap = cv2.VideoCapture(video)
        success, image = vidcap.read()
        count = 1

        while success:
            cv2.imwrite(f'{image_path}/img_%05d.jpg' % count, image)
            success, image = vidcap.read()
            count += 1