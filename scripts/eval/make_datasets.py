from ast import expr_context
import cv2
import os

val_txt = "../../datasets/settings/hmdb51/val_rgb_split1.txt"
data_dir = "../../datasets/hmdb51_frames"
video_path = "../../datasets/HMDB51/"

f_val = open(val_txt, 'r')
val_list = f_val.readlines()

for i, line in enumerate(val_list):
    line_info = line.split(" ")
    clip_path = os.path.join(data_dir, line_info[0])
    
    # make folder
    try:
        if not os.path.exists(clip_path):
            os.mkdir(clip_path)
            # print("make folder!")
    except OSError:
        print("Error")
    
    # Clip video convert to image(sampling rate = 30frame)
    video = cv2.VideoCapture(video_path + f"/{line_info[0]}.avi")
    # print('load complete!!')
    success, image = video.read()
    count = 1
    
    while success:
        cv2.imwrite(f'{clip_path}/img_%05d.jpg' % count, image)
        success, image = video.read()
        count += 1
    # print('change complete!!')