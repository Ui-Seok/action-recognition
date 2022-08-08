import os, sys
from statistics import mean
import numpy as np
import pandas as pd
import pickle

import time
import argparse

# from ptflops import get_model_complexity_info

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
from sklearn.metrics import confusion_matrix

from change_extension import img2video, video2img


datasetFolder="../../datasets"
sys.path.insert(0, "../../")
import models
from VideoSpatialPrediction3D_bert import VideoSpatialPrediction3D_bert

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__all__)

hmdb51_class = ['brush_hair', 'cartwheel', 'catch', 'chew', 'clap',
                'climb', 'climb_stairs', 'dive', 'draw_sword', 'dribble',
                'drink', 'eat', 'fall_floor', 'fencing', 'flic_flac',
                'golf', 'handstand', 'hit', 'hug' ,'jump', 
                'kick', 'kick_ball', 'kiss', 'laugh', 'pick',
                'pour', 'pullup', 'punch', 'push', 'pushup',
                'ride_bike', 'ride_horse', 'run', 'shake_hands', 'shoot_ball',
                'shoot_bow', 'shoot_gun', 'sit', 'situp', 'smile',
                'smoke', 'somersault', 'stand', 'swing_baseball', 'sword',
                'sword_exercise', 'talk', 'throw', 'turn', 'walk',
                'wave']

parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition RGB Test Case')

parser.add_argument('--dataset', '-d', default='hmdb51',
                    choices=["ucf101", "hmdb51"],
                    help='dataset: ucf101 | hmdb51')
parser.add_argument('--arch', '-a', metavar='ARCH', default='rgb_resneXt3D64f101_bert10_FRMB',
                    choices=model_names)

parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')

parser.add_argument('-w', '--window', default=3, type=int, metavar='V',
                    help='validation file index (default: 3)')


parser.add_argument('-v', '--val', dest='window_val', action='store_true',
                    help='Window Validation Selection')

multiGPUTest = False
multiGPUTrain = False
ten_crop_enabled = True
num_seg=16
num_seg_3D=1

result_dict = {}

def buildModel(model_path,num_categories):
    model=models.__dict__[args.arch](modelPath='', num_classes=num_categories,length=num_seg_3D)
    params = torch.load(model_path)

    if multiGPUTest:
        model=torch.nn.DataParallel(model)
        new_dict={"module."+k: v for k, v in params['state_dict'].items()} 
        model.load_state_dict(new_dict)
        
    elif multiGPUTrain:
        new_dict = {k[7:]: v for k, v in params['state_dict'].items()} 
        model_dict=model.state_dict() 
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(params['state_dict'])
    model.cuda()
    model.eval()  
    return model


def main():
    global args
    args = parser.parse_args()
    if '64f' in args.arch:
        length=64
    elif '32f' in args.arch:
        length=32
    elif '8f' in args.arch:
        length=8    
    else:
        length=16
        

    modelLocation="./checkpoint/"+args.dataset+"_"+args.arch+"_split"+str(args.split)

    model_path = os.path.join('../../',modelLocation,'model_best.pth.tar')
    
    start_frame = 0
    if args.dataset=='ucf101':
        num_categories = 101
    elif args.dataset=='hmdb51':
        num_categories = 51
    elif args.dataset=='smtV2':
        num_categories = 174
    elif args.dataset=='window':
        num_categories = 3

    model_start_time = time.time()
    spatial_net=buildModel(model_path,num_categories)
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print("Action recognition model is loaded in %4.4f seconds." % (model_time))

    # input_video = '../../datasets/video_input'
    video_names = [
                #     'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 
                #    'Shoplifting', 'Stealing', 'Vandalism',
                   'Normal_Videos'
                   ]
    # video_names = os.listdir("../../datasets/UCF-Crime/Burglary")
    for video_name in video_names: 
        print(video_name, 'START')
        video_name = os.listdir(f"../../datasets/UCF-Crime/{video_name}")
        for vd in video_name:
            print(vd)
            image_path = os.path.join(datasetFolder, f"UCF-Crime/{vd[:-9]}/{vd}")
            # video to img and save imgs
            # video2img(input_video, image_path)
            
            img_list = os.listdir(image_path)
            len_32 = len(img_list) // 32
            vis_img_list = []
            result_list = []
            if (len_32 / 64) > 5:
                sampling_rate = (len_32 // 64)
                print(sampling_rate)
            else: sampling_rate = 5
            for i in range(1, len(img_list)):
                if i % len_32 == 0:
                    vis_img_list.append(i)

                    spatial_prediction = VideoSpatialPrediction3D_bert(
                        img_l = vis_img_list,
                        vid_name = image_path,
                        net = spatial_net,
                        num_categories = num_categories,
                        architecture_name = args.arch,
                        start_frame = start_frame,
                        num_frames = 0,
                        num_seg = num_seg_3D,
                        length = length,
                        extension = 'img_{0:05d}.jpg',
                        ten_crop = ten_crop_enabled
                    )
                    
                    pred_index, mean_result, top3, top5 = spatial_prediction
                    top_1_classname = hmdb51_class[pred_index]
                    
                    # df = pd.DataFrame(mean_result)
                    # df.to_csv('result.csv', index = False)
                    
                    # print(mean_result)
                    print(top_1_classname)
                    print('------top5------')
                    for top in top5:
                        print(hmdb51_class[top])
                    print(vis_img_list)
                    print('----------------')
                    result_list.append(mean_result)
                    vis_img_list = []
                elif i % sampling_rate == 0:
                    vis_img_list.append(i)
            df = pd.DataFrame(result_list)
            df.to_csv('result.csv', index = False)
            
            with open(f'../pickle/{vd[:-9]}/{vd}.pkl', 'wb') as f:
                pickle.dump(result_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    # print(modelLocation)
            
    # img to video and write pred_index
    # img2video(image_path, top5, hmdb51_class)

if __name__ == "__main__":
    main()

