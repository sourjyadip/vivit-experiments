import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
import cv2
from torchvision import transforms
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import math

def construct_filename(id, start, end):
    new_start = str(start).rjust(6, '0')
    new_end = str(end).rjust(6, '0')
    filename = id + '_' + new_start + '_' +  new_end + '.mp4'
    return filename

def remove_corrupt_files(df, data_dir):
    filenames =  df['youtube_id'].tolist()
    start = df['time_start'].tolist()
    end = df['time_end'].tolist()
    for i in range(len(filenames)):
        f = filenames[i]
        s = start[i]
        e = end[i] 
        file_path = data_dir + construct_filename(f,s,e)
        try:
            vid_cap = cv2.VideoCapture(file_path)
            fps =  vid_cap.get(cv2.CAP_PROP_FPS)
            frames = vid_cap.get(7)
            length = vid_cap.get(7)/fps
            if frames <= 128:
                df = df[df['youtube_id'] != f]
            vid_cap.release()
        except:
            df = df[df['youtube_id'] != f]
    df = df.reset_index()
    return df
    

def timestamp_to_frame(timestamp, frames, length): #VERIFY ONCE :|
    time_frame = round((float(timestamp)/float(length))*frames)
    return time_frame

def read_video(vid_cap, indices, num_frames):
    frames = []
    start_frame = 0
    end_frame = num_frames
    vid_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    i = start_frame
    while i <= end_frame:
        ret, frame = vid_cap.read()
        if i in indices:
            #ret, frame = vid_cap.read()
            if ret is True:
                #frame = np.array(frame)
                #frame = frame.to_ndarray(format="rgb24")
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.array(frame, dtype=np.uint8)
                #frame = frame.to_ndarray(format="rgb24")
                frames.append(frame)

        i +=1
    return np.stack(frames)

def get_start_end_frame( start_time, end_time, fps, total_frames, length): #start and end frame number
    
    start = timestamp_to_frame(start_time, total_frames, length)
    end =  timestamp_to_frame(end_time, total_frames, length)

    return (start, end)

def sample_indices(start, end, num_frames):
    indices = []
    factor = int(math.floor((end-start)/num_frames))
    if factor == 0:
        factor = 1
    i = start
    while(i<=end):
        indices.append(i)
        i += factor
        if (len(indices) == num_frames):
            break
    return indices