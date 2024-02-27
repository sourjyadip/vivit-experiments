import numpy as np
import torch
from transformers import VivitImageProcessor, VivitModel, VivitForVideoClassification, VivitConfig
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score
import torch.optim as optim 
from tqdm import tqdm
import cv2
from torchvision import transforms
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import time
import math
from model import *
import json
from PIL import Image

start_time = time.time()

label2id = {}
with open('label2id.json', 'r') as fp:
    label2id = json.load(fp)

id2label = {}
with open('id2label.json', 'r') as fp:
    id2label = json.load(fp)

train_df = pd.read_csv('/home/sourjyadip-pg/h-vivit/data/kinetics-dataset/k400/annotations/val.csv') #replace with train set when data is avl
train_df = train_df[train_df['label'].isin([id2label['0'], id2label['1'], id2label['2']])]
#train_df, _ = train_test_split(train_df, test_size = 0.9)#reduced to test code
print("Data size before removing corrupt videos:",len(train_df))
train_df = remove_corrupt_files(df = train_df, data_dir = "/home/sourjyadip-pg/h-vivit/data/kinetics-dataset/k400/val/")
print("Data size after removing corrupt videos:",len(train_df))
print("Total time taken:  %s seconds" % (time.time() - start_time))
print("class 0: ", len(train_df[train_df['label'].isin([id2label['0']])]))
print("class 1: ", len(train_df[train_df['label'].isin([id2label['1']])]))
print("class 2: ", len(train_df[train_df['label'].isin([id2label['2']])]))
#train_df, _ = train_test_split(train_df, test_size=0.5)

test_df = pd.read_csv('/home/sourjyadip-pg/h-vivit/data/kinetics-dataset/k400/annotations/test.csv') #replace with train set when data is avl
test_df = test_df[test_df['label'].isin([id2label['0'], id2label['1'], id2label['2']])]
#train_df, _ = train_test_split(train_df, test_size = 0.9)#reduced to test code
print("Data size of test set before removing corrupt videos:",len(test_df))
test_df = remove_corrupt_files(df = test_df, data_dir = "/home/sourjyadip-pg/h-vivit/data/kinetics-dataset/k400/test/")
print("Data size after removing corrupt videos:",len(test_df))
print("Total time taken:  %s seconds" % (time.time() - start_time))
print("class 0: ", len(test_df[test_df['label'].isin([id2label['0']])]))
print("class 1: ", len(test_df[test_df['label'].isin([id2label['1']])]))
print("class 2: ", len(test_df[test_df['label'].isin([id2label['2']])]))
test_df, _ = train_test_split(test_df, test_size=0.6)

train_df = train_df.reset_index()
test_df = test_df.reset_index()

unique_labels = list(train_df['label'].unique())
print("Number of unique labels : ", len(unique_labels))

#building label dictioanries
unique_videos = list(train_df['youtube_id'].unique())
label2id = {item: index for index, item in enumerate(unique_labels)}
id2label = {index: item for index, item in enumerate(unique_labels)}

configs ={
    "total_frames": 128,
    "num_frames":32,
    "factor": int(128/32),
    "architecture": "vivit-baseline",
    "dataset": "kinetics-400 reduced val split",
    "epochs": 50,
    "num_classes": len(unique_labels),
    "tubelet_size": [2, 16, 16],
    "data_dir": "/home/sourjyadip-pg/h-vivit/data/kinetics-dataset/k400/val/",
    "test_data_dir": "/home/sourjyadip-pg/h-vivit/data/kinetics-dataset/k400/test/"
}
print("Initialising model")
image_processor = VivitImageProcessor()
#base_model = VivitModel.from_pretrained("google/vivit-b-16x2")
config = VivitConfig(num_frames = configs['num_frames'], tubelet_size  = configs['tubelet_size'], output_hidden_states=  True, output_attentions = True)

class ActionClassificationDataset(Dataset):
    def __init__(self, df, data_dir):
        self.df = df
        self.video_files = df.youtube_id.tolist()
        self.data = []
        print(len(self.video_files))

        for idx in range(len(self.video_files)):
            filename = self.video_files[idx]
            start = self.df['time_start'][idx]
            end = self.df['time_end'][idx]
            label = self.df['label'][idx]
            vid_path = data_dir + construct_filename(filename, start, end)

            vid_cap = cv2.VideoCapture(vid_path)
            fps =  vid_cap.get(5)
            frames = vid_cap.get(7)
            length = vid_cap.get(7)/fps
            #start_frame, end_frame = get_start_end_frame( start, end, fps, frames, length)
            #print(start_frame, end_frame)
            start = 0
            end  = frames
            indices  = sample_indices(start, end, configs['total_frames'])
            video = read_video(vid_cap, indices=indices, num_frames =  frames)
            #print(video)
            video = video.reshape(configs['num_frames'],configs['factor'],224,224,3)
            #print(video)
            video = np.mean(video, axis = 1 )
            video = video/255.0
            #print(video.shape)
            #video = Image.fromarray((video * 255).astype(np.uint8))
            #print(video)
            inputs = image_processor(list(video), return_tensors="pt", do_rescale=False, offset =  False)

            label_vector  = []
            for i in range(configs['num_classes']):
                label_vector.append(0.0)
            label_vector[label2id[label]] = 1.0

            encoded_label = torch.tensor(label_vector)
            vid_cap.release()
            self.data.append({'input': inputs, 'target': encoded_label})
            if idx%100 == 0:
                print(idx, "Done")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return sample
    
print("Creating train Dataloader...")
train_dataset = ActionClassificationDataset(df=train_df, data_dir =  configs['data_dir'])
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
print("Done")
print("Total time taken:  %s seconds" % (time.time() - start_time))

print("Creating test Dataloader...")
test_dataset = ActionClassificationDataset(df=test_df, data_dir = configs['test_data_dir'])
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
print("Done")
print("Total time taken:  %s seconds" % (time.time() - start_time))

#model = VideoClassifier(config, num_classes = configs['num_classes'])
model = VideoClassifierPytorch(num_classes = configs['num_classes'])
#print(model)
criterion = nn.CrossEntropyLoss()

total_params = sum(
    param.numel() for param in model.parameters()
)
trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)
print("Total params: ",total_params)
print("Trainable params: ",trainable_params)

# Set your hyperparameters
learning_rate = 0.000001  # Initial learning rateconfigs['data_dir']
momentum = 0.9
warmup_epochs = 2  # Number of warm-up epochs
num_epochs = configs['epochs']  # Total number of epochs
total_steps = num_epochs  # Total number of training steps
#batch_size = 4  # Set your batch size
'''
# Define the optimizer with synchronous SGD and momentum
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# Define the learning rate schedule with warm-up
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
'''
'''
lambda_lr = lambda step: step / (warmup_epochs * total_steps) if step <= warmup_epochs * total_steps else 0.5 * (1 + math.cos(
    (step - warmup_epochs * total_steps) / ((1 - warmup_epochs / num_epochs) * total_steps) * 3.1415))
print(lambda_lr)
scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)
'''
optimizer =  torch.optim.AdamW(model.parameters(), lr = learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
model.cuda()
#model.train()



#test function

def test_model(model):
    model.eval()

    predictions = []
    real_values = []

    with torch.no_grad():
        for vid_batch in test_dataloader:
            inputs, label = vid_batch['input'], vid_batch['target']
            inputs = inputs['pixel_values']
            inputs =  inputs.squeeze(1)
            #print(inputs.shape)
            #print(label.shape)
            inputs = inputs.cuda()
            label = label.cuda()
            probs = model(inputs)
            actual_label = torch.argmax(label).item()
            predicted_label = torch.argmax(probs).item()

            real_values.append(actual_label)
            predictions.append(predicted_label)

    print("real values")
    print(real_values)
    print("predictions")
    print(predictions)

    print(classification_report(real_values, predictions,  digits=4))

test_model(model)

# Training loop
print("Starting training loop")

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    total_loss = 0
    i = 1
    for vid_batch in train_dataloader:
        inputs, label = vid_batch['input'], vid_batch['target']
        inputs = inputs['pixel_values']
        inputs =  inputs.squeeze(1)
        #print(inputs.shape)
        #print(label.shape)
        inputs = inputs.cuda()
        label = label.cuda()
        logits = model(inputs)
        #print("label", label.shape)
        #print("pred probs", logits.shape)
        loss = criterion(logits, label)
        loss.backward()
        '''
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f'Parameter: {name}, Gradient norm: {param.grad.norm()}')
        '''
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        #print(loss.item())
        #print(f"{i} batches of {len(train_dataloader)} done")
        i += 1

    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}:")
    print(f"Average Loss: {average_loss}")
    print("Total time taken:  %s seconds" % (time.time() - start_time))
    test_model(model)


