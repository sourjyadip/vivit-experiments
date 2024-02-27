import cv2
import pandas as pd 
from utils import *
import json

df = pd.read_csv('/home/sourjyadip-pg/h-vivit/data/kinetics-dataset/k400/annotations/val.csv')

unique_labels = list(df['label'].unique())
print("Number of unique labels : ", len(unique_labels))

label2id = {item: index for index, item in enumerate(unique_labels)}
id2label = {index: item for index, item in enumerate(unique_labels)}

with open('label2id.json', 'w') as fp:
    json.dump(label2id, fp)

with open('id2label.json', 'w') as fp:
    json.dump(id2label, fp)
