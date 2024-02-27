import numpy as np
import torch
from transformers import VivitImageProcessor, VivitModel, VivitForVideoClassification, VivitConfig
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import cv2
from torchvision import transforms
import pandas as pd
import random
import torch.nn as nn
import torch.nn.functional as F
from vivit import ViViT



class VideoClassifier(nn.Module):
    def __init__(self, config, num_classes):
        super(VideoClassifier, self).__init__()
        self.vivit = VivitModel(config)
        # Add a custom binary classification head for each frame

        self.classification_head = nn.Linear(self.vivit.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        # Forward pass through the ViT backbone
        vivit_outputs = self.vivit(pixel_values=pixel_values)
        # Extract the cls token representation
        cls_token = vivit_outputs.last_hidden_state[:, 0, :]
        #print(vivit_outputs.last_hidden_state.shape)
        #print(cls_token.shape)
        #print(cls_token.shape)
        # Classification head
        output = self.classification_head(cls_token) #logits
        probabilities = F.softmax(output, dim=1)
        return probabilities

class VideoClassifierPytorch(nn.Module):
    def __init__(self, num_classes):
        super(VideoClassifierPytorch, self).__init__()
        self.vivit = ViViT(224, 16, num_classes, 32)
        # Add a custom binary classification head for each frame

        #self.classification_head = nn.Linear(self.vivit.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        # Forward pass through the ViT backbone
        output = self.vivit(pixel_values)
        
        probabilities = F.softmax(output, dim=1)
        return probabilities
