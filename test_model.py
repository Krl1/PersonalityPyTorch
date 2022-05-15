#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from glob import glob
from pathlib import Path
import numpy as np
from tqdm import tqdm
import cv2
from cnn4 import CNN4
from datamodule import Datamodule
from params import LocationConfig, TrainingConfig
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchsummary import summary


# In[2]:


cnn4 = CNN4(lr=TrainingConfig.lr, batch_norm=True, sigmoid=True)
summary(cnn4.cuda(), (208,208, 1) )


# In[4]:


checkpoints_dir = Path(LocationConfig.checkpoints_dir)
list_of_checkpoints = list(checkpoints_dir.glob("*.ckpt"))
latest_checkpoint_path = max(list_of_checkpoints, key=lambda p: p.stat().st_ctime)

cnn4 = CNN4(lr=TrainingConfig.lr, batch_norm=TrainingConfig.batch_norm, sigmoid=TrainingConfig.sigmoid)    .load_from_checkpoint(checkpoint_path=latest_checkpoint_path)
cnn4.eval();
cnn4.state_dict()['conv.conv_0.weight'][0]


# In[6]:


train_data_path = Path(LocationConfig.new_data + 'train')
test_data_path = Path(LocationConfig.new_data + 'test')
dm = Datamodule(
        batch_size=TrainingConfig.batch_size,
        train_dir=train_data_path,
        val_dir=test_data_path,
        )
# dm.setup(val_only=True)
dm.setup()


# In[12]:


acc_class_global_12 = np.zeros(5)
acc_class_global_47 = np.zeros(5)
i=0
for batch in tqdm(dm.val_dataloader()):
    X, Y = batch['normalized'], batch['label']
#     normalize = transforms.Normalize((X.mean()), (X.std()))
#     X = normalize(X)
    Y_pred = cnn4.predict_step(X, None)
    Y_pred_12 = np.where(Y_pred > 1/2, 1, 0)
    acc_class_12 = np.sum(Y_pred_12 == np.array(Y), axis=0) / len(Y)
    acc_class_global_12 += acc_class_12
    
    Y_pred_47 = np.where(Y_pred > 4/7, 1, 0)
    acc_class_47 = np.sum(Y_pred_47 == np.array(Y), axis=0) / len(Y)
    acc_class_global_47 += acc_class_47
    i+=1
acc_class_global_12 /= i
print('1/2')
print(acc_class_global_12)
print(acc_class_global_12.mean())
acc_class_global_47 /= i
print('4/7')
print(acc_class_global_47)
print(acc_class_global_47.mean())


# In[14]:


for batch in dm.val_dataloader():
    X, Y = batch['normalized'], batch['label']
#     normalize = transforms.Normalize((X.mean()), (X.std()))
#     X = normalize(X)
    Y_pred = cnn4(X)
    # Y_pred = np.where(Y_pred > limit, 1, 0)
    print(Y_pred[0:5])
    Y_pred = np.where(Y_pred > 1/2, 1, 0)
    print(Y_pred[0:5])
    print(Y[0:5])
    break

