#!/usr/bin/env python
# coding: utf-8

# In[53]:


from tqdm import tqdm
import torch
from pathlib import Path
import numpy as np

from cnn4 import CNN4
from datamodule import Datamodule
from params import LocationConfig, TrainingConfig


# In[21]:


cnn4 = CNN4(lr=TrainingConfig.lr)
state_dict = torch.load(LocationConfig.best_model)
cnn4.load_state_dict(state_dict)
cnn4.eval();


# In[29]:


train_data_path = Path(LocationConfig.train_data)
test_data_path = Path(LocationConfig.test_data)
dm = Datamodule(
        batch_size=TrainingConfig.batch_size,
        train_dir=train_data_path,
        val_dir=test_data_path,
        )
dm.setup(only_val=True)


# In[107]:


acc_class_global = np.zeros(5)
i=0
for batch in tqdm(dm.val_dataloader()):
    limit = 1/2 # 4/7
    X, Y = batch
    Y_pred = cnn4.predict_step(X, None)
    Y_pred = np.where(Y_pred > limit, 1, 0)
    acc_class = np.sum(Y_pred == Y, axis=0) / len(Y)
    acc_class_global += acc_class
    i+=1
acc_class_global /= i
print(acc_class_global)
print(acc_class_global.mean())

