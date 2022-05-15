import pytorch_lightning as pl
import torch
from torch import nn as nn
from torch.nn import functional as F
from torchmetrics import Accuracy
from statistics import mean
import torchvision.transforms as transforms

class CNN4(pl.LightningModule):
    
    def __init__(
        self,
        lr,
        batch_norm,
        sigmoid,
        negative_slope = 0.0,
        max_pool_ceil_mode = False,
        classification = False
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        self.sigmoid = sigmoid
        self.batch_norm = batch_norm
        self.negative_slope = negative_slope
        self.max_pool_ceil_mode = max_pool_ceil_mode
        self.classification = classification
        
        if self.classification:
            self.train_accuracy = Accuracy()
            self.val_accuracy = Accuracy()
        
        # Conv
        fm_size = 16
        self.conv = nn.Sequential()
        self.conv.add_module(f"conv_0",nn.Conv2d(1, fm_size, (3,3), stride=1, padding='same'))
        if self.batch_norm:
            self.conv.add_module(f"batch_norm_0", nn.BatchNorm2d(fm_size))
        if self.negative_slope == 0.0:
            self.conv.add_module(f"relu_0", nn.ReLU())
        else:
            self.conv.add_module(f"lrelu_0", nn.LeakyReLU(self.negative_slope))
        self.conv.add_module(f"dropout_0", nn.Dropout(0.2))
        self.conv.add_module(f"max_pool_0", nn.MaxPool2d(2), ceil_mode=self.max_pool_ceil_mode)
        
        max_range = 8 if self.max_pool_ceil_mode else 7
        for i in range(1, max_range):
            self.conv.add_module(f"conv_{i}", nn.Conv2d(fm_size, fm_size*2, (3,3), stride=1, padding='same'))
            if self.batch_norm:
                self.conv.add_module(f"batch_norm_{i}", nn.BatchNorm2d(fm_size*2))
            if self.negative_slope == 0.0:
                self.conv.add_module(f"relu_0", nn.ReLU())
            else:
                self.conv.add_module(f"lrelu_{i}", nn.LeakyReLU(self.negative_slope))
            self.conv.add_module(f"dropout_{i}", nn.Dropout(0.2))
            self.conv.add_module(f"max_pool_{i}", nn.MaxPool2d(2), ceil_mode=self.max_pool_ceil_mode)
            fm_size *= 2
        
        # Linear
        self.linear = nn.Sequential() 
        self.linear.add_module("flatten", nn.Flatten())
        self.linear.add_module("linear_0", nn.Linear(fm_size, 128))
        if self.batch_norm:
            self.linear.add_module(f"batch_norm_0", nn.BatchNorm1d(128))
        if self.negative_slope ==0.0:
            self.linear.add_module(f"relu_0", nn.ReLU())
        else:
            self.linear.add_module(f"lrelu_0", nn.LeakyReLU(self.negative_slope))
        self.conv.add_module(f"dropout_0", nn.Dropout(0.1))
        
        self.linear.add_module("linear_1", nn.Linear(128, 64))
        if self.batch_norm:
            self.linear.add_module(f"batch_norm_1", nn.BatchNorm1d(64))
        if self.negative_slope == 0.0:
            self.linear.add_module(f"relu_0", nn.ReLU())
        else:
            self.linear.add_module(f"lrelu_1", nn.LeakyReLU(self.negative_slope))
        self.conv.add_module(f"dropout_1", nn.Dropout(0.15))
        
        self.linear.add_module("linear_2", nn.Linear(64, 5))
        if self.sigmoid:
            self.linear.add_module(f"sigmoid", nn.Sigmoid())

    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.linear(x) 
        return x
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        x, y = batch['normalized'], batch['label']
        y_pred = self.forward(x)
        y = torch.tensor(y, dtype=torch.float32, device=y_pred.device)

        loss = F.mse_loss(y_pred, y)
        if self.classification:
            train_acc_batch = self.train_accuracy(y_pred, y.to(torch.int64))
            self.log('train_acc_batch', train_acc_batch, on_step=True, on_epoch=False)
            
        self.log('train_loss_batch', loss, on_step=True, on_epoch=False)
        
        return loss
    
    def training_epoch_end(self, outputs):
        sum_loss = 0.0
        for i in outputs:
            sum_loss += i['loss'].item()
        self.log('train_loss_epoch', sum_loss/len(outputs))
        if self.classification:
            accuracy = self.train_accuracy.compute()
            print(f"Train Accuracy: {accuracy}")
            self.log('train_acc_epoch', accuracy)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['normalized'], batch['label']
        y_pred = self.forward(x)
        y = torch.tensor(y, dtype=torch.float32, device=y_pred.device)

        loss = F.mse_loss(y_pred, y)
        if self.classification:
            val_acc_batch = self.val_accuracy(y_pred, y.to(torch.int64))

        return loss
    
    def validation_epoch_end(self, outputs):
        self.log('val_loss_epoch', sum(outputs)/len(outputs))
        if self.classification:
            accuracy = self.val_accuracy.compute()
            self.log('val_acc_epoch', accuracy)
