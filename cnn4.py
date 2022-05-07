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
        lr=1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        
        fm_size = 16
        self.conv = nn.Sequential()
        self.conv.add_module(
            f"conv_0", 
            nn.Conv2d(1, fm_size, (3,3), padding=1)
        )
        self.conv.add_module(f"relu_0", nn.LeakyReLU(0.1))
        self.conv.add_module(f"max_pool_0", nn.MaxPool2d(2))
        
        for i in range(1, 8):
            self.conv.add_module(f"conv_{i}", nn.Conv2d(fm_size, fm_size*2, (3,3), stride=1, padding='same'))
            self.conv.add_module(f"relu_{i}", nn.LeakyReLU(0.1))
            self.conv.add_module(f"max_pool_{i}", nn.MaxPool2d((2,2), ceil_mode=True))
            fm_size *= 2
            
        self.linear = nn.Sequential() 
        self.linear.add_module("flatten", nn.Flatten())
        self.linear.add_module("linear_0", nn.Linear(fm_size, 50))
        self.linear.add_module(f"relu_0", nn.LeakyReLU(0.1))
        self.linear.add_module("linear_1", nn.Linear(50, 50))
        self.linear.add_module(f"relu_0", nn.LeakyReLU(0.1))
        self.linear.add_module("linear_2", nn.Linear(50, 5))
        # self.linear.add_module(f"sigmoid", nn.Sigmoid())

    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.linear(x) 
        return x
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        x, y = batch['normalized'], batch['label']
        normalize = transforms.Normalize((x.mean()), (x.std()))
        x = normalize(x)
        y_pred = self.forward(x)
        y = torch.tensor(y, dtype=torch.float32, device=y_pred.device)

        loss = F.mse_loss(y_pred, y)
        train_acc_batch = self.train_accuracy(y_pred, y.to(torch.int64))

        self.log('train_acc_batch', train_acc_batch, on_step=True, on_epoch=False)
        self.log('train_loss_batch', loss, on_step=True, on_epoch=False)
        
        return loss
    
    def training_epoch_end(self, outputs):
        accuracy = self.train_accuracy.compute()
        print(f"Train Accuracy: {accuracy}")
        sum_loss = 0.0
        for i in outputs:
            sum_loss += i['loss'].item()
        self.log('train_loss_epoch', sum_loss/len(outputs))
        self.log('train_acc_epoch', accuracy)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['normalized'], batch['label']
        normalize = transforms.Normalize((x.mean()), (x.std()))
        x = normalize(x)
        y_pred = self.forward(x)
        y = torch.tensor(y, dtype=torch.float32, device=y_pred.device)

        loss = F.mse_loss(y_pred, y)
        val_acc_batch = self.val_accuracy(y_pred, y.to(torch.int64))

        return loss
    
    def validation_epoch_end(self, outputs):
        accuracy = self.val_accuracy.compute()
        self.log('val_loss_epoch', sum(outputs)/len(outputs))
        self.log('val_acc_epoch', accuracy)
