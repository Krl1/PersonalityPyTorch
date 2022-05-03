import pytorch_lightning as pl
import torch
from torch import nn as nn
from torch.nn import functional as F
from torchmetrics import Accuracy

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
            nn.Conv2d(1, fm_size, 3, padding=1)
        )
        self.conv.add_module(f"relu_0", nn.ReLU())
        self.conv.add_module(f"max_pool_0", nn.MaxPool2d(2))
        for i in range(1, 8):
            self.conv.add_module(
                f"conv_{i}", 
                nn.Conv2d(fm_size, fm_size*2, 3, stride=1, padding='same')
            )
            self.conv.add_module(f"relu_{i}", nn.ReLU())
            pad = 1 if i in [4, 5] else 0
            self.conv.add_module(f"max_pool_{i}", nn.MaxPool2d(2, padding=pad))
            fm_size *= 2
        
        self.linear = nn.Sequential() 
        self.linear.add_module("flatten", nn.Flatten())
        self.linear.add_module("linear_0", nn.Linear(fm_size, 50))
        self.linear.add_module(f"relu_0", nn.ReLU())
        self.linear.add_module("linear_1", nn.Linear(50, 50))
        self.linear.add_module(f"relu_0", nn.ReLU())
        self.linear.add_module("linear_2", nn.Linear(50, 5))
        self.linear.add_module(f"sigmoid", nn.Sigmoid())

    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.linear(x) 
        return x
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        y = torch.tensor(y, dtype=torch.float32, device=y_pred.device)

        loss = F.mse_loss(y_pred, y)
        train_acc_batch = self.train_accuracy(y_pred, y.to(torch.int64))

        self.log('train_acc_batch', train_acc_batch)
        self.log('train_loss_batch', loss)
        
        return {'loss' : loss, 'y_pred' : y_pred, 'y_true' : y}
    
    def training_epoch_end(self, outputs):
        accuracy = self.train_accuracy.compute()
        print(f"Train Accuracy: {accuracy}")
        self.log('Train_acc_epoch', accuracy, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        y = torch.tensor(y, dtype=torch.float32, device=y_pred.device)

        loss = F.mse_loss(y_pred, y)
        val_acc_batch = self.val_accuracy(y_pred, y.to(torch.int64))
        
        self.log('val_acc_batch', val_acc_batch, prog_bar=False)
        self.log('val_loss_batch', loss, prog_bar=False)

        return {'loss' : loss, 'y_pred' : y_pred, 'target' : y}
    
    def validation_epoch_end(self, outputs):
        accuracy = self.val_accuracy.compute()

        self.log('val_acc_epoch', accuracy, prog_bar=True)