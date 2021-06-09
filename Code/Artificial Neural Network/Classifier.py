"""
@author: Jinal Shah

This script will host my
model

"""
# Importing needed libraries
import torch.nn as nn
import torch.nn.functional as F

# Building the Models
class Digit_Classifier(nn.Module):
    # Constructor
    def __init__(self):
        super().__init__()

        # Building the layers
        self.input_layer = nn.Linear(784, 500)
        self.hidden1 = nn.Linear(500, 250)
        self.hidden2 = nn.Linear(250, 128)
        self.hidden3 = nn.Linear(128, 100)
        self.hidden4 = nn.Linear(100, 64)
        self.output = nn.Linear(64, 10)
        self.dropout = nn.Dropout(p=0.2)

    # Forward method
    def forward(self, x):
        out = x.view(-1, 784)
        out = out.float()
        out = self.dropout(F.relu(self.input_layer(out)))
        out = self.dropout(F.relu(self.hidden1(out)))
        out = self.dropout(F.relu(self.hidden2(out)))
        out = self.dropout(F.relu(self.hidden3(out)))
        out = self.dropout(F.relu(self.hidden4(out)))
        out = self.output(out)
        fin = nn.LogSoftmax(dim=1)
        return fin(out)
