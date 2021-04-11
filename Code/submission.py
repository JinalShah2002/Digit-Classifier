"""

@author: Jinal Shah

This script will deploy the
model on the submission set

"""
# Importing needed libraries
import pandas as pd
import numpy as np
import torch
from Classifier import Digit_Classifier

# Loading the model
loaded = torch.load('/Users/jinalshah/Jinal/Github Repos/Digit-Classifier/Models/model2.pt')
model = Digit_Classifier()
model.load_state_dict(loaded['weights & biases'])

# Getting the data
submission_data = pd.read_csv('/Users/jinalshah/Jinal/Github Repos/Digit-Classifier/Data/test.csv')
submission_data = submission_data.values
submission_data = torch.from_numpy(submission_data)

# Making predictions
model.eval()
logps = model(submission_data)
_, top_class = torch.exp(logps).topk(1, dim=1)

# Creating submission file
pred = top_class.numpy()
pred_list = []
for element in pred:
    pred_list.append(element[0])

pred_dict = {
    'ImageId': np.arange(start=1, stop=submission_data.shape[0]+1),
    'Label': pred_list,
}
pred_dict = pd.DataFrame(pred_dict)
pred_dict.to_csv('/Users/jinalshah/Jinal/Github Repos/Digit-Classifier/Submissions/submission3.csv', index=False)
print('File ran succesfully!!!')
