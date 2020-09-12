import torch
import pandas as pd
from utils import GetLoader, load_data, plot_image
from torch.utils.data import DataLoader
from models import Baseline, test
import numpy as np
from spectral import *
#%%

BATCH_SIZE = 100
INPUT_CHANNELS = 200
CLASSES = 17
DEVICE = torch.device('cpu')
data_path = 'datasets/Indian_pines_corrected.mat'
label_path = 'datasets/Indian_pines_gt.mat'

data, label = load_data(data_path, label_path, 'indian_pines')

DATA = pd.read_csv('datasets/Indian_pines.csv', header=None).values
data_D = DATA[:,:-1]
data_L = DATA[:,-1]

data_set = GetLoader(data_D, data_L)
data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=False)

net = Baseline(INPUT_CHANNELS, CLASSES, dropout=False)
net.load_state_dict(torch.load('checkpionts/BaseLine_run90_0.6837804878048781.pth'))

pred_labels = test(net, data_loader, DEVICE)
#%%
new_label = []
for i in range(len(pred_labels)):
    new_label.extend(pred_labels[i].tolist())
#%%
# for l in [new_label[i] for i in range(len(new_label))]:
#     print(l,end=',')
#%%
pred_matrix = np.zeros((data.shape[0], data.shape[1]))
count = 0
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if label[i][j] != 0:
            pred_matrix[i][j] = new_label[count]
            count += 1
#%%
save_rgb('gt.jpg', pred_matrix, colors=spy_colors)

