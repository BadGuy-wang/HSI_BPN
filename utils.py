import pandas as pd
import numpy as np
from spectral import *
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torch


def load_data(data_path, label_path, data_name):
    if data_name == 'indian_pines':
        data = loadmat(data_path)['indian_pines_corrected']
        label = loadmat(label_path)['indian_pines_gt']
    elif data_name == 'salinas':
        data = loadmat(data_path)['salinas']
        label = loadmat(data_path)['salinas_gt']
    else:
        print('Please enter right dataset name')

    return np.asarray(data), np.asarray(label)

def get_value_data(data, label):
    new_data_list = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if label[i][j] != 0:
                temp = list(data[i][j])
                temp.append(label[i][j])
                new_data_list.append(temp)

    df = pd.DataFrame(new_data_list)
    df.to_csv('datasets/Indian_pines.csv', header=None, index=None)

def plot_curve(data):
    """
    Show training image

    Args:
        data:Two-dimensional array of image
    """
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()


def plot_image(label, name):
    """
    Show image  tag, name, content
    Args:
        img:train or test image
        label:Image's label
        name:Image's name
    return:None
    """
    fig = plt.figure()
    imshow = spectral.imshow(classes=label.astype(int))
    plt.title(name)
    plt.show()

class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)