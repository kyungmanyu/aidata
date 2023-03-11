
# # <br>0. Hyperparameter Setting
# - data_dir: 데이터의 경로 (해당 실습에서는 csv 파일의 경로를 의미함)
# - batch_size: 학습 및 검증에 사용할 배치의 크기
# - num_classes: 새로운 데이터의 class 개수
# - num_epochs: 학습할 epoch 횟수
# - window_size: input의 시간 길이 (time series data에서 도출한 subsequence의 길이)
# - input_size: 변수 개수
# - hidden_size: 모델의 hidden dimension
# - num_layers: 모델의 layer 개수
# - bidirectional: 모델의 양방향성 여부
# - random_seed: reproduction을 위해 고정할 seed의 값

# 모듈 불러오기
import os
import time
import copy
import random
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Hyperparameter setting
data_dir = '/content/LG_time_series_day11/input/har-data'
batch_size = 32
num_classes = 6
num_epochs = 200
window_size = 50
input_size = 561
hidden_size = 64
num_layers = 2
bidirectional = True

random_seed = 42
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Detect if we have a GPU available

# seed 고정
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


print(torch.cuda.is_available())