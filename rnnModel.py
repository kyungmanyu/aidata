# <br>0. Hyperparameter Setting
# - data_dir: 데이터가 존재하는 경로 (해당 실습에서는 train/test 시계열 데이터가 존재하는 경로를 의미함)
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
from dataAccess import *

import dataAccess as DA

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
import os
from os.path import join

# Hyperparameter setting
data_dir = '/content/LG_time_series_day11/input/har-data'
batch_size = 32
num_classes = 3
num_epochs = 200
window_size = 50
input_size = 10
hidden_size = 64
num_layers = 2
bidirectional = True

random_seed = 42

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # Detect if we have a GPU available

print('device',device)
# seed 고정
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


print(torch.cuda.is_available())

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


historyFileName = 'history.xlsx'
simulationDF = pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])


def store_history_data_to_excel(da):
    if os.path.isdir('history_data') == False:
        os.mkdir('history_data')

    wb = openpyxl.Workbook()
    ws = wb.active

    for r in dataframe_to_rows(da.simulationDF, index=True, header=True):
        ws.append(r)
    wb.save(join('history_data', historyFileName))

def load_history_data_from_excel(da):
    wb = openpyxl.load_workbook(join('history_data', historyFileName))
    sheet = wb.get_sheet_by_name('Sheet')
    df = pd.DataFrame(sheet.values)
    df.columns = df.iloc[0, :]
    df = df.iloc[2:, :]

    simulationDF['datetime'] = df['datetime'].astype(object)
    simulationDF['open'] = df['open'].astype(float)
    simulationDF['high'] = df['high'].astype(float)
    simulationDF['low'] = df['low'].astype(float)
    simulationDF['close'] = df['close'].astype(float)
    simulationDF['volume'] = df['volume'].astype(float)
    
    da.simulationDF = SA.set_rsi_adx_debug(simulationDF)



def create_classification_dataset(window_size, data_dir, batch_size):
    # data_dir에 있는 train/test 데이터 불러오기
    # x = pickle.load(open(os.path.join(data_dir, 'x_train.pkl'), 'rb'))
    # y = pickle.load(open(os.path.join(data_dir, 'state_train.pkl'), 'rb'))
    # x_test = pickle.load(open(os.path.join(data_dir, 'x_test.pkl'), 'rb'))
    # y_test = pickle.load(open(os.path.join(data_dir, 'state_test.pkl'), 'rb'))
    da = DA.dataAccess()
    simDays = 120
    symbol = 'ETH/USDT'
    unitTime = 1
    if os.path.isfile(join('history_data', historyFileName)) == True:
        load_history_data_from_excel(da)
    else:
        da.load_history_data_from_binance(simDays, unitTime ,symbol)
        store_history_data_to_excel(da)
    # da.load_history_data_from_binance(simDays, unitTime ,symbol)
    data = da.simulationDF
    # data['datetime'] =  pd.to_datetime(da.simulationDF['datetime'], unit='ms') + datetime.timedelta(hours=9)
    # data.set_index('datetime',inplace=True)
    # print('data last',data.iloc[-1])
    # dataX = data.drop(['close','datetime'], axis=1)
    
    window = 60
    
    label = []
    for i in range(0,len(data)-window):
        # label.append(data.iloc[i+5]['close'])
        if(i>999):
            for j in range (i+1,i+window):
                currentPrice = data.iloc[i]['close']
                futurePrice = data.iloc[j]['close']
                profit = (futurePrice - currentPrice)/currentPrice * 100
                if(profit > 1):
                    label.append(2)
                    break
                elif(profit < -1):
                    label.append(1)
                    break
                if(j == i+window-1):
                    label.append(0)
                    break
                
    data = data[1000:-window]
                    
    ypd = pd.DataFrame(label,columns=['label'])
    # print('ypd head',ypd.head())
    ypd.set_index(data.index,inplace=True)
    
    data = pd.concat([data,ypd],axis=1)
    
    
    
    data['datetime'] =  pd.to_datetime(data['datetime'], unit='ms') + datetime.timedelta(hours=9)
    data.set_index('datetime',inplace=True)
    
    # print('shape data : ',data.shape())
    
    # self.compareCov(data)
    
    # datay = data['close']
    # y = data['label'].values
    data_y = data['label'].to_numpy()
    # if(ind == 0):
    #     X = data.drop(['close','open','high','low','label'], axis=1,inplace=True)
    # elif(ind == 1):
    #     X = data.drop(['close','open','label'], axis=1,inplace=True)
    X = data.drop(['close','open','label'], axis=1,inplace=True)
    # X = data
    
    
    # window, label = scailing(data)
    # data_x = data[['Open','High','Low','Close', 'Volume', 'ema12', 'ema26']].to_numpy()
    # data_y = data[['label']].to_numpy()
    data_x = data.to_numpy()
    
      
    # data를 시간순으로 8:2:2의 비율로 train/validation/test set으로 분할
    # train_slice = slice(None, int(0.6 * len(data)))
    # valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    # test_slice = slice(int(0.8 * len(data)), None)
    
    # train_data_x = data_x[train_slice]
    # train_data_y = data_y[train_slice]
    
    # normalization
    scaler = MinMaxScaler()
    # scaler = scaler.fit(train_data_x)
    scaler = scaler.fit(data_x)
    data_x = scaler.transform(data_x)
    
    windows = [data_x[i:i + window_size] for i in range(0, len(data_x) - int(window_size))]
    windows = np.transpose(np.array(windows), (0, 2, 1))
    print('windows size',len(windows))
    print('window shape',windows.shape)
    labels = np.roll(data_y, int(-1 * window_size))
    
    print('labels shape',labels.shape)
    
    labels = labels[:len(windows)]
    print('labels re shape',labels.shape)
    print('labels size',len(labels))
    
    # if len(window_total) == 0: 
    #     window_total = window   
    # else : 
    #     window_total = np.concatenate((window_total, window), axis=0)
    # if len(label_total) == 0 : 
    #     label_total = label
    # else : 
    #     label_total = np.concatenate((label_total, label), axis=0)    

    data_x, test_data_x, data_y, test_data_y = train_test_split(windows, labels, test_size=0.2, shuffle=True, stratify=labels)    
    train_data_x, valid_data_x, train_data_y, valid_data_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=True,stratify=data_y)
    
    # train_data_x, train_data_y = windows[train_slice], labels[train_slice]
    # valid_data_x, valid_data_y = windows[valid_slice], labels[valid_slice]
    # test_data_x, test_data_y = windows[test_slice], labels[test_slice]
       
    # train/validation/test 데이터를 기반으로 window_size 길이의 input으로 바로 다음 시점을 예측하는 데이터 생성
    datasets = []
    datasets.append(torch.utils.data.TensorDataset(torch.Tensor(train_data_x), torch.Tensor(train_data_y)))
    datasets.append(torch.utils.data.TensorDataset(torch.Tensor(valid_data_x), torch.Tensor(valid_data_y)))
    datasets.append(torch.utils.data.TensorDataset(torch.Tensor(test_data_x), torch.Tensor(test_data_y)))

    # train/validation/test DataLoader 구축
    trainset, validset, testset = datasets[0], datasets[1], datasets[2]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


class Attention(nn.Module):
    def __init__(self, device, hidden_size):
        super(Attention, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        self.concat_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, rnn_outputs, final_hidden_state):
        # rnn_output.shape:         (batch_size, seq_len, hidden_size)
        # final_hidden_state.shape: (batch_size, hidden_size)
        # NOTE: hidden_size may also reflect bidirectional hidden states (hidden_size = num_directions * hidden_dim)
        batch_size, seq_len, _ = rnn_outputs.shape
        
        attn_weights = self.attn(rnn_outputs) # (batch_size, seq_len, hidden_dim)
        attn_weights = torch.bmm(attn_weights, final_hidden_state.unsqueeze(2))
        
        attn_weights = F.softmax(attn_weights.squeeze(2), dim=1)

        context = torch.bmm(rnn_outputs.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)

        attn_hidden = torch.tanh(self.concat_linear(torch.cat((context, final_hidden_state), dim=1)))

        return attn_hidden, attn_weights
    
    
    
class GRU_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional, device):
        super(GRU_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional == True else 1
        
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.attn = Attention(device, hidden_size * self.num_directions)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        batch_size, _, seq_len = x.shape
        
        # data dimension: (batch_size x input_size x seq_len) -> (batch_size x seq_len x input_size)로 변환
        x = torch.transpose(x, 1, 2)
        
        # initial hidden states 설정
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        
        # out: tensor of shape (batch_size, seq_len, hidden_size)
        rnn_output, hiddens = self.rnn(x, h0)
        final_state = hiddens.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)[-1]
        
        # Handle directions
        final_hidden_state = None
        if self.num_directions == 1:
            final_hidden_state = final_state.squeeze(0)
        elif self.num_directions == 2:
            h_1, h_2 = final_state[0], final_state[1]
            final_hidden_state = torch.cat((h_1, h_2), 1)  # Concatenate both states

        # Push through attention layer
        attn_output, attn_weights = self.attn(rnn_output, final_hidden_state)

        attn_output = self.fc(attn_output)
        return attn_output
    
    



def train_model(model, dataloaders, criterion, num_epochs, optimizer):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # 각 epoch마다 순서대로 training과 validation을 진행
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 training mode로 설정
            else:
                model.eval()   # 모델을 validation mode로 설정

            running_loss = 0.0
            running_corrects = 0
            running_total = 0

            # training과 validation 단계에 맞는 dataloader에 대하여 학습/검증 진행
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device, dtype=torch.long)

                # parameter gradients를 0으로 설정
                optimizer.zero_grad()

                # forward
                # training 단계에서만 gradient 업데이트 수행
                with torch.set_grad_enabled(phase == 'train'):
                    # input을 model에 넣어 output을 도출한 후, loss를 계산함
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # output 중 최댓값의 위치에 해당하는 class로 예측을 수행
                    _, preds = torch.max(outputs, 1)

                    # backward (optimize): training 단계에서만 수행
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # batch별 loss를 축적함
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_total += labels.size(0)

            # epoch의 loss 및 accuracy 도출
            epoch_loss = running_loss / running_total
            epoch_acc = running_corrects.double() / running_total

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # validation 단계에서 validation loss가 감소할 때마다 best model 가중치를 업데이트함
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    # 전체 학습 시간 계산
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # validation loss가 가장 낮았을 때의 best model 가중치를 불러와 best model을 구축함
    model.load_state_dict(best_model_wts)
    
    # best model 가중치 저장
    # torch.save(best_model_wts, '../output/best_model.pt')
    return model, val_acc_history




def test_model(model, test_loader):
    model.eval()   # 모델을 validation mode로 설정
    
    # test_loader에 대하여 검증 진행 (gradient update 방지)
    with torch.no_grad():
        corrects = 0
        total = 0
        preds_long = []
        preds_short = []
        preds_mid = []
        label_long = []
        label_short = []
        label_mid = []
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.long)

            # forward
            # input을 model에 넣어 output을 도출
            outputs = model(inputs)

            # output 중 최댓값의 위치에 해당하는 class로 예측을 수행
            _, preds = torch.max(outputs, 1)

            # batch별 정답 개수를 축적함
            for i in range (0,len(preds)):
                if preds[i] == 2 and preds[i] == labels.data[i]:
                    preds_long.append(1)
                elif preds[i] == 1 and preds[i] == labels.data[i]:
                    preds_short.append(1)
                elif preds[i] == 0 and preds[i] == labels.data[i]:
                    preds_mid.append(1)
                if labels.data[i] == 2:
                    label_long.append(1)
                elif labels.data[i] == 1:
                    label_short.append(1)
                elif labels.data[i] == 0:
                    label_mid.append(1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

    # accuracy를 도출함
    test_acc = corrects.double() / total
    print('Testing Acc: {:.4f}'.format(test_acc))
    
    print('Testing long pred: ', len(preds_long))
    print('Testing long ans: ', len(label_long))
    
    
    print('Testing short pred: ', len(preds_short))
    print('Testing short ans: ', len(label_short))
    
    print('Testing mid pred: ', len(preds_mid))
    print('Testing mid ans: ', len(label_mid))
    


def makeTestLoader(window_size,batch_size):
    da = DA.dataAccess()
    simDays = 120
    symbol = 'ETH/USDT'
    unitTime = 1
    if os.path.isfile(join('history_data', historyFileName)) == True:
        load_history_data_from_excel(da)
    else:
        da.load_history_data_from_binance(simDays, unitTime ,symbol)
        store_history_data_to_excel(da)
    # da.load_history_data_from_binance(simDays, unitTime ,symbol)
    data = da.simulationDF
    # data['datetime'] =  pd.to_datetime(da.simulationDF['datetime'], unit='ms') + datetime.timedelta(hours=9)
    # data.set_index('datetime',inplace=True)
    # print('data last',data.iloc[-1])
    # dataX = data.drop(['close','datetime'], axis=1)
    
    window = 60
    
    label = []
    for i in range(0,len(data)-window):
        # label.append(data.iloc[i+5]['close'])
        if(i>999):
            for j in range (i+1,i+window):
                currentPrice = data.iloc[i]['close']
                futurePrice = data.iloc[j]['close']
                profit = (futurePrice - currentPrice)/currentPrice * 100
                if(profit > 1):
                    label.append(2)
                    break
                elif(profit < -1):
                    label.append(1)
                    break
                if(j == i+window-1):
                    label.append(0)
                    break
                
    data = data[1000:-window]
                    
    ypd = pd.DataFrame(label,columns=['label'])
    # print('ypd head',ypd.head())
    ypd.set_index(data.index,inplace=True)
    
    data = pd.concat([data,ypd],axis=1)
    
    
    
    data['datetime'] =  pd.to_datetime(data['datetime'], unit='ms') + datetime.timedelta(hours=9)
    data.set_index('datetime',inplace=True)
    
    # print('shape data : ',data.shape())
    
    # self.compareCov(data)
    
    # datay = data['close']
    # y = data['label'].values
    data_y = data['label'].to_numpy()
    # if(ind == 0):
    #     X = data.drop(['close','open','high','low','label'], axis=1,inplace=True)
    # elif(ind == 1):
    #     X = data.drop(['close','open','label'], axis=1,inplace=True)
    X = data.drop(['close','open','label'], axis=1,inplace=True)
    # X = data
    
    
    # window, label = scailing(data)
    # data_x = data[['Open','High','Low','Close', 'Volume', 'ema12', 'ema26']].to_numpy()
    # data_y = data[['label']].to_numpy()
    data_x = data.to_numpy()
    
      
    # data를 시간순으로 8:2:2의 비율로 train/validation/test set으로 분할
    # train_slice = slice(None, int(0.6 * len(data)))
    # valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    # test_slice = slice(int(0.8 * len(data)), None)
    
    # train_data_x = data_x[train_slice]
    # train_data_y = data_y[train_slice]
    
    # normalization
    scaler = MinMaxScaler()
    # scaler = scaler.fit(train_data_x)
    scaler = scaler.fit(data_x)
    data_x = scaler.transform(data_x)
    
    windows = [data_x[i:i + window_size] for i in range(0, len(data_x) - int(window_size))]
    windows = np.transpose(np.array(windows), (0, 2, 1))
    print('windows size',len(windows))
    print('window shape',windows.shape)
    labels = np.roll(data_y, int(-1 * window_size))
    
    print('labels shape',labels.shape)
    
    labels = labels[:len(windows)]
    print('labels re shape',labels.shape)
    print('labels size',len(labels))
    
    # if len(window_total) == 0: 
    #     window_total = window   
    # else : 
    #     window_total = np.concatenate((window_total, window), axis=0)
    # if len(label_total) == 0 : 
    #     label_total = label
    # else : 
    #     label_total = np.concatenate((label_total, label), axis=0)    

    # data_x, test_data_x, data_y, test_data_y = train_test_split(windows, labels, test_size=0.2, shuffle=True, stratify=labels)    
    # train_data_x, valid_data_x, train_data_y, valid_data_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=True,stratify=data_y)
    test_data_x, test_data_y = train_test_split(windows, labels, test_size=1, shuffle=True, stratify=labels)    
    # train_data_x, valid_data_x, train_data_y, valid_data_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=True,stratify=data_y)
    
    # train_data_x, train_data_y = windows[train_slice], labels[train_slice]
    # valid_data_x, valid_data_y = windows[valid_slice], labels[valid_slice]
    # test_data_x, test_data_y = windows[test_slice], labels[test_slice]
       
    # train/validation/test 데이터를 기반으로 window_size 길이의 input으로 바로 다음 시점을 예측하는 데이터 생성
    datasets = []
    # datasets.append(torch.utils.data.TensorDataset(torch.Tensor(train_data_x), torch.Tensor(train_data_y)))
    # datasets.append(torch.utils.data.TensorDataset(torch.Tensor(valid_data_x), torch.Tensor(valid_data_y)))
    datasets.append(torch.utils.data.TensorDataset(torch.Tensor(test_data_x), torch.Tensor(test_data_y)))

    # train/validation/test DataLoader 구축
    # trainset, validset, testset = datasets[0], datasets[1], datasets[2]
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
    # valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=False)
    
    return test_loader
    


# Dataloader 구축
# data shape: (batch_size x input_size x seq_len)
train_loader, valid_loader, test_loader = create_classification_dataset(window_size, data_dir, batch_size)
    
    
# GRU 모델 구축
gru = GRU_Attention(input_size, hidden_size, num_layers, num_classes, bidirectional, device)
gru = gru.to(device)
print(gru)
    

# trining 단계에서 사용할 Dataloader dictionary 생성
dataloaders_dict = {
    'train': train_loader,
    'val': valid_loader
}


# loss function 설정
criterion = nn.CrossEntropyLoss()

# # GRU with attention 모델 학습
# gru, gru_val_acc_history = train_model(gru, dataloaders_dict, criterion, num_epochs,
#                                        optimizer=optim.Adam(gru.parameters(), lr=0.001))

    # GRU with attention 모델 검증하기 (Acc: 0.8889)
# Benchmark model인 GRU(Acc: 0.8000)와 비교했을 때, Attetion의 적용이 성능 향상에 도움이 됨을 알 수 있음


test_loader = makeTestLoader(window_size,batch_size)

gru.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu')))

test_model(gru, test_loader)