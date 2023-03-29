# 모듈 불러오기
import os
import time
import copy
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

""" 시각화 """
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameter setting
data_dir = './340930.csv'
batch_size = 64
num_classes = 2
num_epochs = 30
window_size = 36
input_size = 6
hidden_size = 8
num_layers = 4
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

def create_forecasting_dataset(window_size, data_dir, batch_size):
    data = pd.read_csv(data_dir, index_col='Date', parse_dates=True)

    data = data[50:]
    data_x = data[['Open','High','Low','Close', 'ema12', 'ema26']].to_numpy()
    data_y = data[['Close', 'High']].to_numpy()
      
    # data를 시간순으로 8:2:2의 비율로 train/validation/test set으로 분할
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    train_data_x = data_x[train_slice]
    train_data_y = data_y[train_slice]
    
    # normalization
    scaler = StandardScaler()
    scaler = scaler.fit(train_data_x)
    data_x = scaler.transform(data_x)
    
    scaler_y = StandardScaler()
    scaler_y = scaler_y.fit(train_data_y)
    data_y = scaler_y.transform(data_y)
    
    windows = [data_x[i:i + window_size] for i in range(0, len(data_x) - int(window_size))]
    windows = np.transpose(np.array(windows), (0, 2, 1))
    # labels = np.roll(data_y, int(-1 * window_size))
    # labels = labels[:len(windows)].squeeze(-1)
    labels = [data_y[j] for j in range(int(window_size), len(data_y))]    
    labels = np.squeeze(labels)
    
    train_data_x, train_data_y = windows[train_slice], labels[train_slice]
    valid_data_x, valid_data_y = windows[valid_slice], labels[valid_slice]
    test_data_x, test_data_y = windows[test_slice], labels[test_slice]
       
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

    return train_loader, valid_loader, test_loader, scaler_y

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional, rnn_type='rnn'):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.num_directions = 2 if bidirectional == True else 1
        
        # rnn_type에 따른 recurrent layer 설정
        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        
        # bidirectional에 따른 fc layer 구축
        # bidirectional 여부에 따라 hidden state의 shape가 달라짐 (True: 2 * hidden_size, False: hidden_size)
        self.fc = nn.Linear(self.num_directions * hidden_size, num_classes)

    def forward(self, x):
        # data dimension: (batch_size x input_size x seq_len) -> (batch_size x seq_len x input_size)로 변환
        x = torch.transpose(x, 1, 2)
        
        # initial hidden states 설정
        h0 = torch.zeros(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # 선택한 rnn_type의 RNN으로부터 output 도출
        if self.rnn_type in ['rnn', 'gru']:
            out, _ = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        else:
            # initial cell states 설정
            c0 = torch.zeros(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to(device)
            out, _ = self.rnn(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        out = self.fc(out[:, -1, :])
        return out
    
def train_model(model, dataloaders, criterion, num_epochs, optimizer):
    since = time.time()

    val_rmse_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_rmse = 10000000

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
            running_total = 0

            # training과 validation 단계에 맞는 dataloader에 대하여 학습/검증 진행
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # parameter gradients를 0으로 설정
                optimizer.zero_grad()

                # forward
                # training 단계에서만 gradient 업데이트 수행
                with torch.set_grad_enabled(phase == 'train'):
                    # input을 model에 넣어 output을 도출한 후, loss를 계산함
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels)

                    # backward (optimize): training 단계에서만 수행
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # batch별 loss를 축적함
                running_loss += loss.item() * inputs.size(0)
                running_total += labels.size(0)

            # epoch의 loss 및 RMSE 도출
            epoch_loss = running_loss / running_total
            epoch_rmse = np.sqrt(running_loss / running_total)

            print('{} Loss: {:.4f} RMSE: {:.4f}'.format(phase, epoch_loss, epoch_rmse))

            # validation 단계에서 validation loss가 감소할 때마다 best model 가중치를 업데이트함
            if phase == 'val' and epoch_rmse < best_rmse:
                best_rmse = epoch_rmse
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_rmse_history.append(epoch_rmse)

        print()

    # 전체 학습 시간 계산
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val RMSE: {:4f}'.format(best_rmse))

    # validation loss가 가장 낮았을 때의 best model 가중치를 불러와 best model을 구축함
    model.load_state_dict(best_model_wts)
    
    # best model 가중치 저장
    # torch.save(best_model_wts, '../output/best_model.pt')
    return model, val_rmse_history

def test_model(model, test_loader, scaler, criterion):
    model.eval()   # 모델을 validation mode로 설정
    
    trues = []
    preds = []
    
    # test_loader에 대하여 검증 진행 (gradient update 방지)
    with torch.no_grad():
        rmse = 0
        total = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.float)

            # forward
            # input을 model에 넣어 output을 도출한 후, loss를 계산함
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)

            # batch별 loss를 축적함
            rmse += loss.item() * inputs.size(0)
            total += labels.size(0)
            
            true = scaler.inverse_transform(labels.cpu())
            trues += true.squeeze().tolist()
            
            pred = scaler.inverse_transform(outputs.cpu())
            preds += pred.squeeze().tolist()

    # RMSE를 도출함
    test_rmse = np.sqrt(rmse / total)
    print('Testing RMSE: {:.4f}'.format(test_rmse))
    return trues, preds

def RNN_Process():
    # Dataloader 구축하기
    # data dimension: (batch_size x input_size x seq_len)
    train_loader, valid_loader, test_loader, scaler = create_forecasting_dataset(window_size, data_dir, batch_size)

    # Vanilla, LSTM, GRU RNN 모델 구축
    # rnn, lstm, gru
    type = 'gru'
    model = RNN(input_size, hidden_size, num_layers, num_classes, bidirectional, rnn_type = type)
    model = model.to(device)
    # print(model)

    # trining 단계에서 사용할 Dataloader dictionary 생성
    dataloaders_dict = {'train': train_loader, 'val': valid_loader }

    # loss function 설정
    criterion = nn.MSELoss()

    # Vanilla, LSTM, GRU RNN 모델 학습
    model, rnn_val_rmse_history = train_model(model, dataloaders_dict, criterion, num_epochs,
                                            optimizer=optim.Adam(model.parameters(), lr=0.0001))

    # Vanilla, LSTM, GRU RNN 모델 검증 (RMSE: 0.3436)
    # trues, rnn_preds = kc_test_model(model, test_loader, scaler, criterion)
    trues, rnn_preds = test_model(model, test_loader, scaler, criterion)  
      
    return trues, rnn_preds

def plot_data(trues, rnn_preds):
    rnn_df = pd.DataFrame()
    rnn_df['True_c'] = trues[:, 0]
    rnn_df['RNN_c'] = rnn_preds[:, 0]
    
    rnn_df['True_h'] = trues[:, 1]
    rnn_df['RNN_h'] = rnn_preds[:, 1]

    rnn_df.plot(figsize=(18, 6))
    
    plt.title("Model")
    plt.xlabel('time')
    plt.ylabel('value')
    # plt.plot(y_train)
    # plt.plot(tree_prediction)
    # plt.legend(["Original", "Valid", 'Predicted'])
    plt.show()

trues, rnn_preds = RNN_Process()
# print(type(trues),trues.shape)
plot_data(np.array(trues), np.array(rnn_preds))
print('finsish')

