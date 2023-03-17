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


# Hyperparameter setting
data_dir = '/content/LG_time_series_day11/input/har-data'
batch_size = 32
num_classes = 3
num_epochs = 200
window_size = 50
input_size = 5
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



def create_classification_dataset(window_size, data_dir, batch_size):
    # data_dir에 있는 train/test 데이터 불러오기
    x = pickle.load(open(os.path.join(data_dir, 'x_train.pkl'), 'rb'))
    y = pickle.load(open(os.path.join(data_dir, 'state_train.pkl'), 'rb'))
    x_test = pickle.load(open(os.path.join(data_dir, 'x_test.pkl'), 'rb'))
    y_test = pickle.load(open(os.path.join(data_dir, 'state_test.pkl'), 'rb'))

    # train data를 시간순으로 8:2의 비율로 train/validation set으로 분할
    n_train = int(0.8 * len(x))
    n_valid = len(x) - n_train
    n_test = len(x_test)
    x_train, y_train = x[:n_train], y[:n_train]
    x_valid, y_valid = x[n_train:], y[n_train:]

    # train/validation/test 데이터를 window_size 시점 길이로 분할
    datasets = []
    for set in [(x_train, y_train, n_train), (x_valid, y_valid, n_valid), (x_test, y_test, n_test)]:
        T = set[0].shape[-1]
        windows = np.split(set[0][:, :, :window_size * (T // window_size)], (T // window_size), -1)
        windows = np.concatenate(windows, 0)
        labels = np.split(set[1][:, :window_size * (T // window_size)], (T // window_size), -1)
        labels = np.round(np.mean(np.concatenate(labels, 0), -1))
        datasets.append(torch.utils.data.TensorDataset(torch.Tensor(windows), torch.Tensor(labels)))

    # train/validation/test DataLoader 구축
    trainset, validset, testset = datasets[0], datasets[1], datasets[2]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

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
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.long)

            # forward
            # input을 model에 넣어 output을 도출
            outputs = model(inputs)

            # output 중 최댓값의 위치에 해당하는 class로 예측을 수행
            _, preds = torch.max(outputs, 1)

            # batch별 정답 개수를 축적함
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

    # accuracy를 도출함
    test_acc = corrects.double() / total
    print('Testing Acc: {:.4f}'.format(test_acc))
    
    





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

# GRU with attention 모델 학습
gru, gru_val_acc_history = train_model(gru, dataloaders_dict, criterion, num_epochs,
                                       optimizer=optim.Adam(gru.parameters(), lr=0.001))

    # GRU with attention 모델 검증하기 (Acc: 0.8889)
# Benchmark model인 GRU(Acc: 0.8000)와 비교했을 때, Attetion의 적용이 성능 향상에 도움이 됨을 알 수 있음
test_model(gru, test_loader)