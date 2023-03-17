import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# from sklearn.externals import joblib 
import sklearn.externals as extjoblib
import joblib
import xgboost as xgb
import lightgbm as lgb

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from dataAccess import *

import dataAccess as DA

from sklearn.ensemble import StackingRegressor

from sklearn.linear_model import LinearRegression

from sklearn.datasets import make_regression

from sklearn.metrics import f1_score

import datetime

import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
import os
from os.path import join


class blockfine():
    acc_valid = []
    f1_valid = []

    X_train = 0
    X_test=0
    y_test = 0
    y_train = 0
    X_vaild = 0
    y_valid = 0
    test_pred_y = 0

    def __init__(self):
      
        super().__init__()
    
    def makeData(self,da):
        print('make data')
        # data = pd.read_csv('debugnew.csv')
        # X = data.drop('close', axis=1)
        # y = data['close']


        simDays = 365
        symbol = 'ETH/USDT'
        unitTime = 1
        da.load_history_data_from_binance(simDays, unitTime ,symbol)
        data = da.simulationDF
        # data['datetime'] =  pd.to_datetime(da.simulationDF['datetime'], unit='ms') + datetime.timedelta(hours=9)
        # data.set_index('datetime',inplace=True)
        print('data last',data.iloc[-1])
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
        print('ypd head',ypd.head())
        ypd.set_index(data.index,inplace=True)
        
        data = pd.concat([data,ypd],axis=1)
        
        
        
        data['datetime'] =  pd.to_datetime(data['datetime'], unit='ms') + datetime.timedelta(hours=9)
        data.set_index('datetime',inplace=True)
        
        # self.compareCov(data)
        
        # datay = data['close']
        y = data['label'].values
        X = data.drop(['close','open','high','low','label'], axis=1,inplace=True)
        X = data
        
        # X = data.drop(['close','open','high','low'], axis=1)
        
        # ypd = pd.DataFrame(label)
        
        # print(X.head())
        
        # print('-----------')
        # print(ypd.head())
        
        # ypd.set_index(X.index,inplace=True)
        
        # con = pd.concat([X,ypd],axis=1)
        # print('con',con.head())
        print('lenx',len(X))
        print('leny',len(y))
        
        print('head x',X.head())
        # print('head y',y.head())
        # compX = dataX[1000:-5]
        # print('lencompX',len(compX))
        # print('datay',label)
        print('datay len',len(label))
        print('datay sum',sum(label))
        # y = label
        
        # print(X)
        # self.compareCov(con)
        
        
        
        # data = pd.read_csv('motordata.csv')

        # X = data.drop('suction', axis=1)
        # y = data['suction']

        # print(y)

        random_state = 2023
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=3/10, random_state=random_state,shuffle=True,stratify=y)
        # self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train, self.y_train, test_size=2/8,
        #                                                     random_state=random_state,shuffle=False,stratify=self.y_train)
        
        # data_x, test_data_x, data_y, test_data_y = train_test_split(windows, labels, test_size=0.2, shuffle=True, stratify=labels)    
        # train_data_x, valid_data_x, train_data_y, valid_data_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=True,stratify=data_y)
        # max_depths = list(range(1, 10)) + [None]
        # print(max_depths)
        
    def makeMortordata(self):
        data = pd.read_csv('motordata.csv')

        X = data.drop('suction', axis=1)
        y = data['suction']

       

        random_state = 2023
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=2/10, random_state=random_state,shuffle=False)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train, self.y_train, test_size=2/8,
                                                            random_state=random_state,shuffle=False)
        # max_depths = list(range(1, 10)) + [None]
        # print(max_depths)
        
        return X,y
    
    
    def compareCov(self,x):
        correlation_mat = x.corr()
        sns.heatmap(correlation_mat, annot = True)
        plt.show()

    def trainTestonly(self,da):
        
        simDays = 90
        symbol = 'ETH/USDT'
        unitTime = 1
        da.load_history_data_from_binance(simDays, unitTime ,symbol)
        data = da.simulationDF
        data['datetime'] =  pd.to_datetime(da.simulationDF['datetime'], unit='ms') + datetime.timedelta(hours=9)
        data.set_index('datetime',inplace=True)
        


        dataX = data
        
        label = []
        for i in range(0,len(data)-5):
            # label.append(data.iloc[i+5]['close'])
            if(i>999):
                for j in range (i+1,i+5):
                    currentPrice = data.iloc[i]['close']
                    futurePrice = data.iloc[j]['close']
                    profit = (futurePrice - currentPrice)/currentPrice * 100
                    if(profit > 1):
                        label.append(2)
                        break
                    elif(profit < -1):
                        label.append(1)
                        break
                    else:
                        label.append(0)
                        break
                        
                    
        
        # datay = data['close']
        X = dataX[1000:-5]
        print('lenx',len(X))
        # compX = dataX[1000:-5]
        # print('lencompX',len(compX))
        # print('datay',label)
        print('datay len',len(label))
        print('datay sum',sum(label))
        y = label
        
        self.X_test = X
        self.y_test = y
        
        self.compareResult()

    def trainDataRF(self):
        print('train data')
        param = {'n_estimators': [100, 200],
              'oob_score': [True], # compute out of bag error
              'n_jobs':[-1], 
              'max_depth': [3, 5]
              }
        # rf_model = RandomForestRegressor()
        rf_model = RandomForestClassifier()
        

        # hyperparameter search
        # regression
        # grid_search = GridSearchCV(rf_model, param_grid=param, cv=5, scoring='neg_mean_squared_error')
        # classfication
        grid_search = GridSearchCV(rf_model, param_grid=param, cv=5, scoring='f1')
        grid_search.fit( self.X_train, self.y_train)

        print(grid_search.best_params_)

        opt_model = grid_search.best_estimator_
        joblib.dump(opt_model, 'lgb.pkl')

        self.test_pred_y = opt_model.predict(self.X_test)
        
        x = []
        ans = []
        for i in range(len(self.test_pred_y)):
            if(self.test_pred_y[i] == self.y_test[i]):
                ans.append(1)
            x.append(i)
            
        accuracy = sum(ans)/len(self.y_test) * 100
        mae = np.mean(np.abs(self.test_pred_y - self.y_test))
        print('rf acc',accuracy)
        
        return opt_model
    def compareResult(self):
        load_model = joblib.load('lgb.pkl')
        
        self.test_pred_y = load_model.predict(self.X_test)
        x = []
        ans = []
        long = []
        short = []
        long_pred = []
        short_pred = []
        for i in range(len(self.test_pred_y)):
            if self.y_test[i] == 2:
                long.append(self.y_test[i])
                # print('long index:',self.X_test.index[i])
                # print('long actual:',self.y_test[i])
            elif self.y_test[i] == 1:
                short.append(self.y_test[i])
                # print('short index:',self.X_test.index[i])
                # print('short actual:',self.y_test[i])
            if self.test_pred_y[i] == 2:
                long_pred.append(self.test_pred_y[i])
                # print('long index:',self.X_test.index[i])
                # print('long predict:',self.test_pred_y[i])
            elif self.test_pred_y[i] == 1:
                short_pred.append(self.y_test[i])
                # print('short index:',self.X_test.index[i])
                # print('short predic:',self.test_pred_y[i])
            
            if(self.test_pred_y[i] == self.y_test[i]):
                ans.append(1)
            x.append(i)
            
        # accuracy = sum(ans)/len(self.y_test) * 100
        print('long short pred',len(long_pred)+len(short_pred))
        print('long short act',(len(long)+len(short)))
        accuracy = (len(long_pred)+len(short_pred))/(len(long)+len(short)) * 100
        # mae = np.mean(np.abs(self.test_pred_y - self.y_test))
        print('comlgb acc',accuracy)
        print('comlgb long',len(long))
        print('comlgb short',len(short))
        now = datetime.datetime.now()
        print('endtime',str(now))
        print('f1score',f1_score( self.y_test, self.test_pred_y ,average='weighted'))
        
        # lable =  pd.DataFrame(self.y_test,columns=['label'],dtype=float)
        # predict = pd.DataFrame(self.test_pred_y,columns=['predict'],dtype=float)
        # ylable = pd.concat([lable,predict],axis=0)
        # data = self.X_test
        # data = pd.concat([self.X_test,lable],axis=0)
        # data = pd.concat([data,ylable],axis=0)
        # if os.path.isdir('profit') == False:
        #     os.mkdir('profit')  
        # wb = openpyxl.Workbook()
        # ws = wb.active
        # for r in dataframe_to_rows(data, index=True, header=True):
        #     ws.append(r)
        # wb.save(join('profit', 'test.xlsx'))
    def trainDataXGB(self):
        print('train data xgb')
        param = {"max_depth": [25,50,75],
              "min_child_weight" : [1,3,6],
              "n_estimators": [200], 
              "learning_rate": [0.05, 0.1,0.16]  }
        # rf_model = xgb.XGBRegressor()
        rf_model = xgb.XGBClassifier()


        # hyperparameter search
        # grid_search = GridSearchCV(rf_model, param_grid=param, cv=5, scoring='neg_mean_squared_error')
        grid_search = GridSearchCV(rf_model, param_grid=param, cv=5, scoring='f1')
        # grid_search = GridSearchCV(rf_model, param_grid=param, cv=5, scoring='accuracy')
        
        grid_search.fit( self.X_train, self.y_train)

        print(grid_search.best_params_)

        opt_model = grid_search.best_estimator_


        self.test_pred_y = opt_model.predict(self.X_test)
        x = []
        ans = []
        for i in range(len(self.test_pred_y)):
            if(self.test_pred_y[i] == self.y_test[i]):
                ans.append(1)
            x.append(i)
            
        accuracy = sum(ans)/len(self.y_test) * 100
        mae = np.mean(np.abs(self.test_pred_y - self.y_test))
        print('xgb acc',accuracy)
        return opt_model
        
    def trainDataLGB(self):
        print('train data lgb')
        now = datetime.datetime.now()
        print('start time',str(now))
        param = { "objective":['multiclass'], # multiclass, regression
              "max_depth": [25,50, 75],
              "learning_rate" : [0.01,0.05,0.1],
              "num_leaves": [300,900,1200],
              "n_estimators": [200]     }
        # rf_model = lgb.LGBMRegressor()  
        # rf_model = lgb.LGBMClassifier(device='gpu')  
        rf_model = lgb.LGBMClassifier() 


        # hyperparameter search
        # grid_search = GridSearchCV(rf_model, param_grid=param, cv=5, scoring='neg_mean_squared_error')
        grid_search = GridSearchCV(rf_model, param_grid=param, cv=5, scoring='f1_weighted')
        grid_search.fit( self.X_train, self.y_train)

        print(grid_search.best_params_)

        
        opt_model = grid_search.best_estimator_
        joblib.dump(opt_model, 'lgb.pkl')

        self.test_pred_y = opt_model.predict(self.X_test)
        
        # x = []
        # ans = []
        # for i in range(len(self.test_pred_y)):
        #     if(self.test_pred_y[i] == self.y_test[i]):
        #         ans.append(1)
        #     x.append(i)
            
        # accuracy = sum(ans)/len(self.y_test) * 100
        # mae = np.mean(np.abs(self.test_pred_y - self.y_test))
        # print('lgb acc',accuracy)
        return opt_model
        
    def showResult(self):
        x = []
        ans = []
        for i in range(len(self.test_pred_y)):
            if(self.test_pred_y[i] == self.y_test[i]):
                ans.append(1)
            x.append(i)
            
        accuracy = sum(ans)/len(self.y_test) * 100
        mae = np.mean(np.abs(self.test_pred_y - self.y_test))
        print(accuracy)
        plt.figure(figsize=(100,100))
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title(" predict randomforest")
        # plt.plot(test_pred_y,y_test,color=['r','b'])
        plt.plot(self.X_test.index,self.test_pred_y, label='predict')
        plt.plot(self.X_test.index,self.y_test, label='origin')
        plt.legend()
        plt.show()



if __name__ == '__main__':
    da = DA.dataAccess()
    block = blockfine()
    block.makeData(da)
    # block.trainTestonly(da)
    # X,y = block.makeMortordata()
    # rf_model = block.trainDataRF()
    # xgb_model = block.trainDataXGB()
    lgb_model = block.trainDataLGB()
    block.compareResult()
    # model_list = [('rf',rf_model),('xgb',xgb_model),('lgb',xgb_model)]
    #Final model 정의
    # model_final = LinearRegression()

    #Stacking 모델 정의
    # model = StackingRegressor(estimators=model_list,final_estimator=model_final,cv=5)
    # block.showResult()
    
    #Stacking 모델 학습
    # model.fit(block.X_train,block.y_train)
    # block.test_pred_y = model.predict(block.X_test)
    # block.showResult()

    #Stacking 모델 평가
    # print("valid",np.mean(np.abs(model.predict(X)-y)))
    
    