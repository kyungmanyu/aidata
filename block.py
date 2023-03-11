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


        simDays = 60
        symbol = 'ETH/USDT'
        unitTime = 1
        da.load_history_data_from_binance(simDays, unitTime ,symbol)
        data = da.simulationDF
        data['datetime'] =  pd.to_datetime(da.simulationDF['datetime'], unit='ms') + datetime.timedelta(hours=9)
        data.set_index('datetime',inplace=True)
        print('data aaaa',data)
        dataX = data.drop(['close'], axis=1)
        
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
        print('datay',label)
        print('datay len',len(label))
        print('datay sum',sum(label))
        y = label
        
        print(X)
        
        
        
        # data = pd.read_csv('motordata.csv')

        # X = data.drop('suction', axis=1)
        # y = data['suction']

        # print(y)

        random_state = 2023
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=2/10, random_state=random_state,shuffle=False)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train, self.y_train, test_size=2/8,
                                                            random_state=random_state,shuffle=False)
        max_depths = list(range(1, 10)) + [None]
        print(max_depths)
        
    def makeMortordata(self):
        data = pd.read_csv('motordata.csv')

        X = data.drop('suction', axis=1)
        y = data['suction']

       

        random_state = 2023
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=2/10, random_state=random_state,shuffle=False)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train, self.y_train, test_size=2/8,
                                                            random_state=random_state,shuffle=False)
        max_depths = list(range(1, 10)) + [None]
        print(max_depths)
        
        return X,y



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
        for i in range(len(self.test_pred_y)):
            if(self.test_pred_y[i] == self.y_test[i]):
                ans.append(1)
            x.append(i)
            
        accuracy = sum(ans)/len(self.y_test) * 100
        mae = np.mean(np.abs(self.test_pred_y - self.y_test))
        print('comlgb acc',accuracy)
        now = datetime.datetime.now()
        print('endtime',str(now))
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
        
        x = []
        ans = []
        for i in range(len(self.test_pred_y)):
            if(self.test_pred_y[i] == self.y_test[i]):
                ans.append(1)
            x.append(i)
            
        accuracy = sum(ans)/len(self.y_test) * 100
        mae = np.mean(np.abs(self.test_pred_y - self.y_test))
        print('lgb acc',accuracy)
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
        
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title(" predict randomforest")
        # plt.plot(test_pred_y,y_test,color=['r','b'])
        plt.plot(x,self.test_pred_y, label='predict')
        plt.plot(x,self.y_test, label='origin')
        plt.legend()
        plt.show()



if __name__ == '__main__':
    da = DA.dataAccess()
    block = blockfine()
    block.makeData(da)
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
    
    