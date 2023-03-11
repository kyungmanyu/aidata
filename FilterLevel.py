import sklearn
import pandas as pd
import numpy as np
from sklearn import model_selection # cross-validation score를 가져오기 위함
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import BaggingClassifier # bagging
from sklearn.tree import DecisionTreeClassifier # 의사 결정 나무
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from collections import Counter # count
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb

""" 시각화 """
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter("ignore", UserWarning)

# hyperparameters
param_rf = {'n_estimators': [200],
              'oob_score': [True], # compute out of bag error
              'n_jobs':[-1], 
              'max_depth': [25, 50, 75] }

param_xgb = {"max_depth": [25,50,75],
              "min_child_weight" : [1,3,6],
              "n_estimators": [200], 
              "learning_rate": [0.05, 0.1,0.16]  }

param_lgb = { "objective":['regression'], # multiclass, regression
              "max_depth": [25,50, 75],
              "learning_rate" : [0.01,0.05,0.1],
              "num_leaves": [300,900,1200],
              "n_estimators": [200]     }

train_file = './train_data.csv'
test_file = './test_data.csv'


class AI_Filter_Level():
    def File_read(self, type):
        if type == 'train':
            filename = train_file
        else :
            filename = test_file
            
        df = pd.read_csv(filename)    
        df = df.fillna(0)            
        df = df.drop(df[df.suction == 0].index)
               
        return df
        
    def Data_set(self, df, type, split = 0):       
        # df.pressure = np.log(df.pressure)
        # df.rpm = np.log(df.rpm)
        print(df.head(5))        
        # df = df[df['mode'] == 2.0]
        
        # x 변수 선택
        # X = df[['mode', 'pressure', 'rpm', 'orifice']]
        X = df[['mode', 'pressure', 'rpm']]
        # X = df[['pressure', 'rpm']]
        
        if type == 'suction':
            Y = df.suction
        elif type == 'orifice':    
            Y = df.orifice
        elif type == 'level5':
            Y = self.df.level5
            
        # print(X[:5], Y[:5])
        
        if split != 0:  
            train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = split, random_state=0)
            return train_x, test_x, train_y, test_y
        else:
            return X, Y            
                      
    def Trainning_ensenmble(self, type):
        # 1) 모델 선언 & 2) 여러 모델들을 ensemble: randomforest
        if type == 'rf':
            model = RandomForestRegressor()
            parm = param_rf
        elif type == 'xgb':
            model = xgb.XGBRegressor()
            parm = param_xgb
        elif type == 'lgb':
            model = lgb.LGBMRegressor()     
            parm = param_lgb  
        
        # hyperparameter search
        # self.grid_search = GridSearchCV(model, param_grid=parm, cv=5, scoring='f1')
        self.grid_search = GridSearchCV(model, param_grid=parm, cv = 5,scoring="accuracy")
        
        self.grid_search.fit(self.train_x, self.train_y)        
        self.opt_model = self.grid_search.best_estimator_    
        
    def Trainning_MLP(self):        
        self.scaler = StandardScaler()
        # self.scaler = MinMaxScaler()
        self.scaler.fit(self.train_x)
        self.train_x = self.scaler.transform(self.train_x)
        
        self.reg_mlp = MLPRegressor(activation='relu', alpha=0.001, batch_size=10,
                    hidden_layer_sizes=(21,9,3), max_iter=10000, # 16,9,3
                    solver='adam', verbose = True, random_state = 2023)
        
        self.reg_mlp.fit(self.train_x, self.train_y)     
    
        train_loss_values = self.reg_mlp.loss_curve_        
        # print(self.reg_mlp.coefs_)
        
        plt.figure(figsize=(20,10))
        plt.plot(train_loss_values,label='Train Loss')

        plt.legend(fontsize=20)
        plt.title("Learning Curve of trained MLP Regressor", fontsize=18)
        plt.show()        
        
    def Testing(self, type):
        if type == 'ensemble':
            test_pred_y = self.opt_model.predict(self.test_x)
        elif type == 'MLP':
            self.test_x = self.scaler.transform(self.test_x)
            test_pred_y = self.reg_mlp.predict(self.test_x)
            
        mae = np.mean(np.abs(test_pred_y-self.test_y))
        mape = np.mean(np.abs(test_pred_y-self.test_y)/self.test_y)
        
        print('mae:',mae)
        print('mape:',mape)
        
        plt.figure(figsize = (16,8))
        ind = np.argsort(np.array(self.test_y))
        plt.plot(np.array(self.test_y)[ind],label='Real_Value')
        plt.plot(np.array(test_pred_y)[ind],label='Predict_Value')
        plt.legend()
        plt.title('suction power')
        plt.show()
        
    def Check_result_var(self):
        self.opt_model = self.grid_search.best_estimator_
        print(self.opt_model)        
        print(self.opt_model.feature_importances_)
        print(self.opt_model.oob_score_)
        print(self.grid_search.best_params_)      
        
    def Processing(self):
        df_train = self.File_read('train')
        self.train_x, self.train_y = self.Data_set(df_train, 'suction', 0)
        self.Trainning_ensenmble('rf') # model : rf, xgb , lgb
        # self.Trainning_MLP() 
        
        df_test = self.File_read('test')
        self.test_x, self.test_y = self.Data_set(df_test, 'suction', 0)
        self.Testing('ensemble')  
        # self.Testing('MLP')               
        
        print('finish')
        
oAI = AI_Filter_Level()
oAI.Processing()


