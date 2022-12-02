import os
import numpy as np
import pandas as pd

class sharpe:
    def __init__(self,path):
        '''
        class function for calculating sharpe ratio
        load end_price of S&P data

        '''
        self.df=pd.read_csv(path) # endprice for each time
        self.fields=self.df.columns.tolist()[1:]
        self.mat=list()

        for field in self.fields:
            self.mat.append(self.cal_return_summary(self.df[field]))

        self.cov_mat=np.cov(np.array(self.mat))
        np.fill_diagonal(self.cov_mat,1.0)

        self.expected_return=list()

        for li in self.mat:
            self.expected_return.append(np.mean(li))

        self.exp_return=np.array(self.expected_return)

    def __len__(self):
        return len(self.fields)
    
    def cal_return_summary(self,li):
        tmp_li=list()
        for i in range(len(li)-52,len(li)-1): #len(li)-1
            x=(li[i+1]/li[i])-1
            tmp_li.append(x)
        return(tmp_li)

    def __call__(self):
        return self.exp_return, self.cov_mat
