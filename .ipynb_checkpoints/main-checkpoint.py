import os
import pandas as pd
import numpy as np
from util.model import sharpe

filename=input("Enter filename.csv: ")

assert filename[-3:] =='csv', 'expansion should be csv'

data=os.path.join(os.getcwd(),'data',filename)
sharpe_class=sharpe(data)

exp_return,cov_matrix=sharpe_class()

weights = input("Enter {} weights with space: ".format(len(sharpe_class)))

if weights == 'default':
    weights=np.full((len(sharpe_class),),1/(len(sharpe_class)))
else:
    weights=[float(x) for x in weights.split(' ')]

def main(exp_return,cov_matrix,weights):
    '''
    riskfree rate is averageo of annual rate
    '''

    std=np.matmul(np.matmul(weights,cov_matrix),weights)
    riskfree=0.0001

    expected_return=np.dot(exp_return,weights)

    sharperatio=(expected_return-riskfree)/std
    print('This is sharperatio')
    print(sharperatio)

if __name__=="__main__":
    main(exp_return,cov_matrix,weights)
