# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 14:15:04 2021

@author: ghazal
"""

#%%
"""
This code needs to be ran 17 times for each warming level between 0 to 8
In this code @ is representing the saving folder for each warming level and needs to be substitute with numbers between 0 to 8 with .5 intervals
"""

"""
This code needs to be ran in batch format with array elements 0-100, as we have 100 realisations for each warming level
"""
#%%
import Func_Lib as func
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from statsmodels.tsa.arima_model import ARMA
plt.style.use('seaborn-white')
import time
from scipy.stats import pearson3
import os
from glob import glob
import sys
#%%

filename=glob('/cluster/tufts/lamontagnelab/gshabe01/deg_tempC.@/*.csv')

print(filename)
dataframes=[pd.read_csv(f) for f in filename]

#%% importing the data

# importing historical Ar model parameters, lag covariances,transformation bias correction, and historical residuals
Beta=np.load('/cluster/tufts/lamontagnelab/gshabe01/Beta_livneh.npy')
beta_cov= np.load('/cluster/tufts/lamontagnelab/gshabe01/BetaCovariance_livneh.npy')
residual=np.load('/cluster/tufts/lamontagnelab/gshabe01/Residual_livneh.npy')
initial=np.load('/cluster/tufts/lamontagnelab/gshabe01/initial_observed_livneh.npy')
livneh=pd.read_csv('/cluster/tufts/lamontagnelab/gshabe01/livneh.csv')
BCF=np.load('/cluster/tufts/lamontagnelab/gshabe01/BCF.npy')
#%%
livneh['date']=pd.to_datetime(livneh['date'])

livneh['Ar_res']=residual
data=dataframes[int(sys.argv[1])]
data['date']=pd.to_datetime(data['Date'])
data=pd.merge(data,livneh,how='right',on=['date']).dropna()
data['month']=pd.DatetimeIndex(data['Date']).month
data=data.reset_index()
print('merged')
#%%
n=100

m=100
Beta=np.random.multivariate_normal(Beta,beta_cov,m)


high=np.array(data['Ar_res'].loc[data['Qmodel']> 10])
low=np.array(data['Ar_res'].loc[data['Qmodel']<= 10])
E=func.conditional_bootstrap(data[' 2'],low,high,n*m)

lamda=initial
T=len(data)

l=np.zeros((T,m*n))

for i in range(0,m*n):
    
    l[0:4,i]=lamda[0:4]


for i in range(0,m):    
    B=Beta[i,:]
    for t in range(4,T):
        l[t,i*n:((i+1)*n)]=B[0]+B[1]*l[t-1,i*n:((i+1)*n)]+B[2]*l[t-2,i*n:((i+1)*n)]+B[3]*l[t-3,i*n:((i+1)*n)]+E[t,i*n:((i+1)*n)]

        
#%%
Q=np.array(data[' 2'])/np.exp(l.T)



Q=Q*BCF

#%%

name=sys.argv[1]
print('Qruned_%s' %name)

R=np.load('/cluster/tufts/lamontagnelab/gshabe01/QM.npy')

Q_frame=pd.DataFrame(Q.T, index= data['date'])
for i in range (0,Q.shape[0]):
    Q_frame[i]=(Q_frame[i].sort_values()*R).sort_index()

Q=np.array(Q_frame).T

from numpy import savetxt

savetxt('/cluster/tufts/lamontagnelab/gshabe01/outfiles/deg@/Q_%s.csv' %name , Q, delimiter=',')
#%%

annualmax=func.annualmax(Q,data,m*n)

data=data.set_index('date')
annualmax['PRMS']=np.sort(data[' 2'].resample('Y').max())[::-1]
data=data.reset_index()

annual_max=np.zeros((len(annualmax),2))
annual_max[:,1]=np.array(annualmax)[:,0:n*m].max(axis=1)
annual_max[:,0]=np.array(annualmax)[:,0:n*m].min(axis=1)
#%%
print('annualmax_%s' %name)
np.save('/cluster/tufts/lamontagnelab/gshabe01/annualmax/deg@/annualmax_%s.npy' %name , annual_max)
np.save('/cluster/tufts/lamontagnelab/gshabe01/outfiles/deg@/annualmaxbig_%s.npy' %name , annualmax)
#%%
low7=func.low7day(Q,data,n*m)

data=data.set_index('date')
low7['PRMS']=np.sort(data[' 2'].rolling(7).mean().resample('Y').min())[:: -1]
data=data.reset_index()

day7low=np.zeros((len(low7),2))
day7low[:,1]=np.array(low7)[:,0:n*m].max(axis=1)
day7low[:,0]=np.array(low7)[:,0:n*m].min(axis=1)

#%%
print('day7low_%s' %name)
np.save('/cluster/tufts/lamontagnelab/gshabe01/7daylow/deg@/day7low_%s.npy' %name , day7low)
np.save('/cluster/tufts/lamontagnelab/gshabe01/outfiles/deg@/day7lowbig_%s.npy' %name , low7)


ann_max=np.array(annualmax)

    
flood2Y0=np.exp(np.log(pd.DataFrame(ann_max[:,0:10000])).mean()+np.log(pd.DataFrame((ann_max[:,0:10000]))).std()*abs(pearson3.ppf(0.5, np.log(pd.DataFrame(ann_max[:,0:10000])).skew())))
    
    
flood10Y0=np.exp(np.log(pd.DataFrame(ann_max[:,0:10000])).mean()+np.log(pd.DataFrame((ann_max[:,0:10000]))).std()*abs(pearson3.ppf(0.9, np.log(pd.DataFrame(ann_max[:,0:10000])).skew())))

flood50Y0=np.exp(np.log(pd.DataFrame(ann_max[:,0:10000])).mean()+np.log(pd.DataFrame((ann_max[:,0:10000]))).std()*abs(pearson3.ppf(0.98, np.log(pd.DataFrame(ann_max[:,0:10000])).skew())))
    

flood100Y0=np.exp(np.log(pd.DataFrame(ann_max[:,0:10000])).mean()+np.log(pd.DataFrame((ann_max[:,0:10000]))).std()*abs(pearson3.ppf(0.99, np.log(pd.DataFrame(ann_max[:,0:10000])).skew())))
#np.exp(np.log(pd.DataFrame(ann_max[:,0:10000])).mean()+np.log(pd.DataFrame((ann_max[:,0:10000]))).std()*abs(pearson3.ppf(0.99, np.log(pd.DataFrame(ann_max[:,0:10000])).skew())))

flood500Y0=np.exp(np.log(pd.DataFrame(ann_max[:,0:10000])).mean()+np.log(pd.DataFrame((ann_max[:,0:10000]))).std()*abs(pearson3.ppf(0.998, np.log(pd.DataFrame(ann_max[:,0:10000])).skew())))
    
flood1000Y0=np.exp(np.log(pd.DataFrame(ann_max[:,0:10000])).mean()+np.log(pd.DataFrame((ann_max[:,0:10000]))).std()*abs(pearson3.ppf(0.999, np.log(pd.DataFrame(ann_max[:,0:10000])).skew())))
    
 

 


savetxt('/cluster/tufts/lamontagnelab/gshabe01/outfiles/deg@/flood2_%s.csv' %name , flood2Y0, delimiter=',')
savetxt('/cluster/tufts/lamontagnelab/gshabe01/outfiles/deg@/flood10_%s.csv' %name , flood10Y0, delimiter=',')
savetxt('/cluster/tufts/lamontagnelab/gshabe01/outfiles/deg@/flood50_%s.csv' %name , flood50Y0, delimiter=',')
savetxt('/cluster/tufts/lamontagnelab/gshabe01/outfiles/deg@/flood100_%s.csv' %name , flood100Y0, delimiter=',')
savetxt('/cluster/tufts/lamontagnelab/gshabe01/outfiles/deg@/flood500_%s.csv' %name , flood500Y0, delimiter=',')
savetxt('/cluster/tufts/lamontagnelab/gshabe01/outfiles/deg@/flood1000_%s.csv' %name , flood1000Y0, delimiter=',')


np.save('/cluster/tufts/lamontagnelab/gshabe01/outfiles/deg@/medianflood2_%s.npy' %name , flood2Y0.median())
np.save('/cluster/tufts/lamontagnelab/gshabe01/outfiles/deg@/medianflood10_%s.npy' %name , flood10Y0.median())
np.save('/cluster/tufts/lamontagnelab/gshabe01/outfiles/deg@/medianflood50_%s.npy' %name , flood50Y0.median())
np.save('/cluster/tufts/lamontagnelab/gshabe01/outfiles/deg@/medianflood100_%s.npy' %name , flood100Y0.median())
np.save('/cluster/tufts/lamontagnelab/gshabe01/outfiles/deg@/medianflood500_%s.npy' %name , flood500Y0.median())
np.save('/cluster/tufts/lamontagnelab/gshabe01/outfiles/deg@/medianflood1000_%s.npy' %name , flood1000Y0.median())

np.save('/cluster/tufts/lamontagnelab/gshabe01/outfiles/deg@/daily_mean_%s.npy' %name , np.median(Q,axis=1))
