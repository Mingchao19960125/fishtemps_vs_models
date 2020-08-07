# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 17:47:14 2020

Plot time series and scatter of Interpolation methods

@author: Mingchao
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
import math
import datetime
from matplotlib import gridspec
from scipy import stats

def rmse(actual, predicted):
    '''
    get Root Mean Square Error
    actual is observation data,predicted is model's data
    '''
    mse = mean_squared_error(actual, predicted)
    rmse = math.sqrt(mse)
    return float(rmse)

def multipl(a,b):
    '''
    Input two sequences
    Output the sum of the products of the two sequences
    '''
    sumofab = 0.0
    for i in range(len(a)):
        temp = a[i]*b[i]
        sumofab += temp
    return sumofab

def corrcoef(x,y):
    '''
    Input two sequences
    Output Correlation coefficient of two series
    '''
    n = len(x)
    #sum
    sum1 = sum(x)
    sum2 = sum(y)
    #Find the sum of products
    sumofxy = multipl(x,y)
    #Find the sum of squares
    sumofx2 = sum([pow(i,2) for i in x])
    sumofy2 = sum([pow(j,2) for j in y])
    num = sumofxy-(float(sum1)*float(sum2)/n)
    #Calculate Pearson's correlation coefficient
    den = math.sqrt((sumofx2-float(sum1**2)/n)*(sumofy2-float(sum2**2)/n))
    return num/den
###################### Hardcodes #########################
path = 'E:\\Mingchao\\paper\\interpo_comparison_data_final.csv'
path_save = 'E:\\Mingchao\\paper'
seasons = ['WINTER', 'SPRING', 'SUMMER', 'FALL']
start_time = datetime.datetime(2018,8,1,00,00,00) # 8 9
end_time = datetime.datetime(2018,9,1,00,00,00)
###################### Main ##############################
comparison_df = pd.read_csv(path,index_col=0)
comparison_df = comparison_df.dropna()
comparison_df_scatter = comparison_df
comparison_df.index = pd.to_datetime(comparison_df['time'])
for i in comparison_df.index:
    if not start_time<i<end_time:
        comparison_df['time'][i]=np.nan
comparison_df = comparison_df.dropna(subset=['time'])
comparison_df['month'] = comparison_df.index.month
season = ((comparison_df.month % 12 + 3) // 3).map({1:'WINTER', 2: 'SPRING', 3:'SUMMER', 4:'FALL'})
comparison_df['season'] = season.values 
fig = plt.figure()#figsize=(11.69,8.27)
gsc = gridspec.GridSpec(2,2,wspace=0.04, hspace=0.04,width_ratios=[1, 1])
ax1 = fig.add_subplot(gsc[0,0])
ax2 = fig.add_subplot(gsc[0,1])
ax3 = fig.add_subplot(gsc[1,0])
ax4 = fig.add_subplot(gsc[1,1])
ax1.axes.get_xaxis().set_visible(False)
ax2.axes.get_xaxis().set_visible(False)
ax2.axes.get_yaxis().set_visible(False)
ax4.axes.get_yaxis().set_visible(False)
observation_list = list(comparison_df_scatter['observation_T'])#.loc[comparison_df['season']==seasons[i]])
fitting_list = list(comparison_df_scatter['fitting'])#.loc[comparison_df['season']==seasons[i]])
nofitting_list = list(comparison_df_scatter['no_fitting'])#.loc[comparison_df['season']==seasons[i]])
size = min(fig.get_size_inches())
MAX_T=max(comparison_df['observation_T'].values)
MIN_T=min(comparison_df['observation_T'].values)
diff_temp=MAX_T-MIN_T
if diff_temp==0:
    textend_lim=0.1
else:
    textend_lim=diff_temp/8.0
ax1.plot(comparison_df.index,comparison_df['observation_T'].values,color='r',label='Observation')
ax1.plot(comparison_df.index,comparison_df['fitting'].values,color='blue',alpha=0.8,label='Bilinear Interpolation')
ax1.set_ylabel('Celsius',fontsize=3.5*size)
ax1.set_ylim(MIN_T-textend_lim,MAX_T+textend_lim)
ax1.set_title('Bilinear Interpolation')
ax1.legend(prop={'size':2* size})
ax1.tick_params(labelsize=2.5*size)
#plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))
#plt.gcf().autofmt_xdate()
ax2.plot(comparison_df.index,comparison_df['observation_T'].values,color='r',label='Observation')
ax2.plot(comparison_df.index,comparison_df['no_fitting'].values,color='green',alpha=0.8,label='Nearest Interpolation')
ax2.set_title('Nearest Interpolation')
ax2.legend(prop={'size':2* size})
ax2.tick_params(labelsize=2.5*size)

############# scatter #################
a = ax3.hist2d(x=comparison_df_scatter['fitting'].values, y=comparison_df_scatter['observation_T'].values, bins=40, alpha=0.9, cmin=1)
ax3.set_xlim(0,25)
ax3.set_ylim(0,25)
ax3.plot([0,25],[0,25],color='red',linewidth=1,alpha=0.8)
ax3.set_ylabel('Observation',fontsize=3.5*size)
b = ax4.hist2d(x=comparison_df_scatter['no_fitting'].values, y=comparison_df_scatter['observation_T'].values, bins=40, alpha=0.9, cmin=1)
ax4.set_xlim(0,25)
ax4.set_ylim(0,25)
ax4.plot([0,25],[0,25],color='red',linewidth=1,alpha=0.8)
plt.colorbar(a[3], ax=[ax3, ax4])
print('observation mean:%.2f'%np.mean(observation_list),'fitting mean:%.2f'%np.mean(fitting_list),'no_fitting mean:%.2f'%np.mean(nofitting_list))
print('observation std:%.2f'%np.std(observation_list),'fitting std:%.2f'%np.std(fitting_list),'no_fitting std:%.2f'%np.std(nofitting_list))
print('observation var:%.2f'%np.var(observation_list),'fitting var:%.2f'%np.var(fitting_list),'no_fitting var:%.2f'%np.var(nofitting_list))
print('fitting bias:%.2f'%(np.mean(observation_list)-np.mean(fitting_list)),'no_fitting bias:%.2f'%(np.mean(observation_list)-np.var(nofitting_list)))
print('fitting Correlation coefficient:%.2f'%corrcoef(fitting_list,observation_list),'no_fitting Correlation coefficient:%.2f'%corrcoef(nofitting_list,observation_list))
print('fitting RMSE:%.2f'%rmse(observation_list,fitting_list),'no_fitting RMSE:%.2f'%rmse(observation_list,nofitting_list))
fig.suptitle('Comparison of Interpolation methods',fontsize=16)
plt.savefig(os.path.join(path_save,'interpolation_comparison_figure.png'),dpi=300)
