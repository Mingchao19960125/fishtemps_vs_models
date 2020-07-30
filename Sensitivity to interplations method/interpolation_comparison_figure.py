# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 17:47:14 2020

@author: Mingchao
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import gridspec
import math

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
path = 'E:\\Mingchao\\paper\\interpo_comparison_data.csv'
path_save = 'E:\\Mingchao\\paper'
seasons = ['WINTER', 'SPRING', 'SUMMER', 'FALL']
###################### Main ##############################
comparison_df = pd.read_csv(path,index_col=0)
comparison_df = comparison_df.dropna()
comparison_df.index = pd.to_datetime(comparison_df['time'])
comparison_df['month'] = comparison_df.index.month
season = ((comparison_df.month % 12 + 3) // 3).map({1:'WINTER', 2: 'SPRING', 3:'SUMMER', 4:'FALL'})
comparison_df['season'] = season.values 
fig = plt.figure(figsize=(11.69,8.27))#figsize=(11.69,8.27)
size = min(fig.get_size_inches())
gsc = gridspec.GridSpec(2,2,wspace=0.02, hspace=0.09,width_ratios=[1, 1])
ax1 = fig.add_subplot(gsc[0,0])
ax1.axes.get_xaxis().set_visible(False)
ax2 = fig.add_subplot(gsc[0,1])
ax2.axes.get_xaxis().set_visible(False)
ax2.axes.get_yaxis().set_visible(False)
ax3 = fig.add_subplot(gsc[1,0])
ax3.axes.get_xaxis().set_visible(False)
ax4 = fig.add_subplot(gsc[1,1])
ax4.axes.get_yaxis().set_visible(False)
ax4.axes.get_xaxis().set_visible(False)
ax_list = [ax1,ax2,ax3,ax4]
MAX_T=max(comparison_df['observation_T'].values)
MIN_T=min(comparison_df['observation_T'].values)
diff_temp=MAX_T-MIN_T
if diff_temp==0:
    textend_lim=0.1
else:
    textend_lim=diff_temp/8.0
for i in range(4):
    time_list = list(comparison_df.index[comparison_df['season']==seasons[i]])
    observation_list = list(comparison_df['observation_T'].loc[comparison_df['season']==seasons[i]])
    fitting_list = list(comparison_df['fitting'].loc[comparison_df['season']==seasons[i]])
    nofitting_list = list(comparison_df['no_fitting'].loc[comparison_df['season']==seasons[i]])
    ax_list[i].plot(time_list,observation_list,color='r',label='Observation')
    ax_list[i].plot(time_list,fitting_list,linewidth=2,color='blue',label='Bilinear Interpolation')
    ax_list[i].plot(time_list,nofitting_list,linewidth=2,color='green',label='Nearest Interpolation')
    ax_list[i].set_ylabel('Celsius',fontsize=2*size)
    ax_list[i].set_ylim(MIN_T-textend_lim,MAX_T+textend_lim)
    ax_list[i].set_title(seasons[i])
    ax_list[i].legend()
    print(seasons[i]+':')
    print('observation mean:%f'%np.mean(observation_list),'fitting mean:%f'%np.mean(fitting_list),'no_fitting mean:%f'%np.mean(nofitting_list))
    print('observation std:%f'%np.std(observation_list,ddof=1),'fitting std:%f'%np.std(fitting_list,ddof=1),'no_fitting std:%f'%np.std(nofitting_list,ddof=1))
    print('observation var:%f'%np.var(observation_list),'fitting var:%f'%np.var(fitting_list),'no_fitting var:%f'%np.var(nofitting_list))
    print('fitting mean:%f'%corrcoef(fitting_list,observation_list),'no_fitting mean:%f'%corrcoef(nofitting_list,observation_list))
fig.suptitle('Seasonal comparison of Interpolation methods',fontsize=20)
plt.savefig(os.path.join(path_save,'interpolation_comparison_figure.png'))
