# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 12:43:16 2020

@author: Mingchao
"""

import matplotlib.pyplot as plt
from datetime import datetime

#HARDCODES
path = 'E:\\Mingchao\\paper\\'
models_list = ['ESPRESSO (2006 to 2017)', 'DOPPIO (2017 to present)',\
               'GOMOFS (2015 to present)', 'FVCOM-GOM3 (1978 to present)']
models_color = ['g', 'r', 'm', 'b']
start_time = ['1978', '2006', '2015', '2017', '2020', '2021']
time_list = [datetime.strptime(start_time[0], '%Y'), datetime.strptime(start_time[1], '%Y'), datetime.strptime(start_time[2], '%Y'),\
             datetime.strptime(start_time[3], '%Y'), datetime.strptime(start_time[4], '%Y'), datetime.strptime(start_time[5], '%Y')]

#main
fig =plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot_date(time_list[0:-1], [2,2,2,2,2], '-', label=models_list[3])
ax.plot_date(time_list[2:-1], [3,3,3], '-', label=models_list[2])
ax.plot_date(time_list[3:-1], [4,4], '-', label=models_list[1])
ax.plot_date(time_list[1:4], [1,1,1], '-', label=models_list[0])
ax.axes.get_yaxis().set_visible(False)
plt.legend(loc='upper left')
plt.title('MODELS TIME RANGE')
plt.savefig(path+'MODELS_TIME_RANGE.png')
plt.show()