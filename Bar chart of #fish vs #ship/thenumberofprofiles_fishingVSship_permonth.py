# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 01:04:26 2020
This code is for ploting bar chart of fishing VS ship permonth from 2015 to 2018

@author: Mingchao
"""

import pandas as pd
from datetime import datetime,timedelta
import time
import numpy as np
import matplotlib.pyplot as plt
import os

def read_emolt_all(path='https://www.nefsc.noaa.gov/drifter/emolt.dat',endtime=datetime.now()):
#def read_emolt_all(path='/home/jmanning/Mingchao/result/emolt_dat_all.csv',endtime=datetime.now()):
    """read the emolt data and fix a standard format, the return the standard data"""
    while True:
        emolt_df=pd.read_csv(path,sep='\s+',names=['vessel_n','esn','month','day','Hours','minutes','fracyrday',\
                                          'lon','lat','dum1','dum2','depth','rangedepth','timerange','temp','stdtemp','year'])
        if int(emolt_df['year'][len(emolt_df)-2])==endtime.year:
            break
        else:
            print('check the web:https://www.nefsc.noaa.gov/drifter/emolt.dat.')
            time.sleep(600)
    return emolt_df

def read_emolt(start,end,path='https://www.nefsc.noaa.gov/drifter/emolt.dat'):
#def read_emolt(start,end,path='/home/jmanning/Mingchao/result/emolt_dat_all.csv'):
    '''the start and end is represent the sart time and end time, the format is datetime.datetime
    function: return the emolt data, the time during start time and end time.
    '''
    emolt_df=read_emolt_all(path)   #emolt_df means emolt data, this data from website 'https://www.nefsc.noaa.gov/drifter/emolt.dat',we should avoid the update time when we use this function
    #screen out the data of telemetry in interval
    valuable_emolt_df=pd.DataFrame(data=None,columns=['vessel_n','esn','time','lon','lat','depth','temp'])#use to save the data from start time to end time
    for i in range(len(emolt_df)):
        emolt_time_str=str(emolt_df['year'].iloc[i])+'-'+str(emolt_df['month'].iloc[i])+'-'+str(emolt_df['day'].iloc[i])+' '+\
                                         str(emolt_df['Hours'].iloc[i])+':'+str(emolt_df['minutes'].iloc[i])+':'+'00'# the string of observation time
        emolt_time_str=str(emolt_df['year'].iloc[i])+'-'+str(emolt_df['month'].iloc[i])+'-'+str(emolt_df['day'].iloc[i])+' '+str(emolt_df['Hours'].iloc[i])+':'+str(emolt_df['minutes'].iloc[i])+':'+'00'# the string of observation time
        emolt_time=datetime.strptime(emolt_time_str,'%Y-%m-%d %H:%M:%S') #chang the observation time format as datetime.datetime. it is convenient to compare with start time and end time.
        if start<emolt_time<=end:# grab the data that time between start time and end time
                valuable_emolt_df=valuable_emolt_df.append(pd.DataFrame(data=[[emolt_df['vessel_n'][i],emolt_df['esn'][i],emolt_time,emolt_df['lon'][i],\
                #valuable_emolt_df=valuable_emolt_df.append(pd.DataFrame(data=[[emolt_df['vessel_n'][i],emolt_df['esn'][i],datetime.strptime(emolt_time_str,'%Y-%m-%d %H:%M:%S'),emolt_df['lon'][i],\
                                   emolt_df['lat'][i],emolt_df['depth'][i],emolt_df['temp'][i]]],\
                                   columns=['vessel_n','esn','time','lon','lat','depth','temp']))
    
    return valuable_emolt_df

#Hard codes
fishing_path = 'E:\\Mingchao\\result\\emolt_dat_all(only emolt_no_telemetry).csv'
ship_path = 'E:\\Mingchao\\parameter\\original_ship_data.csv'
permonth_save = 'E:\\Mingchao\\result\\'
end_time = datetime.utcnow()
start_str = '2015-1-01 00:00:00'
start_time = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
#main
fishing_df = read_emolt(start=start_time, end=end_time, path='https://www.nefsc.noaa.gov/drifter/emolt.dat')
fishing_df.index = range(len(fishing_df))
ship_df = pd.read_csv(ship_path, skiprows=4100000, usecols=[0,1,2,3], names=['year','A','number','day'])
ship_df = ship_df.drop_duplicates(subset='day')
ship_df.index = range(len(ship_df))
#ship_list = np.unique(ship_df['day'])
ship_df['day'] = pd.to_datetime(ship_df['day'])
#ship_df['day'] = ship_df['day'].map(lambda x:datetime.strptime(x, '%Y-%m-%d %H:%M'))
#fper month
ship_Num=[]
fishing_Num=[]
#for i in range(6):    # 2014~2018,5 years
for i in range(4):
    #ship_Num.append([0]*12)    #12 months
    fishing_Num.append([0]*12)
for i in range(4):
    ship_Num.append([0]*12)

for i in range(len(ship_df['day'])):
    for j in range(4):
        if ship_df['day'][i].year == 2015+j:
            for q in range(12):
                if ship_df['day'][i].month == q+1:
                    ship_Num[j][q]+=1

for i in range(len(fishing_df)):
    #for j in range(6):
    for j in range(4):
        if fishing_df['time'][i].year == 2015+j:
            for q in range(12):
                if fishing_df['time'][i].month == q+1:
                    fishing_Num[j][q]+=1
#plot
width=0.2
#color=['blue','black','red','green','yellow','gray']
color=['blue','red','green','yellow']
fig=plt.figure()
ax1 = fig.add_subplot(1,2,1)
#for i in range(6):
for i in range(4):
    #ax1.bar(np.arange(1,13)+width*(i-3.75),fishing_Num[i],align="center",width=width,color=color[i],label=str(i+2015))
    ax1.bar(np.arange(1,13)+width*(i-1.75),fishing_Num[i],align="center",width=width,color=color[i],label=str(i+2015))
plt.legend(loc='best',fontsize = 'x-small')
plt.xlim([0,13]) 
#plt.ylim([0,600])
plt.ylim([0,400])
#plt.xticks(range(13),fontsize=10)
plt.xticks(np.arange(1,13))#12 monthes
plt.yticks(fontsize=10)
#plt.ylabel('Quantity',fontsize=16)
plt.ylabel('number of bottom temperature observations')
#plt.title('#Fishing profiles per month',fontsize=12)
plt.title('Fishing Vessels',fontsize=12)

ax2 = fig.add_subplot(1,2,2)
for i in range(4):
    #ax2.bar(np.arange(1,13)+width*(i-3.75),ship_Num[i],align="center",width=width,color=color[i] ,label=str(i+2015))
    ax2.bar(np.arange(1,13)+width*(i-1.75),ship_Num[i],align="center",width=width,color=color[i] ,label=str(i+2015))
plt.legend(loc='best',fontsize = 'x-small')
plt.xlim([0,13]) 
#plt.ylim([0,600])
plt.ylim([0,400])
#plt.xticks(range(13),fontsize=10)
plt.xticks(np.arange(1,13))
plt.setp(ax2.get_yticklabels(),visible=False)
fig.text(0.5,0.04,'Month',ha='center', va='center',fontsize=16)
#plt.title('#Ship profiles per month',fontsize=12)
plt.title('Research Vessels',fontsize=12)
if not os.path.exists(permonth_save):
        os.makedirs(permonth_save)
plt.savefig(permonth_save+'fishingVSship_profiles.png', dpi=200)
plt.show()





