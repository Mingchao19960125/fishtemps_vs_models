# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:34:52 2020

This function get time series plotting of fixed vessel's raw data.

Input:
   telemetry_status.csv,such as Excalibur_hours.csv
Output:
   a figure named such as Excalibur_raw_data_time_series_plot.png 

@author: Mingchao
"""

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import zlconversions as zl
import matplotlib.dates as dates
from datetime import datetime,timedelta
import matplotlib.ticker as ticker

######################### Hardcodes ################################
Hours_save = 'E:\\Mingchao\\result\\Hours_data\\'
telemetry_status_path = 'E:\\Mingchao\\parameter\\telemetry_status.csv'
start_time = datetime(2019,8,20,22,30,00)#temp_df.index[0]
end_time = start_time+timedelta(weeks=9)
model_list = ['DOPPIO_T', 'GOMOFS_T', 'FVCOM_T']

def get_Fixed_vessel(telemetry_status_path):
    '''get Fixed vessel's name'''
    telemetrystatus_df = read_telemetrystatus(telemetry_status_path)
    Fixed_vessel_list = []
    for index in range(len(telemetrystatus_df)):
        if telemetrystatus_df['Fixed vs. Mobile'][index]=='Fixed':
            Fixed_vessel_list.append(telemetrystatus_df['Boat'][index])
    return Fixed_vessel_list

def read_telemetrystatus(path_name):
    """read the telementry_status, then return the useful data"""
    data=pd.read_csv(path_name)
    #find the data lines number in the file('telemetry_status.csv')
    for i in range(len(data['vessel (use underscores)'])):
        if data['vessel (use underscores)'].isnull()[i]:
            data_line_number=i
            break
    #read the data about "telemetry_status.csv"
    telemetrystatus_df=pd.read_csv(path_name,nrows=data_line_number)
    as_list=telemetrystatus_df.columns.tolist()
    idex=as_list.index('vessel (use underscores)')
    as_list[idex]='Boat'
    telemetrystatus_df.columns=as_list
    for i in range(len(telemetrystatus_df)):
        telemetrystatus_df['Boat'][i]=telemetrystatus_df['Boat'][i].replace("'","")
        if not telemetrystatus_df['Lowell-SN'].isnull()[i]:
            telemetrystatus_df['Lowell-SN'][i]=telemetrystatus_df['Lowell-SN'][i].replace('，',',')
        if not telemetrystatus_df['logger_change'].isnull()[i]:
            telemetrystatus_df['logger_change'][i]=telemetrystatus_df['logger_change'][i].replace('，',',')
    return telemetrystatus_df

def plot(name,Hours_save,temp_df,start_time,end_time,dpi=300):
        #name = vessel_lists.split('\\')[5].split('_hours')[0]
        for time_index in temp_df.index:
            if not start_time<=temp_df['new_time'][time_index]<=end_time:
                temp_df = temp_df.drop(time_index)
        temp_df.dropna(subset=['lat','lon'],inplace=True)
        if len(temp_df)!=0:#make sure the dataframe have data in timerange
            MIN_T=min(min(temp_df['temp']), min(temp_df['DOPPIO_T']),\
                      min(temp_df['FVCOM_T']))
            MAX_T=max(max(temp_df['temp']), max(temp_df['DOPPIO_T']),\
                      max(temp_df['FVCOM_T']))
            diff_temp=MAX_T-MIN_T
            if diff_temp==0:
                textend_lim=0.1
            else:
                textend_lim=diff_temp/8.0
            fig=plt.figure(figsize=[11.69,8.27])
            size=min(fig.get_size_inches())        
            fig.suptitle(name,fontsize=3*size, fontweight='bold')
            ax1=fig.add_subplot()
            ax1.set_title(str(temp_df.index[0].split(' ')[0])+' to '+str(temp_df.index[-1].split(' ')[0]), fontsize=2*size)
            vessel_index_list=[]#for storing the index that has big difference between last index
            for vessel_index in range(len(temp_df)-1):
                if "{:.4f}".format(temp_df['lon'][vessel_index]) != "{:.4f}".format(temp_df['lon'][vessel_index+1]):
                    vessel_index_list.append(vessel_index+1)
            if len(vessel_index_list) == 0:#the fishing period only once
                 ax1.plot(temp_df['new_time'][:], temp_df['temp'][:], color='b', linewidth=3, label='Observed')
                 ax1.plot(temp_df['new_time'][:], temp_df['DOPPIO_T'][:], color='r', linewidth=3, label='DOPPIO')
                 ax1.plot(temp_df['new_time'][:], temp_df['FVCOM_T'][:], color='y', linewidth=3, label='FVCOM')
                 ax1.plot(temp_df['new_time'][:], temp_df['GOMOFS_T'][:], color='black', linewidth=3, label='GOMOFS')
                 ax1.plot(temp_df['new_time'][:], temp_df['CLIM_T'][:], color='g', linewidth=3, label='Climatology')
            elif len(vessel_index_list) == 1:#the fishing period only twice
                 ax1.plot(temp_df['new_time'][:vessel_index_list[0]], temp_df['temp'][:vessel_index_list[0]], color='b', linewidth=3, label='Observed')
                 ax1.plot(temp_df['new_time'][:vessel_index_list[0]], temp_df['DOPPIO_T'][:vessel_index_list[0]], color='r', linewidth=3, label='DOPPIO')
                 ax1.plot(temp_df['new_time'][:vessel_index_list[0]], temp_df['FVCOM_T'][:vessel_index_list[0]], color='y', linewidth=3, label='FVCOM')
                 ax1.plot(temp_df['new_time'][:vessel_index_list[0]], temp_df['GOMOFS_T'][:vessel_index_list[0]], color='black', linewidth=3, label='GOMOFS')
                 ax1.plot(temp_df['new_time'][:vessel_index_list[0]], temp_df['CLIM_T'][:vessel_index_list[0]], color='g', linewidth=3, label='Climatology')
                 ax1.plot(temp_df['new_time'][vessel_index_list[0]:], temp_df['temp'][vessel_index_list[0]:], color='b', linewidth=3)
                 ax1.plot(temp_df['new_time'][vessel_index_list[0]:], temp_df['DOPPIO_T'][vessel_index_list[0]:], color='r', linewidth=3)
                 ax1.plot(temp_df['new_time'][vessel_index_list[0]:], temp_df['FVCOM_T'][vessel_index_list[0]:], color='y', linewidth=3)
                 ax1.plot(temp_df['new_time'][vessel_index_list[0]:], temp_df['GOMOFS_T'][vessel_index_list[0]:], color='black', linewidth=3)
                 ax1.plot(temp_df['new_time'][vessel_index_list[0]:], temp_df['CLIM_T'][vessel_index_list[0]:], color='g', linewidth=3)
            else:
                for number in range(len(vessel_index_list)):
                    if number == 0:#the first fishing period
                        ax1.plot(temp_df['new_time'][:vessel_index_list[0]], temp_df['temp'][:vessel_index_list[0]], color='b', linewidth=3, label='Observed')
                        ax1.plot(temp_df['new_time'][:vessel_index_list[0]], temp_df['DOPPIO_T'][:vessel_index_list[0]], color='r', linewidth=3, label='DOPPIO')
                        ax1.plot(temp_df['new_time'][:vessel_index_list[0]], temp_df['FVCOM_T'][:vessel_index_list[0]], color='y', linewidth=3, label='FVCOM')
                        ax1.plot(temp_df['new_time'][:vessel_index_list[0]], temp_df['GOMOFS_T'][:vessel_index_list[0]], color='black', linewidth=3, label='GOMOFS')
                        ax1.plot(temp_df['new_time'][:vessel_index_list[0]], temp_df['CLIM_T'][:vessel_index_list[0]], color='g', linewidth=3, label='Climatology')
                    elif number == len(vessel_index_list)-1:#the last fishing period
                        ax1.plot(temp_df['new_time'][vessel_index_list[number]:], temp_df['temp'][vessel_index_list[number]:], color='b', linewidth=3)
                        ax1.plot(temp_df['new_time'][vessel_index_list[number]:], temp_df['DOPPIO_T'][vessel_index_list[number]:], color='r', linewidth=3)
                        ax1.plot(temp_df['new_time'][vessel_index_list[number]:], temp_df['FVCOM_T'][vessel_index_list[number]:], color='y', linewidth=3)
                        ax1.plot(temp_df['new_time'][vessel_index_list[number]:], temp_df['GOMOFS_T'][vessel_index_list[number]:], color='black', linewidth=3)
                        ax1.plot(temp_df['new_time'][vessel_index_list[number]:], temp_df['CLIM_T'][vessel_index_list[number]:], color='g', linewidth=3)
                    else:
                        ax1.plot(temp_df['new_time'][vessel_index_list[number]:vessel_index_list[number+1]], temp_df['temp'][vessel_index_list[number]:vessel_index_list[number+1]], color='b', linewidth=3)
                        ax1.plot(temp_df['new_time'][vessel_index_list[number]:vessel_index_list[number+1]], temp_df['DOPPIO_T'][vessel_index_list[number]:vessel_index_list[number+1]], color='r', linewidth=3)
                        ax1.plot(temp_df['new_time'][vessel_index_list[number]:vessel_index_list[number+1]], temp_df['FVCOM_T'][vessel_index_list[number]:vessel_index_list[number+1]], color='y', linewidth=3)
                        ax1.plot(temp_df['new_time'][vessel_index_list[number]:vessel_index_list[number+1]], temp_df['GOMOFS_T'][vessel_index_list[number]:vessel_index_list[number+1]], color='black', linewidth=3)
                        ax1.plot(temp_df['new_time'][vessel_index_list[number]:vessel_index_list[number+1]], temp_df['CLIM_T'][vessel_index_list[number]:vessel_index_list[number+1]], color='g', linewidth=3)
            ax12=ax1.twinx()
            ax12.set_ylabel('Fahrenheit',fontsize=2*size)
            #conversing the Celius to Fahrenheit
            ax12.set_ylim((MAX_T+textend_lim)*1.8+32,(MIN_T-textend_lim)*1.8+32)
            ax12.invert_yaxis()
            ax1.set_ylabel('Celsius',fontsize=2*size)
            ax1.set_ylim()#MIN_T-textend_lim,MAX_T+textend_lim
            ax1.legend(prop={'size':2* size})
            ax1.tick_params(labelsize=1.5*size)
            plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))
            #fig.autofmt_xdate() 
            plt.gcf().autofmt_xdate()
        else:
            print(name+' does not have valuable data in this time range!')
#        if not os.path.exists(os.path.join(Hours_save+vessel_lists[i].split('/')[6].split('_hours')[0]+'/')):
#            os.makedirs(Hours_save+vessel_lists[i].split('/')[6].split('_hours')[0]+'/')
#        plt.savefig(os.path.join(Hours_save+vessel_lists[i].split('/')[6].split('_hours')[0]+'/')+vessel_lists[i].split('/')[6].split('_hours')[0]+'_hours.ps',dpi=dpi,orientation='landscape')
        plt.savefig(Hours_save+name+'\\'+name+'_raw_data_time_series_plot.png',dpi=dpi,orientation='portait')
        plt.show()
        
####################### main ######################
hours_lists=zl.list_all_files(Hours_save)
vessel_lists=[]#store the path of every vessel's file
Fixed_vessel_list = get_Fixed_vessel(telemetry_status_path)
#Loop every vessel's file and Plot
for file in hours_lists:
   if file[len(file)-9:]=='hours.csv':
     vessel_lists.append(file)                
for i in range(len(vessel_lists)):
    name = vessel_lists[i].split('\\')[5].split('_hours')[0]
    if name in Fixed_vessel_list:
        vessel_df=pd.read_csv(vessel_lists[i],index_col=0)
        vessel_df['new_time'] = vessel_df.index
        vessel_df['new_time'] = pd.to_datetime(vessel_df['new_time'])#change the time style to datetime
        vessel_df.drop_duplicates(subset=['new_time'],inplace=True)
        plot(name=name,Hours_save=Hours_save,temp_df=vessel_df,start_time=start_time,end_time=end_time,dpi=300)
    else:
        continue