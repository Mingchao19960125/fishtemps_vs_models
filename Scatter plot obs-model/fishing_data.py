# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 04:39:09 2020

@author: Mingchao
"""
from datetime import datetime
import os
import conda
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
import pandas as pd
import json
#Hardcodes
path = 'E:\\programe\\aqmain\\dictionary\\dictionary.json'
path_save = 'E:\\Mingchao\\paper\\fishing_data\\'
mindepth = 10
start_time_str = '2015-01-01 00:00:00'
end_time_str = '2020-01-01 00:00:00'
start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')

def check_time(df,time_header,start_time,end_time):
    '''keep the type of time is datetime
    input start time and end time, return the data between start time and end time'''
    for i in range(len(df)):
        if type(df[time_header][i])==str:
            df[time_header][i]=datetime.strptime(df[time_header][i],'%Y-%m-%d %H:%M:%S')
        if start_time<=df[time_header][i]<=end_time:
            continue
        else:
            df=df.drop(i)
    df=df.dropna()
    df.index=range(len(df))
    return df


def check_depth(df,mindepth):
    '''keep the depth is out of mindepth and correct the format of depth for example:-20 '''
    if len(df)>0:  
        for i in df.index:
            if abs(df['depth'][i])<abs(mindepth):
                df=df.drop(i)
        df.index=range(len(df))
    else:
        return df
    for i in range(len(df)):
        if df['depth'][i]>0:
            df['depth'][i]=-1*df['depth'][i]
    df=df.dropna()
    df.index=range(len(df))
    return df

#main
with open(path,'r') as fp:
    dict=json.load(fp)
#all_df = pd.DataFrame(data=None,columns=['time','lat','lon','observation_T','Doppio_T','GoMOLFs_T','FVCOM_T','Clim_T'])
for name in dict.keys():
    if name=='end_time':
        continue
    else: 
        try:
            all_df = pd.DataFrame(data=None,columns=['time','lat','lon','observation_T','Doppio_T','GoMOLFs_T','FVCOM_T','Clim_T'])
            df = pd.DataFrame.from_dict(dict[name])
            df['time'] = df.index
            #df.index = range(len(df))
            tele_df = df[['time','lat','lon','observation_T', 'observation_H']]
            tele_df.rename(columns={'observation_T':'temp','observation_H':'depth'},inplace=True)
            Doppio_df=df[['time','lat','lon','Doppio_T', 'Doppio_H']]
            Doppio_df.rename(columns={'Doppio_T':'temp','Doppio_H':'depth'},inplace=True)
            GoMOLFs_df=df[['time','lat','lon','GoMOLFs_T', 'GoMOLFs_H']]
            GoMOLFs_df.rename(columns={'GoMOLFs_T':'temp','GoMOLFs_H':'depth'},inplace=True)
            FVCOM_df=df[['time','lat','lon','FVCOM_T', 'FVCOM_H']]
            FVCOM_df.rename(columns={'FVCOM_T':'temp','FVCOM_H':'depth'},inplace=True)
            Clim_df=df[['time','lat','lon','Clim_T', 'NGDC_H']]
            Clim_df.rename(columns={'Clim_T':'temp','NGDC_H':'depth'},inplace=True)
            #through the parameter of mindepth to screen the data, make sure the depth is out of ten
            tele_df = check_depth(df=tele_df,mindepth=mindepth) #this dataframe is obervasion data
            Doppio_df = check_depth(df=Doppio_df,mindepth=mindepth)
            GoMOLFs_df = check_depth(df=GoMOLFs_df,mindepth=mindepth)
            FVCOM_df = check_depth(df=FVCOM_df,mindepth=mindepth)
            Clim_df = check_depth(df=Clim_df,mindepth=mindepth)
            #through the start time and end time screen data
            tele_dft=check_time(df=tele_df,time_header='time',start_time=start_time,end_time=end_time)
            Doppio_dft=check_time(df=Doppio_df,time_header='time',start_time=start_time,end_time=end_time)
            GoMOLFs_dft=check_time(df=GoMOLFs_df,time_header='time',start_time=start_time,end_time=end_time)
            FVCOM_dft=check_time(df=FVCOM_df,time_header='time',start_time=start_time,end_time=end_time)
            Clim_dft=check_time(df=Clim_df,time_header='time',start_time=start_time,end_time=end_time)
            all_df['time'] = tele_dft['time']
            all_df['lat'] = tele_dft['lat']
            all_df['lon'] = tele_dft['lon']
            all_df['observation_T'] = tele_dft['temp']
            all_df['Doppio_T'] = Doppio_dft['temp']
            all_df['GoMOLFs_T'] = GoMOLFs_dft['temp']
            all_df['FVCOM_T'] = FVCOM_dft['temp']
            all_df['Clim_T'] = Clim_dft['temp']
            all_df.to_csv(path_save+name+'_fishing_temp_data.csv')
        except:
            print(name)