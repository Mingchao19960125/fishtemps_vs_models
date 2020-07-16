#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:47:29 2020
This function get data from checked folder,then create files include models'predicted temperature.
Input:
   checked folder of check_csv.py , get_clim folder
Output:
   files named like Excalibur_hours.csv.
   The columns of files : ['lat','lon','temp','DOPPIO_T','GOMOFS_T','FVCOM_T','CLIM_T']

@author: Jim&Mingchao
"""
import conversions as cv
import os
import pandas as pd
import zlconversions as zl
from datetime import datetime,timedelta
from pylab import mean, std
import multiple_models as mm
import numpy as np

#Hardcodes
#input_dir='/home/jmanning/leizhao/programe/raw_data_match/result/checked/'
input_dir='E:\\programe\\raw_data_match\\result\\checked\\'
#end_time=datetime.now()
end_time=datetime.utcnow()
start_time=end_time-timedelta(days=313)
#start_time=end_time-timedelta(weeks=1)
#Hours_save='/home/jmanning/Mingchao/result/Hours_data/'
Hours_save='E:\\Mingchao\\result\\Hours_data\\'
getclim_path='E:\\programe\\aqmain\\py\\clim\\'
vessel_name = 'Virginia_Marie'#'Excalibur'    
#main
allfile_lists=zl.list_all_files(input_dir)
file_lists=[]#store the path of every vessel's files
#hoursfile_lists=zl.list_all_files(Hours_save)
#filter the raw files and store in file_lists
for file in allfile_lists:
   if file[len(file)-4:]=='.csv':
     file_lists.append(file)
try:
    for file in file_lists: # loop raw files
        fpath,fname=os.path.split(file)  #get the file's path and name
        if fpath.split('\\')[5] != vessel_name:#get data of vessel that you want
            continue
        time_str=fname.split('.')[0].split('_')[2]+' '+fname.split('.')[0].split('_')[3]
    #GMT time to local time of file
        time_gmt=datetime.strptime(time_str,"%Y%m%d %H%M%S")
        if time_gmt<start_time or time_gmt>end_time:
            continue
    # now, read header and data of every file  
        header_df=zl.nrows_len_to(file,2,name=['key','value']) #only header 
        data_df=zl.skip_len_to(file,2) #only data
        value_data_df=data_df.loc[(data_df['Depth(m)']>0.95*max(data_df['Depth(m)']))]  #filter the data
        value_data_df=value_data_df.iloc[7:]   #delay several minutes to let temperature sensor record the real bottom temp
        value_data_df=value_data_df.loc[(value_data_df['Temperature(C)']>mean(value_data_df['Temperature(C)'])-3*std(value_data_df['Temperature(C)'])) & \
                            (value_data_df['Temperature(C)']<mean(value_data_df['Temperature(C)'])+3*std(value_data_df['Temperature(C)']))]  #Excluding gross error
        value_data_df.index = range(len(value_data_df))  #reindex
        value_data_df['Datet(Y/m/d)']=1 #create a new column for saving another time style of '%Y-%m-%d'
        for i in range(len(value_data_df)):
            value_data_df['Lat'][i],value_data_df['Lon'][i]=cv.dm2dd(value_data_df['Lat'][i],value_data_df['Lon'][i])
            #value_data_df['Datet(Y/m/d)'][i]=datetime.strptime(value_data_df['Datet(GMT)'][i],'%Y-%m-%d %H:%M:%S')
        Hours_df=pd.DataFrame(data=None,columns=['time','lat','lon','temp','DOPPIO_T',\
                                                 'GOMOFS_T','FVCOM_T','CLIM_T'])
        Hours_df['time']=value_data_df['Datet(GMT)']
        Hours_df['lat']=value_data_df['Lat']
        Hours_df['lon']=value_data_df['Lon']
        Hours_df['temp']=value_data_df['Temperature(C)']
        Hours_df['time'] = pd.to_datetime( Hours_df['time'])
        Hours_df['DOPPIO_T']=0.00
        Hours_df['GOMOFS_T']=0.00
        Hours_df['FVCOM_T']=0.00
        Hours_df['CLIM_T']=0.00
        Hours_df.index = Hours_df['time'].values
        resample_df=Hours_df.loc[:,['lat','lon','temp','DOPPIO_T',\
                             'GOMOFS_T','FVCOM_T','CLIM_T']].resample('H',how=np.mean)
        for time_index in resample_df.index:
            try:
                resample_df['DOPPIO_T'][time_index]=mm.get_doppio_fitting(latp=resample_df['lat'][time_index],lonp=resample_df['lon'][time_index],\
                                                   depth='bottom',dtime=time_index,fortype='temperature')
            except:
                resample_df['DOPPIO_T'][time_index]=np.nan
            try:
                resample_df['GOMOFS_T'][time_index]=mm.get_gomofs_zl(dtime=time_index,latp=resample_df['lat'][time_index],lonp=resample_df['lon'][time_index],\
                                                   depth='bottom',mindistance=20,autocheck=True,fortype='temperature')
            except:
                resample_df['GOMOFS_T'][time_index]=np.nan
            try:
                resample_df['FVCOM_T'][time_index]=mm.get_FVCOM_fitting(latp=resample_df['lat'][time_index],lonp=resample_df['lon'][time_index],\
                                                   dtime=time_index,depth='bottom',mindistance=2,fortype='temperature')
            except:
                resample_df['FVCOM_T'][time_index]=np.nan
            try:
                resample_df['CLIM_T'][time_index]=mm.getclim(lat1=resample_df['lat'][time_index],lon1=resample_df['lon'][time_index],\
                                                   path=getclim_path,dtime=time_index,var='Bottom_Temperature\\BT_')
            except:
                resample_df['CLIM_T'][time_index]=np.nan
        if not os.path.exists(Hours_save+file.split('\\')[5]):
            os.makedirs(Hours_save+file.split('\\')[5])
        resample_df.to_csv(os.path.join(Hours_save+file.split('\\')[5]+'\\',file.split('\\')[5]+'#'+file.split('\\')[7]))
        hoursfile_together_lists=zl.list_all_files(Hours_save+file.split('\\')[5]+'\\')
        total_hours_list=[]
        for k in range(len(hoursfile_together_lists)):
            total_hours_list.append(pd.read_csv(hoursfile_together_lists[k],index_col=0))#contact values belong to one vessel
        total_hours_df=pd.concat(total_hours_list)
        total_hours_df.drop_duplicates(subset=['lat','lon','temp'],inplace=True)
        total_hours_df=total_hours_df.sort_index()#sorting by index
        total_hours_df.to_csv(os.path.join(Hours_save+file.split('\\')[5]+'\\',hoursfile_together_lists[0].split('#')[0].split('\\')[4]+'_hours.csv'))
except:
    print(file+'check if the files exists')
         


       
                