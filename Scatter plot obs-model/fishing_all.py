# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 06:11:15 2020

@author: Administrator
"""
import os
import pandas as pd 
#Hardcodes
path = 'E:\\Mingchao\\paper\\fishing_data\\'
save_path = 'E:\\Mingchao\\paper\\'
columns=['time', 'lat', 'lon', 'observation_T',\
         'Doppio_T', 'GoMOLFs_T', 'FVCOM_T', 'Clim_T'
         ]
def list_all_files(rootdir):
    """get all files' path and name in rootdirectory"""
    if rootdir[0] != 'E':
        rootdir=rootdir[1:]
    _files = []
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0,len(list)):
           path = os.path.join(rootdir,list[i])
           if os.path.isdir(path):
              _files.extend(list_all_files(path))
           if os.path.isfile(path):
              _files.append(path)
    return _files

files_lists = list_all_files(path)
vessel_lists = []#store the path of every vessel's file
vessel_dfs = pd.DataFrame(data=None, columns=columns)
for file in files_lists:
    if file.split('.')[1] == 'csv':
        vessel_lists.append(file)
for i in range(len(vessel_lists)):
    vessel_df = pd.read_csv(vessel_lists[i], index_col=0)
    vessel_dfs = pd.concat([vessel_dfs, vessel_df], axis=0, ignore_index=True)
vessel_dfs_D = vessel_dfs.dropna(subset=['Doppio_T'])
vessel_dfs_G = vessel_dfs.dropna(subset=['GoMOLFs_T'])
vessel_dfs_F = vessel_dfs.dropna(subset=['FVCOM_T'])
vessel_dfs_C = vessel_dfs.dropna(subset=['Clim_T'])
vessel_dfs_D.index = range(len(vessel_dfs_D))
vessel_dfs_G.index = range(len(vessel_dfs_G))
vessel_dfs_F.index = range(len(vessel_dfs_F))
vessel_dfs_C.index = range(len(vessel_dfs_C))
vessel_dfs_D.to_csv(save_path+'vessel_dfs_D'+'.csv')
vessel_dfs_G.to_csv(save_path+'vessel_dfs_G'+'.csv')
vessel_dfs_F.to_csv(save_path+'vessel_dfs_F'+'.csv')
vessel_dfs_C.to_csv(save_path+'vessel_dfs_C'+'.csv')