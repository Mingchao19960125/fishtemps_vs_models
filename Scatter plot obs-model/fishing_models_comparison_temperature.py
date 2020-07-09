# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 22:09:40 2020

@author: Mingchao
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

#Hardcodes
path = 'E:\\Mingchao\\paper\\'
save_path='E:\\Mingchao\\paper\\fishing_data\\FISHING and MODELS comparison of temperature.png'
#main
df_D = pd.read_csv(path+'vessel_dfs_D.csv', index_col=0)
df_G = pd.read_csv(path+'vessel_dfs_G.csv', index_col=0)
df_F = pd.read_csv(path+'vessel_dfs_F.csv', index_col=0)
df_C = pd.read_csv(path+'vessel_dfs_C.csv', index_col=0)
models = ['ROMS-DOPPIO', 'ROMS-GOMOFS', 'FVCOM-GOM3', 'CLIM']
columns = ['Doppio_T', 'GoMOLFs_T', 'FVCOM_T', 'Clim_T']
fig = plt.figure(figsize=(8,9))
size=min(fig.get_size_inches()) 
ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)
ax2.axes.get_yaxis().set_visible(False)
ax4.axes.get_yaxis().set_visible(False)
#a = ax1.hist2d(x=df[columns[0]], y=df['observation_T'], bins=90, alpha=0.7, norm=LogNorm())
a = ax1.hist2d(x=df_D[columns[0]], y=df_D['observation_T'], bins=35, alpha=0.8, cmin=1)
plt.colorbar(a[3], ax=[ax1, ax2, ax3, ax4])
b = ax2.hist2d(x=df_G[columns[1]], y=df_G['observation_T'], bins=35, alpha=0.8, cmin=1)
#plt.colorbar(b[3], ax=ax2)
c = ax3.hist2d(x=df_F[columns[2]], y=df_F['observation_T'], bins=35, alpha=0.8, cmin=1)
#plt.colorbar(c[3], ax=ax3)
d = ax4.hist2d(x=df_C[columns[3]], y=df_C['observation_T'], bins=35, alpha=0.8, cmin=1)
#plt.colorbar(d[3], ax=[ax1, ax2, ax3, ax4])
ax1.set_xlim(0,24)
ax1.set_ylim(0,27)
ax2.set_xlim(0,24)
ax2.set_ylim(0,27)
ax3.set_xlim(0,24)
ax3.set_ylim(0,27)
ax4.set_xlim(0,24)
ax4.set_ylim(0,27)
ax1.set_xlabel(models[0], fontsize=1.5*size)
ax2.set_xlabel(models[1], fontsize=1.5*size)
ax3.set_xlabel(models[2], fontsize=1.5*size)
ax4.set_xlabel(models[3], fontsize=1.5*size)
fig.text(0.05, 0.5, 'FISHING TEMPERATURE', ha='center', va='center', rotation='vertical',fontsize=2.5*size)
fig.text(0.5, 0.93, '2018.06 to 2019.12', ha='center', va='center',fontsize=2.5*size)
plt.suptitle('FISHING and MODELS comparison of temperature', fontsize=2.5*size, va='center_baseline')
plt.savefig(save_path)
plt.show()