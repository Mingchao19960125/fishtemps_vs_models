# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 01:18:54 2020

This code simply binning_scatter_modvsobs.py for comparing temperature with observation and other models.

@author: Mingchao

"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import os
import datetime
from scipy.interpolate import griddata

####### Hardcodes
os.environ['PROJ_LIB'] = 'E:\\Mingchao\\paper'
from mpl_toolkits.basemap import Basemap
maxdiff_ignore=8 # ignore when difference is greater than this number
data_list = 'E:\\Mingchao\\test\\ensemble_data.csv'
which_method='binned'      # 'scatter' or 'binned'
gridsize=0.5 # needed when which_method is binned in units of lat/lon
which_mode='models' # 'models' or 'season'
which_model=1 # 0-4 depends on options below used when which_mode= "season"
ensemble = 'off'# 'on' or 'off' for using ensemble
options=[' Objective','Equal','DOPPIO','FVCOM','GOMOFS','CLIMATOLOGY','MODELS']
models=['obj','eq','Doppio_T','FVCOM_T','GoMOLFs_T','Clim_T','MODELS']
seasons = ['WINTER', 'SPRING', 'SUMMER', 'FALL']  
save_path = os.environ['PROJ_LIB']+'\\Map obs-model include ensemble.png'
start_time = datetime.datetime(2019,1,1,0,0,0)
end_time = datetime.datetime(2020,4,1,0,0,0)
time_period=str(start_time.year)#+' to '+str(end_time.year)
bathy=True
gs=50      # number of bins in the x and y direction so,  if you want more detail, make it bigger
ss=100     # subsample input data so, if you want more detail, make it smaller
cont=[-200]    # contour level
mila=36.;mala=44.5;milo=-76.;malo=-66.#min & max lat and lo
#########  END HARDCODES ######################
def sh_bindata(x, y, z, xbins, ybins):
    """
    Bin irregularly spaced data on a rectangular grid.
    returns X,Y,Z lists
    """
    ix=np.digitize(x,xbins)
    iy=np.digitize(y,ybins)
    xb=0.5*(xbins[:-1]+xbins[1:]) # bin x centers
    yb=0.5*(ybins[:-1]+ybins[1:]) # bin y centers
    zb_mean=np.empty((len(xbins)-1,len(ybins)-1),dtype=z.dtype)
    zb_median=np.empty((len(xbins)-1,len(ybins)-1),dtype=z.dtype)
    zb_std=np.empty((len(xbins)-1,len(ybins)-1),dtype=z.dtype)
    zb_num=np.zeros((len(xbins)-1,len(ybins)-1),dtype=int)    
    for iix in range(1,len(xbins)):
        for iiy in range(1,len(ybins)):
            k,=np.where((ix==iix) & (iy==iiy))
            zb_mean[iix-1,iiy-1]=np.mean(z[k])
            zb_median[iix-1,iiy-1]=np.median(z[k])
            zb_std[iix-1,iiy-1]=np.std(z[k])
            zb_num[iix-1,iiy-1]=len(z[k])
    X,Y = np.meshgrid(xb, yb)
    Z=zb_mean.T.flatten()           
    #return xb,yb,zb_mean,zb_median,zb_std,zb_num
    return X,Y,Z
def add_isobath(m,gs,ss,cont):
    # draws an isobath on map given gridsize,subsample rate,and contour level
    # these inputs are typically 50, 100, and 200 for entire shelf low resolution
    url='http://apdrc.soest.hawaii.edu/erddap/griddap/hawaii_soest_794e_6df2_6381.csv?b_bathy[(0000-01-01T00:00:00Z):1:(0000-01-01T00:00:00Z)][(35.0):1:(45.0)][(-76.0):1:(-66.0)]'
    #df=pd.read_csv(url)
    try:                    #Mingchao added 12,May,2020
        df=pd.read_csv(url)
    except:
        print('get some detail bathymetry from USGS have issue')
    df=df.drop('time',axis=1)
    df=df[1:].astype('float')# removes unit row and make all float          
    Xb,Yb=m.makegrid(gs,gs) # where "gs" is the gridsize specified in hardcode section
    Xb,Yb=m(Xb,Yb) # converts grid to basemap coordinates .
    xlo,yla=m(df['longitude'][0:-1:ss].values,df['latitude'][0:-1:ss].values)
    zi = griddata((xlo,yla),df['b_bathy'][0:-1:ss].values,(Xb,Yb),method='linear')
    CS=m.contour(Xb,Yb,zi,cont,zorder=0,linewidths=[1],linestyles=['dashed'])
    plt.clabel(CS, inline=1, fontsize=8,fmt='%d')

def draw_basemap(fig, ax, df_lon, df_lat, list_season, seasons,bathy,gs,ss,milo,malo,mila,mala,cont,which_method,gridsize):
    # if bathy=True, plot bathy
    ax = fig.sca(ax)
    m = Basemap(projection='stere',lon_0=(milo+malo)/2,lat_0=(mila+mala)/2,lat_ts=0,llcrnrlat=mila,urcrnrlat=mala,\
                llcrnrlon=milo,urcrnrlon=malo,rsphere=6371200.,resolution='l',area_thresh=100)
    #m.drawcoastlines()
    m.fillcontinents(color='grey',lake_color='grey')
#    if (ax == ax1) or (ax == ax3) or (ax == ax5): #draw parallels
    parallels = np.arange(mila,mala,3)
    m.drawparallels(parallels, labels=[1,0,0,0], fontsize=10, linewidth=0.0)
#    if (ax == ax3) or (ax==ax4):# draw meridians
    if (ax == ax5) or (ax==ax6):# draw meridians
        meridians = np.arange(milo+2,malo,4)
        m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=10, linewidth=0.0)
    if which_method=='scatter':
        a=m.scatter(df_lon, df_lat, c=list_season, cmap='coolwarm',marker='o', linewidths=0.01, latlon=True)
    else: 
        print('binning data')
        xi = np.arange(milo,malo,gridsize)# longitude grid
        yi = np.arange(mila,mala,gridsize)# latitude grid
        X,Y,Z = sh_bindata(df_lon, df_lat,list_season,xi,yi)
        X,Y = m(X.flatten(),Y.flatten())
        print('finally adding scatter plot ')
        a=m.scatter(X, Y, c=Z, cmap='coolwarm',marker='o', linewidths=0.01)
        a.set_clim(-1*maxdiff_ignore,maxdiff_ignore)        
    if bathy==True: # get some detail bathymetry from USGS
        add_isobath(m,gs,ss,cont)
    mx,my=m(milo+.2,mala-.5)
    #plt.text(mx+50000,my-110000,seasons,color='w',fontsize=18,fontweight='bold')
    plt.text(mx,my,seasons,color='w',fontweight='bold')#fontsize=18,
    plt.text(mx,my-100000,'mean=' +str(np.format_float_positional(np.mean(list_season), precision=2, unique=False, fractional=False)),color='w',fontweight='bold')#fontsize=14,
    plt.text(mx,my-180000,'RMS = '+str(np.format_float_positional(np.sqrt(np.mean(np.square(list_season))), precision=2, unique=False, fractional=False)),color='w',fontweight='bold')#fontsize=14,
    return a
##### MAIN CODE ######
DF=pd.read_csv(data_list, index_col=0)    # reads input file with obs & models
DF = DF.dropna(axis=0)                    # gets rid of null values. 4,May Mingchao
DF['time'] = pd.to_datetime(DF['time'])

DF.index=DF['time']                       # makes time index
DF['month']=DF.index.month
fig = plt.figure(figsize=(10,10))
size=min(fig.get_size_inches())
gsc = gridspec.GridSpec(3, 2,wspace=0.0, hspace=0.05,width_ratios=[1, 1],
                        top=1.-0.3/(2+1), bottom=0.3/(2+1), 
                        left=0.3/(2+1), right=1-0.3/(2+1)) 

ax1 = fig.add_subplot(gsc[0,0])
ax1.axes.get_yaxis().set_visible(False)
ax2 = fig.add_subplot(gsc[0,1])
ax2.axes.get_yaxis().set_visible(False)
ax3 = fig.add_subplot(gsc[1,0])
ax4 = fig.add_subplot(gsc[1,1])
ax5 = fig.add_subplot(gsc[2,0])
ax6 = fig.add_subplot(gsc[2,1])
ax_list = [ax1, ax2, ax3, ax4, ax5, ax6]

for i in range(6):# this runs the other three
    diff=(DF['observation_T']-DF[models[i]]).values
    id=np.where(abs(diff)<maxdiff_ignore)
    a=draw_basemap(fig, ax=ax_list[i], df_lon=DF['lon'].values[id], df_lat=DF['lat'].values[id],\
                   list_season=diff[id], seasons=options[i],bathy=True,gs=gs,ss=ss,milo=milo,\
                   malo=malo,mila=mila,mala=mala,cont=cont,which_method=which_method,gridsize=gridsize)
c3 = fig.colorbar(a, ax=[ax1, ax2, ax3, ax4, ax5, ax6])
c3.set_ticks(np.arange(-1*maxdiff_ignore,maxdiff_ignore))
c3.set_ticklabels(np.arange(-maxdiff_ignore,maxdiff_ignore))
fig.text(0.5, 0.93, 'Observation minus Other Models', ha='center', va='center', fontsize=2.0*size)
plt.suptitle('Comparison of temperature differences', va='center_baseline', fontsize=2.5*size)
plt.savefig(save_path,dpi=300)
plt.show()