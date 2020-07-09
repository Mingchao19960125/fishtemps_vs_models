# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 04:47:06 2020
# Plot boundaries of GOMOFS,DOPPIO and FVCOM-GOM3
@author: Mingchao
"""

import datetime
import math
import netCDF4
import os
import numpy as np 
from mpl_toolkits.basemap import Basemap
import get_fvcom_gom3_grid as gfgg
import matplotlib.pyplot as plt

#Hardcode
path = 'E:\\Mingchao\\result\\'#save the picture

def get_doppio_url(date):
    url='http://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/2017_da/his/runs/History_RUN_2018-11-12T00:00:00Z'
    return url.replace('2018-11-12',date)

def get_FVCOM_url(dtime):
    """dtime: the formate of time is datetime"""
    #if (dtime-datetime.datetime.now())>datetime.timedelta(days=-2):
    if (dtime-datetime.datetime.utcnow())>datetime.timedelta(days=-2):
        url='http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc' 
    elif dtime>=datetime.datetime(2019,1,1):
        #if dtime.month!=datetime.datetime.now().month:
        if dtime.month!=datetime.datetime.utcnow().month:
            url='http://www.smast.umassd.edu:8080/thredds/dodsC/models/fvcom/NECOFS/Archive/NECOFS_GOM/2019/gom4_201907.nc'
            url=url.replace('201907',dtime.strftime('%Y%m'))
            url=url.replace('2019',dtime.strftime('%Y'))
        else:
            url=np.nan      
    else:
        url=np.nan
    return url

def get_gomofs_url(date):
    """
    the format of date is:datetime.datetime(2019, 2, 27, 11, 56, 51, 666857)
    input date and return the url of data
    """
    #date=date+datetime.timedelta(hours=4.5)
    date_str=date.strftime('%Y%m%d%H%M%S')
    hours=int(date_str[8:10])+int(date_str[10:12])/60.+int(date_str[12:14])/3600.
    tn=int(math.floor((hours)/6.0)*6)  ## for examole: t12z the number is 12
    if len(str(tn))==1:
        tstr='t0'+str(tn)+'z'   # tstr in url represent hour string :t00z
    else:
        tstr='t'+str(tn)+'z'
    if round((hours)/3.0-1.5,0)==tn/3:
        nstr='n006'       # nstr in url represent nowcast string: n003 or n006
    else:
        nstr='n003'
    url='http://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/GOMOFS/MODELS/'\
    +date_str[:6]+'/nos.gomofs.fields.'+nstr+'.'+date_str[:8]+'.'+tstr+'.nc'
    return url

#main
date_str = '2020-03-05'#input date to url of models'code to get url
date_time = datetime.datetime.strptime(date_str+' 00:00:00', '%Y-%m-%d %H:%M:%S')
url = get_gomofs_url(date_time)
url_doppio = get_doppio_url(date_str)
nc = netCDF4.Dataset(str(url))
nc_doppio = netCDF4.Dataset(str(url_doppio))
gomofs_lons = nc.variables['lon_rho'][:]
gomofs_lats = nc.variables['lat_rho'][:]
doppio_lons = nc_doppio.variables['lon_rho'][:]
doppio_lats = nc_doppio.variables['lat_rho'][:]
fvcom = gfgg.get_fvcom_gom3_grid(a='server')#get the data of fvcom 
#fvcom = np.load('E:\\Mingchao\\result\\npy\\Grid.npy')
fvcom_lons = fvcom['lon']
fvcom_lats = fvcom['lat']
#use this method from Yizhen Li 
x1 = [gomofs_lons[0][0], gomofs_lons[0][-1], gomofs_lons[-1][-1], gomofs_lons[-1][0], gomofs_lons[0][0]]
y1 = [gomofs_lats[0][0], gomofs_lats[0][-1], gomofs_lats[-1][-1], gomofs_lats[-1][0], gomofs_lats[0][0]]
x1_d = [doppio_lons[0][0], doppio_lons[0][-1], doppio_lons[-1][-1], doppio_lons[-1][0], doppio_lons[0][0]]
y1_d = [doppio_lats[0][0], doppio_lats[0][-1], doppio_lats[-1][-1], doppio_lats[-1][0], doppio_lats[0][0]]
m = Basemap(projection='stere',lon_0=(-74-60)/2,lat_0=(38+46)/2,lat_ts=0,llcrnrlat=31.5,urcrnrlat=46.,\
            llcrnrlon=-78.,urcrnrlon=-55.,rsphere=6371200.,resolution='f',area_thresh=100)
# draw coastlines
m.drawcoastlines(color='gray').set_zorder(1)#set zorder to cover lines of gomofs and doppio
#m.drawmapboundary(fill_color='aqua')
#m.drawlsmask(ocean_color='aqua')
m.fillcontinents(color='gray')
 # draw parallels.
parallels = np.arange(32,47,5)
m.drawparallels(parallels, labels=[1,0,0,0], fontsize=10, linewidth=0.0)
# draw meridians
meridians = np.arange(-77,-55,8)
m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=10, linewidth=0.0)
#for i in range(len(x1)):
    #xy.append(m(x1[i], y1[i]))
#for j in range(len(xy)):
    #x.append(xy[j][0])
    #y.append(xy[j][1])
kvb=np.where(fvcom['nfv']<4) # vertices with 4 or smaller neighboring triangles
m.plot(x1, y1, color='red', latlon=True, zorder=0, label='ROMS-GOMOFS')
m.plot(fvcom_lons[kvb],fvcom_lats[kvb], 'b.', latlon=True, zorder=0, label='FVCOM-GOM3')
m.plot(x1_d,y1_d, color='y', latlon=True, zorder=0, label='ROMS-DOPPIO')
plt.title('MODEL GRID BOUNDARIES')
plt.legend(loc='lower right')
plt.savefig(os.path.join(path+'models_boundary.png'))
plt.show()