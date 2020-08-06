# -*- coding: utf-8 -*-
"""
This code get data for Interpolation method's comparison 
Created on Sun Jul 26 20:28:58 2020

@author: Mingchao
"""
import pandas as pd
import numpy as np
import netCDF4
import datetime
import zlconversions as zl  # this is a set of Lei Zhao's functions that must be in same folder 

#################### Hardcodes ############################
path = 'E:\\Mingchao\\paper\\vessel_dfs.csv'
save_path = 'E:\\Mingchao\\paper\\'
start_time = datetime.datetime(2018,9,30,21,00,00)
end_time = datetime.datetime(2018,12,16,00,00,00)
#end_time = datetime.datetime(2019,5,8,00,00,00)
#Doppio
url = 'http://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/2017_da/his/History_Best'#get_doppio_url(url_time)        
nc = netCDF4.Dataset(url)
lons = nc.variables['lon_rho'][:]
lats = nc.variables['lat_rho'][:]
temp = nc.variables['temp']
doppio_time = nc.variables['time']
##################### Functions ###############################
def get_doppio_url(date): # modification of Lei Zhao code to find the most recent DOPPIO url
    #url='http://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/2017_da/his/runs/History_RUN_2018-11-12T00:00:00Z'
    #return url.replace('2018-11-12',date)
    url='http://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/2017_da/his/History_Best'
    return url

def find_nd(target,lat,lon,lats,lons):
    
    """ Bisection method:find the index of nearest distance"""
    row=0
    maxrow=len(lats)-1
    col=len(lats[0])-1
    while col>=0 and row<=maxrow:
        distance=zl.dist(lat1=lats[row,col],lat2=lat,lon1=lons[row,col],lon2=lon)
        if distance<=target:
            break
        elif abs(lats[row,col]-lat)<abs(lons[row,col]-lon):
            col-=1
        else:
            row+=1
    distance=zl.dist(lat1=lats[row,col],lat2=lat,lon1=lons[row,col],lon2=lon)
    row_md,col_md=row,col  #row_md the row of minimum distance
    #avoid row,col out of range in next step
    if row<3:
        row=3
    if col<3:
        col=3
    if row>maxrow-3:
        row=maxrow-3
    if col>len(lats[0])-4:
        col=len(lats[0])-4
    for i in range(row-3,row+3,1):
        for j in range(col-3,col+3,1):
            distance_c=zl.dist(lat1=lats[i,j],lat2=lat,lon1=lons[i,j],lon2=lon)
            if distance_c<=distance:
                distance=distance_c
                row_md,col_md=i,j
    return row_md,col_md

def fitting(point,lat,lon):
#represent the value of matrix
    ISum = 0.0
    X1Sum = 0.0
    X2Sum = 0.0
    X1_2Sum = 0.0
    X1X2Sum = 0.0
    X2_2Sum = 0.0
    YSum = 0.0
    X1YSum = 0.0
    X2YSum = 0.0

    for i in range(0,len(point)):
        
        x1i=point[i][0]
        x2i=point[i][1]
        yi=point[i][2]

        ISum = ISum+1
        X1Sum = X1Sum+x1i
        X2Sum = X2Sum+x2i
        X1_2Sum = X1_2Sum+x1i**2
        X1X2Sum = X1X2Sum+x1i*x2i
        X2_2Sum = X2_2Sum+x2i**2
        YSum = YSum+yi
        X1YSum = X1YSum+x1i*yi
        X2YSum = X2YSum+x2i*yi

#  matrix operations
# _mat1 is the mat1 inverse matrix
    m1=[[ISum,X1Sum,X2Sum],[X1Sum,X1_2Sum,X1X2Sum],[X2Sum,X1X2Sum,X2_2Sum]]
    mat1 = np.matrix(m1)
    m2=[[YSum],[X1YSum],[X2YSum]]
    mat2 = np.matrix(m2)
    _mat1 =mat1.getI()
    mat3 = _mat1*mat2

# use list to get the matrix data
    m3=mat3.tolist()
    a0 = m3[0][0]
    a1 = m3[1][0]
    a2 = m3[2][0]
    y = a0+a1*lat+a2*lon

    return y

def doppio_coordinate(lat,lon):
    f1=-0.8777722604596849*lat-lon-23.507489034447012>=0
    f2=-1.072648270137022*lat-40.60872567829448-lon<=0
    f3=1.752828434063416*lat-131.70051451008493-lon>=0
    f4=1.6986954871237598*lat-lon-144.67649951783605<=0
    if f1 and f2 and f3 and f4:
        return True
    else:
        return False

def get_doppio_fitting(latp=0,lonp=0,depth='bottom',dtime=datetime.datetime.now(),fortype='temperature',hour_allowed=1):
    """
    notice:
        the format of time is like "%Y-%m-%d %H:%M:%S" this time is utctime or the type of time is datetime.datetime
        the depth is under the bottom depth
    the module only output the temperature of point location
    """
    if not doppio_coordinate(latp,lonp):
        print('the lat and lon out of range in doppio')
        return np.nan,np.nan
    if type(dtime)==str:
        date_time=datetime.datetime.strptime(dtime,'%Y-%m-%d %H:%M:%S') # transform time format
    else:
        date_time=dtime
    for m in range(0,7):
        try:
            url_time=(date_time-datetime.timedelta(days=m)).strftime('%Y-%m-%d')
            url=zl.get_doppio_url(url_time)
            #get the data 
            nc=netCDF4.Dataset(url)
            lons=nc.variables['lon_rho'][:]
            lats=nc.variables['lat_rho'][:]
            doppio_time=nc.variables['time']
            doppio_rho=nc.variables['s_rho']
            doppio_temp=nc.variables['temp']
            doppio_h=nc.variables['h']
        except:
            continue
        #calculate the index of the minimum timedelta
        parameter=(datetime.datetime(2017,11,1,0,0,0)-date_time).days*24+(datetime.datetime(2017,11,1,0,0,0)-date_time).seconds/3600.
        time_delta=abs(doppio_time[:]+parameter)
        min_diff_index=np.argmin(time_delta)
        #calculate the min distance and index
        target_distance=2*zl.dist(lat1=lats[0,0],lon1=lons[0,0],lat2=lats[0,1],lon2=lons[0,1])
        index_1,index_2=find_nd(target=target_distance,lat=latp,lon=lonp,lats=lats,lons=lons)
        #calculate the optimal layer index
        if depth=='bottom':
            layer_index=0  #specify the initial layer index
        else:
            h_distance=abs(doppio_rho[:]*doppio_h[index_1,index_2]+abs(depth))
            layer_index=np.argmin(h_distance)
#        fitting the data through the 5 points
        if index_1==0:
            index_1=1
        if index_1==len(lats)-1:
            index_1=len(lats)-2
        if index_2==0:
            index_2=1
        if index_2==len(lats[0])-1:
            index_2=len(lats[0])-2
        while True:
            point=[[lats[index_1][index_2],lons[index_1][index_2],doppio_temp[min_diff_index,layer_index,index_1,index_2]],\
            [lats[index_1-1][index_2],lons[index_1-1][index_2],doppio_temp[min_diff_index,layer_index,(index_1-1),index_2]],\
            [lats[index_1+1][index_2],lons[index_1+1][index_2],doppio_temp[min_diff_index,layer_index,(index_1+1),index_2]],\
            [lats[index_1][index_2-1],lons[index_1][index_2-1],doppio_temp[min_diff_index,layer_index,index_1,(index_2-1)]],\
            [lats[index_1][index_2+1],lons[index_1][index_2+1],doppio_temp[min_diff_index,layer_index,index_1,(index_2+1)]]]
            break
        point_temp=fitting(point,latp,lonp)
        while True:
            points_h=[[lats[index_1][index_2],lons[index_1][index_2],doppio_h[index_1,index_2]],\
            [lats[index_1-1][index_2],lons[index_1-1][index_2],doppio_h[(index_1-1),index_2]],\
            [lats[index_1+1][index_2],lons[index_1+1][index_2],doppio_h[(index_1+1),index_2]],\
            [lats[index_1][index_2-1],lons[index_1][index_2-1],doppio_h[index_1,(index_2-1)]],\
            [lats[index_1][index_2+1],lons[index_1][index_2+1],doppio_h[index_1,(index_2+1)]]]
            break
        point_temp=fitting(point,latp,lonp)
        point_h=fitting(points_h,latp,lonp)
        if np.isnan(point_temp):
            continue
        if time_delta[min_diff_index]<hour_allowed:
            break        
    if fortype=='tempdepth':
        return point_temp, point_h
    else:
        return point_temp

def angle_conversion(a):
    a = np.array(a)
    return a/180*np.pi

def dist(lat1=0,lon1=0,lat2=0,lon2=0):
    """caculate the distance of two points, return miles"""
    conversion_factor = 0.62137119
    R = 6371.004
    lon1, lat1 = angle_conversion(lon1), angle_conversion(lat1)
    lon2, lat2 = angle_conversion(lon2), angle_conversion(lat2)
    l = R*np.arccos(np.cos(lat1)*np.cos(lat2)*np.cos(lon1-lon2)+\
                        np.sin(lat1)*np.sin(lat2))*conversion_factor
    return l

def get_doppio_no_fitting(lons,lats,temp,doppio_time,lat=0,lon=0,depth=99999,time='2018-11-12 12:00:00'):
    """
    notice:
        the format of time is like "%Y-%m-%d %H:%M:%S"
        the default depth is under the bottom depth
    the module only output the temperature of point location
    """
    import datetime
    #date_time=datetime.datetime.strptime(time,'%Y-%m-%d %H:%M:%S') # transform time format
    date_time=time
    for i in range(0,7): # look back 7 hours for data
        #url_time=(date_time-datetime.timedelta(hours=i)).strftime('%Y-%m-%d')#
#        url = 'http://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/2017_da/his/History_Best'#get_doppio_url(url_time)        
#        nc=netCDF4.Dataset(url)
#        lons=nc.variables['lon_rho'][:]
#        lats=nc.variables['lat_rho'][:]
#        temp=nc.variables['temp']
#        doppio_time=nc.variables['time']
        #doppio_depth=nc.variables['h'][:]
        min_diff_time=abs(datetime.datetime(2017,11,1,0,0,0)+datetime.timedelta(hours=int(doppio_time[0]))-date_time)
        min_diff_index=0
        for i in range(1,157): # 6.5 days and 24
            diff_time=abs(datetime.datetime(2017,11,1,0,0,0)+datetime.timedelta(hours=int(doppio_time[i]))-date_time)
            if diff_time<min_diff_time:
                min_diff_time=diff_time
                min_diff_index=i
                
        min_distance=dist(lat1=lat,lon1=lon,lat2=lats[0][0],lon2=lons[0][0])
        index_1,index_2=0,0
        for i in range(len(lons)):
            for j in range(len(lons[i])):
                if min_distance>dist(lat1=lat,lon1=lon,lat2=lats[i][j],lon2=lons[i][j]):
                    min_distance=dist(lat1=lat,lon1=lon,lat2=lats[i][j],lon2=lons[i][j])
                    index_1=i
                    index_2=j
        if depth==99999:# case of bottom
            S_coordinate=1
        #else:
            #S_coordinate=float(depth)/float(doppio_depth[index_1][index_2])
        if 0<=S_coordinate<1:
            point_temp=temp[min_diff_index][39-int(S_coordinate/0.025)][index_1][index_2]# because there are 0.025 between each later
            #point_depth=doppio_depth[index_1][index_2]
        elif S_coordinate==1:
            point_temp=temp[min_diff_index][0][index_1][index_2]
            #point_depth=doppio_depth[index_1][index_2]
        else:
            return 9999
        if np.isnan(point_temp):
            continue
        if min_diff_time<datetime.timedelta(hours=1):
            break
    return point_temp#,point_depth

#################### main ############################
file = pd.read_csv(path,index_col=0)
#file_df = file.dropna()
comparison_df = pd.DataFrame(data=None,columns=['time','lat','lon','observation_T','fitting','no_fitting'])
comparison_df['lat'] = file['lat']
comparison_df['lon'] = file['lon']
comparison_df['observation_T'] = file['observation_T']
comparison_df['fitting'] = file['Doppio_T']
comparison_df['time'] = pd.to_datetime(file['time'])
comparison_df = comparison_df.dropna(subset=['fitting'])

for i in comparison_df.index:
    if start_time<comparison_df['time'][i]<end_time:
        try:
            #a = get_doppio_fitting(latp=comparison_df['lat'][i],lonp=comparison_df['lon'][i],depth='bottom',dtime=comparison_df['time'][i],fortype='tempdepth')
            b = get_doppio_no_fitting(lons,lats,temp,doppio_time,lat=comparison_df['lat'][i],lon=comparison_df['lon'][i],depth=99999,time=comparison_df['time'][i])
            #comparison_df['fitting'][i] = a[0]
            comparison_df['no_fitting'][i] = b
            print('good data:'+str(comparison_df['time'][i]))
            pass
        except Exception as result:
            print(str(comparison_df['time'][i]))
            print(result)
            continue
    pass        
#comparison_df = comparison_df.dropna()
comparison_df.to_csv(save_path+'interpo_comparison_data.csv')
