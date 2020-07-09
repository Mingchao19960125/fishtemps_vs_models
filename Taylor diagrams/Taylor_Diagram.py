# -*- coding: utf-8 -*-
"""
Created on Sun May 31 15:04:25 2020

@author: Mingchao
"""

import pandas as pd
import numpy as np
import skill_metrics as sm
import datetime
import matplotlib.pyplot as plt

def taylor_statistics(p, r):
    # Calculate bias (B)
    bias = np.mean(p) - np.mean(r)
    # Calculate correlation coefficient
    ccoef = np.corrcoef(p,r)
    ccoef = ccoef[0]
    # Calculate centered root-mean-square (RMS) difference (E')^2
    crmsd = [0.0, sm.centered_rms_dev(p,r)]
    # Calculate standard deviation of predicted field w.r.t N (sigma_p)
    sdevp = np.std(p)    
    # Calculate standard deviation of reference field w.r.t N (sigma_r)
    sdevr = np.std(r)
    sdev = [sdevr, sdevp];
    # Store statistics in a dictionary
    stats = {'ccoef': ccoef, 'crmsd': crmsd, 'sdev': sdev, 'bias': bias}
    return stats

####################### Hardcodes ###########################################3
path = 'E:\\Mingchao\\paper\\vessel_dfs.csv'
save_path = 'E:\\Mingchao\\paper\\'
start_time = datetime.datetime(2019,1,1,0,0,0)
end_time = datetime.datetime(2020,1,1,0,0,0)

####################### MAIN ###########################################3
data = pd.read_csv(path, index_col=0)
data['ENSEMBLE_T'] = 1.000000
data['time'] = pd.to_datetime(data['time'])
data = data.dropna()
for i in data.index:
    data['ENSEMBLE_T'][i] = (data['Doppio_T'][i]+data['FVCOM_T'][i]+data['GoMOLFs_T'][i])/3
    if not start_time<data['time'][i]<end_time:
        data = data.drop(i)
# Calculate statistics for Taylor diagram
taylor_stats1 = taylor_statistics(data.Doppio_T, data.observation_T)
taylor_stats2 = taylor_statistics(data.GoMOLFs_T, data.observation_T)
taylor_stats3 = taylor_statistics(data.FVCOM_T, data.observation_T)
taylor_stats4 = taylor_statistics(data.ENSEMBLE_T, data.observation_T)
 # Store statistics in arrays
sdev = np.array([taylor_stats1['sdev'][0], taylor_stats4['sdev'][1], 
                 taylor_stats2['sdev'][1], taylor_stats3['sdev'][1],
                 taylor_stats1['sdev'][1]])
crmsd = np.array([taylor_stats1['crmsd'][0], taylor_stats4['crmsd'][1], 
                  taylor_stats2['crmsd'][1], taylor_stats3['crmsd'][1],
                  taylor_stats1['crmsd'][1]])
ccoef = np.array([taylor_stats1['ccoef'][0], taylor_stats4['ccoef'][1], 
                  taylor_stats2['ccoef'][1], taylor_stats3['ccoef'][1],
                  taylor_stats1['ccoef'][1]])
bias  = np.array([0, taylor_stats2['bias'], taylor_stats4['bias'],
                  taylor_stats3['bias'], taylor_stats1['bias']])
label = ['Non-Dimensional Observation', 'E','GOMOFS', 'FVCOM', 'DOPPIO']
sm.taylor_diagram(sdev, crmsd, ccoef, markerLabel=label,
                      locationColorBar='EastOutside',
                      markerDisplayed='colorbar', titleColorBar='Bias', 
                      markerLabelColor='r', markerSize=10,
                      markerLegend='off', cmapzdata=bias,
                      colRMS='g', styleRMS=':', widthRMS=2.0, titleRMS='on',
                      colSTD='b', styleSTD='-.', widthSTD=1.0, titleSTD ='on',
                      colCOR='k', styleCOR='--', widthCOR=1.0, titleCOR='on')
# Write plot to file
plt.savefig(save_path+'Taylor_Diagram.png')
# Show plot
plt.show()
