# -*- coding: utf-8 -*-
import TUMforecast as TUM
import pandas as pd
import numpy as np
import os
"""
    File name: Forecast.py
    Author: Wessam El-Baz
    Date created: 01/02/2019
    Date last modified: 06/02/2019
    Python Version: 3.6.3
    For detailed documentation please refer to https://welbaz.github.io/p3/
    
    Installation steps
    1) Install the pcube_PV_Forecast_Installer.exe located in \ForecastingSystemFiles\Redistribution_Installation
    2) Go to C:/Program Files/Technical University of Munich/TUMforecast/application
    2) Run in the installation folder (python setup.py. install) to install the package
"""

__version__ = 0.1

class ForecastingSystem:
    def __init__(self,
                 StartTimeTi="01-May-2016 00:00:00",
                 EndTimeTi="30-Dec-2016 23:59:00",
                 StartTimePi="01-May-2017 00:00:00",
                 EndTimePi="31-May-2017 23:59:00",
                 UTC = 0.0,
                 lat=48.1505119,
                 long = 11.568185099999937,
                 alt = 515.898,
                 ppeak=3.0,
                 tilt = 30.0,
                 azimuth = 200.0,
                 eff = 0.16,
                 No_modules = 12.0,
                 Amodule = 1.67,
                 M_Dir=os.path.dirname(os.path.abspath('./ForecastingSystemFiles/M_Data/TrainingSet_1hr.txt')),
                 WF_Dir = os.path.dirname(os.path.abspath('./ForecastingSystemFiles/WF_Data/WF_Data.mat')),
                 WF_Dir_raw = os.path.dirname(os.path.abspath('./ForecastingSystemFiles/WF_Data/Testdata/20160505T000031_hourlyforecast_10days.mat')),
                 plot_opt=0):
      self.StartTimeTi = StartTimeTi #Training Start time
      self.EndTimeTi = EndTimeTi     #Training End time
      self.StartTimePi = StartTimePi #Prediction start time
      self.EndTimePi = EndTimePi     #Prediction end time
      self.UTC=UTC                   #Time zone
      self.lat=lat                   #Latitude
      self.long=long                 #Longitude
      self.alt=alt                   #Altitude
      self.ppeak=ppeak               #PV Peak Power in kW
      self.tilt=tilt                 #degrees i.e., inclination angle of the PV system ( ground == 0 verticle ==90)
      self.azimuth=azimuth           #wrt North (phi_north = 0)
      self.eff=eff                   #initial guess of system's efficiency
      self.No_modules=No_modules     #number of modules
      self.Amodule=Amodule           #area of each module [m2]
      self.M_Dir=M_Dir               #PV measurements minutely and hourly
      self.WF_Dir=WF_Dir             #Weather forecast data processed
      self.WF_Dir_raw=WF_Dir_raw     #Weather forecast data raw
      self.plot_opt=plot_opt         #Switch on plotter


    #######################################
    ### STEP 1: PV Forecast Generate Results
    #######################################
    def pv(self):
        print('Initializing Forecast compiler')
        pcube=TUM.initialize() # Initialize MATLAB Compiler
        det,prob=pcube.forecastPV(self.StartTimeTi,self.EndTimeTi,self.StartTimePi,\
                              self.EndTimePi,self.UTC,self.lat,self.long,self.alt,\
                              self.ppeak,self.tilt,self.azimuth,self.eff,self.No_modules, \
                              self.Amodule,self.WF_Dir,self.WF_Dir_raw,self.M_Dir,self.plot_opt,nargout=2) # Process forecast
        prob=pd.DataFrame(np.matrix(prob)) # Convert the output to a dataframe
        det=pd.DataFrame(np.matrix(det))   # Convert the output to a dataframe
        return {'prob': prob, \
                'det': det}

