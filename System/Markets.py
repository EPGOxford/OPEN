#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPEN Markets module

A Market class defines an upstream market which the EnergySystem is connected
to. Attributes include the network location, prices of imports and exports
over the simulation time-series, the demand charge paid on the maximum demand
over the simulation time-series and import and export power limits. 

The market class has a method which calculates the total revenue associated
with a particular set of real and reactive power profiles over the simulation
time-series.

"""

#import modules
import copy
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import numpy as np
import picos as pic
import matplotlib.pyplot as plt
from datetime import date, timedelta
import os
import requests
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans

__version__ = "1.0.0"

#Market Base Class
class Market:
    """
    A market class to handle prices and other market associated parameters.

    Parameters
    ----------
    bus_id : int
        id number of the bus in the network
    prices_export : numpy.ndarray
        price paid for exports (£/kWh)
    prices_import : numpy.ndarray
        price charged for imports (£/kWh)
    demand_charge : float
        charge for the maximum demand over the time series (£/kWh)
    Pmax : float
        maximum import power over the time series (kW)
    Pmin : float
        minimum import over the time series (kW)
    dt_market : float
        time interval duration (minutes)
    T_market : int
        number of time intervals
    FR_window : int 
        binary value over time series to indicate when frequency response has 
        been offered (0,1)
    FR_capacity : float
        capacity of frequency response offered (kW)
    FR_SOC_max : float
        max SOC at which frequency response can still be fulfilled if needed
    FR_SOC_min : float
        min SOC at which frequency response can still be fulfilled if needed
    FR_price : float
        price per kW capacity per hour avaiable (£/kW.h)
    

    Returns
    -------
    Market


    """
     
    def __init__(self, bus_id, prices_export, prices_import, demand_charge,
                 Pmax, Pmin, dt_market, T_market, FR_window = None,
                 FR_capacity = None, FR_SOC_max = 0.6,
                 FR_SOC_min = 0.4, FR_price = 5/1000, stochastic_date=None, 
                 daily_connection_charge = 0.13):
        #id number of the bus in the network
        self.bus_id = bus_id 
        #price paid for exports (£/kWh)
        self.prices_export = prices_export 
        #price charged for imports (£/kWh)
        self.prices_import = prices_import 
        #charge for the maximum demand over the time series (£/kWh)
        self.demand_charge = demand_charge 
        #maximum import power over the time series (kW)
        self.Pmax = Pmax 
        #minimum import over the time series (kW)
        self.Pmin = Pmin 
        #time interval duration
        self.dt_market = dt_market 
        #number of time intervals
        self.T_market = T_market
        #time window during which frequency response has been offered
        self.FR_window = FR_window
        #capacity of frequency response offered (kW)
        self.FR_capacity = FR_capacity
        #max SOC at which frequency response can still be fulfilled if needed
        self.FR_SOC_max = FR_SOC_max
        #min SOC at which frequency response can still be fulfilled if needed
        self.FR_SOC_min = FR_SOC_min
        #price per kW capacity per hour avaiable (£/kW.h)
        self.FR_price = FR_price
        #cost from energy supplier for daily connection to the grid
        self.daily_connection_charge = daily_connection_charge
        #Total earnings from offering frequency response (initate as 0)
        self.FR_price_tot = 0
        
    
        
    def calculate_revenue(self, P_import_tot, dt):
        """
        Calculate revenue according to simulation results

        Parameters
        ----------
        P_import_tot : float
            Total import power to the site over the time series (kW)
        dt : float
            simulation time interval duration (minutes)
        c_deg_lin : float
            cost of battery degradation associated with each kWh throughput 
            (£/kWh)

        Returns
        -------
        revenue : float
            Total revenue generated during simulation

        """
        #convert import power to the market time-series
        P_import_market = np.zeros(self.T_market)
        for t_market in range(self.T_market):
            t_indexes = (t_market*self.dt_market/dt \
                         + np.arange(0,self.dt_market/dt)).astype(int)
            P_import_market[t_market] = np.mean(P_import_tot[t_indexes])
        #calcuate the revenue
        P_max_demand = np.max(P_import_market)
        P_import = np.maximum(P_import_market,0)
        P_export = np.maximum(-P_import_market,0)
        revenue = -self.demand_charge*P_max_demand+\
                  sum(-1*self.prices_import[t]*P_import[t]*self.dt_market+\
                  +self.prices_export[t]*P_export[t]*self.dt_market\
                                     for t in range(self.T_market))
        if self.FR_window is not None:
            FR_price_tot = self.FR_price*self.FR_capacity*\
            np.count_nonzero(self.FR_window)*self.dt_market
        else: FR_price_tot = 0
        revenue = float(revenue+FR_price_tot)
        return revenue


#-----------------------------------------------------------------------------#
#### NOT needed for OPEN case studies ####
#-----------------------------------------------------------------------------#
#### NOT needed for OPEN case studies ####
class Distribution_Markets:   
    def __init__(self, dt_market):
        self.dt_market = dt_market #time interval duration
        self.DUOS_daily_connection = 12.05
        
    def DUOS_charges(self, T0):
        green = 0.136/100 #£/kWh
        amber = 0.286/100 #£/kWh
        red = 9.75/100 #£/kWh
        DUOS_addition = green*np.ones(48)
        # (https://www.ukpowernetworks.co.uk/internet/en/about-us/duos/)
        DUOS_addition[(7*2):(23*2)] = amber
        # (https://www.ukpowernetworks.co.uk/internet/en/about-us/duos/)
        DUOS_addition[(16*2):(19*2)] = red
        DUOS_addition = np.tile(DUOS_addition,2)
        DUOS_addition = DUOS_addition[T0*2:(T0*2)+48]
        DUOS_addition_week = green*np.ones((48*7))
        DUOS_addition_week[:(48*5)] = np.tile(DUOS_addition, 5)
        return DUOS_addition_week

#### NOT needed for OPEN case studies ####
class Transmission_Markets:  
    def __init__(self, dt_market):
        self.dt_market = dt_market #time interval duration

#    def TRIAD_charge(self, T0, TRIAD_Location_code = 7, TRIAD_time = (17*2)):
#        TRIAD_charges = [11.36, 14.12, 22.87, 28.86, 29.13, 30.57, 32.56, 33.85, 34.48, 30.86, 37.16, 39.96, 38.47, 36.92]
#        TRIAD_charge = TRIAD_charges[TRIAD_Location_code - 1] #£/kW during triad hours
#        AGIC = 3.22 #£/kW
#        Residual = 14.65 #£/kW
#        TRIAD_total_charge = TRIAD_charge + AGIC + Residual
#        TRIAD_addition = np.zeros(48)
#        TRIAD_addition[TRIAD_time - T0] = TRIAD_total_charge
#        return TRIAD_addition
    def TRIAD_charge(self, T0):
#        from Final TNUoS Tarrifs 2019/2020 https://www.nationalgrideso.com/document/137351/download
#        Transm_cost = 2434e6
#        Tot_demand = 51.3e9
#        Cost_per_kW = Transm_cost/Tot_demand
        Cost_per_kW = 49.9
        Triad_add = np.zeros(int(24/self.dt_market))
        Triad_s = int(17/self.dt_market)
        Triad_e = int(19/self.dt_market)
        Triad_add[Triad_s:Triad_e] = Cost_per_kW
        Triad_add = np.tile(Triad_add,2)
        Triad_add = Triad_add[int(T0/self.dt_market):int((T0/self.dt_market)+
                                  (24/self.dt_market))]
        return Triad_add
    def Triad_days(self, days):
        #https://theenergyst.com/national-grid-confirms-2018-19-triads/
        Triad_days = [23, 344, 326]
        return Triad_days

#### NOT needed for OPEN case studies ####   
class Market_set_up:        
    def getLink(self,d1):
        key = ''
        hostname = 'https://api.bmreports.com'
        version_no = 'v1'
        SD = str(d1)
        period = '*'
        service_type = 'csv'
        ipURL = hostname + '/BMRS/MID/' + version_no \
                    + '?APIKey=' + key \
                    + '&FromSettlementDate=' + SD + '&ToSettlementDate=' + SD \
                    + '&Period=' + period \
                    + '&ServiceType=' + service_type
        return ipURL
    
    def dlCSV(self,ipDate):
        ipLink = self.getLink(ipDate)
        print('Checking date: ' + str(ipDate))
        filename = 'MID_' + str(ipDate) + '.csv'
        exists = os.path.isfile('data/MID/' + filename)
        if not exists:
            link = requests.get(ipLink)
            with open(os.path.join('data/MID/', filename), 'wb') as f:
                content = link.content
                f.write(content[22:])
                print(filename)
                
    def vectorise(self,x):
        n = x.size
        res = np.zeros((n,1))
        for i in range(n):
            res[i] = float( x[i] )
        return res
            
    def dropZeros(self,df):
        a = df[4].values
        if 0 in a:
            idx = np.where(a==0)
            df.drop(df.index[idx], inplace=True)            
        return df
        
    def handleMissing(self,ipDate):
        print('\n========\nINSIDE MISSING\n=========\n')
        
        self.dlCSV(ipDate)
        filename = 'MID_' + str(ipDate) + '.csv'
        data1 = pd.read_csv('data/MID/' + filename, header=None,
                            usecols=[1,2,3,4])
        apxData = data1[data1[1].str.contains('APXMIDP')]
        apxData = self.dropZeros(apxData)
            
        self.dlCSV(ipDate - timedelta(1))
        filename = 'MID_' + str(ipDate - timedelta(1)) + '.csv'
        dataPrev = pd.read_csv('data/MID/' + filename, header=None,
                               usecols=[1,2,3,4])
        apxPrev = dataPrev[dataPrev[1].str.contains('APXMIDP')]
        apxPrev = self.dropZeros(apxPrev)
        
        self.dlCSV(ipDate + timedelta(1))
        filename = 'MID_' + str(ipDate + timedelta(1)) + '.csv'
        dataNext = pd.read_csv('data/MID/' + filename, header=None,
                               usecols=[1,2,3,4])
        apxNext = dataNext[dataNext[1].str.contains('APXMIDP')]
        apxNext = self.dropZeros(apxNext)
        
        apxTrio = apxPrev;
        apxTrio = apxTrio.append(apxData, ignore_index=True)
        apxTrio = apxTrio.append(apxNext, ignore_index=True)
        
        spPrev = self.vectorise(apxPrev[3].values)
        spCurr = self.vectorise(apxData[3].values)
        spNext = self.vectorise(apxNext[3].values)
        
        spPrev = spPrev - 1
        spCurr = spCurr + 47
        spNext = spNext + 95
        
        spStack = np.concatenate((spPrev,spCurr,spNext),axis=0)        
        apxTrio = apxTrio.drop(3,1)
        apxTrio.insert(loc=2, column=3, value=spStack)
        
        sp,mip = self.fillMissing(apxTrio)
        
        return (mip)
    
    def fillMissing(self,apx):    
        sp = apx[3].values
        mip = apx[4].values    
        
        f = interp1d(sp,mip)
        sp_new = np.arange(48., 96.)
        mip_new = f(sp_new)
        
        sp_actual = np.arange(0., 48.)
        
        return (sp_actual, mip_new)
        
        
    def getMIP(self,pDate,delays):
        
        nDays = len(delays)
        prices = np.zeros((nDays,48))
        
        for i in range(nDays):
            
            ipDelay = delays[i]
            ipDate = pDate - timedelta(ipDelay)
            
            self.dlCSV(ipDate)
            filename = 'MID_' + str(ipDate) + '.csv'
            data1 = pd.read_csv('data/MID/' + filename, header=None,
                                    usecols=[1,2,3,4])
            apxData = data1[data1[1].str.contains('APXMIDP')]
    
            mip = apxData[4].values
            nrows = len(apxData.index)
            if nrows != 48 or 0 in mip:
                mip = self.handleMissing(ipDate)
            else:
                mip = apxData[4].values
            
            for j in range(48):
                prices[i,j] = mip[j]
        
        return prices

