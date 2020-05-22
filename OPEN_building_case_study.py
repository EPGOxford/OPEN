#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The building energy management case study focuses on a building with a flexible
HVAC unit which is controlled in order to minimise costs, with the constraint
that the internal temperature remains between 16 and 18 degrees C.
"""

#import modules
import os
from os.path import normpath, join
import copy
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import numpy as np
import picos as pic
import matplotlib.pyplot as plt
from datetime import date, timedelta

import System.Assets as AS
import System.Markets as MK
import System.EnergySystem as ES

import sys

print('Code started.')
#plt.close('all')

############## VERSION ##############


__version__ = "1.1.0"
        
#######################################
###       
### Case Study: Building HVAC flexibility
###        
#######################################

path_string = normpath('Results/Building_Case_Study/')
if not os.path.isdir(path_string):
    os.makedirs(path_string)
        
#######################################
### STEP 0: Load Data
#######################################

choice = 0
while choice not in ['1','2']:
    choice=input('Input season number: \n1 for Summer \n2 for Winter\n');
if choice == '1':
    winterFlag = False
else:
    winterFlag = True

PV_data_path = os.path.join("Data/Building/", "PVpu_1min_2014JAN.csv")    
PVpu_raw_wtr = pd.read_csv(PV_data_path, index_col=0, parse_dates=True).values
Loads_data_path = os.path.join("Data/Building/", "Loads_1min_2014JAN.csv")    
Loads_raw_wtr = pd.read_csv(Loads_data_path, index_col=0, parse_dates=True).values

PV_data_path = os.path.join("Data/Building/", "PVpu_1min_2013JUN.csv")    
PVpu_raw_smr = pd.read_csv(PV_data_path, index_col=0, parse_dates=True).values
Loads_data_path = os.path.join("Data/Building/", "Loads_1min_2013JUN.csv")    
Loads_raw_smr = pd.read_csv(Loads_data_path, index_col=0, parse_dates=True).values

PVtotal_smr = np.sum(PVpu_raw_smr,1)
PVtotal_wtr = np.sum(PVpu_raw_wtr,1)

if winterFlag == False:
    PVpu = PVtotal_smr/np.max(PVtotal_smr)
else:
    PVpu = PVtotal_wtr/np.max(PVtotal_smr)
Loads = Loads_raw_smr

#######################################
### STEP 1: setup parameters
#######################################

dt = 1/60 #1 minute time intervals
T = int(24/dt) #Number of intervals
dt_ems = 15/60 #30 minute EMS time intervals
T_ems = int(T*dt/dt_ems) #Number of EMS intervals
T0 = 8 #from 8 am to 8 am
Ppv_nom = 400 #power rating of the PV generation
#Electric Vehicle (EV) parameters
N_EVs = 120 #number of EVs
Emax_EV = 30  #maximum EV energy level
Emin_EV = 0 #minimum EV energy level
P_max_EV = 7 #maximum EV charging power
P_min_EV = 0 #minimum EV charging power
np.random.seed(1000)
E0_EVs = Emax_EV*np.random.uniform(0,1,N_EVs) #random EV initial energy levels 
ta_EVs = np.random.randint(12*2,22*2,N_EVs) - T0*2 #random EV arrival times between 12pm and 10pm
td_EVs = np.random.randint(29*2,32*2+1,N_EVs) - T0*2 #random EV departure times 5am and 8am
#Ensure EVs can be feasibility charged
for i in range(N_EVs):
    td_EVs[i] = np.max([td_EVs[i],ta_EVs[i]])
    E0_EVs[i] = np.max([E0_EVs[i],Emax_EV-P_max_EV*(td_EVs[i]-ta_EVs[i])])
#Building parameters
Tmax = 18 # degree celsius
Tmin = 16 # degree celsius
T0 = 17 # degree centigrade
heatmax = 90 #kW Max heat supplied
coolmax = 200 #kW Max cooling
CoP_heating = 3# coefficient of performance - heating
CoP_cooling = 1# coefficient of performance - cooling
#Parameters from MultiSAVES
C = 500 # kWh/ degree celsius
R = 0.0337 #degree celsius/kW
#Market parameters
dt_market = dt_ems #market and EMS have the same time-series
T_market = T_ems #market and EMS have same length
prices_export = 0.04*np.ones(T_market) #money received of net exports
prices_import = np.hstack((0.07*np.ones(int(T_market*7/24)), \
                          0.15*np.ones(int(T_market*17/24)))) #price of net imports
demand_charge = 0.10 #price per kW for the maximum demand
Pmax_market = 500*np.ones(T_market) #maximum import power
Pmin_market = -500*np.ones(T_market) #maximum export power

#######################################
### STEP 2: setup the network
#######################################

#(from https://github.com/e2nIEE/pandapower/blob/master/tutorials/minimal_example.ipynb)
network = pp.create_empty_network()
#create buses 
bus1 = pp.create_bus(network, vn_kv=20., name="bus 1")
bus2 = pp.create_bus(network, vn_kv=0.4, name="bus 2")
bus3 = pp.create_bus(network, vn_kv=0.4, name="bus 3")
#create bus elements
pp.create_ext_grid(network, bus=bus1, vm_pu=1.0, name="Grid Connection")
#create branch elements
trafo = pp.create_transformer(network, hv_bus=bus1, lv_bus=bus2, std_type="0.4 MVA 20/0.4 kV", name="Trafo")
line = pp.create_line(network, from_bus=bus2, to_bus=bus3, length_km=0.1, std_type="NAYY 4x50 SE", name="Line")
N_buses = network.bus['name'].size

#######################################
### STEP 3: setup the assets 
#######################################

#initiate empty lists for different types of assets
storage_assets = []
building_assets = []
nondispatch_assets = []

#PV source at bus 3
Pnet = -PVpu*Ppv_nom #100kW PV plant
Qnet = np.zeros(T)
PV_gen_bus3 = AS.NondispatchableAsset(Pnet, Qnet, bus3, dt, T)
nondispatch_assets.append(PV_gen_bus3)

#Load at bus 3
Pnet = np.sum(Loads,1) #summed load across 120 households
Qnet = np.zeros(T)
load_bus3 = AS.NondispatchableAsset(Pnet, Qnet, bus3, dt, T)
nondispatch_assets.append(load_bus3)

#Building asset at bus 3
Tmax_bldg_i = Tmax*np.ones(T_ems)
Tmin_bldg_i = Tmin*np.ones(T_ems)
Hmax_bldg_i = heatmax
Cmax_bldg_i = coolmax
T0_i = T0
C_i = C
R_i = R
CoP_heating_i = CoP_heating
CoP_cooling_i = CoP_cooling
if winterFlag == True:
    Ta_i = 10*np.ones(T_ems)
else:
    Ta_i = 22*np.ones(T_ems)
bus_id_bldg_i = bus3
bldg_i = AS.BuildingAsset(Tmax_bldg_i, Tmin_bldg_i, Hmax_bldg_i, Cmax_bldg_i, T0_i, C_i, R_i, CoP_heating_i, CoP_cooling_i, Ta_i, bus_id_bldg_i, dt, T, dt_ems, T_ems)
building_assets.append(bldg_i)
N_BLDGs = len(building_assets)
    
#######################################
### STEP 4: setup the market
#######################################
    
bus_id_market = bus1
market = MK.Market(bus_id_market, prices_export, prices_import, demand_charge, Pmax_market, Pmin_market, dt_market, T_market)

#######################################
#STEP 5: setup the energy system
#######################################

energy_system = ES.EnergySystem(storage_assets, nondispatch_assets, network, market, dt, T, dt_ems, T_ems, building_assets)

#######################################
### STEP 6: simulate the energy system: 
#######################################

output = energy_system.simulate_network()
#output = energy_system.simulate_network_bldg()

buses_Vpu = output['buses_Vpu']
buses_Vang = output['buses_Vang']
buses_Pnet = output['buses_Pnet']
buses_Qnet = output['buses_Qnet']
Pnet_market = output['Pnet_market']
Qnet_market = output['Qnet_market']
buses_Vpu = output['buses_Vpu']
P_import_ems = output['P_import_ems']
P_export_ems = output['P_export_ems']
P_BLDG_ems = output['P_BLDG_ems']
P_demand_ems = output['P_demand_ems']

P_demand_base = np.zeros(T)
for i in range(len(nondispatch_assets)):
    bus_id = nondispatch_assets[i].bus_id
    P_demand_base += nondispatch_assets[i].Pnet


#######################################
### STEP 7: plot results
#######################################

#x-axis time values
time = dt*np.arange(T)
time_ems = dt_ems*np.arange(T_ems)
timeE = dt*np.arange(T+1)

#Print revenue generated
revenue = market.calculate_revenue(-Pnet_market,dt)
print('Net Revenue: £ ' + str(revenue) )

#Plot the base demand and the total imported power
plt.figure(num=None, figsize=(6, 3), dpi=80, facecolor='w', edgecolor='k')
plt.plot(time,P_demand_base,'--',label='Base Demand')
plt.plot(time,Pnet_market,label='Total Power Imported')
plt.ylabel('Power (kW)')
plt.xlabel('Time (h)')
plt.xlim(0, max(time))
plt.legend()
plt.grid(True,alpha=0.5)
plt.tight_layout()

Pnet_market_30min_avg = np.zeros(T_ems)
for t_ems in range(T_ems):
    t_indexes = (t_ems*dt_ems/dt + np.arange(0,dt_ems/dt)).astype(int)
    Pnet_market_30min_avg[t_ems] = np.mean(Pnet_market[t_indexes])

plt.figure(figsize=(6, 12), dpi=80, facecolor='w')
plt.subplot(4,1,1)
plt.plot(time_ems,market.prices_import)
plt.grid(True,alpha=0.5)
plt.ylabel('Price (£/kWh)')
plt.xlabel('Time (h)')
plt.xlim(0, max(time_ems))
ax0 = plt.subplot(4,1,2)
plt.plot(time_ems,P_demand_ems,label='Base Demand (EMS)')
plt.plot(time_ems,P_demand_ems+np.sum(P_BLDG_ems,1),label='Total Demand (EMS)')
plt.ylabel('Overall Power (kW)')
plt.xlabel('Time (h)')
plt.xlim(0, max(time_ems))
plt.legend()
plt.grid(True,alpha=0.5)
ax1=plt.subplot(4, 1, 3)
for i in range(N_BLDGs):
    ax1.plot(time,building_assets[i].Pnet,color='C0')
ax1.set_ylabel('HVAC Power (kW)')
ax1.set_xlabel('Time (h)')
plt.grid(True,alpha=0.5)
ax2=plt.subplot(4, 1, 4)
for i in range(N_BLDGs):
    ax2.plot(time_ems,building_assets[i].T_int,color='C0')
ax2.plot(time_ems,building_assets[i].Tmax*np.ones(T_ems),'k:')
ax2.plot(time_ems,building_assets[i].Tmin*np.ones(T_ems),'k:')
ax2.set_ylabel('Internal\nTemperature ($^{o}C$)')
ax2.set_xlabel('Time (h)')
ax0.set_xlim(0, max(time_ems))
ax0.set_ylim(-300, 300)
ax1.set_xlim(0, max(time))
ax1.set_ylim(0, coolmax*1.2)
ax2.set_xlim(0, max(time_ems))
ax2.set_ylim(15.5, 18.5)
plt.grid(True,alpha=0.5)
plt.tight_layout()

#plt.figure(num=None, figsize=(6, 3), dpi=80, facecolor='w', edgecolor='k') #6,3.75
#plt.plot(time_ems,P_demand_ems,label='Base Demand (EMS)')
#plt.plot(time_ems,P_demand_ems+np.sum(P_BLDG_ems,1),label='Total Demand (EMS)')
#plt.xticks([0,8,16,23.75],('00:00', '08:00', '16:00', '00:00'))
#plt.xlabel('Time (hh:mm)')
#plt.xlim(0, max(time_ems))
#plt.ylabel('Power (kW)')
#plt.yticks(np.arange(-500, 500, step=100))
#plt.ylim([-400,300])
#plt.grid(True,alpha=0.5)
#plt.legend(loc = 'top right')
#plt.grid(alpha=0.5)
#plt.show ()
#plt.tight_layout()

#Final plots
#Base and total demand
if not winterFlag:
    season_str = '_summer'
else:
    season_str = '_winter'

save_suffix = '.pdf'

plt.figure(num=None, figsize=(6, 2.5), dpi=80, facecolor='w', edgecolor='k') #6,3.75
plt.plot(time,P_demand_base,'--',label='Base Demand')
plt.plot(time,Pnet_market,label='Total Demand')
plt.xticks([0,8,16,max(time)],('00:00', '08:00', '16:00', '00:00'))
plt.xlabel('Time (hh:mm)')
plt.xlim(0, max(time))
plt.ylabel('Power (kW)')
plt.yticks(np.arange(-500, 500, step=100))
plt.ylim([-400,300])
plt.grid(True,alpha=0.5)
plt.legend(loc = 'lower right')
plt.grid(alpha=0.5)
plt.show ()
plt.tight_layout()
plt.savefig(join(path_string, normpath('Demand' + season_str + save_suffix)),
            bbox_inches='tight')
#HVAC Power
if not winterFlag:
    plt.figure(num=None, figsize=(6, 2.5), dpi=80, facecolor='w', edgecolor='k') #6,3.75
    for i in range(N_BLDGs):
        plt.plot(time,building_assets[i].Pnet,color='C0',label='HVAC Cooling',zorder=10)
    plt.hlines(coolmax,0,max(time_ems),linestyle = ':',label = 'Maximum',color = 'red', zorder=11,)
    plt.ylabel('Power (kW)')
    plt.grid(True,alpha=0.5)
    plt.xticks([0,8,16,23.75],('00:00', '08:00', '16:00', '00:00'))
    plt.xlabel('Time (hh:mm)')
    plt.xlim(0, max(time_ems))
    plt.ylim(0, coolmax*1.5)
    plt.grid(True,alpha=0.5)
    plt.legend(loc = 'upper right')
    plt.grid(alpha=0.5)
    plt.show ()
    plt.tight_layout()
    plt.savefig(join(path_string, normpath('HVAC'  + season_str + save_suffix)),
                bbox_inches='tight')
else:
    plt.figure(num=None, figsize=(6, 2.5), dpi=80, facecolor='w', edgecolor='k') #6,3.75
    for i in range(N_BLDGs):
        plt.plot(time,building_assets[i].Pnet,color='C0',label='HVAC Heating',zorder=10)
    plt.hlines(heatmax,0,max(time_ems),linestyle = ':',label = 'Maximum',color = 'red', zorder=11)
    plt.ylabel('Power (kW)')
    plt.grid(True,alpha=0.5)
    plt.xticks([0,8,16,23.75],('00:00', '08:00', '16:00', '00:00'))
    plt.xlabel('Time (hh:mm)')
    plt.xlim(0, max(time_ems))
    plt.ylim(0, heatmax*1.5)
    plt.grid(True,alpha=0.5)
    plt.legend(loc = 'upper right')
    plt.grid(alpha=0.5)
    plt.show ()
    plt.tight_layout()
    plt.savefig(join(path_string, normpath('HVAC'  + season_str + save_suffix)),
                bbox_inches='tight')

#Temperature
plt.figure(num=None, figsize=(6, 2.5), dpi=80, facecolor='w', edgecolor='k') #6,3.75
for i in range(N_BLDGs):
    plt.plot(time_ems,building_assets[i].T_int,color='C0',label = 'Temperature')
plt.plot(time_ems,building_assets[i].Tmax*np.ones(T_ems),'r:',linestyle = ':', zorder=11,label = 'Limits')
plt.plot(time_ems,building_assets[i].Tmin*np.ones(T_ems),'r:',linestyle = ':', zorder=11)
plt.xticks([0,8,16,23.75],('00:00', '08:00', '16:00', '00:00'))
plt.xlabel('Time (hh:mm)')
plt.xlim(0, max(time_ems))
plt.ylabel('Temperature ($^{o}C$)')
plt.xlabel('Time (hh:mm)')
plt.grid(True,alpha=0.5)
plt.legend(loc = 'center right')
plt.grid(alpha=0.5)
plt.show ()
plt.tight_layout()
plt.savefig(join(path_string, normpath('Temp'  + season_str + save_suffix)),
            bbox_inches='tight')



#Plot to check that for each interval of the EMS either P_import = 0 or P_exports = 0
plt.figure(num=None, figsize=(6, 3), dpi=80, facecolor='w', edgecolor='k')
plt.plot(time_ems,P_import_ems,label='Imported Power (EMS)')
plt.plot(time_ems,P_export_ems,label='Exported Power (EMS)')
plt.ylabel('Power (kW)')
plt.xlabel('Time (h)')
plt.legend()
plt.grid(True,alpha=0.5)
plt.tight_layout()

