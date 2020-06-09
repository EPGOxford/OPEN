#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Electric Vehicle Smart Charging case study considers the smart charging of
EVs within an unbalanced three-phase distribution network.

The case study considers a business park where 80 EVs are charged at 6.6 kW
charge points.

The objective is to charge all of the vehicles to their maximum energy level
prior to departure, at lowest cost.
"""

#import modules
import os
from os.path import normpath, join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from System.Network_3ph_pf import Network_3ph
import System.Assets as AS
import System.Markets as MK
import System.EnergySystem as ES

############## VERSION ##############

__version__ = "1.0.0"
        
################################################
###       
### Case Study: Electric Vehicle Smart Charging
###        
################################################
 
################################################
### RUN OPT OR JUST PLOT (IF RESULTS PICKLED)
################################################

run_opt = 1
opt_type = ['open_loop', 'mpc']

path_string = normpath('Results/EV_Case_Study/')
if not os.path.isdir(path_string):
    os.makedirs(path_string)
save_suffix = '.pdf'

def figure_plot(x, N_EVs, P_demand_base_pred_ems, P_compare, P_demand_base,\
                Pnet_market, storage_assets, N_ESs,\
                nondispatch_assets, time_ems, time, timeE, buses_Vpu):   
    
    # plot half hour predicted and actual net load
    title = '' #str(x)
    plt.figure(num=None, figsize=(6, 2.5), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(time_ems,P_demand_base_pred_ems,label=\
             'Predicted net load, 30 mins')
    plt.plot(time_ems,P_compare, label =\
             'Predicted net load + EVs charging, 30 mins')
    plt.ylabel('Power (kW)')
    plt.ylim(0, 2100)
    plt.xticks([0,8,16,23.75],('00:00', '08:00', '16:00', '00:00'))
    plt.xlabel('Time (hh:mm)')
    plt.xlim(0, max(time_ems))
    plt.grid(True,alpha=0.5)
    plt.legend()
    plt.tight_layout()
    ax = plt.gca()
    plt.text(0.02, 0.9, title, transform=ax.transAxes, fontsize=12)
    plt.savefig(join(path_string, normpath('P_ems_'  + str(x) + save_suffix)),
                bbox_inches='tight')

    # plot 5 minute predicted and actual net load
    plt.figure(num=None, figsize=(6, 2.5), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(time,P_demand_base,'--',label=\
             'Base Load')
    plt.plot(time,Pnet_market,label=\
             'Import Power')
    plt.ylabel('Power (kW)')
    plt.ylim(500, 2100)
    plt.xticks([0,8,16,23.916],('00:00', '08:00', '16:00', '00:00'))
    plt.xlabel('Time (hh:mm)')
    plt.xlim(0, max(time))
    plt.grid(True,alpha=0.5)
    plt.legend()
    plt.tight_layout()
    ax = plt.gca()
    plt.text(0.02, 0.9, title, transform=ax.transAxes, fontsize=12)
    plt.savefig(join(path_string, normpath('P_actual_'  + str(x) + save_suffix)),
                bbox_inches='tight')
    
    # plot power for EV charging
    plt.figure(num=None, figsize=(6, 2.5), dpi=80, facecolor='w', edgecolor='k')
    #plt.plot(time,sum(storage_assets[i].Pnet for i in range(N_ESs)))
    for i in range(N_EVs):
        plt.plot(time,storage_assets[i].Pnet)
    plt.xlim(0,24)
    plt.ylim(0,10)
    plt.ylabel('Power (kW)')
    plt.xticks([0,8,16,23.916],('00:00', '08:00', '16:00', '00:00'))
    plt.xlabel('Time (hh:mm)')
    plt.xlim(0, max(time))
    plt.grid(True,alpha=0.5)
    ax = plt.gca()
    plt.tight_layout()
    plt.text(0.02, 0.9, title, transform=ax.transAxes, fontsize=12)
    plt.savefig(join(path_string, normpath('P_EVs_'  + str(x) + save_suffix)),
                bbox_inches='tight')
    
    # plot average battery energy
    plt.figure(num=None, figsize=(6, 2.5), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(timeE,sum(storage_assets[i].E for i in range(N_ESs))/N_EVs)
    plt.ylabel('Average EV Energy (kWh)')
    plt.xticks([0,8,16,23.916],('00:00', '08:00', '16:00', '00:00'))
    plt.yticks(np.arange(0,37,4))
    plt.ylim(12, 36)
    plt.xlabel('Time (hh:mm)')
    plt.xlim(0, max(time))
    plt.grid(True,alpha=0.5)
    ax = plt.gca()
    plt.tight_layout()
    plt.text(0.02, 0.9, title, transform=ax.transAxes, fontsize=12)
    plt.savefig(join(path_string, normpath('E_EVs_'  + str(x) + save_suffix)),
                bbox_inches='tight')
    
    # plot line voltages
    plt.figure(num=None, figsize=(6, 2.5), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(time,np.min(buses_Vpu[:,buses_Vpu[0,:,0]>0,0],1),'-',label='Phase A')
    plt.plot(time,np.min(buses_Vpu[:,buses_Vpu[0,:,1]>0,1],1),'--',label='Phase B')
    plt.plot(time,np.min(buses_Vpu[:,buses_Vpu[0,:,2]>0,2],1),'-.',label='Phase C')
    plt.hlines(0.95,0,24,'r',':','Lower Limit')
    plt.ylabel('Minimum Voltage Mag. (pu)')
    plt.ylim(0.94, 1.00)
    plt.yticks(np.arange(0.95, 1.00, step=0.01))
    plt.xticks([0,8,16,23.916],('00:00', '08:00', '16:00', '00:00'))
    plt.xlabel('Time (hh:mm)')
    plt.xlim(0, max(time))
    plt.grid(True,alpha=0.5)
    plt.legend()
    plt.tight_layout()
    ax = plt.gca()
    plt.text(0.02, 0.9, title, transform=ax.transAxes, fontsize=12)
    plt.savefig(join(path_string, normpath('Vmin_'  + str(x) + save_suffix)),
                bbox_inches='tight')


if run_opt ==1:
           
    #######################################
    ### STEP 0: Load Data
    #######################################
            
    PV_data_path = os.path.join("Data", "PVpu_1min.csv")    
    PVpu_raw = pd.read_csv(PV_data_path, index_col=0, parse_dates=True).values[:,0]
    
    substation_data = pickle.load(open(os.path.join\
                                       ("Data","substation_daily_PQ_data.p"),'rb'))
    T_5min_sub_data = substation_data[0]
    N_days_sub_data = substation_data[1]
    P_raw_days_sub_data = 1*substation_data[2]
    Q_raw_days_sub_data = 1*substation_data[3]
    
    #######################################
    ### STEP 1: setup parameters
    #######################################
    
    dt = 5/60  # 5 minute time intervals
    T = int(24/dt)  # Number of time intervals
    dt_ems = 30/60  # 30 minute EMS time intervals
    T_ems = int(T*dt/dt_ems)  # Number of EMS intervals
    T0 = 0  # from 12 am to 12 am
    N_PVs = 5  # Number of PVs
    P_pv = 200*np.ones(N_PVs)  # PV rated power (kW)
    PV_bus_names = ['634','645','652','671','675']
    PV_phases = [[0,1,2],[1],[0],[0,1,2],[0,1,2]]  # phases at each bus
    
    # Electric Vehicle (EV) parameters
    N_EVs = 80  # number of EVs
    #N_EVs = 2  # number of EVs
    Emax_EV = 36  # maximum EV energy level
    Emin_EV = 0  # minimum EV energy level
    P_max_EV = 6.6  # maximum EV charging power
    P_min_EV = 0  # minimum EV charging power
    
    # EV charge/discharge efficiency
    eff_EV = np.ones(100)
    eff_EV[0:50] = 0.6
    eff_EV[50:70] = 0.75
    eff_EV[70:100] = 0.8
    eff_EV_opt = 1  # fixed efficiency for EVs to use in optimiser
    
    # EV arrival & departure times and energy levels on arrival
    np.random.seed(1000)
    # random EV initial energy levels
    E0_EVs = Emax_EV*np.random.uniform(0.2,0.9,N_EVs)
    # random EV arrival times between 6am and 9am
    ta_EVs = np.random.randint(int(6/dt_ems),int(10/dt_ems),N_EVs) - int(T0/dt_ems)
    # random EV departure times between 5pm and 9pm
    td_EVs = np.random.randint(int(15/dt_ems),\
                               int(21/dt_ems),N_EVs) - int(T0/dt_ems)
    
    # Market parameters
    # market and EMS have the same time-series
    dt_market = dt_ems
    T_market = T_ems
    
    # Import and Export Prices
    prices_export = 0.05*np.ones(T_market)  #(£/kWh)
    prices_import = 0.15*np.ones(T_market)  #(£/kWh)
    demand_charge = 0.1 # (£/kW) for the maximum demand
    
    # Site Power Constraints
    Pmax_market = 100e3*np.ones(T_market)
    Pmin_market = -100e3*np.ones(T_market)
    
    # PV data set up
    N_sub_data = P_raw_days_sub_data.shape[1]
    P_sub0 = np.zeros([T,N_sub_data])
    Q_sub0 = np.zeros([T,N_sub_data])
    P_sub = np.zeros([T,N_sub_data])
    Q_sub = np.zeros([T,N_sub_data])
    dt_raw = 1/60  # 1 minute time intervals
    T_raw = int(24/dt_raw)  # Number of data time intervals
    dt_sub_raw = 5/60  # 5 minute time intervals
    T_sub_raw = int(24/dt_raw)  # Number of data time intervals
    PVpu_8am = np.zeros(T)
    for t in range(T):
        t_raw_indexes = (t*dt/dt_raw + np.arange(0,dt/dt_raw)).astype(int)
        t_sub_indexes = (t*dt/dt_sub_raw + np.arange(0,dt/dt_sub_raw)).astype(int)
        PVpu_8am[t] = np.mean(PVpu_raw[t_raw_indexes])
        P_sub[t,:] = np.mean(P_raw_days_sub_data[t_sub_indexes,:],0)
        Q_sub[t,:] = np.mean(Q_raw_days_sub_data[t_sub_indexes,:],0)
    
    # Shift PV to 12am from 8am start time
    PVpu = np.zeros(T)
    for t in range(T):
        t_sub0 = int((t-8/dt)%T)
        PVpu[t] = PVpu_8am[t_sub0]
    
    #######################################
    ### STEP 2: setup the network
    #######################################
    
    # from https://github.com/e2nIEE/pandapower/blob/
    # master/tutorials/minimal_example.ipynb
    network = Network_3ph()  # IEEE 13 bus by default
    network.capacitor_df = network.capacitor_df[0:0] #removes the capacitors
    network.update_YandZ()
    
    # set bus voltage limits
    network.set_pf_limits(0.95*network.Vslack_ph, 1.05*network.Vslack_ph,
                          2000e3/network.Vslack_ph)
    
    # set up busses
    bus650_num = network.bus_df[network.bus_df['name']=='650'].number.values[0]
    bus634_num = network.bus_df[network.bus_df['name']=='634'].number.values[0]
    bus645_num = network.bus_df[network.bus_df['name']=='645'].number.values[0]
    bus646_num = network.bus_df[network.bus_df['name']=='646'].number.values[0]
    bus652_num = network.bus_df[network.bus_df['name']=='652'].number.values[0]
    bus671_num = network.bus_df[network.bus_df['name']=='671'].number.values[0]
    bus675_num = network.bus_df[network.bus_df['name']=='675'].number.values[0]
    phase_array = np.array([0,1,2])
    N_buses = network.N_buses  # Number of buses
    N_phases = network.N_phases  # Number of phases
    N_load_bus_phases = N_phases*(N_buses-1)  # Number of load buses 
    N_lines = network.N_lines  # Number lines
    N_line_phases = N_lines*N_phases 
    
    
    #######################################
    ### STEP 3: setup the assets 
    #######################################
    
    storage_assets = []
    nondispatch_assets = []
    smooth = True
    
    # Method to smooth actual data to generate equivalent of predicted data
    def smoothing(Pnet, Qnet):
        h = 20
        m = len(Pnet)
        Pnet_pred = np.zeros(m)
        Qnet_pred = np.zeros(m)
        P_cont = np.tile(Pnet,2)
        Q_cont = np.tile(Qnet,2)
        for i in range(m):
            Pnet_pred[i] = sum(P_cont[i:i+h])/(h) 
            Qnet_pred[i] = sum(Q_cont[i:i+h])/(h)
        return{"Pnet_pred": Pnet_pred, "Qnet_pred": Qnet_pred}
    
    # Create loads
    sub_load_index = 0
    # Create loads at bus 634
    for ph_i in range(3):
        Pnet = P_sub[:,sub_load_index]
        Qnet = Q_sub[:,sub_load_index]
        Pnet_pred = None
        Qnet_pred = None
        if smooth ==True:
            out = smoothing(Pnet,Qnet)
            Pnet_pred = out['Pnet_pred']
            Qnet_pred = out['Qnet_pred']
        ND_load_ph = AS.NondispatchableAsset_3ph(Pnet, Qnet, bus634_num, ph_i, dt,
                                                 T, Pnet_pred = Pnet_pred,
                                                 Qnet_pred = Qnet_pred)
        nondispatch_assets.append(ND_load_ph)
        sub_load_index += 1
    # Create loads at bus 645
    for ph_i in [1]:
        Pnet = P_sub[:,sub_load_index]
        Qnet = Q_sub[:,sub_load_index]
        Pnet_pred = Pnet
        Qnet_pred = Qnet
        if smooth ==True:
            out = smoothing(Pnet,Qnet)
            Pnet_pred = out['Pnet_pred']
            Qnet_pred = out['Qnet_pred']
        ND_load_ph = AS.NondispatchableAsset_3ph(Pnet, Qnet, bus645_num, ph_i, dt,
                                                 T, Pnet_pred = Pnet_pred,
                                                 Qnet_pred = Qnet_pred)
        nondispatch_assets.append(ND_load_ph)
        sub_load_index += 1
    # Create loads at bus 646
    for ph_i in [1]:
        Pnet = P_sub[:,sub_load_index]
        Qnet = Q_sub[:,sub_load_index]
        Pnet_pred = Pnet
        Qnet_pred = Qnet
        if smooth ==True:
            out = smoothing(Pnet,Qnet)
            Pnet_pred = out['Pnet_pred']
            Qnet_pred = out['Qnet_pred']
        ND_load_ph = AS.NondispatchableAsset_3ph(Pnet, Qnet, bus645_num, ph_i, dt,
                                                 T, Pnet_pred = Pnet_pred,
                                                 Qnet_pred = Qnet_pred)
        nondispatch_assets.append(ND_load_ph)
        sub_load_index += 1
    # Create loads at bus 652
    for ph_i in [0]:
        Pnet = P_sub[:,sub_load_index]
        Qnet = Q_sub[:,sub_load_index]
        Pnet_pred = Pnet
        Qnet_pred = Qnet
        if smooth ==True:
            out = smoothing(Pnet,Qnet)
            Pnet_pred = out['Pnet_pred']
            Qnet_pred = out['Qnet_pred']
        ND_load_ph = AS.NondispatchableAsset_3ph(Pnet, Qnet, bus652_num, ph_i, dt,
                                                 T, Pnet_pred = Pnet_pred,
                                                 Qnet_pred = Qnet_pred)
        nondispatch_assets.append(ND_load_ph)
        sub_load_index += 1
    # Create loads at bus 671
    for ph_i in range(3):
        for k in range(1):
            Pnet = P_sub[:,sub_load_index]
            Qnet = Q_sub[:,sub_load_index]
            Pnet_pred = Pnet
            Qnet_pred = Qnet
            if smooth ==True:
                out = smoothing(Pnet,Qnet)
                Pnet_pred = out['Pnet_pred']
                Qnet_pred = out['Qnet_pred']
            ND_load_ph = AS.NondispatchableAsset_3ph(Pnet, Qnet, bus671_num, ph_i,
                                                     dt, T, Pnet_pred = Pnet_pred,
                                                     Qnet_pred = Qnet_pred)
            nondispatch_assets.append(ND_load_ph)
            sub_load_index += 1
    # Create loads at bus 675 (3->a, 1->b, 2->c)
    for ph_i in [0]:
        for k in range(1):
            Pnet = P_sub[:,sub_load_index]
            Qnet = Q_sub[:,sub_load_index]
            Pnet_pred = Pnet
            Qnet_pred = Qnet
            if smooth ==True:
                out = smoothing(Pnet,Qnet)
                Pnet_pred = out['Pnet_pred']
                Qnet_pred = out['Qnet_pred']
            ND_load_ph = AS.NondispatchableAsset_3ph(Pnet, Qnet, bus675_num, ph_i,
                                                     dt, T, Pnet_pred = Pnet_pred,
                                                     Qnet_pred = Qnet_pred)
            nondispatch_assets.append(ND_load_ph)
            sub_load_index += 1
    for ph_i in [1]:
        Pnet = P_sub[:,sub_load_index]
        Qnet = Q_sub[:,sub_load_index]
        Pnet_pred = Pnet
        Qnet_pred = Qnet
        if smooth ==True:
            out = smoothing(Pnet,Qnet)
            Pnet_pred = out['Pnet_pred']
            Qnet_pred = out['Qnet_pred']
        ND_load_ph = AS.NondispatchableAsset_3ph(Pnet, Qnet, bus675_num, ph_i, dt,
                                                 T, Pnet_pred = Pnet_pred,
                                                 Qnet_pred = Qnet_pred)
        nondispatch_assets.append(ND_load_ph)
        sub_load_index += 1
    for ph_i in [2]:
        for k in range(1):
            Pnet = P_sub[:,sub_load_index]
            Qnet = Q_sub[:,sub_load_index]
            Pnet_pred = Pnet
            Qnet_pred = Qnet
            if smooth ==True:
                out = smoothing(Pnet,Qnet)
                Pnet_pred = out['Pnet_pred']
                Qnet_pred = out['Qnet_pred']
            ND_load_ph = AS.NondispatchableAsset_3ph(Pnet, Qnet, bus675_num, ph_i,
                                                     dt, T, Pnet_pred = Pnet_pred,
                                                     Qnet_pred = Qnet_pred)
            nondispatch_assets.append(ND_load_ph)
            sub_load_index += 1
    
    # Add PV generation sources
    for i in range(N_PVs):
        Pnet_i = -PVpu*P_pv[i]
        Qnet_i = np.zeros(T)
        Pnet_pred = Pnet
        Qnet_pred = Qnet
        if smooth ==True:
            out = smoothing(Pnet_i,Qnet_i)
            Pnet_pred = out['Pnet_pred']
            Qnet_pred = out['Qnet_pred']
        bus_id_i = network.bus_df[network.bus_df['name']==\
                                  PV_bus_names[i]].number.values[0]
        phases_i = PV_phases[i]
        PV_gen_i = AS.NondispatchableAsset_3ph(Pnet_i, Qnet_i, bus_id_i, phases_i,
                                               dt, T, Pnet_pred = Pnet_pred,
                                               Qnet_pred = Qnet_pred)
        nondispatch_assets.append(PV_gen_i)
    N_NDE = len(nondispatch_assets)
    
    # EVs at bus 634
    for i in range(N_EVs): 
        Emax_ev_i = Emax_EV*np.ones(T_ems)
        Emin_ev_i = Emin_EV*np.ones(T_ems)
        Pmax_ev_i = np.zeros(T_ems)
        Pmin_ev_i = np.zeros(T_ems)
        for t in range(ta_EVs[i],int(min(td_EVs[i],T_ems))):
            Pmax_ev_i[t] = P_max_EV
            Pmin_ev_i[t] = P_min_EV
        bus_id_ev_i = bus634_num
        ev_i = AS.StorageAsset(Emax_ev_i, Emin_ev_i, Pmax_ev_i, Pmin_ev_i,
                               E0_EVs[i], Emax_EV, bus_id_ev_i, dt, T, dt_ems,
                               T_ems, Pmax_abs=P_max_EV, c_deg_lin = 0,
                               eff = eff_EV, eff_opt = eff_EV_opt)
        storage_assets.append(ev_i)
    N_ESs = len(storage_assets)
        
    #######################################
    ### STEP 4: setup the market
    #######################################
        
    bus_id_market = bus650_num
    market = MK.Market(bus_id_market, prices_export, prices_import, demand_charge,
                       Pmax_market, Pmin_market, dt_market, T_market)
    
    #######################################
    #STEP 5: setup the energy system
    #######################################
    
    energy_system = ES.EnergySystem(storage_assets, nondispatch_assets, network,
                                    market, dt, T, dt_ems, T_ems)
    
    #######################################
    ### STEP 6: simulate the energy system: 
    #######################################
    
    i_line_unconst_list = list(range(network.N_lines))
    v_bus_unconst_list = []
    
    for x in opt_type:
        if x == "open_loop": 
            output = energy_system.\
                    simulate_network_3phPF('3ph',\
                                           i_unconstrained_lines=\
                                           i_line_unconst_list,\
                                           v_unconstrained_buses=\
                                           v_bus_unconst_list)
        
        if x == "mpc":
            output = energy_system.\
                    simulate_network_mpc_3phPF('3ph',
                                               i_unconstrained_lines=\
                                               i_line_unconst_list,\
                                               v_unconstrained_buses=\
                                               v_bus_unconst_list)
        PF_network_res = output['PF_network_res']
        P_import_ems = output['P_import_ems']
        P_export_ems = output['P_export_ems']
        P_ES_ems = output['P_ES_ems']
        P_demand_ems = output['P_demand_ems']
            
        P_demand_base = np.zeros(T)
        for i in range(len(nondispatch_assets)):
            bus_id = nondispatch_assets[i].bus_id
            P_demand_base += nondispatch_assets[i].Pnet
            
        P_demand_base_pred = np.zeros(T)
        for i in range(len(nondispatch_assets)):
            bus_id = nondispatch_assets[i].bus_id
            P_demand_base_pred += nondispatch_assets[i].Pnet_pred
        
        Pnet_market = np.zeros(T)
        for t in range(T):
            market_bus_res = PF_network_res[t].res_bus_df.iloc[bus_id_market]
            Pnet_market[t] = np.real\
                            (market_bus_res['Sa']\
                             + market_bus_res['Sb']\
                             + market_bus_res['Sc'])
        
        buses_Vpu = np.zeros([T,N_buses,N_phases])
        for t in range(T):
            for bus_id in range(N_buses):
                bus_res = PF_network_res[t].res_bus_df.iloc[bus_id]
                buses_Vpu[t,bus_id,0] = np.abs(bus_res['Va'])/network.Vslack_ph        
                buses_Vpu[t,bus_id,1] = np.abs(bus_res['Vb'])/network.Vslack_ph                  
                buses_Vpu[t,bus_id,2] = np.abs(bus_res['Vc'])/network.Vslack_ph         
        
        P_demand_base_pred_ems = np.zeros(T_ems)
        for t_ems in range(T_ems):
            t_indexes = (t_ems*dt_ems/dt + np.arange(0,dt_ems/dt)).astype(int)
            P_demand_base_pred_ems[t_ems] = np.mean(P_demand_base_pred[t_indexes])
        
        EVs_tot = sum(P_ES_ems[:,n] for n in range(N_ESs))
        P_compare = P_demand_base_pred_ems + EVs_tot
        #######################################
        ### STEP 7: plot results
        #######################################
        
        #x-axis time values
        time = dt*np.arange(T)
        time_ems = dt_ems*np.arange(T_ems)
        timeE = dt*np.arange(T+1)
        
        #energy cost
        energy_cost = market.calculate_revenue(Pnet_market,dt)
        energy_cost_string = 'Total energy cost: £ %.2f' %(-1*energy_cost)
        print(energy_cost_string)
        
        #save the data
        if x == "open_loop": 
            pickled_data_OL = (N_EVs, P_demand_base_pred_ems, P_compare, P_demand_base,\
                            Pnet_market, storage_assets, N_ESs, nondispatch_assets,\
                            time_ems, time, timeE, buses_Vpu)
            pickle.dump(pickled_data_OL, open(join(path_string, normpath("EV_case_data_open_loop.p")), "wb"))
        
        if x == "mpc":
            pickled_data_MPC = (N_EVs, P_demand_base_pred_ems, P_compare, P_demand_base,\
                            Pnet_market, storage_assets, N_ESs, nondispatch_assets,\
                            time_ems, time, timeE, buses_Vpu)
            pickle.dump(pickled_data_MPC, open(join(path_string, normpath("EV_case_data_mpc.p")), "wb"))
        
        
        figure_plot(x, N_EVs, P_demand_base_pred_ems, P_compare, P_demand_base,\
                    Pnet_market, storage_assets, N_ESs,\
                    nondispatch_assets, time_ems, time, timeE, buses_Vpu)

# Load pickled data and plot
else:
    for x in opt_type:
        if x == "open_loop":
            import_data = pickle.load(open(join(path_string, normpath("EV_case_data_open_loop.p")), "rb"))
        
        if x == "mpc":
            import_data = pickle.load(open(join(path_string, normpath("EV_case_data_mpc.p")), "rb"))

        N_EVs = import_data[0]
        P_demand_base_pred_ems = import_data[1]
        P_compare = import_data[2]
        P_demand_base = import_data[3]
        Pnet_market = import_data[4]
        storage_assets = import_data[5]
        N_ESs = import_data[6]
        nondispatch_assets = import_data[7]
        time_ems = import_data[8]
        time = import_data[9]
        timeE = import_data[10]
        buses_Vpu = import_data[11]
        
        figure_plot(x, N_EVs, P_demand_base_pred_ems, P_compare, P_demand_base,\
                Pnet_market, storage_assets, N_ESs,\
                nondispatch_assets, time_ems, time, timeE, buses_Vpu)



    
