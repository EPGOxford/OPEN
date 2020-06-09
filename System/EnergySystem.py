#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPEN Energy System Module.

The EnergySystem Class has two types of methods
i) energy management system (EMS) methods which implement algorithms to
calculate Asset control references, and
ii) simulation methods which call an EMS method to obtain control
references for Asset objects, update the state of Asset objects by calling
their updatecontrol method and update the state of the Network by calling
its power flow method.
An EnergySystem has two separate time series, one for the EMS, and the
other for simulation.

OPEN includes two EMS methods for controllable Asset objects:
(i) one for multi-period optimisation
with a simple ‘copper plate’ network model, and
(ii) one for multi-period optimisation with a linear multi-phase
distribution network model which includes voltage and current flow
constraints.

Open has simulation methods for:
(i) open-loop optimisation, where the EMS method is run ahead of operation
to obtain controllable Asset references over the EMS time-series; and
(ii) for MPC, where the EMS method is implemented with a receding horizon
so that the flexible Asset references are updated at each step of the EMS
time series.
"""

#import modules
import copy
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import numpy as np
import picos as pic
import matplotlib.pyplot as plt
from System.Network_3ph_pf import Network_3ph
import cvxopt

__version__ = "1.1.0"

class EnergySystem:
    """
    Base Energy Sysem Class

    Parameters
    ----------
    storage_assets : list of objects
        Containing details of each storage asset
    building_assets : list of objects
        Containsing details of each building asset
    nondispatch_assets : list of objects
        Containsing details of each nondispatchable asset
    network : object
        Object containing information about the network
    market : object
        Object containing information about the market
    dt_ems : float
        EMS time interval duration (hours)
    T_ems : int
        Number of EMS time intervals
    dt : float
        time interval duration (hours)
    T : int
        number of time intervals

    Returns
    -------
    EnergySystem


    """
    def __init__(self, storage_assets, nondispatch_assets, network, market,
                 dt, T, dt_ems, T_ems, building_assets=[]):
        self.storage_assets = storage_assets
        self.building_assets = building_assets
        self.nondispatch_assets = nondispatch_assets
        self.network = network
        self.market = market
        self.dt_ems = dt_ems
        self.T_ems = T_ems
        self.dt = dt
        self.T = T

#######################################
### Open Loop Control Methods
#######################################
    def EMS_copper_plate(self):
        """
        Energy management system optimization assuming all assets connected to
        a single node.

        Parameters
        ----------
        self : EnergySystem object
            Object containing information on assets, market, network and time
            resolution.

        Returns
        -------
        Output : dictionary
            The following numpy.ndarrays are present depending upon asset mix:
                P_ES_val : Charge/discharge power for storage assets (kW)
                P_BLDG_val :Builfing power consumption (kW)
                P_import_val :Power imported from central grid (kW)
                P_export_val :Power exported to central grid (kW)
                P_demand_val :System power demand at energy management time
                              resolution

        """
        #setup and run a basic energy optimisation
        #(single copper plate network model)
        #######################################
        ### STEP 0: setup variables
        #######################################
        prob = pic.Problem()
        N_ES = len(self.storage_assets)
        N_BLDG = len(self.building_assets)
        N_INDEPENDENT = N_ES + N_BLDG
        N_nondispatch = len(self.nondispatch_assets)
        P_demand_actual = np.zeros(self.T)
        P_demand = np.zeros(self.T_ems)
        for i in range(N_nondispatch):
            P_demand_actual += self.nondispatch_assets[i].Pnet
        #convert P_demand_actual to EMS time series scale
        for t_ems in range(self.T_ems):
            t_indexes = (t_ems*self.dt_ems/self.dt\
                         + np.arange(0,self.dt_ems/self.dt)).astype(int)
            P_demand[t_ems] = np.mean(P_demand_actual[t_indexes])
        #######################################
        ### STEP 1: set up decision variables
        #######################################
        #controllable asset input powers
        P_ctrl_asset = prob.add_variable('P_ctrl_asset',(self.T_ems,\
                                                         N_INDEPENDENT),\
                                                         vtype='continuous')
        if N_BLDG > 0:
            # cooling power
            P_cooling = prob.add_variable('P_cooling',(self.T_ems,N_BLDG),\
                                          vtype='continuous')
            # heating power
            P_heating = prob.add_variable('P_heating',(self.T_ems,N_BLDG),\
                                          vtype='continuous')
            # internal temperature
            T_bldg = prob.add_variable('T_bldg',(self.T_ems,N_BLDG),\
                                       vtype='continuous')
        #(positive) net power imports
        P_import = prob.add_variable('P_import',(self.T_ems,1),\
                                     vtype='continuous')
        #(positive) net power exports
        P_export = prob.add_variable('P_export',(self.T_ems,1),\
                                     vtype='continuous')
        #(positive) maximum demand dummy variable
        P_max_demand = prob.add_variable('P_max_demand',1,\
                                         vtype='continuous')
        #######################################
        ### STEP 2: set up constraints
        #######################################
        Asum_np = np.tril(np.ones([self.T_ems,self.T_ems])).astype('double')
        #lower triangle matrix summing powers
        Asum = pic.new_param('Asum',Asum_np)

        #lbuilding thermal model constraints
        for i in range(N_BLDG):
            #maximum heating constraint
            prob.add_constraint(P_heating[:,i] <= self.building_assets[i].Hmax)
            #maximum cooling constraint
            prob.add_constraint(P_cooling[:,i] <= self.building_assets[i].Cmax)
            #minimum heating constraint
            prob.add_constraint(P_heating[:,i] >= 0)
            #minimum cooling constraint
            prob.add_constraint(P_cooling[:,i] >= 0)
            #maximum temperature constraint
            prob.add_constraint(T_bldg[:,i] <= self.building_assets[i].Tmax)
            #minimum temperature constraint
            prob.add_constraint(T_bldg[:,i] >= self.building_assets[i].Tmin)
            #power consumption is the sum of heating and cooling
            prob.add_constraint(P_ctrl_asset[:,i] == P_cooling[:,i]\
                                + P_heating[:,i])
            for t in range(self.T_ems):
                if t == 0:
                    # initial temperature constraint
                    prob.add_constraint(T_bldg[t,i] ==\
                                        self.building_assets[i].T0)
                else:
                    # Inside temperature is a function of heating/cooling and
                    # outside temperature. Alpha, beta and gamma are parameters
                    # derived from the R and C values of the building.
                    # Relation between alpha, beta, gamma, R and C can be found
                    # in the BuildingAsset class in the Assets.py file
                    prob.add_constraint(T_bldg[t,i] ==\
                                        self.building_assets[i].\
                                        alpha*T_bldg[t-1,i] \
                                        - self.building_assets[i].\
                                        beta*self.building_assets[i].\
                                        CoP_cooling*P_cooling[t-1,i] \
                                        + self.building_assets[i].\
                                        beta*self.building_assets[i].\
                                        CoP_heating*P_heating[t-1,i] \
                                        + self.building_assets[i].\
                                        gamma*self.building_assets[i].\
                                        Ta[t-1])

        #linear battery model constraints
        for i in range(N_ES):
            #maximum power constraint
            prob.add_constraint(P_ctrl_asset[:,N_BLDG+i] <=\
                                self.storage_assets[i].Pmax)
            #minimum power constraint
            prob.add_constraint(P_ctrl_asset[:,N_BLDG+i] >=\
                                self.storage_assets[i].Pmin)
            #maximum energy constraint
            prob.add_constraint(self.dt_ems*Asum*P_ctrl_asset[:,N_BLDG+i] <=\
                                self.storage_assets[i].Emax\
                                -self.storage_assets[i].E0)
            #minimum energy constraint
            prob.add_constraint(self.dt_ems*Asum*P_ctrl_asset[:,N_BLDG+i] >=\
                                self.storage_assets[i].Emin\
                                -self.storage_assets[i].E0)
            #final energy constraint
            prob.add_constraint(self.dt_ems*Asum[self.T_ems-1,:]\
                                *P_ctrl_asset[:,N_BLDG+i] ==\
                                self.storage_assets[i].ET\
                                -self.storage_assets[i].E0)
        #import/export constraints
        for t in range(self.T_ems):
            # power balance
            prob.add_constraint(sum(P_ctrl_asset[t,:]) + P_demand[t] ==\
                                P_import[t]-P_export[t])
            #maximum import constraint
            prob.add_constraint(P_import[t] <= self.market.Pmax[t])
            #maximum import constraint
            prob.add_constraint(P_import[t] >= 0)
            #maximum import constraint
            prob.add_constraint(P_export[t] <= -self.market.Pmin[t])
             #maximum import constraint
            prob.add_constraint(P_export[t] >= 0)
             #maximum demand dummy variable constraint
            prob.add_constraint(P_max_demand >= P_import[t]-P_export[t])
        if self.market.FR_window is not None:
            FR_window = self.market.FR_window
            FR_SoC_max = self.market.FR_SOC_max
            FR_SoC_min = self.market.FR_SOC_min
            for t in range(self.T_ems):
                if FR_window[t] ==1:
                    for i in range(N_ES):
                        # final energy constraint
                        prob.add_constraint(self.dt_ems
                                            * Asum[t,:]
                                            * P_ctrl_asset[:,N_BLDG+i]
                                            <= (FR_SoC_max
                                                * self.storage_assets[i].Emax)
                                            - self.storage_assets[i].E0)
                        # final energy constraint
                        prob.add_constraint(self.dt_ems
                                            * Asum[t,:]
                                            * P_ctrl_asset[:,N_BLDG+i]
                                            >= (FR_SoC_min
                                                * self.storage_assets[i].Emax)
                                            - self.storage_assets[i].E0)

        #######################################
        ### STEP 3: set up objective
        #######################################
        prob.set_objective('min',self.market.demand_charge*P_max_demand+\
                                 sum(self.market.prices_import[t]*P_import[t]+\
                                     -self.market.prices_export[t]*P_export[t]\
                                     for t in range(self.T_ems)))
        #######################################
        ### STEP 3: solve the optimisation
        #######################################
        print('*** SOLVING THE OPTIMISATION PROBLEM ***')
        prob.solve(verbose = 0)
        print('*** OPTIMISATION COMPLETE ***')
        P_ctrl_asset_val = P_ctrl_asset.value
        P_import_val = P_import.value
        P_export_val = P_export.value
        P_demand_val = P_demand

        if N_BLDG > 0:
            #Store internal temperature inside object
            T_bldg_val = T_bldg.value
            for b in range(N_BLDG):
                self.building_assets[b].T_int = T_bldg_val[:,b]


        if N_ES > 0 and N_BLDG > 0:
            output = {'P_BLDG_val':P_ctrl_asset_val[:,:N_BLDG],\
                'P_ES_val':P_ctrl_asset_val[:,N_BLDG:N_ES+N_BLDG],\
                'P_import_val':P_import_val,\
                'P_export_val':P_export_val,\
                'P_demand_val':P_demand_val}
        elif N_ES == 0 and N_BLDG > 0:
            output = {'P_BLDG_val':P_ctrl_asset_val[:,:N_BLDG],\
                'P_import_val':P_import_val,\
                'P_export_val':P_export_val,\
                'P_demand_val':P_demand_val}
        elif N_ES > 0 and N_BLDG == 0:
            output = {'P_ES_val':P_ctrl_asset_val[:,:N_ES],\
                'P_import_val':P_import_val,\
                'P_export_val':P_export_val,\
                'P_demand_val':P_demand_val}
        else:
            raise ValueError('No dispatchable assets.')



        return output


    def simulate_network(self):
        """
        Run the Energy Management System in open loop and simulate a pandapower
        network.

        Parameters
        ----------
        self : EnergySystem object
            Object containing information on assets, market, network and time
            resolution.

        Returns
        -------
        Output : dictionary
            The following numpy.ndarrays are present depending upon asset mix:
                buses_Vpu : Voltage magnitude at bus (V)
                buses_Vang : Voltage angle at bus (rad)
                buses_Pnet : Real power at bus (kW)
                buses_Qnet : Reactive power at bus (kVAR)
                Pnet_market : Real power seen by the market (kW)
                Qnet_market : Reactive power seen by the market (kVAR)
                P_ES_ems : Charge/discharge power for storage assets at energy
                    management time resolution (kW)
                P_BLDG_ems :Builfing power consumption at energy management
                    time resolution (kW)
                P_import_ems :Power imported from central grid at energy
                    management time resolution (kW)
                P_export_ems :Power exported to central grid at energy
                    management time resolution(kW)
                P_demand_ems :System power demand at energy management time
                    resolution (kW)

        """

        #######################################
        ### STEP 1: solve the optimisation
        #######################################
        t0 = 0
        output_ems = self.EMS_copper_plate()

        N_ESs = len(self.storage_assets) #number of EVs
        N_BLDGs = len(self.building_assets) #number of buildings
        N_nondispatch = len(self.nondispatch_assets) #number of EVs

        P_import_ems = output_ems['P_import_val']
        P_export_ems = output_ems['P_export_val']
        if N_ESs > 0:
            P_ES_ems = output_ems['P_ES_val']
        if N_BLDGs > 0:
            P_BLDG_ems = output_ems['P_BLDG_val']
        P_demand_ems = output_ems['P_demand_val']
        #convert P_ES and P_BLDG signals to system time-series scale
        if N_ESs > 0:
            P_ESs = np.zeros([self.T,N_ESs])
            for t in range(self.T):
                t_ems = int(t/(self.dt_ems/self.dt))
                P_ESs[t,:] = P_ES_ems[t_ems,:]
        if N_BLDGs > 0:
            P_BLDGs = np.zeros([self.T,N_BLDGs])
            for t in range(self.T):
                t_ems = int(t/(self.dt_ems/self.dt))
                P_BLDGs[t,:] = P_BLDG_ems[t_ems,:]
        #######################################
        ### STEP 2: update the controllable assets
        #######################################
        if N_ESs > 0:
            for i in range(N_ESs):
                self.storage_assets[i].update_control(P_ESs[:,i])
        if N_BLDGs > 0:
            for i in range(N_BLDGs):
                self.building_assets[i].update_control(P_BLDGs[:,i])
        #######################################
        ### STEP 3: simulate the network
        #######################################
        N_buses = self.network.bus['name'].size
        P_demand_buses = np.zeros([self.T,N_buses])
        Q_demand_buses = np.zeros([self.T,N_buses])
        if N_ESs > 0:
            #calculate the total real and reactive power demand at each bus
            for i in range(N_ESs):
                bus_id = self.storage_assets[i].bus_id
                P_demand_buses[:,bus_id] += self.storage_assets[i].Pnet
                Q_demand_buses[:,bus_id] += self.storage_assets[i].Qnet
        if N_BLDGs > 0:
            #calculate the total real and reactive power demand at each bus
            for i in range(N_BLDGs):
                bus_id = self.building_assets[i].bus_id
                P_demand_buses[:,bus_id] += self.building_assets[i].Pnet
                Q_demand_buses[:,bus_id] += self.building_assets[i].Qnet
        for i in range(N_nondispatch):
            bus_id = self.nondispatch_assets[i].bus_id
            P_demand_buses[:,bus_id] += self.nondispatch_assets[i].Pnet
            Q_demand_buses[:,bus_id] += self.nondispatch_assets[i].Qnet


        buses_Vpu = np.zeros([self.T,N_buses])
        buses_Vang = np.zeros([self.T,N_buses])
        buses_Pnet = np.zeros([self.T,N_buses])
        buses_Qnet = np.zeros([self.T,N_buses])
        Pnet_market = np.zeros(self.T)
        Qnet_market = np.zeros(self.T)
        #print(P_demand_buses)
        print('*** SIMULATING THE NETWORK ***')
        for t in range(self.T):
            #for each time interval:
            #set up a copy of the network for simulation interval t
            network_t = copy.deepcopy(self.network)
            for bus_id in range(N_buses):
                P_t = P_demand_buses[t,bus_id]
                Q_t = Q_demand_buses[t,bus_id]
                #add P,Q loads to the network copy
                pp.create_load(network_t,bus_id,P_t/1e3,Q_t/1e3)
            #run the power flow simulation
            pp.runpp(network_t,max_iteration=100) # or “nr”
            if t % 100 == 0:
                print('network sim complete for t = '\
                      + str(t) + ' of ' + str(self.T))
            Pnet_market[t] = network_t.res_ext_grid['p_mw'][0]*1e3
            Qnet_market[t] = network_t.res_ext_grid['q_mvar'][0]*1e3
            for bus_i in range(N_buses):
                buses_Vpu[t,bus_i] = network_t.res_bus['vm_pu'][bus_i]
                buses_Vang[t,bus_i] = network_t.res_bus['va_degree'][bus_i]
                buses_Pnet[t,bus_i] = network_t.res_bus['p_mw'][bus_i]*1e3
                buses_Qnet[t,bus_i] = network_t.res_bus['q_mvar'][bus_i]*1e3

        print('*** NETWORK SIMULATION COMPLETE ***')

        if N_ESs > 0 and N_BLDGs > 0:
            output = {'buses_Vpu':buses_Vpu,\
                'buses_Vang':buses_Vang,\
                'buses_Pnet':buses_Pnet,\
                'buses_Qnet':buses_Qnet,\
                'Pnet_market':Pnet_market,\
                'Qnet_market':Qnet_market,\
                'P_ES_ems':P_ES_ems,\
                'P_BLDG_ems':P_BLDG_ems,\
                'P_import_ems':P_import_ems,\
                'P_export_ems':P_export_ems,\
                'P_demand_ems':P_demand_ems}
        elif N_ESs == 0 and N_BLDGs > 0:
            output = {'buses_Vpu':buses_Vpu,\
                'buses_Vang':buses_Vang,\
                'buses_Pnet':buses_Pnet,\
                'buses_Qnet':buses_Qnet,\
                'Pnet_market':Pnet_market,\
                'Qnet_market':Qnet_market,\
                'P_BLDG_ems':P_BLDG_ems,\
                'P_import_ems':P_import_ems,\
                'P_export_ems':P_export_ems,\
                'P_demand_ems':P_demand_ems}
        elif N_ESs > 0 and N_BLDGs == 0:
            output = {'buses_Vpu':buses_Vpu,\
                'buses_Vang':buses_Vang,\
                'buses_Pnet':buses_Pnet,\
                'buses_Qnet':buses_Qnet,\
                'Pnet_market':Pnet_market,\
                'Qnet_market':Qnet_market,\
                'P_ES_ems':P_ES_ems,\
                'P_import_ems':P_import_ems,\
                'P_export_ems':P_export_ems,\
                'P_demand_ems':P_demand_ems}
        else:
            raise ValueError('No dispatchable assets.')


        return output

    # NEEDED FOR OXEMF EV CASE STUDY
    def simulate_network_3phPF(self, ems_type = '3ph',
                               i_unconstrained_lines=[],
                               v_unconstrained_buses = []):
        """
        Run the Energy Management System in open loop and simulate an IEEE 13
        bus network either copper plate or 3ph

        Parameters
        ----------
        self : EnergySystem object
            Object containing information on assets, market, network and time
            resolution.
        ems_type : string
            Identifies whether the system is copper plate or 3ph. Default 3ph
        i_unconstrained_lines : list
            List of network lines which have unconstrained current
        v_unconstrained_buses : list
            List of buses at which the voltage is not constrained

        Returns
        -------
        Output : dictionary
                PF_network_res : Network power flow results stored as a list of
                    objects
                P_ES_ems : Charge/discharge power for storage assets at energy
                    management time resolution (kW)
                P_import_ems :Power imported from central grid at energy
                    management time resolution (kW)
                P_export_ems :Power exported to central grid at energy
                    management time resolution(kW)
                P_demand_ems :System power demand at energy management time
                    resolution (kW)

        """

        #######################################
        ### STEP 1: solve the optimisation
        #######################################
        t0 = 0
        if ems_type == 'copper_plate':
            output_ems = self.EMS_copper_plate_t0(t0)
        else:
            output_ems = self.EMS_3ph_linear_t0(t0,
                                                i_unconstrained_lines,
                                                v_unconstrained_buses)
        P_import_ems = output_ems['P_import_val']
        P_export_ems = output_ems['P_export_val']
        P_ES_ems = output_ems['P_ES_val']
        P_demand_ems = output_ems['P_demand_val']
        #convert P_EV signals to system time-series scale
        N_ESs = len(self.storage_assets)
        N_nondispatch = len(self.nondispatch_assets)
        P_ESs = np.zeros([self.T,N_ESs])
        for t in range(self.T):
            t_ems = int(t/(self.dt_ems/self.dt))
            P_ESs[t,:] = P_ES_ems[t_ems,:]
        #######################################
        ### STEP 2: update the controllable assets
        #######################################
        for i in range(N_ESs):
            self.storage_assets[i].update_control(P_ESs[:,i])
        #######################################
        ### STEP 3: simulate the network
        #######################################
        N_buses = self.network.N_buses
        N_phases = self.network.N_phases
        P_demand_buses = np.zeros([self.T,N_buses,N_phases])
        Q_demand_buses = np.zeros([self.T,N_buses,N_phases])
        #calculate the total real and reactive power demand at each bus phase
        for i in range(N_ESs):
            bus_id = self.storage_assets[i].bus_id
            phases_i = self.storage_assets[i].phases
            N_phases_i = np.size(phases_i)
            for ph_i in np.nditer(phases_i):
                P_demand_buses[:,bus_id,ph_i] +=\
                self.storage_assets[i].Pnet/N_phases_i
                Q_demand_buses[:,bus_id,ph_i] +=\
                self.storage_assets[i].Qnet/N_phases_i
        for i in range(N_nondispatch):
            bus_id = self.nondispatch_assets[i].bus_id
            phases_i = self.nondispatch_assets[i].phases
            N_phases_i = np.size(phases_i)
            for ph_i in np.nditer(phases_i):
                P_demand_buses[:,bus_id,ph_i] +=\
                self.nondispatch_assets[i].Pnet/N_phases_i
                Q_demand_buses[:,bus_id,ph_i] +=\
                self.nondispatch_assets[i].Qnet/N_phases_i
        #Store power flow results as a list of network objects

        PF_network_res = []
        print('*** SIMULATING THE NETWORK ***')
        for t in range(self.T):
            #for each time interval:
            #set up a copy of the network for simulation interval t
            network_t = copy.deepcopy(self.network)
            network_t.clear_loads()
            for bus_id in range(N_buses):
                for ph_i in range(N_phases):
                    Pph_t = P_demand_buses[t,bus_id,ph_i]
                    Qph_t = Q_demand_buses[t,bus_id,ph_i]
                    #add P,Q loads to the network copy
                    network_t.set_load(bus_id,ph_i,Pph_t,Qph_t)
            #run the power flow simulation
            network_t.zbus_pf()
            PF_network_res.append(network_t)
        print('*** NETWORK SIMULATION COMPLETE ***')

        return {'PF_network_res' :PF_network_res,\
                'P_ES_ems':P_ES_ems,\
                'P_import_ems':P_import_ems,\
                'P_export_ems':P_export_ems,\
                'P_demand_ems':P_demand_ems}

#######################################
### Model Predictive Control Methods
#######################################

    def EMS_copper_plate_t0(self, t0):
        """
        Setup and run a basic energy optimisation (single copper plate network
        model) for MPC interval t0
        """

        #######################################
        ### STEP 0: setup variables
        #######################################

        t0_dt = int(t0*self.dt_ems/self.dt)
        T_mpc = self.T_ems-t0
        T_range = np.arange(t0,self.T_ems)
        prob = pic.Problem()
        N_ES = len(self.storage_assets)
        N_nondispatch = len(self.nondispatch_assets)
        P_demand_actual = np.zeros(self.T)
        P_demand_pred = np.zeros(self.T)
        P_demand = np.zeros(T_mpc)

        for i in range(N_nondispatch):
            P_demand_actual += self.nondispatch_assets[i].Pnet
            P_demand_pred += self.nondispatch_assets[i].Pnet_pred

        # Assemble P_demand out of P actual and P predicted and convert to EMS
        # time series scale
        for t_ems in T_range:
            t_indexes = ((t_ems * self.dt_ems / self.dt
                          + np.arange(0, self.dt_ems / self.dt)).astype(int))
            if t_ems == t0:
                P_demand[t_ems-t0] = np.mean(P_demand_actual[t_indexes])
            else:
                P_demand[t_ems-t0] = np.mean(P_demand_pred[t_indexes])

        # get total ES system demand (before optimisation)
        Pnet_ES_sum = np.zeros(self.T)
        for i in range(N_ES):
            Pnet_ES_sum += self.storage_assets[i].Pnet
        #get the maximum (historical) demand before t0
        if t0 > 0:
            P_max_demand_pre_t0 = np.max(P_demand_actual[0:t0_dt]\
                                         + Pnet_ES_sum[0:t0_dt])
        else:
            P_max_demand_pre_t0 = 0

        #######################################
        ### STEP 1: set up decision variables
        #######################################
        # energy storage system input powers
        P_ES = prob.add_variable('P_ES', (T_mpc,N_ES), vtype='continuous')
        # energy storage system input powers
        P_ES_ch = prob.add_variable('P_ES_ch', (T_mpc,N_ES),
                                    vtype='continuous')
        # energy storage system output powers
        P_ES_dis = prob.add_variable('P_ES_dis', (T_mpc,N_ES),
                                     vtype='continuous')
        # (positive) net power imports
        P_import = prob.add_variable('P_import', (T_mpc,1), vtype='continuous')
        # (positive) net power exports
        P_export = prob.add_variable('P_export', (T_mpc,1), vtype='continuous')
        # (positive) maximum demand dummy variable
        P_max_demand = prob.add_variable('P_max_demand', 1, vtype='continuous')
        # (positive) minimum terminal energy dummy variable
        E_T_min = prob.add_variable('E_T_min', 1, vtype='continuous')

        #######################################
        ### STEP 2: set up constraints
        #######################################
        #lower triangle matrix summing powers
        Asum = pic.new_param('Asum',np.tril(np.ones([T_mpc,T_mpc])))
        eff_opt = self.storage_assets[i].eff_opt
        # linear battery model constraints
        for i in range(N_ES):
            # maximum power constraint
            prob.add_constraint((P_ES_ch[:,i] - P_ES_dis[:,i])\
                                <= self.storage_assets[i].Pmax[T_range])
            # minimum power constraint
            prob.add_constraint((P_ES_ch[:,i] - P_ES_dis[:,i])\
                                >= self.storage_assets[i].Pmin[T_range])
            # maximum energy constraint
            prob.add_constraint((self.dt_ems
                                 * Asum
                                 * (P_ES_ch[:,i] - P_ES_dis[:,i]))\
                                <= (self.storage_assets[i].Emax[T_range]
                                    - self.storage_assets[i].E[t0_dt]))
            # minimum energy constraint
            prob.add_constraint((self.dt_ems
                                 * Asum
                                 * (P_ES_ch[:,i] - P_ES_dis[:,i]))\
                                >= (self.storage_assets[i].Emin[T_range]
                                    - self.storage_assets[i].E[t0_dt]))
            # final energy constraint
            prob.add_constraint((self.dt_ems
                                 * Asum[T_mpc-1,:]
                                 * (P_ES_ch[:,i] - P_ES_dis[:,i])
                                 + E_T_min)\
                                >= (self.storage_assets[i].ET
                                    - self.storage_assets[i].E[t0_dt]))

            eff_opt = self.storage_assets[i].eff_opt

            # P_ES_ch & P_ES_dis dummy variables
            for t in range(T_mpc):
                prob.add_constraint(P_ES[t, i] == (P_ES_ch[t, i]
                                                  / eff_opt
                                                  - P_ES_dis[t, i]
                                                  * eff_opt))
                prob.add_constraint(P_ES_ch[t, i] >= 0)
                prob.add_constraint(P_ES_dis[t, i] >= 0)

        # import/export constraints
        for t in range(T_mpc):
            # net import variables
            prob.add_constraint((sum(P_ES[t, :]) + P_demand[t])\
                                == (P_import[t] - P_export[t]))
            # maximum import constraint
            prob.add_constraint(P_import[t] <= self.market.Pmax[t0+t])
            # maximum import constraint
            prob.add_constraint(P_import[t] >= 0)
            # maximum import constraint
            prob.add_constraint(P_export[t] <= -self.market.Pmin[t0+t])
            # maximum import constraint
            prob.add_constraint(P_export[t] >= 0)
            #maximum demand dummy variable constraint
            prob.add_constraint((P_max_demand + P_max_demand_pre_t0)\
                                >= (P_import[t] - P_export[t]) )
            # maximum demand dummy variable constraint
            prob.add_constraint(P_max_demand  >= 0)
        if self.market.FR_window is not None:
            FR_window = self.market.FR_window
            FR_SoC_max = self.market.FR_SOC_max
            FR_SoC_min = self.market.FR_SOC_min
            for t in range(t0,self.T_ems):
                if FR_window[t] ==1:
                    for i in range(N_ES):
                        # final energy constraint
                        prob.add_constraint((self.dt_ems
                                             * Asum[t, :]
                                             * (P_ES_ch[:, i]
                                                - P_ES_dis[:, i]))\
                                            <= (FR_SoC_max
                                                * self.storage_assets[i].Emax)
                                            - self.storage_assets[i].E[t0_dt])
                        # final energy constraint
                        prob.add_constraint((self.dt_ems
                                             * Asum[t, :]
                                             * (P_ES_ch[:,i] - P_ES_dis[:,i]))\
                                            >= (FR_SoC_min
                                                * self.storage_assets[i].Emax)
                                            - self.storage_assets[i].E[t0_dt])
        # minimum terminal energy dummy variable  constraint
        prob.add_constraint(E_T_min >= 0)
        #######################################
        ### STEP 3: set up objective
        #######################################
        prices_import = pic.new_param('prices_import',
                                      self.market.prices_import)
        prices_export = pic.new_param('prices_export',
                                      self.market.prices_export)
        terminal_const = 1e12 # coeff for objective terminal soft constraint
        prob.set_objective('min', (self.market.demand_charge * P_max_demand +\
                                   sum(sum(self.dt_ems
                                           * self.storage_assets[i].c_deg_lin
                                           * (P_ES_ch[t, i] + P_ES_dis[t,i])\
                                           for i in range(N_ES))
                                       + self.dt_ems
                                       * prices_import[t0 + t]
                                       * P_import[t]
                                       - self.dt_ems
                                       * prices_export[t0 + t]
                                       * P_export[t]\
                                       for t in range(T_mpc))
                                    + terminal_const * E_T_min))
        #######################################
        ### STEP 3: solve the optimisation
        #######################################
        print('*** SOLVING THE OPTIMISATION PROBLEM ***')
        prob.solve(verbose = 0)
        print('*** OPTIMISATION COMPLETE ***')
        P_ES_val = np.matrix(P_ES.value)
        P_import_val = np.matrix(P_import.value)
        P_export_val = np.matrix(P_export.value)
        P_demand_val = np.matrix(P_demand)
        E_T_min_val = np.matrix(E_T_min.value)
        # P_ES_val = np.array(P_ES.value)
        # P_import_val = np.array(P_import.value)
        # P_export_val = np.array(P_export.value)
        # P_demand_val = np.array(P_demand)
        # E_T_min_val = np.array(E_T_min.value)
        return {'P_ES_val':P_ES_val,\
                'P_import_val':P_import_val,\
                'P_export_val':P_export_val,\
                'P_demand_val':P_demand_val,\
                'E_T_min_val':E_T_min_val}

    def EMS_copper_plate_t0_c1deg(self, t0):
        """
        setup and run a basic energy optimisation (single copper plate network
        model) for MPC interval t0
        """
        #######################################
        ### STEP 0: setup variables
        #######################################
        t0_dt = int(t0 * self.dt_ems / self.dt)
        T_mpc = self.T_ems - t0
        T_range = np.arange(t0,self.T_ems)
        prob = pic.Problem()
        N_ES = len(self.storage_assets)
        N_nondispatch = len(self.nondispatch_assets)
        P_demand_actual = np.zeros(self.T)
        P_demand_pred = np.zeros(self.T)
        P_demand = np.zeros(T_mpc)
        for i in range(N_nondispatch):
            P_demand_actual += self.nondispatch_assets[i].Pnet
            P_demand_pred += self.nondispatch_assets[i].Pnet_pred
        # Assemble P_demand out of P actual and P predicted and convert to
        # EMS time series scale
        for t_ems in T_range:
            t_indexes = (t_ems
                         * self.dt_ems
                         / self.dt
                         + np.arange(0, self.dt_ems / self.dt)).astype(int)
            if t_ems == t0:
                P_demand[t_ems-t0] = np.mean(P_demand_actual[t_indexes])
            else:
                P_demand[t_ems-t0] = np.mean(P_demand_pred[t_indexes])
        #get total ES system demand (before optimisation)
        Pnet_ES_sum = np.zeros(self.T)
        for i in range(N_ES):
            Pnet_ES_sum += self.storage_assets[i].Pnet
        #get the maximum (historical) demand before t0
        if t0 > 0:
            P_max_demand_pre_t0 = (np.max(P_demand_actual[0:t0_dt]
                                   + Pnet_ES_sum[0: t0_dt]))
        else:
            P_max_demand_pre_t0 = 0
        #######################################
        ### STEP 1: set up decision variables
        #######################################
        # energy storage system input powers
        P_ES = prob.add_variable('P_ES', (T_mpc,N_ES), vtype='continuous')
        # energy storage system input powers
        P_ES_ch = prob.add_variable('P_ES_ch', (T_mpc,N_ES),
                                    vtype='continuous')
        # energy storage system output powers
        P_ES_dis = prob.add_variable('P_ES_dis', (T_mpc,N_ES),
                                     vtype='continuous')
        # (positive) net power imports
        P_import = prob.add_variable('P_import', (T_mpc,1), vtype='continuous')
        # (positive) net power exports
        P_export = prob.add_variable('P_export', (T_mpc,1), vtype='continuous')
        # (positive) maximum demand dummy variable
        P_max_demand = prob.add_variable('P_max_demand', 1, vtype='continuous')
        # (positive) minimum terminal energy dummy variable
        E_T_min = prob.add_variable('E_T_min', 1, vtype='continuous')
        #######################################
        ### STEP 2: set up constraints
        #######################################

        # lower triangle matrix summing powers
        Asum = pic.new_param('Asum', np.tril(np.ones([T_mpc,T_mpc])))
#        Asum =  cvxopt.matrix(np.tril(np.ones([T_mpc,T_mpc])), (T_mpc,T_mpc),
#                              'd')
        # linear battery model constraints
        for i in range(N_ES):
            # maximum power constraint
            prob.add_constraint((P_ES_ch[:, i] - P_ES_dis[:, i])\
                                <= self.storage_assets[i].Pmax[T_range])
            # minimum power constraint
            prob.add_constraint((P_ES_ch[:, i] - P_ES_dis[:, i])\
                                >= self.storage_assets[i].Pmin[T_range])
            # maximum energy constraint
            prob.add_constraint((self.dt_ems
                                 * Asum
                                 * (P_ES_ch[:,i] - P_ES_dis[:,i]))\
                                <= (self.storage_assets[i].Emax[T_range]
                                    - self.storage_assets[i].E[t0_dt]))
            # minimum energy constraint
            prob.add_constraint((self.dt_ems
                                 * Asum
                                 * (P_ES_ch[:,i] - P_ES_dis[:,i]))\
                                >= (self.storage_assets[i].Emin[T_range]
                                    - self.storage_assets[i].E[t0_dt]))
            # final energy constraint
            prob.add_constraint((self.dt_ems
                                 * Asum[T_mpc-1, :]
                                 * (P_ES_ch[:, i] - P_ES_dis[:,i])
                                 + E_T_min)\
                                >= (self.storage_assets[i].ET
                                    - self.storage_assets[i].E[t0_dt]))

            eff_opt = self.storage_assets[i].eff_opt

            #P_ES_ch & P_ES_dis dummy variables
            for t in range(T_mpc):
                prob.add_constraint(P_ES[t, i] == (P_ES_ch[t, i]
                                                   / eff_opt
                                                   - P_ES_dis[t, i]
                                                   * eff_opt))
                prob.add_constraint(P_ES_ch[t, i] >= 0)
                prob.add_constraint(P_ES_dis[t, i] >= 0)

        #import/export constraints
        for t in range(T_mpc):
            # net import variables
            prob.add_constraint(sum(P_ES[t, :]) + P_demand[t]\
                                == P_import[t] - P_export[t])
            # maximum import constraint
            prob.add_constraint(P_import[t] <=  self.market.Pmax[t0+t])
            # maximum import constraint
            prob.add_constraint(P_import[t] >= 0)
            # maximum import constraint
            prob.add_constraint(P_export[t] <= -self.market.Pmin[t0 + t])
            # maximum import constraint
            prob.add_constraint(P_export[t] >= 0)
            # maximum demand dummy variable constraint
            prob.add_constraint(P_max_demand + P_max_demand_pre_t0\
                                >= P_import[t] - P_export[t])
            # maximum demand dummy variable constraint
            prob.add_constraint(P_max_demand >= 0)
            # minimum terminal energy dummy variable  constraint
            prob.add_constraint(E_T_min[:] >= 0)

       #if FFR energy constraints
        if self.market.FR_window is not None:
            FR_window = self.market.FR_window
            FR_SoC_max = self.market.FR_SOC_max
            FR_SoC_min = self.market.FR_SOC_min
            for t in range(len(T_mpc)):
                if FR_window[t] == 1:
                    for i in range(N_ES):
                        # final energy constraint
                        prob.add_constraint((self.dt_ems
                                             * Asum[t, :]
                                             * P_ES[:, i])\
                                            <= ((FR_SoC_max
                                                 * self.storage_assets[i].Emax)
                - self.storage_assets[i].E[t0_dt]))
                        # final energy constraint
                        prob.add_constraint((self.dt_ems
                                             * Asum[t, :]
                                             * P_ES[:, i])\
                                            >= ((FR_SoC_min
                                                * self.storage_assets[i].Emax)
                - self.storage_assets[i].E[t0_dt]))

        #######################################
        ### STEP 3: set up objective
        #######################################
        prices_import = pic.new_param('prices_import',
                                      self.market.prices_import)
        prices_export = pic.new_param('prices_export',
                                      self.market.prices_export)
        terminal_const = 1e12 #coeff for objective terminal soft constraint
        prob.set_objective('min', (self.market.demand_charge
                                   * P_max_demand
                                   + sum(sum(self.dt_ems
                                             * self.storage_assets[i].c_deg_lin
                                             * (P_ES_ch[t,i] + P_ES_dis[t,i])\
                                             for i in range(N_ES))
                                         + self.dt_ems
                                         * prices_import[t0 + t]
                                         * P_import[t]
                                         + -self.dt_ems
                                         * prices_export[t0 + t]
                                         * P_export[t]\
                                         for t in range(T_mpc))
                                    + terminal_const
                                    * E_T_min))
        #######################################
        ### STEP 3: solve the optimisation
        #######################################
        print('*** SOLVING THE OPTIMISATION PROBLEM ***')
        #prob.solve(verbose = 0,solver='cvxopt')
        prob.solve(verbose = 0)
        print('*** OPTIMISATION COMPLETE ***')
        P_ES_val = np.array(P_ES.value)
        P_import_val = np.array(P_import.value)
        P_export_val = np.array(P_export.value)
        P_demand_val = np.array(P_demand)
        return {'opt_prob':prob,\
                'P_ES_val':P_ES_val,\
                'P_import_val':P_import_val,\
                'P_export_val':P_export_val,\
                'P_demand_val':P_demand_val}



    # NEEDED FOR OXEMF EV CASE
    def EMS_3ph_linear_t0(self, t0, i_unconstrained_lines=[],
                          v_unconstrained_buses = []):
        """
        Energy management system optimization assuming 3 phase linear network
        model for Model Predictive Control interval t0

        Parameters
        ----------
        self : EnergySystem object
            Object containing information on assets, market, network and time
            resolution.
        t0 : int
            Interval in Model Predictive Control. If open loop, t0 = 0
        i_unconstrained_lines : list
            List of network lines which have unconstrained current
        v_unconstrained_buses : list
            List of buses at which the voltage is not constrained

        Returns
        -------
        Output : dictionary
            The following numpy.ndarrays are present depending upon asset mix:
                P_ES_val : Charge/discharge power for storage assets (kW)
                P_import_val : Power imported from central grid (kW)
                P_export_val : Power exported to central grid (kW)
                P_demand_val : System power demand at energy management time
                    resolution (kW)
                PF_networks_lin : Network 3ph list of objects, one for each
                    optimisation interval, storing the linear power
                    flow model used to formulate netowrk
                    constraints

        """

        #######################################
        ### STEP 0: setup variables
        #######################################
        prob = pic.Problem()
        t0_dt = int(t0*self.dt_ems/self.dt)
        T_mpc = self.T_ems-t0
        T_range = np.arange(t0,self.T_ems)
        N_buses = self.network.N_buses
        N_phases = self.network.N_phases
        N_ES = len(self.storage_assets)
        N_nondispatch = len(self.nondispatch_assets)
        P_demand_actual = np.zeros([self.T,N_nondispatch])
        P_demand_pred = np.zeros([self.T,N_nondispatch])
        P_demand = np.zeros([T_mpc,N_nondispatch])
        Q_demand_actual = np.zeros([self.T,N_nondispatch])
        Q_demand_pred = np.zeros([self.T,N_nondispatch])
        Q_demand = np.zeros([T_mpc,N_nondispatch])
        for i in range(N_nondispatch):
            P_demand_actual[:,i] = self.nondispatch_assets[i].Pnet
            P_demand_pred[:,i] = self.nondispatch_assets[i].Pnet_pred
            Q_demand_actual[:,i] = self.nondispatch_assets[i].Qnet
            Q_demand_pred[:,i] = self.nondispatch_assets[i].Qnet_pred
        #Assemble P_demand out of P actual and P predicted and convert to EMS
        #time series scale
        for i in range(N_nondispatch):
            for t_ems in T_range:
                t_indexes = (t_ems*self.dt_ems/self.dt +
                             np.arange(0,self.dt_ems/self.dt)).astype(int)
                if t_ems == t0:
                    P_demand[t_ems-t0,i] =\
                    np.mean(P_demand_actual[t_indexes,i])
                    Q_demand[t_ems-t0,i] = \
                    np.mean(Q_demand_actual[t_indexes,i])
                else:
                    P_demand[t_ems-t0,i] = np.mean(P_demand_pred[t_indexes,i])
                    Q_demand[t_ems-t0,i] = np.mean(Q_demand_pred[t_indexes,i])
        #get total ES system demand (before optimisation)
        Pnet_ES_sum = np.zeros(self.T)
        for i in range(N_ES):
            Pnet_ES_sum += self.storage_assets[i].Pnet
        #get the maximum (historical) demand before t0
        if t0 == 0:
            P_max_demand_pre_t0 = 0
        else:
            if N_nondispatch == 0: P_max_demand_pre_t0 = Pnet_ES_sum[0:t0_dt]
            else:
                P_demand_act_sum = sum(P_demand_actual[0:t0_dt,i] \
                                       for i in range(N_nondispatch))
                P_max_demand_pre_t0 = np.max(P_demand_act_sum +
                                                 Pnet_ES_sum[0:t0_dt])

        #Set up Matrix linking nondispatchable assets to their bus and phase
        G_wye_nondispatch = np.zeros([3*(N_buses-1),N_nondispatch])
        G_del_nondispatch = np.zeros([3*(N_buses-1),N_nondispatch])
        for i in range(N_nondispatch):
            asset_N_phases = self.nondispatch_assets[i].phases.size
            bus_id = self.nondispatch_assets[i].bus_id
            # check if Wye connected
            wye_flag = self.network.bus_df[self.\
                                           network.bus_df['number']==\
                                           bus_id]['connect'].values[0]=='Y'
            for ph in np.nditer(self.nondispatch_assets[i].phases):
                bus_ph_index = 3*(bus_id-1) + ph
                if wye_flag is True:
                    G_wye_nondispatch[bus_ph_index,i] = 1/asset_N_phases
                else:
                    G_del_nondispatch[bus_ph_index,i] = 1/asset_N_phases
        #Set up Matrix linking energy storage assets to their bus and phase
        G_wye_ES = np.zeros([3*(N_buses-1),N_ES])
        G_del_ES = np.zeros([3*(N_buses-1),N_ES])
        for i in range(N_ES):
            asset_N_phases = self.storage_assets[i].phases.size
            bus_id = self.storage_assets[i].bus_id
            # check if Wye connected
            wye_flag = self.network.bus_df[self.\
                                           network.bus_df['number']==\
                                           bus_id]['connect'].values[0]=='Y'
            for ph in np.nditer(self.storage_assets[i].phases):
                bus_ph_index = 3*(bus_id-1) + ph
                if wye_flag is True:
                    G_wye_ES[bus_ph_index,i] = 1/asset_N_phases
                else:
                    G_del_ES[bus_ph_index,i] = 1/asset_N_phases
        G_wye_nondispatch_PQ = np.concatenate((G_wye_nondispatch,
                                               G_wye_nondispatch),axis=0)
        G_del_nondispatch_PQ = np.concatenate((G_del_nondispatch,
                                               G_del_nondispatch),axis=0)
        G_wye_ES_PQ = np.concatenate((G_wye_ES,G_wye_ES),axis=0)
        G_del_ES_PQ = np.concatenate((G_del_ES,G_del_ES),axis=0)
        #######################################
        ### STEP 1: set up decision variables
        #######################################

        # energy storage system input powers
        P_ES = prob.add_variable('P_ES',
                                 (T_mpc,N_ES), vtype='continuous')
        # energy storage system input powers
        P_ES_ch = prob.add_variable('P_ES_ch',
                                    (T_mpc,N_ES), vtype='continuous')
        # energy storage system output powers
        P_ES_dis = prob.add_variable('P_ES_dis',
                                     (T_mpc,N_ES), vtype='continuous')
        # (positive) net power imports
        P_import = prob.add_variable('P_import',
                                     (T_mpc,1), vtype='continuous')
        # (positive) net power exports
        P_export = prob.add_variable('P_export',
                                     (T_mpc,1), vtype='continuous')
        # (positive) maximum demand dummy variable
        P_max_demand = prob.add_variable('P_max_demand',
                                         1, vtype='continuous')
        # (positive) minimum terminal energy dummy variable
        E_T_min = prob.add_variable('E_T_min',
                                    N_ES, vtype='continuous')

        #######################################
        ### STEP 2: set up linear power flow models
        #######################################
        PF_networks_lin = []
        P_lin_buses = np.zeros([T_mpc,N_buses,N_phases])
        Q_lin_buses = np.zeros([T_mpc,N_buses,N_phases])
        for t in range(T_mpc):
            #Setup linear power flow model:
            for i in range(N_nondispatch):
                bus_id = self.nondispatch_assets[i].bus_id
                phases_i = self.nondispatch_assets[i].phases
                for ph_i in np.nditer(phases_i):
                    bus_ph_index = 3*(bus_id-1) + ph_i
                    P_lin_buses[t,bus_id,ph_i] +=\
                    (G_wye_nondispatch[bus_ph_index,i]+\
                     G_del_nondispatch[bus_ph_index,i])*P_demand[t,i]
                    Q_lin_buses[t,bus_id,ph_i] +=\
                    (G_wye_nondispatch[bus_ph_index,i]+\
                     G_del_nondispatch[bus_ph_index,i])*Q_demand[t,i]
            #set up a copy of the network for MPC interval t
            network_t = copy.deepcopy(self.network)
            network_t.clear_loads()
            for bus_id in range(N_buses):
                for ph_i in range(N_phases):
                    Pph_t = P_lin_buses[t,bus_id,ph_i]
                    Qph_t = Q_lin_buses[t,bus_id,ph_i]
                    #add P,Q loads to the network copy
                    network_t.set_load(bus_id,ph_i,Pph_t,Qph_t)
            network_t.zbus_pf()
            v_lin0 = network_t.v_net_res
            S_wye_lin0 = network_t.S_PQloads_wye_res
            S_del_lin0 = network_t.S_PQloads_del_res
            network_t.linear_model_setup(v_lin0,S_wye_lin0,S_del_lin0)
            # note that phases need to be 120degrees out for good results
            network_t.linear_pf()
            PF_networks_lin.append(network_t)
        #######################################
        ### STEP 3: set up constraints
        #######################################
        # lower triangle matrix summing powers
        Asum = pic.new_param('Asum',np.tril(np.ones([T_mpc,T_mpc])))

        # energy storage asset constraints
        for i in range(N_ES):
            # maximum power constraint
            prob.add_constraint(P_ES[:,i] <=
                                self.storage_assets[i].Pmax[T_range])
            # minimum power constraint
            prob.add_constraint(P_ES[:,i] >=
                                self.storage_assets[i].Pmin[T_range])
            # maximum energy constraint
            prob.add_constraint(self.dt_ems * Asum * (P_ES_ch[:,i] -
                                                  P_ES_dis[:,i]) <=
                                self.storage_assets[i].Emax[T_range] -
                                self.storage_assets[i].E[t0_dt])
            # minimum energy constraint
            prob.add_constraint(self.dt_ems * Asum * (P_ES_ch[:,i] -
                                                  P_ES_dis[:,i]) >=
                                self.storage_assets[i].Emin[T_range] -
                                self.storage_assets[i].E[t0_dt])
            # final energy constraint
            prob.add_constraint(self.dt_ems * Asum[T_mpc-1,:] * (P_ES_ch[:,i] -
                                P_ES_dis[:,i]) + E_T_min[i] >=
                                self.storage_assets[i].ET -
                                self.storage_assets[i].E[t0_dt])

            eff_opt = self.storage_assets[i].eff_opt

            #P_ES_ch & P_ES_dis dummy variables
            for t in range(T_mpc):
                prob.add_constraint(P_ES[t,i] == P_ES_ch[t,i]/eff_opt -
                                    P_ES_dis[t,i] * eff_opt)
                prob.add_constraint(P_ES_ch[t,i] >= 0)
                prob.add_constraint(P_ES_dis[t,i] >= 0)

        #import/export constraints
        for t in range(T_mpc):
            # maximum import constraint
            prob.add_constraint(P_import[t] <= self.market.Pmax[t0 + t])
            # maximum import constraint
            prob.add_constraint(P_import[t] >= 0)
            # maximum import constraint
            prob.add_constraint(P_export[t] <= -self.market.Pmin[t0 + t])
            # maximum import constraint
            prob.add_constraint(P_export[t] >= 0)
            # maximum demand dummy variable constraint
            prob.add_constraint(P_max_demand + P_max_demand_pre_t0 >=
                                P_import[t]-P_export[t])
            # maximum demand dummy variable constraint
            prob.add_constraint(P_max_demand  >= 0)

        # Network constraints
        for t in range(T_mpc):
            network_t = PF_networks_lin[t]
            # Note that linear power flow matricies are in units of W (not kW)
            PQ0_wye = np.concatenate((np.real(network_t.S_PQloads_wye_res),\
                                      np.imag(network_t.S_PQloads_wye_res)))\
                                      *1e3
            PQ0_del = np.concatenate((np.real(network_t.S_PQloads_del_res),\
                                      np.imag(network_t.S_PQloads_del_res)))\
                                      *1e3
            A_Pslack = (np.matmul\
                        (np.real(np.matmul\
                                 (network_t.vs.T,\
                                  np.matmul(np.conj(network_t.Ysn),\
                                            np.conj(network_t.M_wye)))),\
                                      G_wye_ES_PQ)\
                         + np.matmul\
                         (np.real(np.matmul\
                                  (network_t.vs.T,\
                                   np.matmul(np.conj(network_t.Ysn),\
                                             np.conj(network_t.M_del)))),\
                                      G_del_ES_PQ))
            b_Pslack =  np.real(np.matmul\
                                (network_t.vs.T,\
                                 np.matmul(np.conj\
                                           (network_t.Ysn),\
                                           np.matmul(np.conj\
                                                     (network_t.M_wye),\
                                                     PQ0_wye))))\
                        +np.real(np.matmul\
                                 (network_t.vs.T,\
                                  np.matmul(np.conj\
                                            (network_t.Ysn),\
                                            np.matmul(np.conj\
                                                      (network_t.M_del),
                                                      PQ0_del))))\
                        +np.real(np.matmul\
                                 (network_t.vs.T,\
                                  (np.matmul(np.conj\
                                             (network_t.Yss),\
                                             np.conj(network_t.vs))\
                                 + np.matmul(np.conj\
                                             (network_t.Ysn),\
                                             np.conj(network_t.M0)))))
            # net import variables
            prob.add_constraint(P_import[t]-P_export[t] ==\
                                (np.sum(A_Pslack[i]*P_ES[t,i]\
                                        *1e3 for i in range(N_ES))\
                                + b_Pslack)/1e3)

            # Voltage magnitude constraints
            A_vlim = np.matmul(network_t.K_wye,G_wye_ES_PQ)\
                    + np.matmul(network_t.K_del,G_del_ES_PQ)
            b_vlim = network_t.v_lin_abs_res
            #get max/min bus voltages, removing slack and reshaping in a column
            v_abs_max_vec = network_t.v_abs_max[1:,:].reshape(-1,1)
            v_abs_min_vec = network_t.v_abs_min[1:,:].reshape(-1,1)
            for bus_ph_index in range(0,N_phases*(N_buses-1)):
                if int(bus_ph_index/3) not in (np.array\
                      (v_unconstrained_buses)-1):
                    prob.add_constraint(sum(A_vlim[bus_ph_index,i]\
                                            *(P_ES[t,i])\
                                            *1e3 for i in range(N_ES))\
                                        + b_vlim[bus_ph_index] <=\
                                        v_abs_max_vec[bus_ph_index])
                    prob.add_constraint(sum(A_vlim[bus_ph_index,i]\
                                            *(P_ES[t,i])\
                                            *1e3 for i in range(N_ES))\
                                        + b_vlim[bus_ph_index] >=\
                                        v_abs_min_vec[bus_ph_index])

            # Line current magnitude constraints:
            for line_ij in range(network_t.N_lines):
                if line_ij not in i_unconstrained_lines:
                    iabs_max_line_ij = network_t.i_abs_max[line_ij,:] #3 phases
                    # maximum current magnitude constraint
                    A_line = np.matmul(network_t.Jabs_dPQwye_list[line_ij],\
                                       G_wye_ES_PQ)\
                                       + np.matmul(network_t.\
                                                   Jabs_dPQdel_list[line_ij],\
                                                   G_del_ES_PQ)
                    for ph in range(N_phases):
                        prob.add_constraint(sum(A_line[ph,i]\
                                                * P_ES[t,i]\
                                                * 1e3 for i in range(N_ES))\
                                           + network_t.\
                                           Jabs_I0_list[line_ij][ph] <=\
                                           iabs_max_line_ij[ph])
       #if FFR energy constraints
        if self.market.FR_window is not None:
            FR_window = self.market.FR_window
            FR_SoC_max = self.market.FR_SOC_max
            FR_SoC_min = self.market.FR_SOC_min
            for t in range(len(T_mpc)):
                if FR_window[t] ==1:
                    for i in range(N_ES):
                        # final energy constraint
                        prob.add_constraint((self.dt_ems
                                             * Asum[t, :]
                                             * P_ES[:,i])\
                                            <= ((FR_SoC_max
                                                * self.storage_assets[i].Emax)
                                           - self.storage_assets[i].E[t0_dt]))
                        # final energy constraint
                        prob.add_constraint((self.dt_ems
                                             * Asum[t,:]
                                             * P_ES[:,i])\
                                            >= ((FR_SoC_min
                                                * self.storage_assets[i].Emax)
                                           - self.storage_assets[i].E[t0_dt]))

        #######################################
        ### STEP 4: set up objective
        #######################################
        # minimum terminal energy dummy variable  constraint
        prob.add_constraint(E_T_min[i] >= 0)
        #coeff for objective terminal soft constraint
        terminal_const = 1e3
        prices_import = pic.new_param('prices_import',
                                      self.market.prices_import)
        prices_export = pic.new_param('prices_export',
                                      self.market.prices_export)


        prob.set_objective('min', self.market.demand_charge*\
                           (P_max_demand+P_max_demand_pre_t0) +
                           sum(sum(self.dt_ems*self.storage_assets[i].\
                                   c_deg_lin*(P_ES_ch[t,i]+
                                              P_ES_dis[t,i])\
                                              for i in range(N_ES))\
                            + self.dt_ems*prices_import[t0+t]*P_import[t]\
                            - self.dt_ems*prices_export[t0+t]*P_export[t]
                             for t in range(T_mpc))\
                            + sum(terminal_const*E_T_min[i]\
                                  for i in range(N_ES)))

        #######################################
        ### STEP 5: solve the optimisation
        #######################################
        print('*** SOLVING THE OPTIMISATION PROBLEM ***')
        prob.solve(verbose = 0)
        print('*** OPTIMISATION COMPLETE ***')
        P_ES_val = np.matrix(P_ES.value)
        P_import_val = np.matrix(P_import.value)
        P_export_val = np.matrix(P_export.value)
        P_demand_val = np.matrix(P_demand)
        return {'P_ES_val':P_ES_val,
                'P_import_val':P_import_val,
                'P_export_val':P_export_val,
                'P_demand_val':P_demand_val,
                'PF_networks_lin':PF_networks_lin}

    # NEEDED FOR OXEMF EV CASE
    def simulate_network_mpc_3phPF(self, ems_type = '3ph',
                                   i_unconstrained_lines=[],
                                   v_unconstrained_buses = []):
        """
        Run the Energy Management System using Model Predictive Control (MPC)
        and simulate an IEEE 13 bus network either copper plate or 3ph

        Parameters
        ----------
        self : EnergySystem object
            Object containing information on assets, market, network and time
            resolution.
        ems_type : string
            Identifies whether the system is copper plate or 3ph. Default 3ph
        i_unconstrained_lines : list
            List of network lines which have unconstrained current
        v_unconstrained_buses : list
            List of buses at which the voltage is not constrained

        Returns
        -------
        Output : dictionary
                PF_network_res : Network power flow results stored as a list of
                    objects
                P_ES_ems : Charge/discharge power for storage assets at energy
                    management time resolution (kW)
                P_import_ems :Power imported from central grid at energy
                    management time resolution (kW)
                P_export_ems :Power exported to central grid at energy
                    management time resolution(kW)
                P_demand_ems :System power demand at energy management time
                    resolution (kW)

        """

        #######################################
        ### STEP 0: setup variables
        #######################################
        N_ESs = len(self.storage_assets) #number of EVs
        N_nondispatch = len(self.nondispatch_assets) #number of EVs
        P_import_ems = np.zeros(self.T_ems)
        P_export_ems = np.zeros(self.T_ems)
        P_ES_ems = np.zeros([self.T_ems,N_ESs])
        if ems_type == 'copper_plate':
            P_demand_ems = np.zeros(self.T_ems)
        else:
            P_demand_ems = np.zeros([self.T_ems,N_nondispatch])
        N_buses = self.network.N_buses
        N_phases = self.network.N_phases
        P_demand_buses = np.zeros([self.T,N_buses,N_phases])
        Q_demand_buses = np.zeros([self.T,N_buses,N_phases])
        PF_network_res = []
        #######################################
        ### STEP 1: MPC Loop
        #######################################
        print('*** MPC SIMULATION START ***')
        for t_mpc in range(self.T_ems):
            print('************************')
            print('MPC Interval '+ str(t_mpc)+ ' of '+ str(self.T_ems))
            print('************************')
            #######################################
            ### STEP 1.1: Optimisation
            #######################################
            if ems_type == 'copper_plate':
                output_ems = self.EMS_copper_plate_t0_c1deg(t_mpc)
                P_demand_ems[t_mpc] = output_ems['P_demand_val'][0]
            else:
                output_ems = self.EMS_3ph_linear_t0(t_mpc,
                                                    i_unconstrained_lines,
                                                    v_unconstrained_buses)
                P_demand_ems[t_mpc,:] = output_ems['P_demand_val'][0,:]
            P_import_ems[t_mpc] = output_ems['P_import_val'][0]
            P_export_ems[t_mpc] = output_ems['P_export_val'][0]
            P_ES_ems[t_mpc,:] = output_ems['P_ES_val'][0,:]
            # convert P_EV signals to system time-series scale
            T_interval = int(self.dt_ems/self.dt)
            P_ESs = np.zeros([T_interval,N_ESs])
            for t in range(T_interval):
                P_ESs[t,:] = P_ES_ems[t_mpc,:]
            #######################################
            ### STEP 1.2: update the controllable assets
            #######################################
            t0 = int(t_mpc*(self.dt_ems/self.dt))
            # get the simulation time intervals within each EMS time interval
            # and implement the ES system control for them
            t_range = np.arange(t0,t0+T_interval)
            for i in range(N_ESs):
                for t_index in range(T_interval):
                    t = t_range[t_index]
                    self.storage_assets[i].update_control_t(P_ESs[t_index,i],t)
            #######################################
            ### STEP 1.3: simulate the network
            #######################################
            # total real and reactive power demand at each bus phase
            for t_index in range(T_interval):
                t = t_range[t_index]
                for i in range(N_ESs):
                    bus_id = self.storage_assets[i].bus_id
                    phases_i = self.storage_assets[i].phases
                    N_phases_i = np.size(phases_i)
                    for ph_i in phases_i:
                        P_demand_buses[t,bus_id,ph_i] +=\
                        self.storage_assets[i].Pnet[t]/N_phases_i
                        Q_demand_buses[t,bus_id,ph_i] +=\
                        self.storage_assets[i].Qnet[t]/N_phases_i
                for i in range(N_nondispatch):
                    bus_id = self.nondispatch_assets[i].bus_id
                    phases_i = self.nondispatch_assets[i].phases
                    N_phases_i = np.size(phases_i)
                    for ph_i in np.nditer(phases_i):
                        P_demand_buses[t,bus_id,ph_i] +=\
                        self.nondispatch_assets[i].Pnet[t]/N_phases_i
                        Q_demand_buses[t,bus_id,ph_i] +=\
                        self.nondispatch_assets[i].Qnet[t]/N_phases_i
                # set up a copy of the network for simulation interval t
                network_t = copy.deepcopy(self.network)
                network_t.clear_loads()
                for bus_id in range(N_buses):
                    for ph_i in range(N_phases):
                        Pph_t = P_demand_buses[t,bus_id,ph_i]
                        Qph_t = Q_demand_buses[t,bus_id,ph_i]
                        #add P,Q loads to the network copy
                        network_t.set_load(bus_id,ph_i,Pph_t,Qph_t)
                # run the power flow simulation
                network_t.zbus_pf()
                # store power flow results as a list of network objects
                PF_network_res.append(network_t)
        print('*** MPC SIMULATION COMPLETE ***')
        return {'PF_network_res' :PF_network_res,\
                'P_ES_ems':P_ES_ems,\
                'P_import_ems':P_import_ems,\
                'P_export_ems':P_export_ems,\
                'P_demand_ems':P_demand_ems}



    def simulate_network_3phPF_lean(self, ems_type = '3ph'):
        """
        run the EMS in open loop and simulate a 3-phase AC network
        """
        #######################################
        ### STEP 1: solve the optimisation
        #######################################
        t0 = 0
        if ems_type == 'copper_plate':
            # self.EMS_copper_plate()
            output_ems = self.EMS_copper_plate_t0_c1deg(t0)
        else:
            # self.EMS_copper_plate()
            output_ems = self.EMS_3ph_linear_t0(t0)

        #output_ems = self.EMS_copper_plate
        P_import_ems = output_ems['P_import_val']
        P_export_ems = output_ems['P_export_val']
        P_ES_ems = output_ems['P_ES_val']
        P_demand_ems = output_ems['P_demand_val']

        #convert P_EV signals to system time-series scale
        N_ESs = len(self.storage_assets) #number of EVs
        N_nondispatch = len(self.nondispatch_assets) #number of EVs
        P_ESs = np.zeros([self.T,N_ESs])
        for t in range(self.T):
            t_ems = int(t/(self.dt_ems/self.dt))
            P_ESs[t,:] = P_ES_ems[t_ems,:]
        #######################################
        ### STEP 2: update the controllable assets
        #######################################
        for i in range(N_ESs):
            self.storage_assets[i].update_control(P_ESs[:,i])
        #######################################
        ### STEP 3: simulate the network
        #######################################
        N_buses = self.network.N_buses
        N_phases = self.network.N_phases
        P_demand_buses = np.zeros([self.T,N_buses,N_phases])
        Q_demand_buses = np.zeros([self.T,N_buses,N_phases])
        #calculate the total real and reactive power demand at each bus phase
        for i in range(N_ESs):
            bus_id = self.storage_assets[i].bus_id
            phases_i = self.storage_assets[i].phases
            N_phases_i = np.size(phases_i)
            for ph_i in np.nditer(phases_i):
                P_demand_buses[:,bus_id,ph_i] += (self.storage_assets[i].Pnet
                                                  / N_phases_i)
                Q_demand_buses[:,bus_id,ph_i] += (self.storage_assets[i].Qnet
                                                  / N_phases_i)
        for i in range(N_nondispatch):
            bus_id = self.nondispatch_assets[i].bus_id
            phases_i = self.nondispatch_assets[i].phases
            N_phases_i = np.size(phases_i)
            for ph_i in np.nditer(phases_i):
                P_demand_buses[:, bus_id, ph_i]\
                    += (self.nondispatch_assets[i].Pnet / N_phases_i)
                Q_demand_buses[:, bus_id, ph_i]\
                    += (self.nondispatch_assets[i].Qnet / N_phases_i)
        #Store power flow results as a list of network objects

        PF_network_res = []
        print('*** SIMULATING THE NETWORK ***')
        for t in range(self.T):
            #for each time interval:
            #set up a copy of the network for simulation interval t
            network_t = copy.deepcopy(self.network)
            network_t.clear_loads()
            for bus_id in range(N_buses):
                for ph_i in range(N_phases):
                    Pph_t = P_demand_buses[t,bus_id,ph_i]
                    Qph_t = Q_demand_buses[t,bus_id,ph_i]
                    #add P,Q loads to the network copy
                    network_t.set_load(bus_id,ph_i,Pph_t,Qph_t)
            #run the power flow simulation
            network_t.zbus_pf()
            if t % 1 == 0:
                print('network sim complete for t = '
                      + str(t) + ' of ' + str(self.T))
            PF_network_res.append(network_t.res_bus_df)
        print('*** NETWORK SIMULATION COMPLETE ***')

        return {'PF_network_res' :PF_network_res,\
                'P_ES_ems':P_ES_ems,\
                'P_import_ems':P_import_ems,\
                'P_export_ems':P_export_ems,\
                'P_demand_ems':P_demand_ems}


