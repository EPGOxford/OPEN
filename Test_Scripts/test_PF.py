
#######################################
### STEP 0: setup variables
#######################################
prob = pic.Problem()
t0_dt = int(t0*dt_ems/dt)
T_mpc = T_ems-t0
T_range = np.arange(t0,T_ems)
N_buses = network.N_buses
N_phases = network.N_phases
N_ES = len(storage_assets)
N_nondispatch = len(nondispatch_assets)
P_demand_actual = np.zeros([T,N_nondispatch])
P_demand_pred = np.zeros([T,N_nondispatch])
P_demand = np.zeros([T_mpc,N_nondispatch])
Q_demand_actual = np.zeros([T,N_nondispatch])
Q_demand_pred = np.zeros([T,N_nondispatch])
Q_demand = np.zeros([T_mpc,N_nondispatch])
for i in range(N_nondispatch):
    P_demand_actual[:,i] = nondispatch_assets[i].Pnet
    P_demand_pred[:,i] = nondispatch_assets[i].Pnet_pred
    Q_demand_actual[:,i] = nondispatch_assets[i].Qnet
    Q_demand_pred[:,i] = nondispatch_assets[i].Qnet_pred
#Assemble P_demand out of P actual and P predicted and convert to EMS time series scale
for i in range(N_nondispatch):
    for t_ems in T_range:
        t_indexes = (t_ems*dt_ems/dt + np.arange(0,dt_ems/dt)).astype(int)
        if t_ems == t0:
            P_demand[t_ems-t0,i] = np.mean(P_demand_actual[t_indexes,i])
            Q_demand[t_ems-t0,i] = np.mean(Q_demand_actual[t_indexes,i])
        else:
            P_demand[t_ems-t0,i] = np.mean(P_demand_pred[t_indexes,i])
            Q_demand[t_ems-t0,i] = np.mean(Q_demand_pred[t_indexes,i])
#get total ES system demand (before optimisation)
Pnet_ES_sum = np.zeros(T)
for i in range(N_nondispatch):
    Pnet_ES_sum += storage_assets[i].Pnet  
#get the maximum (historical) demand before t0
if t0 > 0:
    P_max_demand_pre_t0 = np.max(P_demand_actual[0:t0_dt]+Pnet_ES_sum[0:t0_dt]) 
else:
    P_max_demand_pre_t0 = 0
#Set up Matrix linking nondispatchable assets to their bus and phase
G_wye_nondispatch = np.zeros([3*(N_buses-1),N_nondispatch])
G_del_nondispatch = np.zeros([3*(N_buses-1),N_nondispatch])
for i in range(N_nondispatch):
    asset_N_phases = nondispatch_assets[i].phases.size
    bus_id = nondispatch_assets[i].bus_id
    wye_flag = network.bus_df[network.bus_df['number']==bus_id]['connect'].values[0]=='Y' #check if Wye connected
    for ph in nondispatch_assets[i].phases:
        bus_ph_index = 3*bus_id + ph
        if wye_flag is True:
            G_wye_nondispatch[bus_ph_index,i] = 1/asset_N_phases   
        else:
            G_del_nondispatch[bus_ph_index,i] = 1/asset_N_phases   
#Set up Matrix linking energy storage assets to their bus and phase
G_wye_ES = np.zeros([3*(N_buses-1),N_ES])  
G_del_ES = np.zeros([3*(N_buses-1),N_ES])   
for i in range(N_ES):
    asset_N_phases = storage_assets[i].phases.size
    bus_id = storage_assets[i].bus_id
    wye_flag = network.bus_df[network.bus_df['number']==bus_id]['connect'].values[0]=='Y' #check if Wye connected
    for ph in storage_assets[i].phases:
        bus_ph_index = 3*bus_id + ph
        if wye_flag is True:
            G_wye_ES[bus_ph_index,i] = 1/asset_N_phases   
        else:
            G_del_ES[bus_ph_index,i] = 1/asset_N_phases   
G_wye_nondispatch_PQ = np.concatenate((G_wye_nondispatch,G_wye_nondispatch),axis=0)  
G_del_nondispatch_PQ = np.concatenate((G_del_nondispatch,G_del_nondispatch),axis=0)  
G_wye_ES_PQ = np.concatenate((G_wye_ES,G_wye_ES),axis=0)   
G_del_ES_PQ = np.concatenate((G_del_ES,G_del_ES),axis=0)   
#######################################
### STEP 1: set up decision variables
#######################################
P_ES = prob.add_variable('P_ES',(T_mpc,N_ES), vtype='continuous') #energy storage system input powers
P_import = prob.add_variable('P_import',(T_mpc,1), vtype='continuous') #(positive) net power imports 
P_export = prob.add_variable('P_export',(T_mpc,1), vtype='continuous') #(positive) net power exports 
P_max_demand = prob.add_variable('P_max_demand',1, vtype='continuous') #(positive) maximum demand dummy variable 
E_T_min = prob.add_variable('E_T_min',1, vtype='continuous') #(positive) minimum terminal energy dummy variable 
#######################################
### STEP 2: set up linear power flow models
#######################################
PF_networks_lin = []
P_lin_buses = np.zeros([T_mpc,N_buses,N_phases])
Q_lin_buses = np.zeros([T_mpc,N_buses,N_phases])
for t in range(T_mpc):
    #Setup linear power flow model:
    for i in range(N_nondispatch):
        bus_id = nondispatch_assets[i].bus_id
        phases_i = nondispatch_assets[i].phases
        for ph_i in phases_i:
            bus_ph_index = 3*bus_id + ph_i
            P_lin_buses[t,bus_id,ph_i] += (G_wye_nondispatch[bus_ph_index,i]+G_del_nondispatch[bus_ph_index,i])*P_demand[t,i]
            Q_lin_buses[t,bus_id,ph_i] += (G_wye_nondispatch[bus_ph_index,i]+G_del_nondispatch[bus_ph_index,i])*Q_demand[t,i]
    #set up a copy of the network for MPC interval t
    network_t = copy.deepcopy(network)
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
    network_t.linear_model_setup(v_lin0,S_wye_lin0,S_del_lin0) #note that phases need to be 120degrees out for good results
    network_t.linear_pf()
    PF_networks_lin.append(network_t)
#######################################
### STEP 3: set up constraints
#######################################
Asum = pic.new_param('Asum',np.tril(np.ones([T_mpc,T_mpc]))) #lower triangle matrix summing powers
#linear battery model constraints
for i in range(N_ES):
    prob.add_constraint(P_ES[:,i] <= storage_assets[i].Pmax[T_range]) #maximum power constraint
    prob.add_constraint(P_ES[:,i] >= storage_assets[i].Pmin[T_range]) #minimum power constraint
    prob.add_constraint(dt_ems*Asum*P_ES[:,i] <= storage_assets[i].Emax[T_range]-storage_assets[i].E[t0_dt]) #maximum energy constraint
    prob.add_constraint(dt_ems*Asum*P_ES[:,i] >= storage_assets[i].Emin[T_range]-storage_assets[i].E[t0_dt]) #minimum energy constraint
    prob.add_constraint(dt_ems*Asum[T_mpc-1,:]*P_ES[:,i] + E_T_min >= storage_assets[i].ET-storage_assets[i].E[t0_dt]) #final energy constraint
#import/export constraints
for t in range(T_mpc):
    prob.add_constraint(P_import[t] <= market.Pmax[t0+t]) #maximum import constraint
    prob.add_constraint(P_import[t] >= 0) #maximum import constraint
    prob.add_constraint(P_export[t] <= -market.Pmin[t0+t]) #maximum import constraint
    prob.add_constraint(P_export[t] >= 0) #maximum import constraint
    prob.add_constraint(P_max_demand + P_max_demand_pre_t0 >= P_import[t]-P_export[t]) #maximum demand dummy variable constraint
    prob.add_constraint(P_max_demand  >= 0) #maximum demand dummy variable constraint
    prob.add_constraint(E_T_min[:] >= 0) #minimum terminal energy dummy variable  constraint
#Network constraints
for t in range(T_mpc):
    network_t = PF_networks_lin[t]
    #Power balance constraint: Pimport - Pexport = P_ES + P_demand + Losses = Power Flow at slack bus 
    #Note that linear power flow matricies are in units of W (not kW)
    PQ0_wye = np.concatenate((np.real(network_t.S_PQloads_wye_res),np.imag(network_t.S_PQloads_wye_res)))*1e3
    PQ0_del = np.concatenate((np.real(network_t.S_PQloads_del_res),np.imag(network_t.S_PQloads_del_res)))*1e3
    A_Pslack = (np.matmul(np.real(np.matmul(network_t.vs.T,np.matmul(np.conj(network_t.Ysn),np.conj(network_t.M_wye)))),G_wye_ES_PQ)\
                 + np.matmul(np.real(np.matmul(network_t.vs.T,np.matmul(np.conj(network_t.Ysn),np.conj(network_t.M_del)))),G_del_ES_PQ))
    b_Pslack =   np.real(np.matmul(network_t.vs.T,np.matmul(np.conj(network_t.Ysn),np.matmul(np.conj(network_t.M_wye),PQ0_wye))))\
                +np.real(np.matmul(network_t.vs.T,np.matmul(np.conj(network_t.Ysn),np.matmul(np.conj(network_t.M_del),PQ0_del))))\
                +np.real(np.matmul(network_t.vs.T,(np.matmul(np.conj(network_t.Yss),np.conj(network_t.vs))+np.matmul(np.conj(network_t.Ysn),np.conj(network_t.M0)))))
    prob.add_constraint(P_import[t]-P_export[t] == (np.sum(A_Pslack[i]*P_ES[t,i]*1e3 for i in range(N_ES)) + b_Pslack)/1e3) #net import variables
    #Voltage magnitude constraints
    A_vlim = np.matmul(network_t.K_wye,G_wye_ES_PQ) + np.matmul(network_t.K_del,G_del_ES_PQ)
    b_vlim = network_t.v_lin_abs_res 
    #get max/min bus voltages, removing slack and reshaping in a column
    v_abs_max_vec = network_t.v_abs_max[1:,:].reshape(-1,1)
    v_abs_min_vec = network_t.v_abs_min[1:,:].reshape(-1,1)
    for bus_ph_index in range(0,N_phases*(N_buses-1)):
        prob.add_constraint(sum(A_vlim[bus_ph_index,i]*(P_ES[t,i])*1e3 for i in range(N_ES)) + b_vlim[bus_ph_index] <= v_abs_max_vec[bus_ph_index])
        prob.add_constraint(sum(A_vlim[bus_ph_index,i]*(P_ES[t,i])*1e3 for i in range(N_ES)) + b_vlim[bus_ph_index] >= v_abs_min_vec[bus_ph_index])
    #Line current magnitude constraints:
    for line_ij in range(network_t.N_lines):
        iabs_max_line_ij = network_t.i_abs_max[line_ij,:] #3 phases
        #maximum current magnitude constraint
        A_line = np.matmul(network_t.Jabs_dPQwye_list[line_ij],G_wye_ES_PQ) + np.matmul(network_t.Jabs_dPQdel_list[line_ij],G_del_ES_PQ)   
        for ph in range(N_phases):
            prob.add_constraint(sum(A_line[ph,i]*P_ES[t,i]*1e3 for i in range(N_ES))+ network_t.Jabs_I0_list[line_ij][ph] <= iabs_max_line_ij[ph])
        
#######################################
### STEP 4: set up objective
#######################################
terminal_const = 1e12 #coeff for objective terminal soft constraint
print(E_T_min)
prob.set_objective('min',market.demand_charge*P_max_demand+\
                   sum(market.prices_import[t]*P_import[t]+\
                     -market.prices_export[t]*P_export[t]\
                     for t in range(T_mpc)) + terminal_const*E_T_min)#terminal_const*sum(E_T_min[i] for i in range(N_ES)))
#######################################
### STEP 5: solve the optimisation
#######################################
print('*** SOLVING THE OPTIMISATION PROBLEM ***')
prob.solve(verbose = 0)
print('*** OPTIMISATION COMPLETE ***')
P_ES_val = P_ES.value
P_import_val = P_import.value
P_export_val = P_export.value
P_demand_val = P_demand
