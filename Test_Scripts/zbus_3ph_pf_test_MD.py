# -*- coding: utf-8 -*-
import sys, os

# path = os.path.dirname(os.path.dirname(__file__)) 
path = os.path.dirname(os.path.dirname(sys.argv[0]))
sys.path.insert(0, os.path.join(path,'System'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# from System.Network_3ph_pf import Network_3ph 
from Network_3ph_pf import Network_3ph 
import copy
import time

feeder = '13BusOxEmf'

ntwxs = os.path.join(path,'Data','Networks')
dir0 = os.path.join(ntwxs,feeder+'_dss')
sn0 = os.path.join(dir0,feeder)

net_ieee13 = Network_3ph()

line_df0 = net_ieee13.line_df
trn_df = net_ieee13.transformer_df

bus_columns = ['name','number','v_base','load_type','connect','Pa','Pb','Pc','Qa','Qb','Qc']
bus_index = range(net_ieee13.N_buses)
# #bus_df = pd.DataFrame(index = bus_index,columns = bus_columns)
# #bus_df.iloc[0]= {'name':'650','number':0,  'load_type':'S', 'connect':'Y','Pa':  0,'Pb':  0,'Pc':  0,'Qa':  0,'Qb':  0,'Qc':  0}
# #bus_df.iloc[1]= {'name':'632','number':1,  'load_type':'PQ','connect':'Y','Pa':  0,'Pb':  0,'Pc':  0,'Qa':  0,'Qb':  0,'Qc':  0}
# #bus_df.iloc[2]= {'name':'645','number':2,  'load_type':'PQ','connect':'Y','Pa':  0,'Pb':150,'Pc':  0,'Qa':  0,'Qb': 50,'Qc':  0}
# #bus_df.iloc[3]= {'name':'646','number':3,  'load_type':'PQ','connect':'D','Pa':  0,'Pb':150,'Pc':  0,'Qa':  0,'Qb': 50,'Qc':  0}
# #bus_df.iloc[4]= {'name':'633','number':4,  'load_type':'PQ','connect':'Y','Pa':  0,'Pb':  0,'Pc':  0,'Qa':  0,'Qb':  0,'Qc':  0}
# #bus_df.iloc[5]= {'name':'634','number':5,  'load_type':'PQ','connect':'Y','Pa':200,'Pb':200,'Pc':200,'Qa': 50,'Qb': 50,'Qc': 50}
# #bus_df.iloc[6]= {'name':'671','number':6,  'load_type':'PQ','connect':'D','Pa':150,'Pb':150,'Pc':150,'Qa': 50,'Qb': 50,'Qc': 50}
# #bus_df.iloc[7]= {'name':'680','number':7,  'load_type':'PQ','connect':'Y','Pa':  0,'Pb':  0,'Pc':  0,'Qa':  0,'Qb':  0,'Qc':  0}
# #bus_df.iloc[8]= {'name':'684','number':8,  'load_type':'PQ','connect':'Y','Pa':  0,'Pb':  0,'Pc':  0,'Qa':  0,'Qb':  0,'Qc':  0}
# #bus_df.iloc[9]= {'name':'611','number':9,  'load_type':'PQ','connect':'Y','Pa':  0,'Pb':  0,'Pc':150,'Qa':  0,'Qb':  0,'Qc': 50}
# #bus_df.iloc[10]={'name':'652','number':10, 'load_type':'PQ','connect':'Y','Pa':150,'Pb':  0,'Pc':  0,'Qa': 50,'Qb':  0,'Qc':  0}
# #bus_df.iloc[11]={'name':'692','number':11, 'load_type':'PQ','connect':'D','Pa':  0,'Pb':  0,'Pc':150,'Qa':  0,'Qb':  0,'Qc': 50}
# #bus_df.iloc[12]={'name':'675','number':12, 'load_type':'PQ','connect':'Y','Pa':150,'Pb':150,'Pc':150,'Qa': 50,'Qb': 50,'Qc': 50}

bus_df = pd.DataFrame(index = bus_index,columns = bus_columns)
bus_df.iloc[0]= {'name':'650','number':0,'v_base':net_ieee13.Vslack_ph,  'load_type':'S', 'connect':'Y','Pa':  0,'Pb':  0,'Pc':  0,'Qa':  0,'Qb':  0,'Qc':  0}
bus_df.iloc[1]= {'name':'632','number':1,'v_base':net_ieee13.Vslack_ph,  'load_type':'PQ','connect':'Y','Pa':  0,'Pb':  0,'Pc':  0,'Qa':  0,'Qb':  0,'Qc':  0}
bus_df.iloc[2]= {'name':'645','number':2,'v_base':net_ieee13.Vslack_ph,  'load_type':'PQ','connect':'Y','Pa':  0,'Pb':150,'Pc':  0,'Qa':  0,'Qb': 50,'Qc':  0}
bus_df.iloc[3]= {'name':'646','number':3,'v_base':net_ieee13.Vslack_ph,  'load_type':'PQ','connect':'D','Pa':  0,'Pb':150,'Pc':  0,'Qa':  0,'Qb': 50,'Qc':  0}
bus_df.iloc[4]= {'name':'633','number':4,'v_base':net_ieee13.Vslack_ph,  'load_type':'PQ','connect':'Y','Pa':  0,'Pb':  0,'Pc':  0,'Qa':  0,'Qb':  0,'Qc':  0}
bus_df.iloc[5]= {'name':'634','number':5,'v_base':net_ieee13.Vslack_ph,  'load_type':'PQ','connect':'Y','Pa':200,'Pb':200,'Pc':200,'Qa': 50,'Qb': 50,'Qc': 50}
bus_df.iloc[6]= {'name':'671','number':6,'v_base':net_ieee13.Vslack_ph,  'load_type':'PQ','connect':'D','Pa':150,'Pb':150,'Pc':150,'Qa': 50,'Qb': 50,'Qc': 50}
bus_df.iloc[7]= {'name':'680','number':7,'v_base':net_ieee13.Vslack_ph,  'load_type':'PQ','connect':'Y','Pa':  0,'Pb':  0,'Pc':  0,'Qa':  0,'Qb':  0,'Qc':  0}
bus_df.iloc[8]= {'name':'684','number':8,'v_base':net_ieee13.Vslack_ph,  'load_type':'PQ','connect':'Y','Pa':  0,'Pb':  0,'Pc':  0,'Qa':  0,'Qb':  0,'Qc':  0}
bus_df.iloc[9]= {'name':'611','number':9,'v_base':net_ieee13.Vslack_ph,  'load_type':'PQ','connect':'Y','Pa':  0,'Pb':  0,'Pc':0,'Qa':  0,'Qb':  0,'Qc': 0}
bus_df.iloc[10]={'name':'652','number':10,'v_base':net_ieee13.Vslack_ph,  'load_type':'PQ','connect':'Y','Pa':0,'Pb':  0,'Pc':  0,'Qa': 0,'Qb':  0,'Qc':  0}
bus_df.iloc[11]={'name':'692','number':11,'v_base':net_ieee13.Vslack_ph,  'load_type':'PQ','connect':'D','Pa':  0,'Pb':  0,'Pc':0,'Qa':  0,'Qb':  0,'Qc': 0}
bus_df.iloc[12]={'name':'675','number':12,'v_base':net_ieee13.Vslack_ph,  'load_type':'PQ','connect':'Y','Pa':150,'Pb':150,'Pc':150,'Qa': 50,'Qb': 50,'Qc': 50}

net_ieee13.bus_df=bus_df

# net_ieee13.zbus_pf()
# RES = net_ieee13.res_bus_df



bus_df2 = pd.read_csv(sn0+"_bus_df.csv",index_col=0)
bus_df2.index=np.arange(len(bus_df2))
bus_df2.loc[:,'load_type'] = 'PQ'
bus_df2.loc[0,'load_type'] = 'S'
bus_df2['name'] = bus_df2['name'].astype("str") # <------ required

# for i in range(bus_df2.shape[0]): # MWE showing that we can't easily convert to an int, for some annoying reason (CF lines_df2)
    # bus_df2.iloc[i,1] = 0
    # print(type(0))
    # print(type(bus_df2.iloc[i,1]))

line_df2 = pd.read_csv(sn0+"_line_df.csv",index_col=0)
line_df2.index=np.arange(len(line_df2))
line_df2.loc[:,'busA'] = list(map(str,line_df2.loc[:,'busA'])) # cannot save numbers as 'str' type when saving as csv
line_df2.loc[:,'busB'] = list(map(str,line_df2.loc[:,'busB'])) # cannot save numbers as 'str' type when saving as csv

# fix up all of the complex numbers to be complex:
for i in range(2,line_df2.shape[1]):
    for j in range(line_df2.shape[0]):
        line_df2.iloc[j,i] = np.complex_(np.complex(line_df2.iloc[j,i])) # convert to (exactly) same format

line_df = net_ieee13.line_df

trn_df2 = pd.DataFrame(columns=trn_df.columns)

# shift = []
# for bus in buses2:
    # shift = shift + np.nonzero(bus==buses)[0].tolist()

net_ieee13.bus_df = bus_df2
net_ieee13.line_df = line_df2
net_ieee13.transformer_df = trn_df2

# net_ieee13.bus_df = bus_df # for testing
# net_ieee13.line_df = line_df
# net_ieee13.transformer_df = trn_df

net_ieee13.update_YandZ()
print(sum(sum(net_ieee13.Y))) # check that this is not zero
net_ieee13.zbus_pf()

RES = net_ieee13.res_bus_df

plt.subplot(131)
plt.plot(abs(RES['Va'])/(4160/np.sqrt(3)),'kx')
plt.ylim((0.85,1.15))
plt.subplot(132)
plt.plot(abs(RES['Vb'])/(4160/np.sqrt(3)),'rx')
plt.ylim((0.85,1.15))
plt.subplot(133)
plt.plot(abs(RES['Vc'])/(4160/np.sqrt(3)),'bx')
plt.ylim((0.85,1.15))
plt.show()

# # bus_name_ev_station = '634'
# # bus_index_ev_station = np.argwhere(net_ieee13.bus_df['name']==bus_name_ev_station)[0,0]

# # #Linear power flow
# # P_ev_station_lin = 500e3
# # P_ev_station_base = 450e3
# # #net_ieee13.bus_df.loc[bus_index_ev_station,'Pa'] = P_ev_station/3/1e3
# # #net_ieee13.bus_df.loc[bus_index_ev_station,'Pb'] = P_ev_station/3/1e3
# # #net_ieee13.bus_df.loc[bus_index_ev_station,'Pc'] = P_ev_station/3/1e3
# # net_ieee13.bus_df.loc[bus_index_ev_station,'Pa'] = P_ev_station_base/3/1e3 
# # net_ieee13.bus_df.loc[bus_index_ev_station,'Pb'] = P_ev_station_lin/1e3 + P_ev_station_base/3/1e3
# # net_ieee13.bus_df.loc[bus_index_ev_station,'Pc'] = P_ev_station_base/3/1e3
# # net_ieee13.bus_df.loc[bus_index_ev_station,'Qa'] = 0
# # net_ieee13.bus_df.loc[bus_index_ev_station,'Qb'] = 0
# # net_ieee13.bus_df.loc[bus_index_ev_station,'Qc'] = 0

# # clock_start = time.time()
# # net_ieee13.zbus_pf()
# # clock_elapsed = time.time() - clock_start
# # print('Time for power flow = '+ str(clock_elapsed)+ 's')


# # #v_lin0 = net_ieee13.v_flat()
# # v_lin0 = net_ieee13.v_net_res
# # S_wye_lin0 = net_ieee13.S_PQloads_wye_res
# # S_del_lin0 = net_ieee13.S_PQloads_del_res

# # net_ieee13.linear_model_setup(v_lin0,S_wye_lin0,S_del_lin0) #note that phases need to be 120degrees out for good results
# # net_ieee13.linear_pf()

# # pf_model =  copy.deepcopy(net_ieee13) 

# # #Actual power flow solution
# # P_ev_station = P_ev_station_lin + 100e3
# # #net_ieee13.bus_df.loc[bus_index_ev_station,'Pa'] = P_ev_station/3/1e3
# # #net_ieee13.bus_df.loc[bus_index_ev_station,'Pb'] = P_ev_station/3/1e3
# # #net_ieee13.bus_df.loc[bus_index_ev_station,'Pc'] = P_ev_station/3/1e3
# # net_ieee13.bus_df.loc[bus_index_ev_station,'Pa'] = P_ev_station_base/3/1e3
# # net_ieee13.bus_df.loc[bus_index_ev_station,'Pb'] = P_ev_station/1e3 + P_ev_station_base/3/1e3
# # net_ieee13.bus_df.loc[bus_index_ev_station,'Pc'] = P_ev_station_base/3/1e3
# net_ieee13.zbus_pf()

# # clock_start_lpf = time.time()
# # # net_ieee13.linear_pf()
# # # clock_elapsed_lpf = time.time() - clock_start_lpf
# # # print('Time for linear power flow = '+ str(clock_elapsed)+ 's')

 # # plt.figure(1)
 # # plt.clf()
 # # for bus_i in range(0,len(net_ieee13.bus_df)):
     # # aph_index = (bus_i)*3
     # # bph_index = (bus_i)*3+1
     # # cph_index = (bus_i)*3+2
     # # a_ph_hand = plt.scatter(bus_i,np.abs(net_ieee13.v_net_res[aph_index])/net_ieee13.Vslack_ph,color='C0',marker='x')
     # # b_ph_hand = plt.scatter(bus_i,np.abs(net_ieee13.v_net_res[bph_index])/net_ieee13.Vslack_ph,color='C1',marker='x')
     # # c_ph_hand = plt.scatter(bus_i,np.abs(net_ieee13.v_net_res[cph_index])/net_ieee13.Vslack_ph,color='C2',marker='x')
# # for bus_i in range(0,len(net_ieee13.bus_df)):
    # # aph_index = (bus_i)*3
    # # bph_index = (bus_i)*3+1
    # # cph_index = (bus_i)*3+2
    # # a_ph_lin_hand = plt.scatter(bus_i,net_ieee13.v_net_lin_abs_res[aph_index]/net_ieee13.Vslack_ph,s=35,facecolors='none',edgecolor='C0')
    # # b_ph_lin_hand = plt.scatter(bus_i,net_ieee13.v_net_lin_abs_res[bph_index]/net_ieee13.Vslack_ph,s=35,facecolors='none',edgecolor='C1')
    # # c_ph_lin_hand = plt.scatter(bus_i,net_ieee13.v_net_lin_abs_res[cph_index]/net_ieee13.Vslack_ph,s=35,facecolors='none',edgecolor='C2')

# # #plt.plot(net_ieee13.v_abs_min/net_ieee13.Vslack_ph,'--')
# # #plt.plot(net_ieee13.v_abs_max/net_ieee13.Vslack_ph,'--')

# # plt.xlabel('Bus Number')
# # plt.ylabel('V (pu)')
# # plt.legend([a_ph_hand,b_ph_hand,c_ph_hand,a_ph_lin_hand,b_ph_lin_hand,c_ph_lin_hand]\
           # # ,["Nonlinear Phase A","Nonlinear Phase B","Nonlinear Phase C","Linear Phase A","Linear Phase B","Linear Phase C"])


# # #Check linear model for slack bus and voltage limits
# # #Linear slack bus power flow limit
# # ev_station_phase_index_a= 15-3
# # ev_station_phase_index_b= 16-3
# # ev_station_phase_index_c= 17-3


# # ev_station_phase_index=ev_station_phase_index_b

# # d_P_ev = (P_ev_station-P_ev_station_lin)
# # M_wye = pf_model.M_wye
# # M_del = pf_model.M_del
# # M_wye_ph = M_wye[:,ev_station_phase_index]
# # M_del_ph = M_del[:,ev_station_phase_index]
# # Ysn = pf_model.Ysn
# # PQ0_wye = np.concatenate((np.real(pf_model.S_PQloads_wye_res),np.imag(pf_model.S_PQloads_wye_res)))*1e3
# # PQ0_del = np.concatenate((np.real(pf_model.S_PQloads_del_res),np.imag(pf_model.S_PQloads_del_res)))*1e3
# # A_Pslack = np.real(np.matmul(pf_model.vs.T,np.matmul(np.conj(Ysn),np.conj(M_wye_ph))))
# # b_Pslack = np.real(np.matmul(pf_model.vs.T,np.matmul(np.conj(Ysn),np.matmul(np.conj(M_wye),PQ0_wye))))\
          # # +np.real(np.matmul(pf_model.vs.T,np.matmul(np.conj(Ysn),np.matmul(np.conj(M_del),PQ0_del))))\
          # # +np.real(pf_model.M0[0])
# # P_slack_calc = (A_Pslack*d_P_ev+b_Pslack)/1e3           
# # P_slack_actual = np.real(np.sum(net_ieee13.S_net_res[0:3]))
# # print('P slack linear = ' + str(np.around(P_slack_calc)) + 'kW')
# # print('P slack actual = ' + str(np.around(P_slack_actual)) + 'kW')
# # print('P slack calc error = ' + str((np.abs(P_slack_calc-P_slack_actual)/P_slack_actual*100)) + '%')

# # #linear voltage limits 
# # A_vlim = pf_model.K_wye[:,ev_station_phase_index]
# # b_vlim = pf_model.v_lin_abs_res #- pf_model.K_wye[:,ev_station_phase_index]*P_ev_station_lin 

# # V_calc = A_vlim*d_P_ev + b_vlim
# # V_actual = np.abs(net_ieee13.v_res)
# # for i in range((net_ieee13.N_buses-1)*3):
    # # print('Bus ' + str(int(i/3)+1) + ' Phase ' + str(i%3))
    # # if V_actual[i] ==0:
        # # print('not connected')
    # # else:
        # # print('V abs error = ' + str(V_calc[i]-V_actual[i]) + 'V')
        # # print('V abs error = ' + str((V_calc[i]-V_actual[i])/V_actual[i]*100) + '%')


# # #check line current calculations
# # #for each bus, check current injected and line currents sum to zero on each phase
# # Ibus_inj = np.zeros([net_ieee13.N_buses,3],dtype=np.complex_)
# # Ibus_lines = np.zeros([net_ieee13.N_buses,3],dtype=np.complex_)
# # for bus_i in range(net_ieee13.N_buses):
    # # bus_i_df = net_ieee13.res_bus_df.iloc[bus_i]
    # # Ibus_inj[bus_i,0] = bus_i_df['Ia'] 
    # # Ibus_inj[bus_i,1] = bus_i_df['Ib']
    # # Ibus_inj[bus_i,2] = bus_i_df['Ic']
    # # for line_ij in range(net_ieee13.N_lines):
        # # line_ij_df = net_ieee13.res_lines_df.iloc[line_ij]
        # # if line_ij_df['busA'] == bus_i_df['name']:
            # # Ibus_lines[bus_i,0] += line_ij_df['Ia'] #current FROM bus i
            # # Ibus_lines[bus_i,1] += line_ij_df['Ib']
            # # Ibus_lines[bus_i,2] += line_ij_df['Ic']
        # # elif line_ij_df['busB'] == bus_i_df['name']:
            # # Ibus_lines[bus_i,0] -= line_ij_df['Ia'] #current INTO bus i
            # # Ibus_lines[bus_i,1] -= line_ij_df['Ib']
            # # Ibus_lines[bus_i,2] -= line_ij_df['Ic']
# # #plt.figure()
# # #plt.plot(np.abs(Ibus_lines-Ibus_inj))

# # #Check linear model for line currents
# # I_lines = np.zeros([net_ieee13.N_lines,3],dtype=np.complex_)
# # I_lines_calc = np.zeros([net_ieee13.N_lines,3],dtype=np.complex_)
# # Iabs_lines = np.zeros([net_ieee13.N_lines,3],dtype=np.complex_)
# # Iabs_lines_calc = np.zeros([net_ieee13.N_lines,3],dtype=np.complex_)
# # for line_ij in range(net_ieee13.N_lines):
    # # line_ij_df = net_ieee13.res_lines_df.iloc[line_ij]
    # # I_lines[line_ij,0] = line_ij_df['Ia']
    # # I_lines[line_ij,1] = line_ij_df['Ib']
    # # I_lines[line_ij,2] = line_ij_df['Ic']
    # # Iabs_lines[line_ij,0] = np.abs(line_ij_df['Ia'])
    # # Iabs_lines[line_ij,1] = np.abs(line_ij_df['Ib'])
    # # Iabs_lines[line_ij,2] = np.abs(line_ij_df['Ic'])
    # # I_lines_calc[line_ij,0] = net_ieee13.J_dPQwye_list[line_ij][0,ev_station_phase_index]*d_P_ev + net_ieee13.J_I0_list[line_ij][0]
    # # I_lines_calc[line_ij,1] = net_ieee13.J_dPQwye_list[line_ij][1,ev_station_phase_index]*d_P_ev + net_ieee13.J_I0_list[line_ij][1]
    # # I_lines_calc[line_ij,2] = net_ieee13.J_dPQwye_list[line_ij][2,ev_station_phase_index]*d_P_ev + net_ieee13.J_I0_list[line_ij][2]
    # # Iabs_lines_calc[line_ij,0] = net_ieee13.Jabs_dPQwye_list[line_ij][0,ev_station_phase_index]*d_P_ev + net_ieee13.Jabs_I0_list[line_ij][0]
    # # Iabs_lines_calc[line_ij,1] = net_ieee13.Jabs_dPQwye_list[line_ij][1,ev_station_phase_index]*d_P_ev + net_ieee13.Jabs_I0_list[line_ij][1]
    # # Iabs_lines_calc[line_ij,2] = net_ieee13.Jabs_dPQwye_list[line_ij][2,ev_station_phase_index]*d_P_ev + net_ieee13.Jabs_I0_list[line_ij][2]

# # I_lines_calc2 = np.zeros([net_ieee13.N_lines,3],dtype=np.complex_)
# # I_lines_calc3 = np.zeros([net_ieee13.N_lines,3],dtype=np.complex_)
# # Iabs_lines_calc2 = np.zeros([net_ieee13.N_lines,3])
# # v_lin_calc = np.append(net_ieee13.vs,M_wye_ph*d_P_ev+np.matmul(M_wye,PQ0_wye)+np.matmul(M_del,PQ0_del) + pf_model.M0)#np.append(net_ieee13.vs,net_ieee13.v_lin_res) #
# # for line_ij in range(net_ieee13.N_lines):
    # # line_ij_df = net_ieee13.res_lines_df.iloc[line_ij]
    # # Yseries = net_ieee13.list_Yseries[line_ij]
    # # Yshunt = net_ieee13.list_Yshunt[line_ij]
    # # bus_i = net_ieee13.bus_df[net_ieee13.bus_df['name']==line_ij_df['busA']]['number'].values[0]
    # # bus_j = net_ieee13.bus_df[net_ieee13.bus_df['name']==line_ij_df['busB']]['number'].values[0]
    # # v_i_abc_lin = v_lin_calc[3*bus_i:3*(bus_i+1)]
    # # v_i_abc = net_ieee13.v_net_res[3*(bus_i):3*(bus_i+1)]
    # # v_j_abc_lin = v_lin_calc[3*bus_j:3*(bus_j+1)]
    # # v_j_abc = net_ieee13.v_net_res[3*(bus_j):3*(bus_j+1)]       
    # # I_lines_calc2[line_ij,0:3] = np.matmul(Yshunt+Yseries,v_i_abc_lin)-np.matmul(Yseries,v_j_abc_lin)
    # # I_lines_calc3[line_ij,0:3] = np.matmul(Yshunt+Yseries,v_i_abc)-np.matmul(Yseries,v_j_abc)
# # #plt.figure()
# # #plt.plot(np.abs(v_lin_calc))
# # #plt.plot(np.abs(net_ieee13.v_net_res),'--')
    


# # plt.figure(2)
# # for line_ij in range(net_ieee13.N_lines):
    # # a_ph_hand = plt.scatter(line_ij,np.abs(I_lines_calc3[line_ij,0]),color='C0',marker='x')
    # # b_ph_hand = plt.scatter(line_ij,np.abs(I_lines_calc3[line_ij,1]),color='C1',marker='x')
    # # c_ph_hand = plt.scatter(line_ij,np.abs(I_lines_calc3[line_ij,2]),color='C2',marker='x')
    # # a_ph_lin_hand = plt.scatter(line_ij,Iabs_lines_calc[line_ij,0],s=35,facecolors='none',edgecolor='C0')
    # # b_ph_lin_hand = plt.scatter(line_ij,Iabs_lines_calc[line_ij,1],s=35,facecolors='none',edgecolor='C1')
    # # c_ph_lin_hand = plt.scatter(line_ij,Iabs_lines_calc[line_ij,2],s=35,facecolors='none',edgecolor='C2')
# # plt.legend([a_ph_hand,b_ph_hand,c_ph_hand,a_ph_lin_hand,b_ph_lin_hand,c_ph_lin_hand]\
           # # ,["Nonlinear Phase A","Nonlinear Phase B","Nonlinear Phase C","Linear Phase A","Linear Phase B","Linear Phase C"])
# # plt.xlabel('Line Number')
# # plt.ylabel('Current Magnitude (A)')



