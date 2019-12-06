# The main file of interest is Network_3ph_pf.py - this is used to create a Network_3ph object. The Network_3ph_pf.setup_network_ieee13() sets up a network object as the ieee13 bus. As discussed, the main idea would be to have an alternative function which could replace this method, to load in bus & line data from an external file.

# Two main panda dataframes need to be set:
# bus_df, with columns ['name','number','load_type','connect','Pa','Pb','Pc','Qa','Qb','Qc’], and 
# line_df with columns ['busA','busB','Zaa','Zbb','Zcc','Zab','Zac','Zbc','Baa','Bbb','Bcc','Bab','Bac','Bbc’]

# The file zbus_3ph_pf_test.py can be used for testing the Network_3ph class.


# ADVISED ACTIONS prior to conversion =======================
# 1. Transformers at the head of the feeder to 1:1 at the secondary voltage
# 2. Transformer source impedance based on xfmr ratio 
# - Have this as a 'source' line impedance as in ieee_tn (from previous ptech18 folder)
# If there are downstream transformers:
# 3. Convert to 1:1 on primary
# 4. Convert downstream impedances and voltages
# ============================================================


# WARNING: be very careful about making sure scripts set the default base frequency if there are capacitance values in the circuit that are not trivial; particularly when changing from script to script (OpenDSS uses a global base frequency which changes when you run, for example, the EU LV network, and it is not trivial to keep track of what it is).

import sys, os
# path = os.path.dirname(os.path.dirname(sys.argv[0]))

import pandas as pd
import win32com.client
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Network_3ph_pf import Network_3ph
from miscDssFncs import get_ckt, tp2mat, tp_2_ar, get_sYsD



def dssConvert(feeder,dssName):
    path = os.path.dirname(os.path.dirname(sys.argv[0]))
    if feeder[0]!='n':
        ntwxs = os.path.join(path,'Data','networks')
        fn_ckt = os.path.join(ntwxs,'opendssNetworks',feeder,dssName)
        dir0 = os.path.join(ntwxs,feeder+'_dss')
        sn0 = os.path.join(dir0,feeder)
    else:
        ntwk_no = feeder.split('_')[0][1:]
        fdr_no = feeder.split('_')[1]
        ntwxs = os.path.join(path,'Data','networks')
        fn_ckt = os.path.join(ntwxs,'opendssNetworks','network_'+ntwk_no,'Feeder_'+fdr_no,dssName)
        dir0 = os.path.join(ntwxs,feeder+'_dss')
        sn0 = os.path.join(dir0,feeder)
    if os.path.exists(os.path.dirname(fn_ckt)):
        print('Starting ', feeder)
        saveModel = True
        # saveModel = False

        DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
        DSSText=DSSObj.Text
        DSSCircuit = DSSObj.ActiveCircuit
        DSSSolution=DSSCircuit.Solution

        DSSText.Command='Compile ('+fn_ckt+'.dss)'
        DSSText.Command='Set Controlmode=off' # turn all regs/caps off

        LDS = DSSCircuit.Loads
        CAP = DSSCircuit.Capacitors
        LNS = DSSCircuit.Lines
        TRN = DSSCircuit.Transformers
        ACE = DSSCircuit.ActiveElement
        SetACE = DSSCircuit.SetActiveElement
        SetABE = DSSCircuit.SetActiveBus
        ADE = DSSCircuit.ActiveDSSElement
        ABE = DSSCircuit.ActiveBus
        SRC = DSSCircuit.Vsources

        # create solution DF
        slnColumns = ['bus','vLN','sInjkW']

        YNodeV = tp_2_ar(DSSCircuit.YNodeVarray)
        sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)
        sInj = 1e-3*YNodeV*(iTot.conj())

        sList = list(sInj)
        yList = list(DSSCircuit.YNodeOrder)
        vList = tp_2_ar(DSSCircuit.YNodeVarray).tolist()
        yvList = list(map(list, zip(*[yList,vList,sList])))
        solution_df = pd.DataFrame(data=yvList,columns=slnColumns)

        # create source DF
        SRC.First
        srcColumns = ['kvBaseLL','pu']
        kvbase = SRC.BasekV
        src_df = pd.DataFrame(data=[[kvbase,SRC.pu]],columns=srcColumns,index=[SRC.Name])

        nLds = LDS.Count
        nCap = CAP.Count
        nTrn = TRN.Count
        nLns = LNS.Count

        lineColumns = ['busA','busB','Zaa','Zbb','Zcc','Zab','Zac','Zbc','Baa','Bbb','Bcc','Bab','Bac','Bbc']
        line_df = pd.DataFrame(data=np.zeros((nLns,len(lineColumns)),dtype=complex), index=LNS.AllNames, columns=lineColumns)

        # Create Line DF from lines and transformers ====
        i = LNS.First
        Yprm = {}
        Yprm0 = {}
        YprmErrDiag = []
        while i:
            lineName = LNS.Name
            line_df.loc[lineName,'busA']=LNS.Bus1.split('.')[0]
            line_df.loc[lineName,'busB']=LNS.Bus2.split('.')[0]
            lineLen = LNS.Length
            if len(LNS.Geometry)>0:
                zmat0 = (tp2mat(LNS.Rmatrix) + 1j*tp2mat(LNS.Xmatrix)) # ohms
                bmat0 = 1j*2*np.pi*60*tp2mat(LNS.Cmatrix)*1e-9 # ohms
            else:
                zmat0 = (tp2mat(LNS.Rmatrix) + 1j*tp2mat(LNS.Xmatrix))*lineLen # ohms
                bmat0 = 1j*2*np.pi*60*tp2mat(LNS.Cmatrix)*1e-9*lineLen # ohms
            
            SetACE('Line.'+LNS.Name)
            
            nPh = ACE.NumPhases
            phs = list(map(int,ACE.BusNames[0].split('.')[1:]))
            phsIdx = np.array(phs)-1
            
            Zmat = np.zeros((3,3),dtype=complex)
            Bmat = np.zeros((3,3),dtype=complex)
            
            Yprm0_Y = tp_2_ar(LNS.Yprim)
            Yprm0[lineName] = Yprm0_Y.reshape((np.sqrt(Yprm0_Y.shape)[0].astype('int32'),np.sqrt(Yprm0_Y.shape)[0].astype('int32')))[0:nPh,0:nPh]
            
            if nPh==1:
                Zmat[phs[0]-1,phs[0]-1] = zmat0[0,0]
                Bmat[phs[0]-1,phs[0]-1] = bmat0[0,0]
                Yprm[lineName] = 1/Zmat[phsIdx,phsIdx]
            if nPh==2:
                Zmat[phs[0]-1,phs[0]-1] = zmat0[0,0]
                Zmat[phs[1]-1,phs[0]-1] = zmat0[0,1]
                Zmat[phs[0]-1,phs[1]-1] = zmat0[0,1]
                Zmat[phs[1]-1,phs[1]-1] = zmat0[1,1]
                Yprm[lineName] = np.linalg.inv(Zmat[phsIdx][:,phsIdx])
                
                Bmat[phs[0]-1,phs[0]-1] = bmat0[0,0]
                Bmat[phs[1]-1,phs[0]-1] = bmat0[0,1]
                Bmat[phs[0]-1,phs[1]-1] = bmat0[0,1]
                Bmat[phs[1]-1,phs[1]-1] = bmat0[1,1]
            if nPh==3:
                Zmat = zmat0
                Bmat = bmat0
                Yprm[lineName] = np.linalg.inv(Zmat)
            
            line_df.loc[lineName,'Zaa']=Zmat[0,0]
            line_df.loc[lineName,'Zbb']=Zmat[1,1]
            line_df.loc[lineName,'Zcc']=Zmat[2,2]
            line_df.loc[lineName,'Zab']=Zmat[0,1]
            line_df.loc[lineName,'Zac']=Zmat[0,2]
            line_df.loc[lineName,'Zbc']=Zmat[1,2]
            
            line_df.loc[lineName,'Baa']=Bmat[0,0]
            line_df.loc[lineName,'Bbb']=Bmat[1,1]
            line_df.loc[lineName,'Bcc']=Bmat[2,2]
            line_df.loc[lineName,'Bab']=Bmat[0,1]
            line_df.loc[lineName,'Bac']=Bmat[0,2]
            line_df.loc[lineName,'Bbc']=Bmat[1,2]
            
            Yprm0_Y = tp_2_ar(LNS.Yprim)
            Yprm0[lineName] = Yprm0_Y.reshape((np.sqrt(Yprm0_Y.shape)[0].astype('int32'),np.sqrt(Yprm0_Y.shape)[0].astype('int32')))[0:nPh,0:nPh]
            
            YprmDiag = np.linalg.inv(zmat0) + 0.5*bmat0
            YprmErrDiag= YprmErrDiag + [np.linalg.norm(Yprm0[lineName] - YprmDiag)/np.linalg.norm(Yprm0[lineName])] # for validation/checking
            i = LNS.Next

        # print('Primitive impedance errors:',YprmErrDiag) # error checking


        if nTrn > 0:
            trnColumns = ['busA','busB','typeA','typeB','Zseries','Zshunt','kV_A','kV_B']
            trn_df = pd.DataFrame(data=np.zeros((nTrn,len(trnColumns)),dtype=complex), index=TRN.AllNames, columns=trnColumns)

            # self.transformer_df.append({'busA':'633','busB':'634','typeA':'wye-g','typeB':’wye-g','Zseries':0.381+0.692j,'Zshunt':0},ignore_index=True)  
            # The connection types (‘typeA’ and ‘typeB’) can be 'wye-g', ‘wye' or ‘delta’.

            # Now: go through each of the transformers and use to create the transformer dataframe.

            i = TRN.First # only Lines implemented at this stage.
            while i:
                trnName = TRN.Name
                SetACE('Transformer.'+TRN.Name)
                
                # Winding 1
                TRN.Wdg=1
                trn_df.loc[trnName,'busA'] = ADE.Properties('bus').Val
                trn_df.loc[trnName,'kV_A'] = ADE.Properties('kV').Val
                conn1 = ADE.Properties('conn').Val
                grnded1 = float(ADE.Properties('Rneut').Val)
                kV1 = float(ADE.Properties('kV').Val)
                S1 = float(ADE.Properties('kva').Val)
                zbase1 = (kV1**2)/(S1*1e-3)
                zSh1Pct = float(ADE.Properties('%Noloadloss').Val) + 1j*float(ADE.Properties('%imag').Val)
                zSr1Pct = float(ADE.Properties('%r').Val) + 0.5*1j*float(ADE.Properties('xhl').Val)
                
                # find how many nodes there are at the connected bus
                SetABE(ADE.Properties('bus').Val)
                numNodes = len(ABE.Nodes)
                
                if conn1=='wye ' and grnded1==-1 and numNodes==4:
                    trn_df.loc[trnName,'typeA'] = 'wye'
                elif conn1=='wye ':
                    trn_df.loc[trnName,'typeA'] = 'wye-g'
                elif conn1=='delta ':
                    trn_df.loc[trnName,'typeA'] = 'delta'
                
                ADE.Properties('wdg').Val
                
                # Winding 2
                TRN.Wdg=2
                trn_df.loc[trnName,'busB'] = ADE.Properties('bus').Val
                trn_df.loc[trnName,'kV_B'] = ADE.Properties('kV').Val
                conn2 = ADE.Properties('conn').Val
                grnded2 = float(ADE.Properties('Rneut').Val)
                kV2 = float(ADE.Properties('kV').Val)
                S2 = float(ADE.Properties('kva').Val)
                zbase2 = (kV2**2)/(S2*1e-3)
                zSh2Pct = float(ADE.Properties('%Noloadloss').Val) + 1j*float(ADE.Properties('%imag').Val)
                zSr2Pct = float(ADE.Properties('%r').Val) + 0.5*1j*float(ADE.Properties('xhl').Val) # only half of leakage
                
                if conn1=='wye ' and grnded1==-1 and numNodes==4:
                    trn_df.loc[trnName,'typeB'] = 'wye'
                elif conn2=='wye ':
                    trn_df.loc[trnName,'typeB'] = 'wye-g'
                elif conn2=='delta ':
                    trn_df.loc[trnName,'typeB'] = 'delta'
                
                Zsr = 0.01*(zbase1*zSr1Pct + zbase2*zSr2Pct)
                Zsh = 0.01*(zbase1*zSh1Pct + zbase2*zSh2Pct)
                
                trn_df.loc[trnName,'Zseries'] = Zsr
                if Zsh==0:
                    trn_df.loc[trnName,'Zshunt'] = 0
                
                i = TRN.Next


        # create bus_df from loads and capacitors ======
        nBus = DSSCircuit.NumBuses
        busColumns = ["name","number","v_base","load_type","connect","Pa","Pb","Pc","Qa","Qb","Qc"]
        bus_df = pd.DataFrame(data=np.zeros((nBus,len(busColumns))), index=DSSCircuit.AllBusNames, columns=busColumns)

        bus_df['name'] = DSSCircuit.AllBusNames
        bus_df['number'] = np.arange((nBus))

        # Find the slack bus:
        VSRC = DSSCircuit.Vsources
        VSRC.First
        SetACE('Vsource.'+VSRC.Name)
        bus_df.loc[:,'connect'] = 'Y'
        bus_df.loc[ACE.BusNames[0],'load_type'] = 'S'
        bus_df.loc[:,'v_base'] = 1e3*kvbase/np.sqrt(3)

        i = LDS.First

        while i:
            SetACE('Load.'+LDS.Name)
            actBus = ACE.BusNames[0].split('.')[0]
            
            if LDS.Model==1:
                load_type = 'PQ'
            elif LDS.Model==2:
                load_type = 'Z'
            elif LDS.Model==5:
                load_type = 'I'
            else:
                print('Warning! Load: ',LDS.Name,'Load model not a ZIP load. Setting as PQ.')
                load_type = 'PQ'
            
            if bus_df.loc[actBus,'load_type']==0 or bus_df.loc[actBus,'load_type']==load_type:
                bus_df.loc[actBus,'load_type'] = load_type
            else:
                bus_df.loc[actBus,'load_type'] = 'Mxd'

            nPh = ACE.NumPhases
            phs = ACE.BusNames[0].split('.')[1:]
            if LDS.IsDelta:
                bus_df.loc[actBus,'connect'] = 'D'
                if nPh==1:
                    if '1' in phs and '2' in phs:
                        bus_df.loc[actBus,'Pa'] = LDS.kW + bus_df.loc[actBus,'Pa']
                        bus_df.loc[actBus,'Qa'] = LDS.kvar + bus_df.loc[actBus,'Qa']
                    if '2' in phs and '3' in phs:
                        bus_df.loc[actBus,'Pb'] = LDS.kW + bus_df.loc[actBus,'Pb']
                        bus_df.loc[actBus,'Qb'] = LDS.kvar + bus_df.loc[actBus,'Qb']
                    if '3' in phs and '1' in phs:
                        bus_df.loc[actBus,'Pc'] = LDS.kW + bus_df.loc[actBus,'Pc']
                        bus_df.loc[actBus,'Qc'] = LDS.kvar + bus_df.loc[actBus,'Qc']
                if nPh==3:
                    bus_df.loc[actBus,'Pa'] = LDS.kW/3 + bus_df.loc[actBus,'Pa']
                    bus_df.loc[actBus,'Pb'] = LDS.kW/3 + bus_df.loc[actBus,'Pb']
                    bus_df.loc[actBus,'Pc'] = LDS.kW/3 + bus_df.loc[actBus,'Pc']
                    bus_df.loc[actBus,'Qa'] = LDS.kvar/3 + bus_df.loc[actBus,'Qa']
                    bus_df.loc[actBus,'Qb'] = LDS.kvar/3 + bus_df.loc[actBus,'Qb']
                    bus_df.loc[actBus,'Qc'] = LDS.kvar/3 + bus_df.loc[actBus,'Qc']
                if nPh==2:
                    print('Warning! Load: ',LDS.Name,'2 phase Delta loads not yet implemented.')
            else:
                bus_df.loc[actBus,'connect'] = 'Y'
                if '1' in phs or phs==[]:
                    bus_df.loc[actBus,'Pa'] = LDS.kW/nPh + bus_df.loc[actBus,'Pa']
                    bus_df.loc[actBus,'Qa'] = LDS.kvar/nPh + bus_df.loc[actBus,'Qa']
                if '2' in phs or phs==[]:
                    bus_df.loc[actBus,'Pb'] = LDS.kW/nPh + bus_df.loc[actBus,'Pb']
                    bus_df.loc[actBus,'Qb'] = LDS.kvar/nPh + bus_df.loc[actBus,'Qb']
                if '3' in phs or phs==[]:
                    bus_df.loc[actBus,'Pc'] = LDS.kW/nPh + bus_df.loc[actBus,'Pc']
                    bus_df.loc[actBus,'Qc'] = LDS.kvar/nPh + bus_df.loc[actBus,'Qc']
            
            i = LDS.Next

        # create bus_df from loads and capacitors ======

        if nCap>0:
            capColumns = ["name","number","bus","kVln","connect","Qa","Qb","Qc"]
            cap_df = pd.DataFrame(data=np.zeros((nCap,len(capColumns))), index=CAP.AllNames, columns=capColumns)
            cap_df.loc[:,'name'] = CAP.AllNames
            i = CAP.First
            while i:
                capName = CAP.Name
                cap_df.loc[capName,'number'] = i-1 # not too clear rn what 'number' does?
                
                SetACE('Capacitor.'+capName)
                nPh = ACE.NumPhases
                phs = ACE.BusNames[0].split('.')[1:]
                
                actBus = ACE.BusNames[0].split('.')[0]
                cap_df.loc[capName,'bus'] = actBus
                if CAP.IsDelta:
                    cap_df.loc[capName,'connect'] = 'D'
                    cap_df.loc[capName,'kVln'] = CAP.kV/np.sqrt(3)
                    if nPh==1:
                        if '1' in phs and '2' in phs:
                            cap_df.loc[capName,'Qa'] = CAP.kvar + cap_df.loc[capName,'Qa']
                        if '2' in phs and '3' in phs:
                            cap_df.loc[capName,'Qb'] = CAP.kvar + cap_df.loc[capName,'Qb']
                        if '3' in phs and '1' in phs:
                            cap_df.loc[capName,'Qc'] = CAP.kvar + cap_df.loc[capName,'Qc']
                    if nPh==3:
                        cap_df.loc[capName,'Qa'] = CAP.kvar/3 + cap_df.loc[capName,'Qa']
                        cap_df.loc[capName,'Qb'] = CAP.kvar/3 + cap_df.loc[capName,'Qb']
                        cap_df.loc[capName,'Qc'] = CAP.kvar/3 + cap_df.loc[capName,'Qc']
                    if nPh==2:
                        print('Warning! Cap: ',CAP.Name,'2 phase Delta loads not yet implemented.')
                else:
                    cap_df.loc[capName,'connect'] = 'Y'
                    if nPh==3:
                        cap_df.loc[capName,'kVln'] = CAP.kV/np.sqrt(3) # NB: kV depends on the connection type + no. phases.
                    elif nPh==1:
                        cap_df.loc[capName,'kVln'] = CAP.kV
                    if '1' in phs or phs==[]:
                        cap_df.loc[capName,'Qa'] = CAP.kvar/nPh + cap_df.loc[capName,'Qa']
                    if '2' in phs or phs==[]:
                        cap_df.loc[capName,'Qb'] = CAP.kvar/nPh + cap_df.loc[capName,'Qb']
                    if '3' in phs or phs==[]:
                        cap_df.loc[capName,'Qc'] = CAP.kvar/nPh + cap_df.loc[capName,'Qc']
                i = CAP.Next


        if saveModel:
            if not os.path.exists(dir0):
                os.makedirs(dir0)
            src_df.to_csv(sn0+"_src_df.csv")
            bus_df.to_csv(sn0+"_bus_df.csv")
            line_df.to_csv(sn0+"_line_df.csv")
            solution_df.to_csv(sn0+"_solution_df.csv")
            if nTrn > 0:
                trn_df.to_csv(sn0+"_trn_df.csv")
            if nCap > 0:
                cap_df.to_csv(sn0+"_cap_df.csv")
            print('\nFiles saved in:\n',dir0+'\\','\n')


feeder = "13BusOxEmf"
dssName = "IEEE13Nodeckt_z_oxemf"
feeder = "13BusXfmr"
dssName = "IEEE13Nodeckt"
# feeder = "eulv"
# dssName = "master_z_oxemf"
# feeder = "4Bus-DY-Bal"
# dssName = "4Bus-DY-Bal_oxemf"
# feeder = "4Bus-YY-Bal"
# dssName = "4Bus-YY-Bal_oxemf"
# feeder = "bmatrixVal"
# dssName = "bmatrixVal"
# feeder = 'n10_2'
# feederSet = ['n10_3','n10_4','n10_5','n10_6']

feeder = "4Bus-YY-Bal-xfmr"
dssName = "4Bus-YY-Bal"

# for iFdr in range(1,6):
    # for jFdr in range(11):
        # feeder = 'n'+str(iFdr)+'_'+str(jFdr)
        # dssName = "master_oxemf"

# feeder = 'n2_4'
# dssName = "master_oxemf"
# RUN ME HERE:
dssConvert(feeder,dssName)



