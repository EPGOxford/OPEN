import numpy as np
import os
from scipy import sparse
from cvxopt import matrix

def tp_2_ar(tuple_ex):
    ar = np.array(tuple_ex[0::2]) + 1j*np.array(tuple_ex[1::2])
    return ar

def tp2mat(tuple_ex):
    n = int(np.sqrt(len(tuple_ex)))
    mat = np.zeros((n,n))
    for i in range(n):
        mat[i] = tuple_ex[i*n:(i+1)*n]
    
    return mat
    
def s_2_x(s):
    return np.concatenate((s.real,s.imag))

def vecSlc(vec_like,new_idx):
    if type(vec_like)==tuple:
        vec_slc = tuple(np.array(vec_like)[new_idx].tolist())
    elif type(vec_like)==list:
        vec_slc = np.array(vec_like)[new_idx].tolist()
    return vec_slc

def yzD2yzI(yzD,n2y):
    yzI = []
    for bus in yzD:
        yzI = yzI+find_node_idx(n2y,bus,False)
    return yzI

def idx_shf(x_idx,reIdx):
    x_idx_i = []
    for idx in x_idx:
        x_idx_i.append(reIdx.index(idx))
    
    x_idx_new = np.array([],dtype=int)
    
    x_idx_srt = x_idx_i.copy()
    x_idx_srt.sort()
    x_idx_shf = np.array([],dtype=int)
    for i in x_idx_srt:
        x_idx_shf=np.concatenate((x_idx_shf,[x_idx_i.index(i)]))
        x_idx_new=np.concatenate((x_idx_new,[reIdx[i]]))
    
    return x_idx_shf,x_idx_new

def add_generators(DSSObj,genBuses,delta):
    # NB: nominal power is 10 kW.
    genNames = []
    for genBus in genBuses:
        DSSObj.ActiveCircuit.SetActiveBus(genBus)
        if not delta: # ie wye
            genName = genBus.replace('.','_')
            genKV = str(DSSObj.ActiveCircuit.ActiveBus.kVbase)
            DSSObj.Text.command='new generator.'+genName+' phases=1 bus1='+genBus+' kV='+genKV+' kW=10.0 pf=1.0 model=1 vminpu=0.33 vmaxpu=3.0 conn=wye'
        elif delta:
            genKV = str(DSSObj.ActiveCircuit.ActiveBus.kVbase*np.sqrt(3))
            if genBus[-1]=='1':
                genBuses = genBus+'.2'
            if genBus[-1]=='2':
                genBuses = genBus+'.3'
            if genBus[-1]=='3':
                genBuses = genBus+'.1'
            genName = genBuses.replace('.','_')
            DSSObj.Text.command='new generator.'+genName+' phases=1 bus1='+genBuses+' kV='+genKV+' kW=10.0 pf=1.0 model=1 vminpu=0.33 vmaxpu=3.0 conn=wye'
        genNames = genNames+[genName]
    return genNames

def set_generators(DSSCircuit,genNames,S):
    i = 0
    for genName in genNames:
        DSSCircuit.Generators.Name=genName
        DSSCircuit.Generators.kW = S[i]
        i+=1
    return


def ld_vals( DSSCircuit ):
    ii = DSSCircuit.FirstPCElement()
    S=[]; V=[]; I=[]; B=[]; D=[]; N=[]
    while ii!=0:
        if DSSCircuit.ActiveElement.Name[0:4].lower()=='load':
            DSSCircuit.Loads.Name=DSSCircuit.ActiveElement.Name.split(sep='.')[1]
            S.append(tp_2_ar(DSSCircuit.ActiveElement.Powers))
            V.append(tp_2_ar(DSSCircuit.ActiveElement.Voltages))
            I.append(tp_2_ar(DSSCircuit.ActiveElement.Currents))
            B.append(DSSCircuit.ActiveElement.BusNames)
            N.append(DSSCircuit.Loads.Name)
            if B[-1][0].count('.')==1:
                D.append(False)
            else:
                D.append(DSSCircuit.Loads.IsDelta)
        ii=DSSCircuit.NextPCElement()
    jj = DSSCircuit.FirstPDElement()
    while jj!=0:
        if DSSCircuit.ActiveElement.Name[0:4].lower()=='capa':
            DSSCircuit.Capacitors.Name=DSSCircuit.ActiveElement.Name.split(sep='.')[1]
            S.append(tp_2_ar(DSSCircuit.ActiveElement.Powers))
            V.append(tp_2_ar(DSSCircuit.ActiveElement.Voltages))
            I.append(tp_2_ar(DSSCircuit.ActiveElement.Currents))
            B.append(DSSCircuit.ActiveElement.BusNames)
            D.append(DSSCircuit.Capacitors.IsDelta)
            N.append(DSSCircuit.Capacitors.Name)
        jj=DSSCircuit.NextPDElement()
    return S,V,I,B,D,N

def find_node_idx(n2y,bus,D):
    idx = []
    BS = bus.split('.',1)
    bus_id,ph = [BS[0],BS[-1]] # catch cases where there is no phase
    if ph=='1.2.3' or bus.count('.')==0:
        idx.append(n2y.get(bus_id+'.1',None))
        idx.append(n2y.get(bus_id+'.2',None))
        idx.append(n2y.get(bus_id+'.3',None))
    elif ph=='0.0.0':
        idx.append(n2y.get(bus_id+'.1',None))
        idx.append(n2y.get(bus_id+'.2',None))
        idx.append(n2y.get(bus_id+'.3',None))
    elif ph=='1.2.3.4': # needed if, e.g. transformers are grounded through a reactor
        idx.append(n2y.get(bus_id+'.1',None))
        idx.append(n2y.get(bus_id+'.2',None))
        idx.append(n2y.get(bus_id+'.3',None))
        idx.append(n2y.get(bus_id+'.4',None))
    elif D:
        if bus.count('.')==1:
            idx.append(n2y[bus])
        else:
            idx.append(n2y[bus[0:-2]])
    else:
        idx.append(n2y[bus])
    return idx
    
def calc_sYsD( YZ,B,I,V,S,D,n2y ): # YZ as YNodeOrder
    iD = np.zeros(len(YZ),dtype=complex); sD = np.zeros(len(YZ),dtype=complex)
    iY = np.zeros(len(YZ),dtype=complex); sY = np.zeros(len(YZ),dtype=complex)
    for i in range(len(B)):
        for bus in B[i]:
            idx = find_node_idx(n2y,bus,D[i])
            BS = bus.split('.',1)
            bus_id,ph = [BS[0],BS[-1]] # catch cases where there is no phase
            if D[i]:
                if bus.count('.')==2:
                    iD[idx] = iD[idx] + I[i][0]
                    sD[idx] = sD[idx] + S[i].sum()
                else:
                    iD[idx] = iD[idx] + I[i]*np.exp(1j*np.pi/6)/np.sqrt(3)
                    VX = np.array( [V[i][0]-V[i][1],V[i][1]-V[i][2],V[i][2]-V[i][0]] )
                    sD[idx] = sD[idx] + iD[idx].conj()*VX*1e-3
            else:
                if ph[0]!='0':
                    if bus.count('.')>0:
                        iY[idx] = iY[idx] + I[i][0]
                        sY[idx] = sY[idx] + S[i][0]
                    else:
                        iY[idx] = iY[idx] + I[i][0:3]
                        sY[idx] = sY[idx] + S[i][0:3]
    return iY, sY, iD, sD

def node_to_YZ(DSSCircuit):
    n2y = {}
    YNodeOrder = DSSCircuit.YNodeOrder
    for node in DSSCircuit.AllNodeNames:
        n2y[node]=YNodeOrder.index(node.upper())
    return n2y

def get_sYsD(DSSCircuit):
    S,V,I,B,D,N = ld_vals( DSSCircuit )
    n2y = node_to_YZ(DSSCircuit)
    V0 = tp_2_ar(DSSCircuit.YNodeVarray)*1e-3 # kV
    YZ = DSSCircuit.YNodeOrder
    iY, sY, iD, sD = calc_sYsD( YZ,B,I,V,S,D,n2y )
    H = create_Hmat(DSSCircuit)
    H = H[iD.nonzero()]
    sD = sD[iD.nonzero()]
    yzD = [YZ[i] for i in iD.nonzero()[0]]
    iD = iD[iD.nonzero()]
    iTot = iY + (H.T).dot(iD)
    # chka = abs((H.T).dot(iD.conj())*V0 + sY - V0*(iTot.conj()))/abs(sY) # 1a error, kW
    # sD0 = ((H.dot(V0))*(iD.conj()))
    # chkb = abs(sD - sD0)/abs(sD) # 1b error, kW
    # print('Y- error:')
    # print_node_array(YZ,abs(chka))
    # print('D- error:')
    # print_node_array(yzD,abs(chkb))
    return sY,sD,iY,iD,yzD,iTot,H
    
def create_Hmat(DSSCircuit):
    n2y = node_to_YZ(DSSCircuit)
    Hmat = np.zeros((DSSCircuit.NumNodes,DSSCircuit.NumNodes))
    for bus in DSSCircuit.AllBusNames:
        idx = find_node_idx(n2y,bus,False)
        if idx[0]!=None and idx[1]!=None:
            Hmat[idx[0],idx[0]] = 1
            Hmat[idx[0],idx[1]] = -1
        if idx[1]!=None and idx[2]!=None:
            Hmat[idx[1],idx[1]] = 1
            Hmat[idx[1],idx[2]] = -1
        if idx[2]!=None and idx[0]!=None:
            Hmat[idx[2],idx[2]] = 1
            Hmat[idx[2],idx[0]] = -1        
    return Hmat
    
def cpf_get_loads(DSSCircuit):
    SS = {}
    BB = {}
    i = DSSCircuit.Loads.First
    while i!=0:
        SS[i]=DSSCircuit.Loads.kW + 1j*DSSCircuit.Loads.kvar
        BB[i]=DSSCircuit.Loads.Name
        i=DSSCircuit.Loads.next
    imax = DSSCircuit.Loads.Count
    j = DSSCircuit.Capacitors.First
    while j!=0:
        SS[imax+j]=1j*DSSCircuit.Capacitors.kvar
        BB[imax+j]=DSSCircuit.Capacitors.Name
        j = DSSCircuit.Capacitors.Next
    return BB,SS

def cpf_set_loads(DSSCircuit,BB,SS,k):
    i = DSSCircuit.Loads.First
    while i!=0:
        # DSSCircuit.Loads.Name=BB[i]
        DSSCircuit.Loads.kW = k*SS[i].real
        DSSCircuit.Loads.kvar = k*SS[i].imag
        i=DSSCircuit.Loads.Next
    imax = DSSCircuit.Loads.Count
    j = DSSCircuit.Capacitors.First
    while j!=0:
        DSSCircuit.Capacitors.Name=BB[j+imax]
        DSSCircuit.Capacitors.kVar=k*SS[j+imax].imag
        j=DSSCircuit.Capacitors.next
    return

def find_tap_pos(DSSCircuit):
    TC_No=[]
    i = DSSCircuit.RegControls.First
    while i!=0:
        TC_No.append(DSSCircuit.RegControls.TapNumber)
        i = DSSCircuit.RegControls.Next
    return TC_No

def fix_tap_pos(DSSCircuit, TC_No):
    i = DSSCircuit.RegControls.First
    while i!=0:
        DSSCircuit.RegControls.TapNumber = TC_No[i-1]
        i = DSSCircuit.RegControls.Next

        
def create_tapped_ybus_very_slow( DSSObj,fn_y,TC_No0 ):
    DSSObj.Text.command='Compile ('+fn_y+')'
    fix_tap_pos(DSSObj.ActiveCircuit, TC_No0)
    DSSObj.Text.command='set controlmode=off'
    DSSObj.ActiveCircuit.Solution.Solve()
    
    SysY = DSSObj.ActiveCircuit.SystemY
    SysY_dct = {}
    i = 0
    for i in range(len(SysY)):
        if i%2 == 0:
            Yi = SysY[i] + 1j*SysY[i+1]
            if abs(Yi)!=0.0:
                j = i//2
                SysY_dct[j] = Yi
    del SysY
    
    SysYV = np.array(list(SysY_dct.values()))
    SysYK = np.array(list(SysY_dct.keys()))
    Ybus0 = sparse.coo_matrix((SysYV,(SysYK,np.zeros(len(SysY_dct),dtype=int))))
    n = int(np.sqrt(Ybus0.shape[0]))
    Ybus_ = Ybus0.reshape((n,n))
    Ybus_ = Ybus_.tocsc()
    
    Ybus = Ybus_[3:,3:]
    YNodeOrder_ = DSSObj.ActiveCircuit.YNodeOrder
    YNodeOrder = YNodeOrder_[0:3]+YNodeOrder_[6:];
    return Ybus, YNodeOrder
        
        
def create_tapped_ybus_slow( DSSObj,fn_y,TC_No0 ):
    DSSObj.Text.command='Compile ('+fn_y+')'
    fix_tap_pos(DSSObj.ActiveCircuit, TC_No0)
    DSSObj.Text.command='set controlmode=off'
    DSSObj.ActiveCircuit.Solution.Solve()
    Ybus0 = tp_2_ar(DSSObj.ActiveCircuit.SystemY)
    n = int(np.sqrt(len(Ybus0)))
    Ybus_ = Ybus0.reshape((n,n))
    Ybus = Ybus_[3:,3:]
    YNodeOrder_ = DSSObj.ActiveCircuit.YNodeOrder
    YNodeOrder = YNodeOrder_[0:3]+YNodeOrder_[6:];
    return Ybus, YNodeOrder

def create_tapped_ybus( DSSObj,fn_y,fn_ckt,TC_No0 ):
    DSSObj.Text.command='Compile ('+fn_y+')'
    fix_tap_pos(DSSObj.ActiveCircuit, TC_No0)
    DSSObj.Text.command='set controlmode=off'
    DSSObj.ActiveCircuit.Solution.Solve()
    Ybus_,YNodeOrder_,n = build_y(DSSObj,fn_ckt)
    Ybus = Ybus_[3:,3:]
    YNodeOrder = YNodeOrder_[0:3]+YNodeOrder_[6:];
    return Ybus, YNodeOrder

def build_y(DSSObj,fn_ckt):
    # DSSObj.Text.command='Compile ('+fn_z+'.dss)'
    YNodeOrder = DSSObj.ActiveCircuit.YNodeOrder
    DSSObj.Text.command='show Y'
    os.system("TASKKILL /F /IM notepad.exe")
    
    fn_y = fn_ckt+'\\'+DSSObj.ActiveCircuit.Name+'_SystemY.txt'
    fn_csv = fn_ckt+'\\'+DSSObj.ActiveCircuit.Name+'_SystemY_csv.txt'

    file_r = open(fn_y,'r')
    stream = file_r.read()

    stream=stream.replace('[','')
    stream=stream.replace('] = ',',')
    stream=stream.replace('j','')
    stream=stream[89:]
    stream=stream.replace('\n','j\n')
    stream=stream.replace(' ','')

    file_w = open(fn_csv,'w')
    file_w.write(stream)

    file_r.close()
    file_w.close()

    rc_data = np.loadtxt(fn_csv,delimiter=',',dtype=complex)
    n_y = int(rc_data[-1,0].real)

    I = np.concatenate((rc_data[:,0]-1,rc_data[:,1]-1)).real
    J = np.concatenate((rc_data[:,1]-1,rc_data[:,0]-1)).real
    V = np.concatenate((rc_data[:,2],rc_data[:,2]))
    Ybus = sparse.coo_matrix((V,(I,J)),shape=(n_y,n_y),dtype=complex).tocsr()
    
    n = len(YNodeOrder)
    for i in range(n):
        Ybus[i,i] = Ybus[i,i]/2
    
    os.remove(fn_y)
    os.remove(fn_csv)
    return Ybus, YNodeOrder, n

            # splt = DSSCircuit.ActiveElement.BusNames[0].upper().split('.')
def get_idxs(e_idx,DSSCircuit,ELE):
    i = ELE.First
    while i:
        for BN in DSSCircuit.ActiveElement.BusNames:
            splt = BN.upper().split('.')
            if len(splt) > 1:
                for j in range(1,len(splt)):
                    if splt[j]!='0': # ignore ground
                        e_idx.append(DSSCircuit.YNodeOrder.index(splt[0]+'.'+splt[j]))
            else:
                try:
                    e_idx.append(DSSCircuit.YNodeOrder.index(splt[0]))
                except:
                    for ph in range(1,4):
                        e_idx.append(DSSCircuit.YNodeOrder.index(splt[0]+'.'+str(ph)))
        i = ELE.next
    return e_idx

def get_element_idxs(DSSCircuit,ele_types):
    e_idx = []
    for ELE in ele_types:
        e_idx = get_idxs(e_idx,DSSCircuit,ELE)
    return e_idx

def get_Yvbase(DSSCircuit):
    Yvbase = []
    for yz in DSSCircuit.YNodeOrder:
        bus_id = yz.split('.')
        i = DSSCircuit.SetActiveBus(bus_id[0]) # return needed or this prints a number
        Yvbase.append(1e3*DSSCircuit.ActiveBus.kvbase)
    return np.array(Yvbase)
    
def feeder_to_fn(WD,feeder):
    paths = []
    paths.append(WD+'\\manchester_models\\network_'+feeder[3]+'\\Feeder_'+feeder[1])
    paths.append(WD+'\\manchester_models\\network_'+feeder[3]+'\\Feeder_'+feeder[1]+'\master')
    return paths
    
def print_node_array(YZ,thing):
    for i in range(len(YZ)):
        print(YZ[i]+': '+str(thing[i]))

def get_ckt(WD,feeder):
    fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1',feeder]
    # fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1'] # for editing
    ckts = {'feeder_name':['fn_ckt','fn']}
    ckts[fdrs[0]]=[WD+'\\LVTestCase_copy',WD+'\\LVTestCase_copy\\master_z']
    ckts[fdrs[1]]=feeder_to_fn(WD,fdrs[1])
    ckts[fdrs[2]]=feeder_to_fn(WD,fdrs[2])
    ckts[fdrs[3]]=feeder_to_fn(WD,fdrs[3])
    ckts[fdrs[4]]=feeder_to_fn(WD,fdrs[4])
    ckts[fdrs[5]]=[WD+'\\ieee_tn\\13Bus_copy',WD+'\\ieee_tn\\13Bus_copy\\IEEE13Nodeckt_z']
    ckts[fdrs[6]]=[WD+'\\ieee_tn\\34Bus_copy',WD+'\\ieee_tn\\34Bus_copy\\ieee34Mod1_z']
    ckts[fdrs[7]]=[WD+'\\ieee_tn\\37Bus_copy',WD+'\\ieee_tn\\37Bus_copy\\ieee37_z']
    ckts[fdrs[8]]=[WD+'\\ieee_tn\\123Bus_copy',WD+'\\ieee_tn\\123Bus_copy\\IEEE123Master_z']
    ckts[fdrs[9]]=[WD+'\\ieee_tn\\8500-Node_copy',WD+'\\ieee_tn\\8500-Node_copy\\Master-unbal_z']
    ckts[fdrs[10]]=[WD+'\\ieee_tn\\37Bus_copy',WD+'\\ieee_tn\\37Bus_copy\\ieee37_z_mod']
    ckts[fdrs[11]]=[WD+'\\ieee_tn\\13Bus_copy',WD+'\\ieee_tn\\13Bus_copy\\IEEE13Nodeckt_regMod3rg_z']
    ckts[fdrs[12]]=[WD+'\\ieee_tn\\13Bus_copy',WD+'\\ieee_tn\\13Bus_copy\\IEEE13Nodeckt_regModRx_z']
    ckts[fdrs[13]]=[WD+'\\ieee_tn\\13Bus_copy',WD+'\\ieee_tn\\13Bus_copy\\IEEE13Nodeckt_regModSng_z']
    ckts[fdrs[14]]=[WD+'\\ieee_tn\\usLv',WD+'\\ieee_tn\\usLv\\master_z']
    ckts[fdrs[15]]=[WD+'\\ieee_tn\\123Bus_copy',WD+'\\ieee_tn\\123Bus_copy\\IEEE123MasterMod_z']
    ckts[fdrs[16]]=[WD+'\\ieee_tn\\13Bus_copy',WD+'\\ieee_tn\\13Bus_copy\\IEEE13NodecktMod_z']
    ckts[fdrs[17]]=[WD+'\\ieee_tn\\ckt5',WD+'\\ieee_tn\\ckt5\\Master_ckt5_z']
    ckts[fdrs[18]]=[WD+'\\ieee_tn\\ckt7',WD+'\\ieee_tn\\ckt7\\Master_ckt7_z']
    
    ckts[fdrs[20]]=[WD+'\\ieee_tn\\k1',WD+'\\ieee_tn\\k1\\Master_NoPV_z']
    ckts[fdrs[21]]=[WD+'\\ieee_tn\\m1',WD+'\\ieee_tn\\m1\\Master_NoPV_z']
    
    if not feeder in ckts.keys() and len(feeder)==3:
        dir0 = WD+'\\manchester_models\\batch_manc_ntwx\\network_'+str(int(feeder[0:2]))+'\\Feeder_'+feeder[-1]
        ckts[fdrs[-1]]=[dir0,dir0+'\\Master']
    return ckts[feeder]

def loadLinMagModel(feeder,lin_point,WD,lp_taps):
    # lp_taps either 'Nmt' or 'Lpt'.
    stt = WD+'\\lin_models\\'+feeder+'\\'+feeder+lp_taps
    end = str(np.round(lin_point*100).astype(int)).zfill(3)+'.npy'
    LM = {}
    LM['Ky'] = np.load(stt+'Ky'+end)
    
    LM['bV'] = np.load(stt+'bV'+end)
    LM['xhy0'] = np.load(stt+'xhy0'+end)
    LM['vKvbase'] = np.load(stt+'vKvbase'+end)
    LM['vYNodeOrder'] = np.load(stt+'vYNodeOrder'+end)
    LM['SyYNodeOrder'] = np.load(stt+'SyYNodeOrder'+end)
    LM['v_idx'] = np.load(stt+'v_idx'+end)
    try:
        LM['Kd'] = np.load(stt+'Kd'+end)
        LM['xhd0'] = np.load(stt+'xhd0'+end)
        LM['SdYNodeOrder'] = np.load(stt+'SdYNodeOrder'+end)
    except:
        LM['Kd'] = np.empty(shape=(LM['Ky'].shape[0],0))
        LM['xhd0'] = np.array([])
        LM['SdYNodeOrder'] = np.array([])
    try: 
        LM['Kt'] = np.load(stt+'Kt'+end)
    except:
        LM['Kt'] = np.empty(shape=(LM['Ky'].shape[0],0))
    return LM
    
def loadLtcModel(feeder,lin_point,WD,lp_taps):
    # lp_taps either 'Nmt' or 'Lpt'.
    stt = WD+'\\lin_models\\'+feeder+'\\ltc_model\\'+feeder+lp_taps+'Ltc'
    end = str(np.round(lin_point*100).astype(int)).zfill(3)+'.npy'
    LM = {}
    LM['A'] = np.load(stt+'A'+end)
    LM['B'] = np.load(stt+'B'+end)
    LM['s_idx'] = np.load(stt+'s_idx'+end)
    LM['v_idx'] = np.load(stt+'v_idx'+end)
    LM['Vbase'] = np.load(stt+'Vbase'+end)
    LM['xhy0'] = np.load(stt+'xhy0'+end)
    LM['xhd0'] = np.load(stt+'xhd0'+end)
    LM['YZ'] = np.load(stt+'YZ'+end)
    LM['SyYNodeOrder'] = np.load(stt+'SyYNodeOrder'+end)
    LM['SdYNodeOrder'] = np.load(stt+'SdYNodeOrder'+end)
    return LM
    
def getMu_Kk(feeder,ltcOn):
    # fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1',feeder]
    if feeder=='13bus' and ltcOn:
        mu_kk = 0.9 # 13 BUS with LTC
    if feeder=='34bus' and ltcOn:
        mu_kk = 0.4 # 34 BUS with LTC
    if feeder=='123bus' and ltcOn:        
        mu_kk = 3.0 # 123 BUS with LTC
    if feeder=='eulv':
        mu_kk = 0.6 # EU LV
    if feeder=='epriK1' and not ltcOn:
        mu_kk = 0.7 # EPRI K1, no LTC
    if feeder=='epriK1' and ltcOn:
        mu_kk = 1.20 # EPRI K1, with LTC
    if feeder=='epri7' and not ltcOn:
        mu_kk = 0.5 # EPRI ckt7
    if feeder=='usLv':
        mu_kk = 1.75 # US LV
    if feeder=='041':
        mu_kk = 1.5 # 041
    if feeder=='011':
        mu_kk = 0.7 # 011
    if feeder=='193':
        mu_kk = 0.5 # 193 # NOT WORKING for DSS Solve
    if feeder=='213':
        mu_kk = 0.9 # 213 # NOT WORKING for DSS Solve
    if feeder=='162':
        mu_kk = 0.7 # 162
    if feeder=='031':
        mu_kk = 0.6 # 031 # NOT WORKING for DSS Solve
    if feeder=='024':
        mu_kk = 0.45 # 024
    if feeder=='021':
        mu_kk = 0.6 # 021
    if feeder=='074':
        mu_kk = 0.45 # 074
    return mu_kk