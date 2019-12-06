import sys, os, time
from importlib import reload
import matplotlib.pyplot as plt
import numpy as np

path = os.path.dirname(os.path.dirname(sys.argv[0]))
sys.path.insert(0, os.path.join(path,'System'))

import Network_3ph_pf


def main(networkName="13BusXfmr",testModel=True,testPlot=False):
    reload(Network_3ph_pf)
    network3ph = Network_3ph_pf.Network_3ph()
    network3ph.loadDssNetwork(networkName,updateYZ=True,testModel=testModel,testPlot=testPlot)
    return network3ph

ntwkName = "13BusOxEmf"
# ntwkName = "13BusXfmr"
# ntwkName = "4Bus-DY-Bal"
# ntwkName = "4Bus-YY-Bal"
#ntwkName = "4Bus-YY-Bal-xfmr"
# ntwkName = "eulv"
# ntwkName = "bmatrixVal"
# ntwkName = "n1_1"
network3ph = main(ntwkName,testPlot=True)

self = main(ntwkName)
self.voltageComparison()
print(self.dssEmfBusNode)

plt.plot(np.abs(self.vOxemfVolts))
plt.plot(np.abs(self.vDssVolts))
plt.show()

# # uncomment VVV below to run all feeders
# for iFdr in range(1,6):
    # for jFdr in range(1,10):
        # ntwkName = 'n'+str(iFdr)+'_'+str(jFdr)
        # try:
            # self = main(ntwkName)
        # except:
            # pass