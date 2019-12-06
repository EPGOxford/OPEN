import os
import matplotlib as plt
#set the directory one level up to be able to import from OxEMF_Files folders
path = os.path.dirname(os.path.dirname(__file__)) 
os.chdir(path) 
#os.environ['DYLD_LIBRARY_PATH'] ='/Applications/MATLAB/MATLAB_Runtime/v94/runtime/maci64:/Applications/MATLAB/MATLAB_Runtime/v94/sys/os/maci64:/Applications/MATLAB/MATLAB_Runtime/v94/bin/maci64'

import System.ForecastingSystem as FS
forecast=FS.ForecastingSystem()
Forecast_output=forecast.pv()
Forecast_det= Forecast_output['det'].rename(index=str, columns={0: "Deterministic"})
Forecast_prob =Forecast_output['prob'].rename(index=str, columns={0: "q=10%",1: "q=20%",2: "q=30%",3: "q=40%",4: "q=50%",5: "q=60%",6: "q=70%",7: "q=80%",8: "q=90%"})

