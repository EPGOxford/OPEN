import os, sys

WD = os.path.dirname(sys.argv[0])

for i in range(1,11):
    DN = os.path.join(WD,'network_'+str(i))
    if os.path.exists(DN):
        try:
            os.remove(os.path.join(DN,'masterExample.dss'))
        except OSError:
            pass
        try:
            os.remove(os.path.join(DN,'masterNetwork'+str(i)+'.dss'))
        except OSError:
            pass
        
        # ALSO remove rogue .dbl + DSV files
        fileList = os.listdir(DN)
        for file in fileList:
            if file.endswith(".dbl") or file.endswith(".DSV"):
                os.remove(os.path.join(DN,file))
        
        for j in range(10):
            FDN = os.path.join( DN,'Feeder_'+str(j) )
            if os.path.exists(FDN):
                toDelete = ['Master.dss','Master - Copy.dss','LoadsCopyUnq.txt',
                            'LinesUnq.txt','connectivity_matrix.xls','Master_y - Copy.dss',
                            'Master_y.dss','Transformers_y.txt','XY_Position.csv',
                            'Feeder_Data.xls','XY_Position.xls','Monitors.txt']
                for file in toDelete:
                    try:
                        os.remove( os.path.join( FDN, file ) )
                    except OSError:
                        pass
                fileList = os.listdir(FDN)
                for file in fileList:
                    if file.endswith(".dbl") or file.endswith(".DSV"):
                        os.remove(os.path.join(FDN,file))