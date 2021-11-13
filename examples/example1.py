from numpy.random import rand
import numpy as np
import SpecDiag
from SpecDiag import SD


obs,y1,y2,y3,y4 = 0,0,0,0,0
n = np.arange(0,256)
N = n.shape[0]
k1 =3
#
fase1 = np.pi
fase2 = np.pi+np.pi/4 #45
fase3 = np.pi+np.pi/2 #180
fase4 = 2*np.pi
fase5 = 2*np.pi

amp1 =1.0
obs = amp1* np.cos((2*np.pi*k1*n/N)+(fase1)) #+np.random.normal(0,1,N)
y1 =  amp1 * np.cos((2*np.pi*k1*n/N)+(fase2))#+np.random.normal(0,1,N)
y2 =  amp1 * np.cos((2*np.pi*k1*n/N)+(fase3))
y3 =  amp1 * np.cos((2*np.pi*k1*n/N)+(fase4))
##y4 =  amp1 * cos((2*np.pi*k1*n/N)+(fase5))
##y5 =  amp1 * cos((2*np.pi*k1*n/N)+(fase5))

series = {}
series[0] = obs
series[1] = y1
series[2] = y2
series[3] = y3
f = 3/256.

series_names =['OBS','pi/4','pi/2','pi']
SD(series,series_names,f,'power')
