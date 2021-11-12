#!/usr/bin/python
# _*_ coding: latin-1 -*-
# Spectral Diagram
# "The Spectral Diagram as a new tool for model assessment in the frequency domain:
# Application to a global ocean general circulation model with tides"
# Computers & Geosciences
# Available online 8 November 2021, 104977
# DOI : https://doi.org/10.1016/j.cageo.2021.104977
# author: Mabel Calim Costa
# CCST - INPE
# 08/11/2021
import numpy as np
from numpy import ma
from matplotlib import pyplot as plt
import scipy.stats as stats

def coherence(ts1,ts2):

        """ Coherence function
                ranging from -1 to 1 (the first and second quadrant of the diagram).
                Time series should have the same length

                _____________________________________________________________________
                Inputs:
                        ts1 - list with time series to analyze
                        ts2 = list with observed data or reference time series

                Outputs:
                        coerdata =  - dict with amplitude, phase, power spectrum
                Call function:
                coerdata = coherence(series[i],series[0])
                _____________________________________________________________________

         """

        # Check if the time series have the same lenth
        n, X, Y, sxy, coer, sxx, syy = len(ts1),np.fft.fft(ts1), np.fft.fft(ts2), [], [], [],[]
        # power spectrum
        variance1 = np.var(ts1)
        variance2 = np.var(ts2)
        sxx = (X * np.conj(X))/n
        syy = (Y * np.conj(Y))/n
        sxy = (X * np.conj(Y))/n

        # coherence function
        coer = (sxy ** 2) / (sxx * syy)
        #coer = np.abs(coer.real) # for squared-coherency
        coer = (coer[int(np.ceil( n/2.)):]) #ON 0.72, OFF 1.0
        # phase lag
        nc = len (sxy)
        fi = ma.zeros((nc),float)
        fi = np.arctan2(-sxy.imag, sxy.real)
        fi = fi[int(np.ceil( n/2.)):]
        f = -np.fft.fftfreq(n)[int(np.ceil( n/2.)):]

        sxx  =np.abs (sxx.real)
        sxx  = sxx[int(np.ceil( n/2.)):]

        syy  =np.abs (syy.real)
        syy  = syy[int(np.ceil( n/2.)):]

        amp = 2* 1/n*np.abs(X)
        amp = amp[int(np.ceil( n/2.)):]

        coerdata = {'coer':coer.real, 'fi':fi, 'f':f, 'sxx':sxx, 'syy':syy, 'amp':amp.real}
        return coerdata


def Spectral_diagram(series,s_name,f,namefig):
    """ Taylor Diagram : obs is reference data sample
        in a full diagram (0 --> npi)
        --------------------------------------------------------------------------
        Input: ref_phi - freq of reference  -- > lead to a phase reference
               cohere  - raw of coherence for test (call def cohere)
               phi     - raw of phases for test    (call def cohere)

    """
    import numpy as np
    from numpy import ma
    from matplotlib.projections import PolarAxes
    import mpl_toolkits.axisartist.grid_finder as GF
    import mpl_toolkits.axisartist.floating_axes as FA
    import matplotlib.pyplot as plt
    import statistics
    import matplotlib.cm as cm
    import pylab


    #f_ref = np.abs(f - series[0]['period'][:]).argmin()

    coerdata,coer,std ={},{},{}
    #coerdata2 ={}
    amp,fi = {},{}

    for i in series.keys():
        series[i]   = np.hstack(series[i]) #make sure stack arrays in sequence horizontally (column wise)
        coerdata[i] = coherence(series[i],series[0])
        f_ref       = np.abs(coerdata[0]['f'][:] - f).argmin()
        n           = len(coerdata[i]['sxx'])
        std[i]      = np.sqrt(1.0/n * pow((coerdata[i]['sxx'][f_ref] -np.mean(coerdata[i]['sxx'])),2))
        coer[i]     = coerdata[i]['coer'][f_ref].real
        amp[i]      = coerdata[i]['amp'][f_ref].real
        fi[i]       = coerdata[i]['fi'][f_ref].real

    ref = 1 #1
    #mean values to be R0
    numbers = [coer[key] for key in coer]
    MED = statistics.mean(numbers)
    MAX = np.max(numbers)

    rlocs = np.around(np.concatenate((np.arange(0,-10,-0.2),[-0.95,-0.99],np.arange(0,10,0.2),[0.95,0.99])),2)
    str_rlocs = np.concatenate((np.arange(0,10,0.2),[0.95,0.99],np.arange(0,10,0.2),[0.95,0.99]))
    tlocs = np.arccos(rlocs)        # Conversion to polar angles
    gl1 = GF.FixedLocator(tlocs)    # Positions
    tf1 = GF.DictFormatter(dict(zip(tlocs, map(str,rlocs))))


    str_locs2 = np.arange(-10,11,0.5)
    #rlocs2 = np.concatenate((np.arange(10)/10.,[0.95,0.99]))
    #rlocs2 = np.arange(-10,11,1)
    tlocs2 =  np.arange(-10,11,0.5)      # Conversion to polar angles

    g22 = GF.FixedLocator(tlocs2)
    tf2 = GF.DictFormatter(dict(zip(tlocs2, map(str,str_locs2))))




    tr = PolarAxes.PolarTransform()

    smin = 0
    smax =2.1

    ghelper = FA.GridHelperCurveLinear(tr,
                                           extremes=(0,np.pi, # 1st quadrant  np.pi/2
                                                     smin,smax),
                                           grid_locator1=gl1,
                                           #grid_locator2=g11,
                                           tick_formatter1=tf1,
                                           tick_formatter2=tf2,
                                           )
    fig = plt.figure(figsize=(15,10), dpi=300)
    ax = FA.FloatingSubplot(fig, 323, grid_helper=ghelper)

    fig.add_subplot(ax)
    ax.axis["top"].set_axis_direction("bottom")
    ax.axis["top"].toggle(ticklabels=True, label=True)
    ax.axis["top"].major_ticklabels.set_axis_direction("top")
    ax.axis["top"].label.set_axis_direction("top")
    ax.axis["top"].label.set_text("Coherence function")
    ax.axis["top"].label.set_fontsize(15)

    ax.axis["left"].set_axis_direction("bottom")
    ax.axis["left"].label.set_text("Standard Deviation of power")
    ax.axis["left"].label.set_fontsize(15)

    ax.axis["right"].set_axis_direction("top")
    ax.axis["right"].toggle(ticklabels=True, label=True)
    ax.axis["right"].set_visible(True)
    ax.axis["right"].major_ticklabels.set_axis_direction("bottom")


    ax.axis["bottom"].set_visible(False)

    ax.grid(True)

    ax = ax.get_aux_axes(tr)

    t = np.linspace(0, np.pi)
    r = np.zeros_like(t) + ref

    ax.plot(t,r, 'k--', label='_')


    rs,ts = np.meshgrid(np.linspace(smin,smax),
                            np.linspace(0,np.pi))


    #rms = np.sqrt(ref**2 + rs**2 - 2*ref*rs*np.cos(ts))
    #rms = (4*(1+np.cos(ts))**4/((rs+1/rs)**2 *(1+MAX)**4))#/std[0]
    rms = (4*(1+np.cos(ts))/((rs+1/rs)**2 *(1+MAX)))#/std[0]

    #rms = 1 *e**(-(ref-ts)*np.pi/180)
    CS =ax.contour(ts, rs,rms,[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],cmap = cm.bone)
    plt.clabel(CS, inline=1, fontsize=12)


    ax.plot(np.arccos(0.999999),ref,'k',marker='*',ls='', ms=15)
    aux = range(1,len(series.keys()))
    #del aux[ref]


    cd = ['blue','blue','blue','red', 'red','g','g']
    m = ['o','o','o','o','o']*10
    alf = [1,0.5,1,0.5,1,]*10
    rmse =[]
    for i in aux:
        std[i]=std[i]/std[0]
        ax.plot(np.arccos(coer[i]),std[i],marker=m[i],ms=15,alpha=alf[i],label='%s'%s_name[i])
        #rmse = np.sqrt (round((1**2 + (std[i])**2 - 2*1*(std[i])*coer[i]),3))
        skill = 4*(1+(coer[i]))/((std[i]+1/std[i])**2 *(1+MAX))

        #x = [np.arccos(coer[1]),np.arccos(coer[2])]
        #y = [std[1]/std[0],std[2]/std[0]]
        #ax.plot(x,y,'blue', linewidth = 0.3, alpha=0.7)

        #x = [np.arccos(coer[3]),np.arccos(coer[4])]
        #y = [std[3]/std[0],std[4]/std[0]]
        #ax.plot(x,y,'red', linewidth = 0.3, alpha=0.7)
        print ('name','cohere','STD/STD[0]','SKILL')
        print (s_name[i],coer [i], std[i], skill)
        print ('freq= %s'%f_ref)
        #print rs

    #legend(bbox_to_anchor=(1.1, 1),prop=dict(size='large'),loc='best')
    # -- explain -- Fig 3
    #plt.axhspan(-0.1, 0.2, facecolor='0.5', alpha=0.2)
    t1 = np.linspace(0,2)
    tx = np.linspace(0, np.pi/10)
    t2 =  np.zeros_like(tx) + 2.1
    #ax.fill_between(tx,t2,color='b',facecolor='0.3', alpha=0.2)
    font = {'size'   : 12}
    #plt.text(0.75, 0.35, '$<-- amplitude -->$',color='b',fontdict=font,rotation=10)
    r1 = np.zeros_like(t) + 0.75
    r2 = np.zeros_like(t) + ref+0.25
    #ax.fill_between(t, r1, r2,color ='r',facecolor='0.3', alpha=0.2)
    #plt.text(-0.46, 1.1, '$phase$',color='r',fontdict=font, rotation=0)

    # --- polar ---

    ax = fig.add_subplot(427, projection='polar')
    #r = [1,1,1]
    #theta = [np.pi/2,np.pi/4,np.pi]
    #area = [100,100,100]
    #colors = theta
    #c = ax.scatter(theta, r,s=area, cmap='hsv', alpha=0.75)
    scale_factor=100
    for i in aux:

        ax.scatter(fi[i], std[i],s=amp[i]*scale_factor, cmap='hsv', alpha=1)
        print(fi[i],amp[i])


    # --- power spectrum  --
    cl = ['black','b:', 'b', 'r:','r','g:','g']

    line = [0.5,2,2,2,2,2]
    alfa = [0.5,1,1,1,1,1]

    # time series
    ax = plt.subplot(3,2,2)
    ax.plot(series[0], 'k--')
    for i in aux:
        ax.plot(series[i])


    ax = plt.subplot(3,2,4)
    #ax = plt.subplot(4,2,6)
    for i in aux:

            ax.plot(coerdata[i]['f'][:],coerdata[i]['sxx'][:],alpha=alfa[i],label='%s'%s_name[i], linewidth =line[i] )

            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.set_yscale('log')
            #ax.set_xscale('log')
            plt.xlabel('frequency ',fontsize=15)
            plt.ylabel('Power (log)',fontsize=15)
            ax.set_title('Power Spectrum',fontsize=15)




    ax.legend(bbox_to_anchor=(1, 1),prop=dict(size='large'),loc='best')

    #plt.tight_layout()
    # -- STD vs TD -- Fig4
    fig.text (.5, .8,"(A)",horizontalalignment='right',fontsize=16)
    fig.text (.2, .7,"(B)",horizontalalignment='right',fontsize=16)
    fig.text (.78, .25,"(C)",horizontalalignment='right',fontsize=16)
    fig.text (.2, .2,"(D)",horizontalalignment='right',fontsize=16)
    plt.savefig('%s.png'%namefig,bbox_inches='tight')
    #fig.text (.5, .95,"SPECTRAL Taylor Diagram",horizontalalignment='center',fontsize=16)
    pylab.show()
    return


#from numpy.random import rand
#obs,y1,y2,y3,y4 = 0,0,0,0,0
#n = np.arange(0,256)
#N = n.shape[0]
#k1 =3
#
#fase1 = np.pi
#fase2 = np.pi+np.pi/4 #45
#fase3 = np.pi+np.pi/2 #180
#fase4 = 2*np.pi
#fase5 = 2*np.pi

#amp1 =1.0
#obs = amp1* np.cos((2*np.pi*k1*n/N)+(fase1)) #+np.random.normal(0,1,N)
#y1 =  amp1 * np.cos((2*np.pi*k1*n/N)+(fase2))#+np.random.normal(0,1,N)
#y2 =  amp1 * np.cos((2*np.pi*k1*n/N)+(fase3))
#y3 =  amp1 * np.cos((2*np.pi*k1*n/N)+(fase4))
##y4 =  amp1 * cos((2*np.pi*k1*n/N)+(fase5))
##y5 =  amp1 * cos((2*np.pi*k1*n/N)+(fase5))

#series = {}
#series[0] = obs
#series[1] = y1
#series[2] = y2
#series[3] = y3
#f = 3/256.

#series_names =['OBS','pi/4','pi/2','pi']
#Spectral_diagram(series,series_names,f,'power')
