# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 11:37:55 2021

@author: ruzhu
"""
import pandas as pd
import numpy as np
from numpy import matlib as mb
import rainflow
import math
#import random
#import matplotlib.pyplot as pltfrom docplex.mp.model import Model


def Linear_Part(
            whole_SoC, kdelta1, kdelta2, kdelta3, 
            ksigma, sigma_ref, kT, Tref, kti, 
            rf_DoD, rf_SoC, rf_count, days
        ): 
            ld = 0
            ld_t = 0
            for j in range(0, len(rf_DoD)):
                if rf_DoD[j] == 0:
                   S_DoD = 0
                else:
                   S_DoD = (kdelta1*rf_DoD[j]**kdelta2+kdelta3)**(-1)*rf_count[j]
                
                S_SoC = math.exp(ksigma*(rf_SoC[j]-sigma_ref))
                S_T = math.exp(kT*(20-Tref)*Tref/20)
                
                ld_t = ld_t+S_DoD*S_SoC*S_T  # cycling degradation

            S_time = kti*((days+1)*24*3600)
            S_SoC = math.exp(ksigma*(sum(whole_SoC)/len(whole_SoC)-sigma_ref))
            ld = ld_t+S_time*S_SoC*S_T   
            return ld, ld_t
        
def Non_Linear_Part(alpha, beta, ld, Ini_nld, pre_nld, ld1, nld1, ld_t):
    if (pre_nld<0.08):
        nld = 1-alpha*math.exp(-ld*beta)-(1-alpha)*math.exp(-ld) + Ini_nld
        ld1 = ld
        nld1 = nld
    else:
        #nld1 = 1-alpha*math.exp(-ld1*beta)-(1-alpha)*math.exp(-ld1)
        nld = 1 - (1 - nld1)*math.exp(-(ld-ld1))
    nld_t = 1-alpha*math.exp(-ld_t*beta)-(1-alpha)*math.exp(-ld_t)
    return nld, nld_t, ld1, nld1


def cycle_calculation(day_num, ld):
    
       
    #model 3: through ld
    t = day_num
    S_time = 4.14e-10*((t+1)*24*3600)
    S_T = math.exp(0.0693*(20-25)*25/20)
    cycles = (ld - S_time*S_T)/S_T*17000
    cycles = pd.DataFrame([cycles], columns=['cycles'])

        
    return cycles

def Deg_Model(whole_SoC, Ini_nld, pre_nld, ld1, nld1, days): 
    alpha = 0.0575
    beta = 121
    kdelta1 = 140000
    kdelta2 = -0.5010
    kdelta3 = -123000
    ksigma = 1.04
    sigma_ref = 0.5
    kT = 0.0693
    Tref = 25
    kti = 4.14e-10
            
            
    rf_DoD = list()
    rf_SoC = list()
    rf_count = list()
    for rng, mean, count, i_start, i_end in rainflow.extract_cycles(whole_SoC):#SoC time serise must be list
        rf_DoD.append(rng)
        rf_SoC.append(mean)
        rf_count.append(count)
            
    ld, ld_t = Linear_Part(
    whole_SoC, kdelta1, kdelta2, kdelta3, 
    ksigma, sigma_ref, kT, Tref, kti, 
    rf_DoD, rf_SoC, rf_count, days
    )
    
    cycles = cycle_calculation(days, ld)
            
    nld, nld_t, ld1, nld1 = Non_Linear_Part(alpha, beta, ld, Ini_nld, pre_nld, ld1, nld1, ld_t)
            
    return ld, nld, ld1, nld1, rf_DoD, rf_SoC, rf_count, nld_t, cycles


def slope_update(whole_Pdis, whole_Pcha, whole_nld, days, up_freq, T, DI, ad_all): 
    throughput = (whole_Pdis.iloc[-T:,0] + whole_Pcha.iloc[-T:,0]).sum()
    if throughput ==0:
        ad = ad_all.iloc[-1,0]
    else:        
        if days<up_freq+1:
           #ad = (whole_nld.iloc[-1,0] - whole_nld.iloc[-2,0])/(whole_Pdis.iloc[-T:,0] + whole_Pcha.iloc[-T:,0]).sum()/DI
           ad = ad_all.iloc[0,0]
        else:
           ad = (whole_nld.iloc[-1,0] - whole_nld.iloc[-(up_freq+1),0])/(whole_Pdis.iloc[-T*up_freq:,0] + whole_Pcha.iloc[-T*up_freq:,0]).sum()/DI
        if ad<0:
            ad=1e-8            
        
    return ad









