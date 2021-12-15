# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:02:09 2019

@author: Santos
"""
vSS=0
var_bounds=(0.01,10)
import numpy as np
import scipy.stats as st
from scipy.optimize import minimize, fmin_slsqp
from normalarml import *
from summarystats import *
import pandas as pd
from datetime import datetime
import scipy.optimize as opt
#from arch.univariate import ARX, GARCH, Normal 
#faster computation time
def GetParameters(vP):
    vP= vP.reshape(np.size(vP),)
    dC= vP[0]
    dPhi= vP[1]
    dOmega = vP[2]
    dAlpha= vP[3]
    dBeta=vP[4]
    dS= vP[5]
    
    return dC,dPhi  ,dOmega, dAlpha,dBeta,dS

def getxar(y,iP,c):   
    n=y.shape[0]
    y=y.reshape(-1)
    if (c==0):
        mX= np.ones((n-iP,iP )) 
        for p in range(1,iP+1): 
            pp=iP-p
            mX[:,p-1]= y[pp:n-p]
    
    if(c!=0):
        mX= np.ones((n-iP,iP +1 )) 
        for p in range(1,iP+1): 
            pp=iP-p
            mX[:,p]= y[pp:n-p]

    return mX


def regressionLLgarch(vP):
    global vSS
    (iN, iK)= mX.shape
    dC,dPhi  ,dOmega, dAlpha,dBeta,dS=GetParameters(vP)
    
    
    vB= np.hstack((dC, dPhi))
    
    N=vY.shape[0]
    vE=vY.reshape(N,1) - (mX @ vB).reshape(N,1) #residuals
    
    vS = np.ones((N,1)) 
    for t in range(N):
        if t == 0:
            vS[t] = dS
            if(vS[t]<10**-12): #robust code, to avoid dividing through zero, because computer might round off to zero
                vS[t]=10**-12
            #vS[t]=boundS(vS[t],var_bounds)
        else:
            vS[t] = dOmega + dAlpha*vE[t-1]**2 + dBeta*vS[t-1]
            if(vS[t]<10**-12):
                vS[t]=10**-12
            #vS[t]=boundS(vS[t],var_bounds)
                
    #print(vS)
    vLL=st.norm.logpdf(vE,0,vS) #normal distrubution
    vLL=-0.5 * (np.log(2 * np.pi) + np.log(vS) + vE ** 2.0 / vS)
    vSS=vS
    
    dLL=-np.average(vLL)
    return dLL



def estimatenormalGarch(y,h):  #garch(1,1)
    beta=AR(y,1,h)
    global vY, mX
    #print(beta)
    
    vY=y.values
    mX=getxar(vY,1,1)
    h=h-1
    
    vY=vY[1+h:]
    (n,k)=mX.shape
    
  
    mX=mX[:mX.shape[0]-h]
    
    pp=1
    qq=1
    garchnn=pp+qq + 1

    vP= np.ones(k+garchnn+1)
    #vP[0]=0.20398061                        #c           
    #vP[1]=0.9684539                     #Phi
   # vP[2]=0.00167115                        #omega
   # vP[3]=0.05                    #alpha
   # vP[4]=0.93                       #beta
   # vP[5]=0.20
    vP[0]=beta[0]                        #c           
    vP[1]=beta[1]                     #Phi
    vP[2]=0.2                        #omega
    vP[3]=0.05                    #alpha
    vP[4]=0.90                       #beta
    vP[5]=0.5
    
    
    #vP[0]=0.1617                       #c           
    #vP[1]=0.9738                   #Phi
    #vP[2]=0.0019597                     #omega
    #vP[3]=0.05                    #alpha
    #vP[4]=0.9631                      #beta
    
    
    #print(regressionLLgarch(vP))
    sv=vP
    #dC,dPhi  ,dOmega, dAlpha,dBeta=GetParameters(vP)
    
    cb=(0,8)
    
    pb=(0,1)
    
    vOmega_bnds   = (0, 1)

    vAlpha_bnds   = (0, 1)

    vBeta_bnds    = (0, 1)

    vS_bnds      = (0, 5)
    
    bounds=[cb,pb,vOmega_bnds, vAlpha_bnds, vBeta_bnds,vS_bnds]
    
    
    
    res = opt.minimize(regressionLLgarch, sv, method='SLSQP',bounds=bounds)
    
    #print(res)
    #print(res.x)
    #print('ll')
    #print(-n*res.fun)
    #print("ML estimator: c: ", "%.4f" % vTheta_ML[0])

    #print("ML estimator: phi: ", "%.4f" % vTheta_ML[1])
    
    #print("ML estimator: omega: ", "%.4f" % vTheta_ML[0])

    #print("ML estimator: alpha: ", "%.4f" % vTheta_ML[1])

    #print("ML estimator: beta:  ", "%.4f" % vTheta_ML[2])

    #print("ML estimator: nu:    ", "%.4f" % vTheta_ML[3])
    
    
    

   

    
    #print(vP)
    
    return vP[0:2]


def normalparamsgarch(y,h) :
    mX=getxar(y,1,1)
    h=h-1
    yy=y[1+h:]
    xx=mX[:mX.shape[0]-h]
 
    ag = ARX(yy,xx,lags=0,constant=False)  
    ag.distribution = Normal()  #
    ag.volatility = GARCH(p=1, q=1)  

    result = ag.fit(disp='off')

    
    betacoef=result.params[0:2].values
    
    return betacoef


