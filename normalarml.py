# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:02:09 2019

@author: Santos
"""
vS=0
import numpy as np
import scipy.stats as st
from scipy.optimize import minimize, fmin_slsqp
from summarystats import *
import pandas as pd
from datetime import datetime
import scipy.optimize as opt

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



def GetParameters(vP):
    iK= np.size(vP)-2
    # Force vP to be a 1D matrix
    vP= vP.reshape(iK+2,)
    dS= vP[0]
    dBeta0= vP[1]
    dBeta1= vP[2]

    return dS, dBeta0,dBeta1




def transformP(vP0):
    dS, dBeta0,dBeta1=GetParameters(vP0)
    dSTr= np.log(dS)
    #dBeta0= np.log(dBeta0)
    dBeta1Tr= np.log(1-dBeta1)
    
    vPTr= np.hstack((dSTr, dBeta0, dBeta1Tr))
    
    return vPTr
    
    
def transformBackP(vPTr):
    dSTr, dBeta0,dBeta1Tr=GetParameters(vPTr)
    dS= np.exp(dSTr)
    dBeta1= 1- np.exp(dBeta1Tr)
    
    
    vP= np.hstack((dS, dBeta0, dBeta1))
    
    return vP

def regressionLL(vP, vY, mX):
    global vS
    (iN, iK)= mX.shape
    if (np.size(vP) != iK+1):         # Check if vP is as expected
        print ("Warning: wrong size vP= ", vP)
    
    dS, dBeta0,dBeta1=GetParameters(vP)
    if (dS <= 0):
        print ("sigma<=0", end="")
        return -math.inf
    if (dBeta1 >= 1):
        print ("beta1>=1", end="")
        return -math.inf
    vBeta= np.hstack((dBeta0, dBeta1))
    vSs=vS
    LL=st.norm.logpdf(vY,mX @ vBeta,vSs)
    
    #print (".", end="")   
          # Give sign of life
    
    
    return LL


def estimateARml(y,h):
    vY=y.values
    mX=getxar(vY,1,1)
    h=h-1
    
    vY=vY[1+h:]
    (n,k)=mX.shape
    
  
    mX=mX[:mX.shape[0]-h]
    (iN,iK)=mX.shape
    
    vP0= np.ones(iK+1)
    vP0[2]=0.5
    vP0Tr= transformP(vP0)
    vP1= transformBackP(vP0Tr)
    if (np.max(np.fabs(vP0 - vP1)) > 1e-3):
        print ("Something wrong in the transformation?", vP0, vP1)
    else:
        print('transformation succesful')
        
    minavgLL= lambda vPTr: -np.average(regressionLL(transformBackP(vPTr), vY, mX))
    
    dLL= -iN*minavgLL(vP0Tr)
    res= opt.minimize(minavgLL, vP0Tr, method="BFGS")
   
    vPTr= res.x
    vP= transformBackP(vPTr)
    sMess= res.message
    dLL= -iN*res.fun
    
    print ("\nBFGS results in ", sMess, "\nPars: ", vP, "\nLL= ", dLL, ", f-eval= ", res.nfev)
    
    #mHq= hessian_2sided(minavgLL, vPTr)
    #mHtr= -iN*mHq                        
    #mS2tr= np.linalg.inv(-mHtr)    
    #mJ= jacobian_2sided(transformBackP, vPTr)
    #mS2= mJ@mS2tr@mJ.T                    
    #vS= np.sqrt(np.diag(mS2)) 
    print(vP)
    return vP



def estimateARGG(y,h,oo):
    global vS
    vS=oo
    vY=y.values
    mX=getxar(vY,1,1)
    h=h-1
    
    vY=vY[1+h:]
    (n,k)=mX.shape
    
  
    mX=mX[:mX.shape[0]-h]
    (iN,iK)=mX.shape
    
    vP0= np.ones(iK+1)
    vP0[2]=0.5
    vP0Tr= transformP(vP0)
    vP1= transformBackP(vP0Tr)
    if (np.max(np.fabs(vP0 - vP1)) > 1e-3):
        print ("Something wrong in the transformation?", vP0, vP1)
    else:
        print('transformation succesful')
        
    minavgLL= lambda vPTr: -np.average(regressionLL(transformBackP(vPTr), vY, mX))
    
    dLL= -iN*minavgLL(vP0Tr)
    res= opt.minimize(minavgLL, vP0Tr, method="BFGS")
   
    vPTr= res.x
    vP= transformBackP(vPTr)
    sMess= res.message
    dLL= -iN*res.fun
    
    print ("\nBFGS results in ", sMess, "\nPars: ", vP, "\nLL= ", dLL, ", f-eval= ", res.nfev)
    
    #mHq= hessian_2sided(minavgLL, vPTr)
    #mHtr= -iN*mHq                        
    #mS2tr= np.linalg.inv(-mHtr)    
    #mJ= jacobian_2sided(transformBackP, vPTr)
    #mS2= mJ@mS2tr@mJ.T                    
    #vS= np.sqrt(np.diag(mS2)) 
    print(vP)
    return vP


def main():
    # Magic numbers
    # Initialisation
    excelFile = 'data.xlsx'
    
    df = pd.read_excel(excelFile)
    
    
    df=integertodate(df)
    df = df[(df['Year'].dt.year <= 2000) & (df['Year'].dt.year >= 1985)] 
    df.drop(df.columns[[1]], axis=1, inplace=True)
    
    

    
    dftrain=getForecastDataframe(df)
    
    dftrain=dftrain[6]
   
    
    
    estimateARml(dftrain)
    #print(dftrain)
    
    








if __name__ == "__main__":
    main()

def _gh_stepsize(vP):
    vh = 1e-8*(np.fabs(vP)+1e-8)  
    vh= np.maximum(vh, 5e-6)       
    return vh


def jacobian_2sided(fun, vP, *args):
    p = np.size(vP)
    vP= vP.reshape(p)      
    vF = fun(vP, *args)     
    iN= vF.size
    vh= _gh_stepsize(vP)
    mh = np.diag(vh)        
    mGp = np.zeros((iN, p))
    mGm = np.zeros((iN, p))

    for i in range(p):     
        mGp[:,i] = fun(vP+mh[i], *args)
        mGm[:,i] = fun(vP-mh[i], *args)

    vhr = (vP + vh) - vP    
    vhl = vP - (vP - vh)    
    mG= (mGp - mGm) / (vhr + vhl)
    return mG


def hessian_2sided(fun, vP, *args):
    p = np.size(vP,0)
    vP= vP.reshape(p)    
    f = fun(vP, *args)
    vh= _gh_stepsize(vP)
    vPh = vP + vh
    vh = vPh - vP
    mh = np.diag(vh)            
    fp = np.zeros(p)
    fm = np.zeros(p)
    for i in range(p):
        fp[i] = fun(vP+mh[i], *args)
        fm[i] = fun(vP-mh[i], *args)

    fpp = np.zeros((p,p))
    fmm = np.zeros((p,p))
    for i in range(p):
        for j in range(i,p):
            fpp[i,j] = fun(vP + mh[i] + mh[j], *args)
            fpp[j,i] = fpp[i,j]
            fmm[i,j] = fun(vP - mh[i] - mh[j], *args)
            fmm[j,i] = fmm[i,j]

    vh = vh.reshape((p,1))
    mhh = vh @ vh.T             
    mH = np.zeros((p,p))
    for i in range(p):
        for j in range(i,p):
            mH[i,j] = (fpp[i,j] - fp[i] - fp[j] + f + f - fm[i] - fm[j] + fmm[i,j])/mhh[i,j]/2
            mH[j,i] = mH[i,j]

    return mH
