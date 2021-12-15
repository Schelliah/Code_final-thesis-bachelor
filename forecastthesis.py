# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:22:38 2019

@author: Santos

"""
import numpy as np
import pandas as pd
import math
#from numpy.linalg import inv
from datetime import datetime

import matplotlib.pyplot as plt
from summarystats import *
#from studentgarch import *
#from normalgarch import *

from studentgarch import *
from normalgarch import *
labda=0.0609


def loadingvarnelson(T,labda): #loading
    exp_lt = np.exp(-T*labda)
    factor2 = (1 - exp_lt) / (T*labda)
    factor3 = factor2 - exp_lt
       
    factor1=np.ones(np.size(T,0))  #constant
    factormatrix=np.concatenate((factor1.reshape(-1,1),factor2.reshape(-1,1) ,factor3.reshape(-1,1)), 1)
       
    return factormatrix


def getxforbeta(yields=0):
    labda=0.0609
    
    if(yields==0):
        list = [3,6,9,12,15,18,21,24,30,36,48,60,72,84,96,108,120]
    elif(yields==1):
        list=[3,12, 36, 60, 120]
        
    T=np.array(list)
    t=T.astype(float)
    
    x=loadingvarnelson(t,labda)
    x=x.astype(float)
    
    return x

def arforecast(beta,x):
    x_hat=np.asscalar(beta[0]) + np.asscalar(beta[1])* x
    return x_hat


def varforecast(betavar,x):
    x_hat=np.add(betavar[:,0].reshape(betavar.shape[0],1),betavar[:,1:] @ x.T)           
    return x_hat  #(3,1) shape
    


def nelsonarforecast(beta1,beta2,beta3,dfbeta):
    dfbeta=dfbeta.tail(n=1)
    
    dfbeta.drop(dfbeta.columns[[0]], axis=1, inplace=True)
    oldbeta=dfbeta.values
    oldbeta=oldbeta.astype(float)

    b1=arforecast(beta1,oldbeta[0,0])
    b2=arforecast(beta2,oldbeta[0,1])    
    b3=arforecast(beta3,oldbeta[0,2])
     
    y=nelsonsiegelforecasttoyield(b1,b2,b3)

    return y
    
    
def pcforecast(beta1,beta2,beta3,q1,q2,q3,tail):
    oldbeta=tail.values

    x1=arforecast(beta1,oldbeta[0,0])
    x2=arforecast(beta2,oldbeta[0,1])    
    x3=arforecast(beta3,oldbeta[0,2])
     
    y=pctoyield(x1,x2,x3,q1,q2,q3)

    return y

def pctoyield(x1,x2,x3,q1,q2,q3):
    y=q1*x1 + x2*q2 + x3*q3
    y=y.reshape(-1)
    y3m,y1,y3,y5,y10=y[0], y[3], y[9] , y[11] , y[16]
    
    vY=np.array([y3m, y1, y3,y5,y10])
    return vY
    
    
    

def nelsonvarforecast(betavar,dfbeta):
    dfbeta=dfbeta.tail(n=1)
    dfbeta.drop(dfbeta.columns[[0]], axis=1, inplace=True)
    oldbeta=dfbeta.values
    
    b=varforecast(betavar,oldbeta)
    b1=b[0,0]
    b2=b[1,0]
    b3=b[2,0]
    y=nelsonsiegelforecasttoyield(b1,b2,b3)

    return y              
                  

def nelsonsiegelforecasttoyield(b1,b2,b3):
    x=getxforbeta(1)
    b = np.array([b1,b2,b3]).reshape(3,1)
    
    vY=x @ b
    vY=vY.reshape(-1)
    
    return vY

def AR(column,p,h,c=1): # c =0 for no constant , else there is constant 
    y=column.values
    h=h-1
    x=getxar(y,p,c)
    y=y[p+h:]
    n=y.shape[0]
    y=y.reshape(n,1)
    if (p==1 and c==0):
        nn=x.shape[0]
        x=x.reshape(nn,1)
    
    x=x[:x.shape[0]-h]
   
    beta= np.linalg.inv(x.T@ x)  @ x.T @ y
    
    return beta


def getxar(y,iP,c):
    n=y.shape[0]
    #print(n)
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


def EstimateVAR(mY, p,h=1,value=0):
    if(value==0):
        mY=mY.values
    mY=mY.T
    h=h-1
    mZ= getZ(mY, p)
    
    mY=mY[:,p+h:]
    mX=mZ[:,p:mZ.shape[1]-h]
    
    mB= mY @ mX.T @ np.linalg.inv(mX@mX.T)
    return mB

def getZ(mY, iP, dMiss= np.nan):
    (iK, iT)= mY.shape
    mZ= np.ones((1+iK*iP, iT)) * dMiss
    mZ[0,:]= 1
    for p in range(iP):
        mZ[1+p*iK:1+(p+1)*iK,p+1:]= mY[:,:iT-p-1]
    
    return mZ

def EstimateChangeVar(mY,h=1):
    mY=mY.values

    dY=np.diff(mY,axis=0)
    beta=EstimateVAR(dY, 1,1,2)
    
    return beta, dY

def EstimateECM1(mY,h=1):
    mY=mY.values

    vY3=mY[:mY.shape[0]-1,0]
    y=mY[1:,:]

    dZ=y - vY3.reshape(vY3.shape[0],1)    
    beta=EstimateVAR(dZ, 1,1,2)
    
    return beta, dZ

def EstimateECM2(mY,h=1):
    mY=mY.values
    
    vY3=mY[:mY.shape[0]-1,0] #vector with maturity 3 for difference
    vY3=vY3.reshape(vY3.shape[0],1)
    vY12=mY[:mY.shape[0]-1,1] #vector with maturity 12 for difference
    vY12=vY12.reshape(vY12.shape[0],1)
    
    y=mY[1:,:]
    dZ=y - vY3
    dZ[:,1]=dZ[:,1] + vY3.reshape(-1)- vY12.reshape(-1)
    
    beta=EstimateVAR(dZ, 1,1,2)
    
    return beta, dZ
    
    

def sloperegression(column, column3,h=1):
    h=h-1
    vY=column.values
    vY3=column3.values
    n=vY.shape[0]
    
    x=np.ones((n-1,2))
    x[:,1]=vY[:n-1] - vY3[:n-1]
    y=np.diff(vY)
    y[h:]
    x[:x.shape[0]-h]
    
    beta= np.linalg.inv(x.T@ x)  @ x.T @ y
    return beta


def FBregression(column, columnfr,h=1):
    h=h-1
    vY=column.values
    vFr=columnfr.values
    n=vY.shape[0]
    
    x=np.ones((n-1,2))
    x[:,1]=vFr[:n-1] - vY[:n-1]
    y=np.diff(vY)
    y[h:]
    x[:x.shape[0]-h]
    
    beta= np.linalg.inv(x.T@ x)  @ x.T @ y
    return beta



def AR(column,p,h,c=1): # c =0 for no constant , else there is constant 
    y=column.values
    h=h-1
    x=getxar(y,p,c)
    y=y[p+h:]
    n=y.shape[0]
    y=y.reshape(n,1)
    if (p==1 and c==0):
        nn=x.shape[0]
        x=x.reshape(nn,1)
    
    x=x[:x.shape[0]-h]
   
    
    
    beta= np.linalg.inv(x.T@ x)  @ x.T @ y
    
    
    return beta


def getForecastDataframe(df):
    dfforecast=df.copy()
    
    dfforecast=dfforecast[['Year',3,12,36,60,120]]
    
            
    return dfforecast



def forecastrandomwalk(df):
    iForecastmonths=7*12
    
    alist = [1, 6, 12] 
    blist = [3,12, 36, 60, 120]
    
    dfforecast=getForecastDataframe(df)
    dfy=dfforecast[(dfforecast['Year'].dt.year >=1994)]
    dfres=newzerodataframe(dfy)
    
    for h in alist:
        split=288-h 
        for i in range(1,iForecastmonths+1):
            dftrain= splitdata(dfforecast,split,i-1)
            for m in blist:
               
                #estimate estimate dftrain, return beta
                y_hat=dftrain[m].tail(n=1)
                b=np.asscalar(y_hat)
                
                
                dfres[m].loc[i+287]= dfy[m].loc[i+287] - b
            
            #go to next month for next iteration
            split=split+1
        sd=getSummaryForecast(dfres,h)
        if (h==1):
            sm1=sd
            e1=dfres.copy()
        elif(h==6):
            sm6=sd
        elif(h==12):
            sm12=sd
            e12=dfres.copy()
    return sm1,sm6,sm12,e1,e12
        
    

def forecastlevelar1(df):
    iForecastmonths=7*12
    
    alist = [1, 6, 12] 
    blist = [3,12, 36, 60, 120]
    
    dfforecast=getForecastDataframe(df)
    dfy=dfforecast[(dfforecast['Year'].dt.year >=1994)].copy()
    
    dfres=newzerodataframe(dfy)
    
    
    dfforecast.iloc
    for h in alist:
        split=288-h 
        for i in range(1,iForecastmonths+1):
            dftrain= splitdata(dfforecast,split,i-1)
            
            for m in blist:
                beta=AR(dftrain[m].astype(float),1,h)
                tail=dftrain[m].tail(n=1).values
                b=np.asscalar(beta[0]) + np.asscalar (beta[1])* np.asscalar(tail)
                
              #  if(h!=1):
               #     for j in range(1,h):
                #        b= np.asscalar(beta[0]) + np.asscalar(beta[1])*b
                
                dfres[m].loc[i+287]= dfy[m].loc[i+287] - b
                
            
            #go to next month for next iteration
            split=split+1
            
        sd=getSummaryForecast(dfres,h) 
        if (h==1):
            sm1=sd
        elif(h==6):
            sm6=sd
        elif(h==12):
            sm12=sd
        
    return sm1,sm6,sm12






def forecastlevelVAR(df):
    iForecastmonths=7*12
    
    alist = [1, 6, 12] 
    blist = [3,12, 36, 60, 120]
    
    dfforecast=getForecastDataframe(df)
    dfy=dfforecast[(dfforecast['Year'].dt.year >=1994)].copy()
    dfres=newzerodataframe(dfy)
    
    dfforecast.iloc
    for h in alist:
        split=288-h
        for i in range(1,iForecastmonths+1):
            dftrain= splitdata(dfforecast.copy(),split,i-1)
            dftrain=dftrain.drop(columns=['Year'])
            sigma=EstimateVAR(dftrain, 1,h)
            yy=dftrain.tail(n=1).values
            y=varforecast(sigma,yy)
            count=0
            for m in blist: 
                dfres[m].loc[i+287]= dfy[m].loc[i+287] - y[count]
                count=count+1
            
            #go to next month for next iteration
            split=split+1
        sd=getSummaryForecast(dfres,h)
        if (h==1):
            sm1=sd
        elif(h==6):
            sm6=sd
        elif(h==12):
            sm12=sd
        
    return sm1,sm6,sm12


def forecastchangeVAR(df):
    iForecastmonths=7*12
    
    alist = [1, 6, 12] 
    blist = [3,12, 36, 60, 120]
    
    dfforecast=getForecastDataframe(df)
    dfy=dfforecast[(dfforecast['Year'].dt.year >=1994)].copy()
    dfres=newzerodataframe(dfy)
    
    dfforecast.iloc
    for h in alist:
        split=288-h
        for i in range(1,iForecastmonths+1):
            dftrain= splitdata(dfforecast,split,i-1)
            dftrain=dftrain.drop(columns=['Year'])
            
            sigma,dY=EstimateChangeVar(dftrain,h)
            tail=dftrain.tail(n=1)
            rr=dY[dY.shape[0]-1,:].reshape(5,1)
            yy=rr.T

            y=varforecast(sigma,yy) + tail.values.T

            count=0
            for m in blist: 
                
                dfres[m].loc[i+287]= dfy[m].loc[i+287] - y[count]
                count=count+1
            
            #go to next month for next iteration
            split=split+1
        sd=getSummaryForecast(dfres,h)
        if (h==1):
            sm1=sd
        elif(h==6):
            sm6=sd
        elif(h==12):
            sm12=sd
        
    return sm1,sm6,sm12

def forecastECM1(df):
    iForecastmonths=7*12
    
    alist = [1, 6, 12] 
    blist = [3,12, 36, 60, 120]
    
    dfforecast=getForecastDataframe(df)
    dfy=dfforecast[(dfforecast['Year'].dt.year >=1994)].copy()
    dfres=newzerodataframe(dfy)
    
    dfforecast.iloc
    for h in alist:
        split=288-h
        for i in range(1,iForecastmonths+1):
            dftrain= splitdata(dfforecast,split,i-1)
            dftrain=dftrain.drop(columns=['Year'])
            
            sigma,dY=EstimateECM1(dftrain,h)
            rr=dY[dY.shape[0]-1,:].reshape(5,1)
            yy=rr.T
            tail=dftrain[3].tail(n=1)
            y=varforecast(sigma,yy) + tail.values.T
            count=0
            for m in blist: 
                
                dfres[m].loc[i+287]= dfy[m].loc[i+287] - y[count]
                count=count+1
            
            #go to next month for next iteration
            split=split+1
        sd=getSummaryForecast(dfres,h)
        if (h==1):
            sm1=sd
        elif(h==6):
            sm6=sd
        elif(h==12):
            sm12=sd
        
    return sm1,sm6,sm12



def forecastECM2(df):
    iForecastmonths=7*12
    
    alist = [1, 6, 12] 
    blist = [3,12, 36, 60, 120]
    
    dfforecast=getForecastDataframe(df)
    dfy=dfforecast[(dfforecast['Year'].dt.year >=1994)].copy()
    dfres=newzerodataframe(dfy)
    
    for h in alist:
        split=288-h
        for i in range(1,iForecastmonths+1):
            dftrain= splitdata(dfforecast,split,i-1)
            dftrain=dftrain.drop(columns=['Year'])
            
            sigma,dY=EstimateECM2(dftrain,h)
            tail=np.asscalar(dftrain[3].tail(n=1))
            tail2=np.asscalar(dftrain[12].tail(n=1))
            compvector=np.array([tail,tail2,tail,tail,tail])
            cv=compvector.reshape(5,1)
            rr=dY[dY.shape[0]-1,:].reshape(5,1)
            yy=rr.T
            y=varforecast(sigma,yy) + cv #add compvector,compensation vector
            count=0
            for m in blist: 
                
                dfres[m].loc[i+287]= dfy[m].loc[i+287] - y[count]
                count=count+1
            
            #go to next month for next iteration
            split=split+1
        sd=getSummaryForecast(dfres,h)
        if (h==1):
            sm1=sd
        elif(h==6):
            sm6=sd
        elif(h==12):
            sm12=sd
        
    return sm1,sm6,sm12




def forecastsloperegression(df):
    iForecastmonths=7*12
    
    alist = [1, 6, 12] 
    blist = [12, 36, 60, 120]
    
    dfforecast=getForecastDataframe(df)
    dfy=dfforecast[(dfforecast['Year'].dt.year >=1994)].copy()
   
    
    dfres=newzerodataframe(dfy)
    
    
    dfforecast.iloc
    for h in alist:
        split=288-h
        for i in range(1,iForecastmonths+1):
            dftrain= splitdata(dfforecast.copy(),split,i)
            for m in blist:
                beta=sloperegression(dftrain[m],dftrain[3],h)
                tail=dftrain.tail(n=1)
                xx=tail[m] - tail[3]
                b=np.asscalar(beta[0]) + np.asscalar (beta[1])* np.asscalar(xx) +np.asscalar(tail[m])
            
                dfres[m].loc[287+i]= dfy[m].loc[287+i] - b
                
            
            #go to next month for next iteration
            split=split+1
            
        sd=getSummaryForecast(dfres,h) 
        if (h==1):
            sm1=sd
        elif(h==6):
            sm6=sd
        elif(h==12):
            sm12=sd
        
    return sm1,sm6,sm12
        

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')       


def forecastnelsonsiegelar(df,dfbeta2,l=0.0609):
    global labda
    iForecastmonths=7*12
    
    labda=l
    
    alist = [1, 6, 12] 
    blist = [3,12, 36, 60, 120]
    
    dfforecast=getForecastDataframe(df)
    #print_full(dfbeta2)
    dfy=dfforecast[(dfforecast['Year'].dt.year >=1994)].copy()
    dfres=newzerodataframe(dfy)
    
    
    
    dfforecast.iloc
    for h in alist:
        split=288-h
        for i in range(1,iForecastmonths+1):
            #dftrain, dftest= splitdata(dfforecast,split)
            dfbeta= splitdata(dfbeta2, split,i-1) 
            #print(dfbeta)
            beta1coef=AR(dfbeta['Beta1'],1,h) #lag 1, constant included            
            beta2coef=AR(dfbeta['Beta2'],1,h)            
            beta3coef=AR(dfbeta['Beta3'],1,h)
            
            
            y=nelsonarforecast(beta1coef,beta2coef,beta3coef,dfbeta)
            count=0
            for m in blist: 
                dfres[m].loc[287+i]= dfy[m].loc[287+i] - y[count]
                count=count+1
            
            #go to next month for next iteration
            split=split+1
        sd=getSummaryForecast(dfres,h) 
        if (h==1):
            sm1=sd
            e1=dfres.copy()
        elif(h==6):
            sm6=sd
        elif(h==12):
            sm12=sd
            e12=dfres.copy()
    return sm1,sm6,sm12,e1,e12
      
         
    
        
        

def forecastnelsonsiegelvar(df,dfbeta2):
    iForecastmonths=7*12
    
    alist = [1, 6, 12] 
    blist = [3,12, 36, 60, 120]
    
    dfforecast=getForecastDataframe(df)
    dfy=dfforecast[(dfforecast['Year'].dt.year >=1994)].copy()
    dfres=newzerodataframe(dfy)
    
    dfforecast.iloc
    for h in alist:
        split=288-h
        for i in range(1,iForecastmonths+1):
            #dftrain, dftest= splitdata(dfforecast,split)
            dfbeta= splitdata(dfbeta2,split,i-1)
            
           
            varbeta=dfbeta[['Beta1','Beta2','Beta3']]
            betavar=EstimateVAR(varbeta, 1)
            y=nelsonvarforecast(betavar,dfbeta)
            count=0
            for m in blist: 
                
                dfres[m].loc[i+287]= dfy[m].loc[i+287] - y[count]
                count=count+1
            
            #go to next month for next iteration
            split=split+1
        sd=getSummaryForecast(dfres,h)
        if (h==1):
            sm1=sd
        elif(h==6):
            sm6=sd
        elif(h==12):
            sm12=sd
        
    return sm1,sm6,sm12

def forecastPCA(df):
    iForecastmonths=7*12
    
    alist = [1, 6, 12] 
    blist = [3,12, 36, 60, 120]
    
    dfforecast=getForecastDataframe(df)
    
    dfy=dfforecast[(dfforecast['Year'].dt.year >=1994)].copy()
    dfres=newzerodataframe(dfy)

    for h in alist:
        split=288-h
        for i in range(1,iForecastmonths+1):
            dfpca= splitdata(df, split,0)
            dftrain= splitdata(dfforecast, split,0)
            xdf,q1,q2,q3=PCA(dfpca)
            
            
            beta1coef=AR(xdf['x1'],1,h) #lag 1, constant included
            
            beta2coef=AR(xdf['x2'],1,h)
            
            beta3coef=AR(xdf['x3'],1,h)
            
            y=pcforecast(beta1coef,beta2coef,beta3coef,q1,q2,q3,xdf.tail(n=1))
            count=0
            for m in blist: 
                
                dfres[m].loc[i+287]= dfy[m].loc[i+287] - y[count]
                count=count+1
            
            #go to next month for next iteration
            split=split+1
        sd=getSummaryForecast(dfres,h) 
        if (h==1):
            sm1=sd
        elif(h==6):
            sm6=sd
        elif(h==12):
            sm12=sd
        
    return sm1,sm6,sm12

def forecastFB(df,dfforwardrates):
    iForecastmonths=7*12
    
    alist = [1, 6, 12] 
    blist = [3,12, 36, 60, 120]
    
    dfforecast=getForecastDataframe(df)
    dfy=dfforecast[(dfforecast['Year'].dt.year >=1994)].copy()
     
    dfres=newzerodataframe(dfy)
       
    dfforecast.iloc
    for h in alist:
        split=288-h
        for i in range(1,iForecastmonths+1):
            dftrain= splitdata(dfforecast.copy(),split,i)
            dftrainfr= splitdata(dfforwardrates.copy(),split-180,i)
            for m in blist:
                beta=sloperegression(dftrain[m],dftrainfr[m],h)
                tail=dftrain.tail(n=1)
                tail2=dftrainfr.tail(n=1)
                xx= np.asscalar(tail[m]) - np.asscalar(tail2[m])
                b=np.asscalar(beta[0]) + np.asscalar (beta[1])* xx +np.asscalar(tail[m])
            
                dfres[m].loc[287+i]= dfy[m].loc[287+i] - b
                
            
            #go to next month for next iteration
            split=split+1
            
        sd=getSummaryForecast(dfres,h) 
        if (h==1):
            sm1=sd
            e1=dfres.copy()
        elif(h==6):
            sm6=sd
        elif(h==12):
            sm12=sd
            e12=dfres.copy()
    return sm1,sm6,sm12,e1,e12

def forecastgarchnormal(df,dfbeta2):
    iForecastmonths=7*12
    
    alist = [1, 6, 12] 
    blist = [3,12, 36, 60, 120]
    
    dfforecast=getForecastDataframe(df)
    #print_full(dfbeta2)
    dfy=dfforecast[(dfforecast['Year'].dt.year >=1994)].copy()
    dfres=newzerodataframe(dfy)
    
    
    
    dfforecast.iloc
    for h in alist:
        split=288-h
        for i in range(1,iForecastmonths+1):
            #dftrain, dftest= splitdata(dfforecast,split)
            dfbeta= splitdatas(dfbeta2, split,i-1)   
            vP=normalparamsgarch(dfbeta['Beta1'].values,h)
            beta1coef=vP.reshape(2,1)
            beta2coef=AR(dfbeta['Beta2'],1,h)
            beta3coef=AR(dfbeta['Beta3'],1,h)
            
            y=nelsonarforecast(beta1coef,beta2coef,beta3coef,dfbeta)
            count=0
            for m in blist: 
                dfres[m].loc[287+i]= dfy[m].loc[287+i] - y[count]
                count=count+1
            
            #go to next month for next iteration
            split=split+1
        sd=getSummaryForecast(dfres,h) 
        if (h==1):
            sm1=sd
            e1=dfres.copy()
        elif(h==6):
            sm6=sd
        elif(h==12):
            sm12=sd
            e12=dfres.copy()
    return sm1,sm6,sm12,e1,e12

def forecastgarchstudent(df,dfbeta2):
    iForecastmonths=7*12
    
    alist = [1, 6, 12] 
    blist = [3,12, 36, 60, 120]
    
    dfforecast=getForecastDataframe(df)
    #print_full(dfbeta2)
    dfy=dfforecast[(dfforecast['Year'].dt.year >=1994)].copy()
    dfres=newzerodataframe(dfy)
    
    
    
    dfforecast.iloc
    for h in alist:
        split=288-h
        for i in range(1,iForecastmonths+1):
            #dftrain, dftest= splitdata(dfforecast,split)
            dfbeta= splitdatas(dfbeta2, split,i-1)   
            vP=studentparamsgarch(dfbeta['Beta1'].values,h)
            beta1coef=vP.reshape(2,1)
            beta2coef=AR(dfbeta['Beta2'],1,h)
            beta3coef=AR(dfbeta['Beta3'],1,h)
            
            y=nelsonarforecast(beta1coef,beta2coef,beta3coef,dfbeta)
            count=0
            for m in blist: 
                dfres[m].loc[287+i]= dfy[m].loc[287+i] - y[count]
                count=count+1
            
            #go to next month for next iteration
            split=split+1
        sd=getSummaryForecast(dfres,h) 
        if (h==1):
            sm1=sd
            e1=dfres.copy()
        elif(h==6):
            sm6=sd
        elif(h==12):
            sm12=sd
            e12=dfres.copy()
    return sm1,sm6,sm12,e1,e12

