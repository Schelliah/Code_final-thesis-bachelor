# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:26:13 2019

@author: Santos
"""
### Imports
import numpy as np
import pandas as pd
import math
from summarystats import *
#from numpy.linalg import inv
from datetime import datetime

def integertodate(df):
    df['Year'] = df['Year'].apply(str)
    df['Year']=[datetime.strptime(el, '%Y%m%d' ) for el in (df['Year'])]    
    return df
    
    
    
def datetointeger(df):    
    df['tyear']= df['Year'].dt.year
    df['tmonth']= df['Year'].dt.month
    df['tday']= df['Year'].dt.day
    df['tyear']=df['tyear'].astype(str)
    df['tmonth']=df['tmonth'].astype(str)
    df['tmonth']=df['tmonth'].apply('{:0>2}'.format)
    df['tday']=df['tday'].astype(str)
    df['tday']=df['tday'].apply('{:0>2}'.format)
    
    df['tdate']=  df['tyear'] + df['tmonth'] + df['tday'] 
    df['tdate']=df['tdate'].astype(int)
    df['Year']=df['tdate']
    df=df.drop(columns=['tyear', 'tmonth', 'tday' ,'tdate'])
        
    return df


def varforecast(betavar,x):
    x_hat=np.add(betavar[:,0].reshape(betavar.shape[0],1),betavar[:,1:] @ x.reshape(x.shape[0],1))           
    return x_hat  #(2,1) shape




def varhforecast(beta,mY,h):
    mZ=np.zeros((mY.shape[0],mY.shape[1]))
    mS=mY
    
    for jj in range(0,192):
        print(mY[jj,:])
        mZ[jj,:]=varforecast(beta,mY[jj,:]).reshape(-1)
        

    mZ=mZ[:mZ.shape[0]-h,:]
    mY=mY[h:,:]
    
    mU=mY-mZ
    mU=mU.T
    #print(mZ)
    
    return mU

#############################################################################################
def EstimateVARwithSE(mY, p,h,value=0):
    if(value==0):
        mY=mY.values
        
    mY=mY.T
    h=h-1
    mZ= getZ(mY, p)
    
    
    mY=mY[:,p+h:]
    mX=mZ[:,p:mZ.shape[1]-h]
    
    (k, n)= mY.shape
    
    mB= mY @ mX.T @ np.linalg.inv(mX@mX.T)
    
    mU= mY - mB@mX
 
    mCov= mU@mU.T/n
    print(mCov)
    
    #waldtest=waldtest(mBeta,mCov)
    
    xx=np.diag(np.linalg.inv(mX@mX.T))
    SE=np.zeros((mCov.shape[0],xx.shape[0]))
    ii=0
    for i in np.diag(mCov):
        a=np.sqrt(i*xx)
        SE[ii,:]=a.reshape(-1)
        ii=ii+1
    
    mS=SE
    result=stackBetaandSE(mB,mS)
    
    return result

def stackBetaandSE(mB,mS):
    result=np.zeros((mB.shape[0]*2,mB.shape[1]))
    j=0
    for i in range(0,mB.shape[0]):
        result[j,:]=mB[i,:]
        result[j+1,:]=mS[i,:]
        j=j+2
    
    return result
##########################################################################################
def getZ(mY, iP, dMiss= np.nan):
    (iK, iT)= mY.shape
    mZ= np.ones((1+iK*iP, iT)) * dMiss
    mZ[0,:]= 1
    for p in range(iP):
        mZ[1+p*iK:1+(p+1)*iK,p+1:]= mY[:,:iT-p-1]
    
    return mZ


def VARdecomp(mB,mU,mY,mX,mCov,h):  #implementaton variance decomposition for each var
    (kk, n)= mY.shape

    theta0, theta1, sumtheta=impulseresponsevar1(mB,mCov)
    L=theta0
   
    A1=mB[:,1:]

    mse=np.zeros((kk))
    
    G=np.zeros((kk,kk))
    e=np.identity(kk)
    
    
    thet=np.identity(kk)
    mse=(thet @ mCov @ thet.T)
    for ii in range(1,h):
        thet= np.linalg.matrix_power(A1,ii)
        mse= mse +  (thet @ mCov @ thet.T)
    
 
    
    mse=np.diagonal(mse)
    
    
    for j in range(0,kk):
        for k in range(0,kk):
            thett=np.identity(kk) @ L
            up=(e[:,j].T  @ thett @   e[:,k])**2
            for iii in range(1,h):
                thett= (np.linalg.matrix_power(A1,iii) @ L)
                up= up + (e[:,j].T  @ thett @      e[:,k])**2
                
            G[k,j]=up/mse[j]
    decomp=G.T 
    print(decomp)
      
    return G
            

def impulseresponsevar1(mB,mCov):  #get impulse response
    c=mB[:,0]
    a1=mB[:,1:]
    L = np.linalg.cholesky(mCov)
    
    theta0=L
    theta1=L @ a1
    sumtheta=L + L @ a1
    
    return  theta0, theta1, sumtheta

