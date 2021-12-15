# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:39:10 2019

@author: Santos
"""

labda=0.019655172413793102 # found optimized value through gridsearch method below, here called Gridfive
labda2=0.49827586206896546 #


### Imports
import numpy as np
import pandas as pd
import math
#from numpy.linalg import inv
from datetime import datetime

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

from summarystats import *


def makeautocorrplot(dfcolumn,maxlag):
    
    x=np.arange(1,maxlag+1)
    y=np.ones(maxlag)
    
    
    
    for p in range(1,maxlag+1):          
        y[p-1]=autocorr(dfcolumn,p)
    
    
    plt.plot(x,y)
    plt.fill_between(x,y)
    plt.show
    
    
    
    
def splitdata(data,index,i):
    split = data.index.get_loc(index)
    
    df1=data.iloc[0:split+1:,:]
    
    #df2=data.iloc[split+1:,:]
    return df1


def plotdata(df):
    df = df.set_index("Year")
    x,y = np.meshgrid(df.columns.astype(float), df.index)
    z = df.values

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    #rgb = LightSource(270, 45).shade(z, cmap=plt.cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(x, y, z, 
                       linewidth=1,  antialiased=False, shade=False)

    plt.show()


    

def makeslopecurve(df_old):
    df=df_old.copy()

    df['level']=df.iloc[:,[17]]
    df['month3']=df.iloc[:,[1]]
    df['month24']=df.iloc[:,[8]]
    
    df['Slope']=df['level'] - df['month3']
    df['Curvature']=df['month24'] + df['month24'] - df['level'] - df['month3']
    
    df=df.drop(columns=['level', 'month3', 'month24'])
    
    return df


    
        
def loadingvarnelsonFive(T,labda,labda2):
    exp_lt = np.exp(-T*labda)
    factor2 = (1 - exp_lt) / (T*labda)
    factor3 = factor2 - exp_lt
    
    ex = np.exp(-T*labda2)
    factor4 = (1 - ex) / (T*labda2)
    factor5 = factor4 - ex
    
    
    factor1=np.ones(np.size(T,0))  #constant
    factormatrix=np.concatenate((factor1.reshape(-1,1),factor2.reshape(-1,1) ,factor3.reshape(-1,1),factor4.reshape(-1,1),factor5.reshape(-1,1)), 1)
    
    
    return factormatrix


def getxforbetaFive(yields=0):
    global labda, labda2
    
    if(yields==0):
        list = [3,6,9,12,15,18,21,24,30,36,48,60,72,84,96,108,120]
    elif(yields==1):
        list=[3,12, 36, 60, 120]
        
    T=np.array(list)
    t=T.astype(float)
    
    x=loadingvarnelsonFive(t,labda,labda2)
    x=x.astype(float)
    
    return x




    

def NelsolsiegelFive(df):
    x=getxforbetaFive()
        
    temp_y=df.values
    y=temp_y[:,1:]
    y=y.astype(float)
    y=y.T
    beta2= np.linalg.inv(x.T@ x)  @ x.T @ y
    Beta=beta2.T
               
    #print(np.mean(Beta[:,1]))
    dfbeta = pd.DataFrame(columns=['Year', 'Beta1','Beta2','Beta3','Beta4','Beta5'])
    dfbeta['Year']=df['Year']
    dfbeta['Beta1']=Beta[:,0]
    dfbeta['Beta2']=Beta[:,1]
    dfbeta['Beta3']=Beta[:,2]
    dfbeta['Beta4']=Beta[:,3]
    dfbeta['Beta5']=Beta[:,4]
    
    
    df = df.set_index("Year")

    
    dfpred=newzerodataframe(df)
    dfforward=newzerodataframe(df)
    
    dfres=newzerodataframe(df)
    
    
    i=0
    for index in df.index:     
        dfpred.loc[index]=x @ Beta[i,:].T
        dfres.loc[index] = df.loc[index].values - dfpred.loc[index].values 
        i=i+1
        
    
    print(getSummaryStatsbeta2(dfbeta))  #beta
    
    #makeautocorrplot(dfbeta['Beta3'],60)
    
    ADF(dfbeta['Beta3'],1,0,1)
   
    
    
    print(getSummaryStatsbeta(dfres))   #residuals
    alist = [1, 6, 12]
    for h in alist:
        beta1coef=AR(dfbeta['Beta1'],1,h)
    
    return dfbeta


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

def nelsonFiveforecast(beta1,beta2,beta3,beta4,beta5,dfbeta):
    dfbeta=dfbeta.tail(n=1)
    
    dfbeta.drop(dfbeta.columns[[0]], axis=1, inplace=True)
    oldbeta=dfbeta.values

    b1=arforecast(beta1,oldbeta[0,0])
    b2=arforecast(beta2,oldbeta[0,1])    
    b3=arforecast(beta3,oldbeta[0,2])
    b4=arforecast(beta4,oldbeta[0,3])
    b5=arforecast(beta5,oldbeta[0,4])
     
    y=nelsonsiegelforecasttoyield(b1,b2,b3,b4,b5)

    return y


def nelsonsiegelforecasttoyield(b1,b2,b3,b4,b5):
    x=getxforbetaFive(1)
    b = np.array([b1,b2,b3,b4,b5]).reshape(5,1)
    
    vY=x @ b
    
    return vY
 
def arforecast(beta,x):
    x_hat=np.asscalar(beta[0]) + np.asscalar(beta[1])* x
    return x_hat


def getForecastDataframe(df):
    dfforecast=df.copy()
    for i in range(1,18):
        if (i!=1 and i!=4 and  i!=10 and  i!=12 and  i!=17):
            dfforecast.drop(df.columns[[i]], axis=1, inplace=True)
            
    return dfforecast

def forecastnelsonsiegelFive(df):
    iForecastmonths=7*12
    dfbeta2=NelsolsiegelFive(df)
    
    alist = [1, 6, 12] 
    blist = [3,12, 36, 60, 120]
    
    dfforecast=getForecastDataframe(df)
    
    dfy=dfforecast[(dfforecast['Year'].dt.year >=1994)]
    dfres=newzerodataframe(dfy)
    
    
    
    dfforecast.iloc
    for h in alist:
        split=288-h
        for i in range(1,iForecastmonths+1):
            #dftrain, dftest= splitdata(dfforecast,split)
            dfbeta= splitdata(dfbeta2, split,i-1)
            
            beta1coef=AR(dfbeta['Beta1'],1,h) #lag 1, constant included
            
            beta2coef=AR(dfbeta['Beta2'],1,h)
            
            beta3coef=AR(dfbeta['Beta3'],1,h)
            
            beta4coef=AR(dfbeta['Beta4'],1,h)
            
            beta5coef=AR(dfbeta['Beta5'],1,h)
            
            y=nelsonFiveforecast(beta1coef,beta2coef,beta3coef,beta4coef,beta5coef,dfbeta)
            count=0
            for m in blist: 
                
                dfres[m].loc[i+287]= dfy[m].loc[i+287] - y[count]
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
      
        
    
def gridFive(df):
    global labda, labda2
    
    #grid=np.linspace(0.0001,0.15,120)
    #grid2=np.linspace(0.0001,0.9,100)
    
    grid=np.linspace(0.01,0.15,30)
    grid2=np.linspace(0.01,0.6,30)
    result=np.ones((grid.shape[0]*grid2.shape[0],3))
    j=0
    
    for i in np.nditer(grid):
        for ii  in np.nditer(grid2):
            if(i!=ii):
                labda=i
                labda2=ii
                result[j,0]=i
                result[j,1]=ii
                result[j,2]=NelsolsiegelFivegrid(df)
                
            else:
                result[j,0]=i
                result[j,1]=ii
                result[j,2]=500
            j=j+1
    
    minvalue=np.argmin(result[:,2])
    #print(result)
    bestLabda=result[minvalue,0]
    bestLabda2=result[minvalue,1]
    
    print('bestlabda')
    print(bestLabda)
    print(bestLabda2)
    labda=bestLabda
    labda2=bestLabda2
    
    return bestLabda
        

def NelsolsiegelFivegrid(df):
    x=getxforbetaFive()
        
    temp_y=df.values
    y=temp_y[:,1:]
    y=y.astype(float)
    y=y.T
    beta2= np.linalg.inv(x.T@ x)  @ x.T @ y
    Beta=beta2.T
               
    #print(np.mean(Beta[:,1]))
    dfbeta = pd.DataFrame(columns=['Year', 'Beta1','Beta2','Beta3','Beta4','Beta5'])
    dfbeta['Year']=df['Year']
    dfbeta['Beta1']=Beta[:,0]
    dfbeta['Beta2']=Beta[:,1]
    dfbeta['Beta3']=Beta[:,2]
    dfbeta['Beta4']=Beta[:,3]
    dfbeta['Beta5']=Beta[:,4]
    
    
    df = df.set_index("Year")

    
    dfpred=newzerodataframe(df)
    
    dfres=newzerodataframe(df)
    
    
    i=0
    for index in df.index:     
        dfpred.loc[index]=x @ Beta[i,:].T
        dfres.loc[index] = df.loc[index].values - dfpred.loc[index].values 
        i=i+1
    
   
    
    sd=getSummaryStatsbeta(dfres)
    sumrmse=sd['RMSE'].values

    
    result=np.sum(sumrmse)
    #result=sumrmse[16]

    
    return result    

def plot4whole(df,dfbeta):
    
    230 #1989-03-31
    df1=df.loc[230]
    dfb1=dfbeta.loc[230]
    plot4yield(df1,dfb1,'fivensplot1.png')
    
    234 #1989-07-31
    df2=df.loc[234]
    dfb2=dfbeta.loc[234]
    plot4yield(df2,dfb2,'fivensplot2.png')
    
    328 #1997-05-30
    df3=df.loc[328]
    dfb3=dfbeta.loc[328]
    plot4yield(df3,dfb3,'fivensplot3.png')
    
    343 #1998-08-31
    df4=df.loc[343]
    dfb4=dfbeta.loc[343]
    plot4yield(df4,dfb4,'fivensplot4.png')

def plot4yield(df,dfbeta,string):
    list = [3,6,9,12,15,18,21,24,30,36,48,60,72,84,96,108,120]
    T=np.array(list)
    t=T.astype(float)
    
    real=df.values
    beta=dfbeta.values
    
    real=real[1:].astype(float)
    beta=beta[1:].astype(float)
    print(real)
    print(beta)
    
    tt=np.linspace(0,120,120)
    x=getmeanforbeta(tt)
    
    y=x @ beta
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.xlabel('maturity', fontsize=24)
    plt.ylabel('yields', fontsize=24)
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 

    
    plt.xlim(0,120)
    plt.plot(tt,y,linestyle='--')
    plt.scatter(t,real)

    
    plt.savefig(string)
    plt.show()

def getmeanforbeta(t):
    global labda, labda2
    x=loadingvarnelsonFive(t,labda,labda2)
    meanpred=x.astype(float)
    
    return meanpred


def main():
    # Magic numbers


   

    # Initialisation
    excelFile = 'data.xlsx'
    
    df = pd.read_excel(excelFile)
    
    
    df=integertodate(df)
    df = df[(df['Year'].dt.year <= 2000) & (df['Year'].dt.year >= 1985)] 
    #df.drop(df.columns[[1]], axis=1, inplace=True)
    df2 = df[(df['Year'].dt.year < 1993)] 


    df=datetointeger(df)
    
    
    plotdata(df)
    df=integertodate(df)
    
    gridFive(df2)
    dfbeta=NelsolsiegelFive(df)
    plot4whole(df,dfbeta)
 
    

    dfbeta=NelsolsiegelFive(df)
    #print_full(dfbeta)
    forecastnelsonsiegelFive(df)
    
    


    

    # transform data + plot







if __name__ == "__main__":
    main()

