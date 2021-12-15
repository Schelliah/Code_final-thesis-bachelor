# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:39:10 2019

@author: Santos
"""



### Imports
import numpy as np
import pandas as pd
import math
#from numpy.linalg import inv
from datetime import datetime

from summarystats import *

#labda=0.0609
labda=0.05055454545454546


def loadingvarnelsonTwo(T,labda):
    exp_lt = np.exp(-T*labda)
    factor2 = (1 - exp_lt) / (T*labda)
    
    factor1=np.ones(np.size(T,0))  #constant
    factormatrix=np.concatenate((factor1.reshape(-1,1),factor2.reshape(-1,1)), 1)
    
    
    return factormatrix


def getxforbetaTwo(yields=0):
    #labda=0.0609
    global labda
    
    if(yields==0):
        list = [3,6,9,12,15,18,21,24,30,36,48,60,72,84,96,108,120]
    elif(yields==1):
        list=[3,12, 36, 60, 120]
        
    T=np.array(list)
    t=T.astype(float)
    
    x=loadingvarnelsonTwo(t,labda)
    x=x.astype(float)
    
    return x




    

def NelsolsiegelTwo(df):
    x=getxforbetaTwo()
        
    temp_y=df.values
    y=temp_y[:,1:]
    y=y.astype(float)
    y=y.T
    print(y.shape)
    print(x.shape)
    beta2= np.linalg.inv(x.T@ x)  @ x.T @ y
    Beta=beta2.T
               
    #print(np.mean(Beta[:,1]))
    dfbeta = pd.DataFrame(columns=['Year', 'Beta1','Beta2'])
    dfbeta['Year']=df['Year']
    dfbeta['Beta1']=Beta[:,0]
    dfbeta['Beta2']=Beta[:,1]
    
    
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
    
   
    
    
    print(getSummaryStatsbeta(dfres))   #residuals
    
    return dfbeta


def gridtwo(df):
    global labda
    
    grid=np.linspace(0.0001,0.10,100)
    result=np.ones((grid.shape[0],2))
    j=0
    
    for i in np.nditer(grid):
        labda=i
        result[j,0]=i
        result[j,1]=NelsolsiegelTwogrid(df)
        j=j+1
        
    plt.plot(result[:,0],result[:,1])
    plt.show()
    
    minvalue=np.argmin(result[:,1])
    bestLabda=result[minvalue,0]
    
    print(bestLabda)
    labda=bestLabda
    return bestLabda
        

def NelsolsiegelTwogrid(df):
    x=getxforbetaTwo()
        
    temp_y=df.values
    y=temp_y[:,1:]
    y=y.astype(float)
    y=y.T
    beta2= np.linalg.inv(x.T@ x)  @ x.T @ y
    Beta=beta2.T
               
    #print(np.mean(Beta[:,1]))
    dfbeta = pd.DataFrame(columns=['Year', 'Beta1','Beta2'])
    dfbeta['Year']=df['Year']
    dfbeta['Beta1']=Beta[:,0]
    dfbeta['Beta2']=Beta[:,1]
    
    
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

    
    return result
    


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

def nelsonTwoforecast(beta1,beta2,dfbeta):
    dfbeta=dfbeta.tail(n=1)
    
    dfbeta.drop(dfbeta.columns[[0]], axis=1, inplace=True)
    oldbeta=dfbeta.values

    b1=arforecast(beta1,oldbeta[0,0])
    b2=arforecast(beta2,oldbeta[0,1])    

     
    y=nelsonsiegelforecasttoyield(b1,b2)

    return y


def nelsonsiegelforecasttoyield(b1,b2):
    x=getxforbetaTwo(1)
    b = np.array([b1,b2]).reshape(2,1)
    
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

def forecastnelsonsiegelTwo(df):
    iForecastmonths=7*12
    dfbeta2=NelsolsiegelTwo(df)
    
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
            
            
            y=nelsonTwoforecast(beta1coef,beta2coef,dfbeta)
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
      
        
def plot4whole(df,dfbeta):
    
    230 #1989-03-31
    df1=df.loc[230]
    dfb1=dfbeta.loc[230]
    plot4yield(df1,dfb1,'twoplot1.png')
    
    234 #1989-07-31
    df2=df.loc[234]
    dfb2=dfbeta.loc[234]
    plot4yield(df2,dfb2,'twoplot2.png')
    
    328 #1997-05-30
    df3=df.loc[328]
    dfb3=dfbeta.loc[328]
    plot4yield(df3,dfb3,'twoplot3.png')
    
    343 #1998-08-31
    df4=df.loc[343]
    dfb4=dfbeta.loc[343]
    plot4yield(df4,dfb4,'twoplot4.png')

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
    global labda
    x=loadingvarnelsonTwo(t,labda)
    meanpred=x.astype(float)
    
    return meanpred
    


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')



def main():
    # Magic numbers


   

    # Initialisation
    excelFile = 'data.xlsx'
    
    df = pd.read_excel(excelFile)
    
    
    df=integertodate(df)
    df = df[(df['Year'].dt.year <= 2000) & (df['Year'].dt.year >= 1985)] 
    df2 = df[(df['Year'].dt.year < 1993)] 

    df=datetointeger(df)
    
    gridtwo(df)
    dfbeta=NelsolsiegelTwo(df)
    plot4whole(df,dfbeta)
    #plotdata(df)
    df=integertodate(df)
    
    
 
    

    
    #print_full(dfbeta)
    
    #forecastnelsonsiegelTwo(df)
    
    


    

    # transform data + plot







if __name__ == "__main__":
    main()

