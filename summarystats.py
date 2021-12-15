# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:09:27 2019

@author: Santos
"""
import numpy as np
import pandas as pd
import math

from datetime import datetime

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

def JB(y): #jarque bera test
    n = y.size
    y_mean = y.mean()
    dy = y - y_mean
    T = n / 6 * (skewness(dy)**2 + (kurtosis(dy) - 3)**2 / 4)
    
    return T

def skewness(y):
    n = y.size
    y_mean = y.mean()
    dy = y - y_mean
    return (1 / n * np.sum(dy**3)) / (1 / n * np.sum(dy**2))**(3 / 2.)
    

def kurtosis(y):
    n = y.size
    y_mean = y.mean()
    dy = y - y_mean
    return (1 / n * np.sum(dy**4)) / (1 / n * np.sum(dy**2))**2


def getForecastDataframe(df): #dataframe with right maturity for out of sample h step forecasts
    dfforecast=df.copy()
    for i in range(1,18):
        if (i!=1 and i!=4 and  i!=10 and  i!=12 and  i!=17):
            dfforecast.drop(df.columns[[i]], axis=1, inplace=True)
            
    return dfforecast


def autocovar(y, p): #autocovariance
    n = np.size(y)
    y_mean = np.average(y)
    ac = 0
    for i in np.arange(0, n-p):
        ac += ((y[i+p])-y_mean)*(y[i]-y_mean)
    return (1/(n-1))*ac


def makeautocorrplot(dfcolumn,maxlag,string): #autocorrelation plots
    
    x=np.arange(1,maxlag+1)
    y=np.ones(maxlag)

    for p in range(1,maxlag+1):          
        y[p-1]=autocorr(dfcolumn,p)
    
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.xlim(0,60)
    
    
   
    
    
    plt.plot(x,y)
    
    plt.fill_between(x,y)
    plt.savefig(string)
    plt.show()
    
    
    
    
def splitdata(data,index,i): #split data for rolling window
    split = data.index.get_loc(index)
    
    rollingwindow=12*8
    
    #split+1-rollingwindow
    
    df1=data.iloc[0:split+1]
    return df1


def splitdatas(data,index,i): #split data for rolling window
    split = data.index.get_loc(index)
    
    rollingwindow=12*8
    
    df1=data.iloc[0:split+1]
    return df1


def plotdata(df): #plot yields
    df = df.set_index("Year")
    x,y = np.meshgrid(df.columns.astype(float), df.index)
    z = df.values

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    #rgb = LightSource(270, 45).shade(z, cmap=plt.cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(x, y, z, 
                       linewidth=1,  antialiased=False, shade=False)

    plt.show()


def integertodate(df): #turns date values from integer to date type
    df['Year'] = df['Year'].apply(str)
    df['Year']=[datetime.strptime(el, '%Y%m%d' ) for el in (df['Year'])]    
    return df
    
    
    
def datetointeger(df):    #turns date values from date to integer type
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
    

def makeslopecurve(df_old):   #makes slope and curvature for summary descriptive stats
    df=df_old.copy()

    df['level']=df.iloc[:,[17]]
    df['month3']=df.iloc[:,[1]]
    df['month24']=df.iloc[:,[8]]
    
    df['Slope']=df['level'] - df['month3']
    df['Curvature']=df['month24'] + df['month24'] - df['level'] - df['month3']
    
    df=df.drop(columns=['level', 'month3', 'month24'])
    
    return df


def autocorr(dfcolumn,k):  #p lag , autocorrelation function
    y=dfcolumn.values
    n=y.shape[0]
    ymean=np.mean(y)
    
    y1=y[0:n-k]
    y2=y[k:]
   
    y1=y1 - ymean
    y2=y2-ymean
    y3=y-ymean
    
    corr= (np.dot(y1,y2))/np.dot(y3,y3)
    return corr

def rmse(residual):
    return np.sqrt((residual ** 2).mean())

def mae(residual):
    return (np.abs(residual)).mean()


def autocorcoll(sd,df,lag,columnname,): #calculate autocorrelation for each maturity
    for index in sd.index:
        sd.loc[index,columnname]=   autocorr(df[index],lag)
    return sd    

def msecoll(sd,df,methodname,columnname): #calculate mse for each maturity
    for index in sd.index:
        sd.loc[index,columnname]=  methodname(df[index])
    return sd 


def ADFcoll(sd,df,columnname,a,b): #calculate ADF statistic for each maturity
    maxlag=20
    for index in sd.index:
        sd.loc[index,columnname]=  ADF(df[index].values,maxlag,a,b) 
    return sd 


def getSummaryStats(df): #descriptive summary
    df=makeslopecurve(df)
    
    sd=df.describe()
    sd=sd.loc[['mean', 'std','min', 'max' ]].T
    sd['Autocorr1']=0 
    sd['Autocorr12']=0 
    sd['Autocorr30']=0 
    
    sd=autocorcoll(sd,df,1,'Autocorr1')
    sd=autocorcoll(sd,df,12,'Autocorr12')
    sd=autocorcoll(sd,df,30,'Autocorr30')
    
    
    print(sd)
    return sd

def getSummaryStatsbeta(df): #descriptive stats residual
    sd=df.describe()
    sd=sd.loc[['mean', 'std']].T
    sd['MAE']=0
    sd['RMSE']=0
    sd['Autocorr1']=0 
    sd['Autocorr12']=0 
    sd['Autocorr30']=0 
    
    sd=autocorcoll(sd,df,1,'Autocorr1')
    sd=autocorcoll(sd,df,12,'Autocorr12')
    sd=autocorcoll(sd,df,30,'Autocorr30')
    sd=msecoll(sd,df,rmse,'RMSE')
    sd=msecoll(sd,df,mae,'MAE')
    
    #print(sd.to_latex(float_format="%.3f"))
    return sd


def getSummaryStatsbeta2(df): #descriptive stats estimated betas
    sd=df.describe()
    sd=sd.loc[['mean', 'std','min', 'max' ]].T
    
    sd['Autocorr1']=0 
    sd['Autocorr12']=0 
    sd['Autocorr30']=0 
    sd['ADF']=0
    
    
    sd=autocorcoll(sd,df,1,'Autocorr1')
    sd=autocorcoll(sd,df,12,'Autocorr12')
    sd=autocorcoll(sd,df,30,'Autocorr30')
    sd=ADFcoll(sd,df,'ADF',1,0)
    
    print(sd.to_latex(float_format="%.3f"))

    return sd
    
def getSummaryForecast(df,h):  #summary for each h-step forecast done with rolling window
    sd=df.describe()
    sd=sd.loc[['mean', 'std']].T
    sd['RMSE']=0
    
    if(h==1):
        string1='Autocorr1'
        string2='Autocorr12'
        s1=1
        s2=12
    elif(h==6):
        string1='Autocorr6'
        string2='Autocorr18'
        s1=6
        s2=18
    elif(h==12):
        string1='Autocorr12'
        string2='Autocorr24'
        s1=12
        s2=24
    
    sd[string1]=0 
    sd[string2]=0 
  
    sd=msecoll(sd,df,rmse,'RMSE')
    sd=autocorcoll(sd,df,s1,string1)
    sd=autocorcoll(sd,df,s2,string2)
    
    print(sd)
    print(sd.to_latex(float_format="%.3f"))

def ADF(y,maxlag,a=1,b=0): #augmented dickey fuller test
    if (a==0 and b==1):
        print("not possible, no adf with trend and no constant")
        return 0
    
    vBestlag=np.ones((maxlag,2))
    
    for p in range(1,maxlag+1):
        tstat,bic=DFUnit(y,p,a,b)
        vBestlag[p-1,0]=bic
        vBestlag[p-1,1]=tstat
    
    minvalue=np.argmin(vBestlag[:,0])
    tstat=vBestlag[minvalue,1]
    
    print(tstat)
    return tstat    
    
    
    
def DFUnit(y,p=1,a=1,b=0):  #a for constant, b for trend , p lag, can be used for augmented dicker fuller test with an informatio criterion
    iP=p
    y_level=y
    y_level=y_level[p:y_level.shape[0]-1]
    y=np.diff(y)
    
    n=y.shape[0]
    y=y.reshape(-1)
    if (a==0 and b==0):
        mX= np.ones((n-iP,iP+1 )) 
        mX[:,0]=y_level
        for p in range(1,iP+1): 
            pp=iP-p
            mX[:,p]= y[pp:n-p]
    
    if(a==1 and b==0):
        mX= np.ones((n-iP,iP +2 )) 
        mX[:,1]=y_level
        for p in range(1,iP+1): 
            pp=iP-p
            mX[:,p+1]= y[pp:n-p]
            
    #if (a==0 and b==1):
     #   mX= np.ones((n-iP,iP+2 )) 
      #  mX[:,0]=np.arange(1,n-iP+1)
       # mX[:,1]=y_level
        #for p in range(1,iP+1): 
         #   pp=iP-p
          #  mX[:,p+1]= y[pp:n-p]
    
    if(a==1 and b==1):
        mX= np.ones((n-iP,iP +3 )) 
        mX[:,1]=np.arange(1,n-iP+1)
        mX[:,2]=y_level
        for p in range(1,iP+1): 
            pp=iP-p
            mX[:,p+2]= y[pp:n-p]
            
    y=y[p:]
    x=mX
    
    n=y.shape[0]
    y=y.reshape(n,1)
    
    beta= np.linalg.inv(x.T@ x)  @ x.T @ y
    y_hat=mX @ beta
    e=y-y_hat
    e=y-y_hat.reshape(y_hat.shape[0],1)
    
    SE=(e.T @ e) /(e.shape[0] - (p+a+b+1))
    mCov=np.linalg.inv(x.T@ x)
    s=np.diagonal(mCov)
    se=np.sqrt((SE*s))
    beta, se = beta.reshape(-1) , se.reshape(-1)
    beta1=beta[a+b]
    se1=se[a+b]
    tstat=beta1/se1
    
    bic=BIC(e,p+a+b+1)   
    
    return tstat, bic 
    
def BIC(residual,k):  #Bayesian information criterion, k parameters
    sqresidual=np.square(residual)
    n=residual.shape[0]
    sumres=np.sum(sqresidual) #SSE
    
    bic=n*math.log(sumres*(1/n))+k*math.log(n)
    return bic 

def AIC(residual,k):  #Akaike information criterion
    sqresidual=np.square(residual)
    n=residual.shape[0]
    sumres=np.sum(sqresidual) #SSE
    
    aic= 2*k - 2*math.log(sumres)
    
    return aic
    


def newzerodataframe(df_old):    #creates a dataframe with simililar structure and same dates, but with zero values
    df=df_old.copy()
    for col in df.columns:
        if(col not in ['Year']):
            df[col].values[:] = 0
      
    return df



def rse(residual):
    return np.sqrt((residual ** 2))
                   
def se(residual):
    return (residual ** 2)


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


def PCAgeteigenvector(dff,n): #gets eigenvectors for PCA
    df=dff.copy()
    df.drop(df.columns[[0]], axis=1, inplace=True)
    mX=df.values
    mean=np.mean(mX, axis=0)
    mean=mean.reshape(17,1).T
    Xmean=np.repeat(mean,mX.shape[0],axis=0)

    X= mX - Xmean   #subscract mean so datapoints are centered around origin
    Cov_X = np.cov(X,rowvar = False)
    eigenvalues,eigenvectors = np.linalg.eig(Cov_X)
    
    sortorder = np.argsort(eigenvalues)[::-1]  #sort from high eigenvalue to low
    eigenvalues = eigenvalues[sortorder]
    eigenvectors = eigenvectors[:,sortorder]
    

    eigenvalues=eigenvalues[0:n]
    eigenvectors=eigenvectors[:,0:n]
    
    return eigenvectors
    
    
def PCA(df): #PCA 
    eigenvectors =PCAgeteigenvector(df,3)
    q1,q2,q3=eigenvectors[:,0],eigenvectors[:,1],eigenvectors[:,2]
    q1,q2,q3=q1.reshape(17,1) ,q2.reshape(17,1), q3.reshape(17,1)
    q1,q2,q3=q1.T,q2.T,q3.T
    
    xdf=df.drop(columns=['Year'])
    Yt=xdf.values.T
    x1,x2,x3=q1 @ Yt,q2 @ Yt,q3 @ Yt
   
    xdf['x1'],xdf['x2'],xdf['x3'] =x1.T ,x2.T  ,x3.T        
    xdf=xdf[['x1','x2','x3']]

    return xdf, q1 ,q2 ,q3


def DMtest(e1,e2,h,criterionfunc): #Diebold Mariano test ,criterion func is the function you give, in our case, we give the se function as argument
    n=e1.shape[0]
    
    e1=e1.astype(float)
    e2=e2.astype(float)
    e1,e2=criterionfunc(e1),criterionfunc(e2)
    d=e1 - e2
    d=d.astype(float)
    d_mean=np.average(d)
    fd=fd0(d,h)
   # print(d_mean)
    #print(fd)
    #print("----")
    dmterm=((fd/n)**(1/2))
    
    DM_stat=d_mean/dmterm
    #DM_stat=DM_stat*((n+1-2*h+h*(h-1)/n)/n)**(0.5) #correction term for small sample
    
    return DM_stat

def fd0(d,h):
    if(h==1):
        return   autocovar(d,0)
    else:
        term1=autocovar(d,0)
        term2=0
        for lag in range(1,h):
            temp= 2*autocovar(d,lag)
            term2=term2 + temp
        return (term1 + term2)


def Compare2modelforecasts(a1,a12,b1,b12): #compare two models using the diebold-mariano test
    a1,a12,b1,b12=a1.values,a12.values,b1.values,b12.values
    a1,a12,b1,b12=a1[:,1:],a12[:,1:],b1[:,1:],b12[:,1:]
    
    cm1=np.zeros(a1.shape[1])
    cm12=np.zeros(a1.shape[1])
    
    for i in range(0,a1.shape[1]): 
        dm=DMtest(a1[:,i],b1[:,i],1,se)
        cm1[i]=dm
    
    for ii in range(0,a1.shape[1]): 
        dm=DMtest(a12[:,ii],b12[:,ii],1,se)
        cm12[ii]=dm
    
    
    
    df = pd.DataFrame({'h step forecast':cm1,'h- step forecast':cm12})
    print(df.to_latex(float_format="%.3f")) 
    return df

def plotloadingnelson(T,labda): 
    exp_lt = np.exp(-T*labda)
    
    factor2 = (1 - exp_lt) / (T*labda)
    factor3 = factor2 - exp_lt    
    factor1=np.ones(np.size(T,0))  #constant
    factor1,factor2,factor3=factor1.reshape(-1,1),factor2.reshape(-1,1) ,factor3.reshape(-1,1)
    
    return factor1,factor2,factor3

def plottheoreticalnelson(): #plot loading 3 factor nelson siegel model
    labda=0.0609
    T=np.linspace(0,120,120)
    factor1,factor2,factor3=plotloadingnelson(T,labda)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.xlim(0.01,120)
    plt.plot(T,factor1,linestyle='--')
    plt.plot(T,factor2)
    plt.plot(T,factor3,linestyle='-.')
    plt.legend(['B1 Loading', 'B2 Loading','B3 Loading'], loc='upper right')
    
    plt.savefig('nelsontheorie.png')
    plt.show()


def plotloadingnelsonSven(T,labda,labda2):
    factor1=np.ones(np.size(T,0))
    exp_lt = np.exp(-T*labda)
    factor2 = (1 - exp_lt) / (T*labda)
    factor3 = factor2 - exp_lt
    
    ex = np.exp(-T*labda2)
    fa = (1 - ex) / (T*labda2)
    factor4 = fa - ex  #constant
    
    factor1,factor2,factor3,factor4=factor1.reshape(-1,1),factor2.reshape(-1,1) ,factor3.reshape(-1,1),factor4.reshape(-1,1)
    
    return factor1,factor2,factor3,factor4

def plottheoreticalnelsonSven():
    labda=0.09909090909090908
    labda2=0.01
    T=np.linspace(0,120,120)
    factor1,factor2,factor3,factor4=plotloadingnelsonSven(T,labda,labda2)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.xlim(0.01,120)
    plt.plot(T,factor1,linestyle='--')
    plt.plot(T,factor2)
    plt.plot(T,factor3,linestyle='-.')
    plt.plot(T,factor4,linestyle='-.')
    plt.legend(['B1 Loading', 'B2 Loading','B3 Loading','B4 Loading'], loc='upper right')
    
    plt.savefig('nelsontheoriesven.png')
    plt.show()


def plotloadingnelsonBjork(T,labda,labda2):
    factor1=np.ones(np.size(T,0))
    exp_lt = np.exp(-T*labda)
    factor2 = (1 - exp_lt) / (T*labda)
    factor3 = factor2 - exp_lt
    
    ex = np.exp(-T*labda2)
    factor4 = (1 - ex) / (T*labda2)
    
    factor1,factor2,factor3,factor4=factor1.reshape(-1,1),factor2.reshape(-1,1) ,factor3.reshape(-1,1),factor4.reshape(-1,1)
    
    return factor1,factor2,factor3,factor4

def plottheoreticalnelsonBjork():
    labda=0.06146845637583894
    labda2=2*labda
    T=np.linspace(0,120,120)
    factor1,factor2,factor3,factor4=plotloadingnelsonBjork(T,labda,labda2)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.xlim(0.01,120)
    plt.plot(T,factor1,linestyle='--')
    plt.plot(T,factor2)
    plt.plot(T,factor4)
    plt.plot(T,factor3,linestyle='-.')
    plt.legend(['B1 Loading', 'B2 Loading','B3 Loading','B4 Loading'], loc='upper right')
    
    plt.savefig('nelsontheoriebjork.png')
    plt.show()
    
def plotloadingnelsonFive(T,labda,labda2):
    factor1=np.ones(np.size(T,0))
    exp_lt = np.exp(-T*labda)
    factor2 = (1 - exp_lt) / (T*labda)
    factor3 = factor2 - exp_lt
    
    ex = np.exp(-T*labda2)
    factor4 = (1 - ex) / (T*labda2)
    factor5 = factor4 - ex  #constant
    
    factor1,factor2,factor4,factor3,factor5=factor1.reshape(-1,1),factor2.reshape(-1,1) ,factor3.reshape(-1,1),factor4.reshape(-1,1),factor5.reshape(-1,1)
    
    return factor1,factor2,factor3,factor4,factor5

def plottheoreticalnelsonFive():
    labda=0.49827586206896546
    labda2=0.019655172413793102
    T=np.linspace(0,120,120)
    factor1,factor2,factor3,factor4,factor5=plotloadingnelsonFive(T,labda,labda2)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.xlim(0.01,120)
    plt.plot(T,factor1,linestyle='--')
    plt.plot(T,factor2)
    plt.plot(T,factor4)
    plt.plot(T,factor3,linestyle='-.')
    plt.plot(T,factor5,linestyle='-.')
    plt.legend(['B1 Loading', 'B2 Loading','B3 Loading','B4 Loading','B5 Loading'], loc='upper right')
    
    plt.savefig('nelsontheorieFive.png')
    plt.show()
    
def AR(column,p,h,c=1,returnmorethenbeta=0): #AR(p) method, c =0 for no constant , else there is constant 
    if(returnmorethenbeta==0):
        y=column.values
    if(returnmorethenbeta!=0):
        y=column
    
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
    
    if(returnmorethenbeta!=0):
        y_pred=x @ beta
        squaredresiduals=(y-y_pred)**2
        return beta, squaredresiduals
    else:
        return beta
    
    



def getxar(y,iP,c): # get mX matrix used for (OLS) regression
    
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

def Engletestforbeta(dfbeta,maxlag): # engle test to test for arch effects to justify use of Garch(1,1)
    b=dfbeta.values
    b=b[:,1:]
    e=b.shape[1]
    result=np.ones((maxlag,e))
    
    for i in range(0,e):
        y=b[:,i]#.astype(float)
        y=y.astype(float)
        beta,sqres=AR(y,1,1,1,1)
        result[:,i]=LM(sqres,maxlag).reshape(-1)
        print(result[:,i])
        #het_arch(np.sqrt(sqres), maxlag=e)
               
    return result


def LM(residuals,maxlag): #lagrane multiplier test
    TR2=np.zeros((maxlag,1))
    
    for p in range(1,maxlag+1):
        beta,sqres=AR(residuals,p,1,1,1)
        SSR=np.sum(sqres)
        res_mean=np.mean(residuals[p:])
        SST=(residuals[p:] - res_mean)**2
        R2=1-(SSR/(np.sum(SST)))
        n=sqres.shape[0]
        TR2[p-1]=(n)*R2
    
    return TR2
    
    