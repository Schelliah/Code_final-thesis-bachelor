# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:24:31 2019

@author: Santos
"""
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
from forecastthesis import *
from forecastthesisfinal import *
from nelsonsiegelSvensson import *
from nelsonsiegelBjork import *
from nelsonsiegelFive import *
from nelsonsiegelTwo import *
from studentgarch import *
from normalgarch import *
from normalarml import *
#from macrovartest import *
pd.options.mode.chained_assignment=None

#labda=0.0609

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

def autocovar(y, p):
    n = np.size(y)
    y_mean = np.average(y)
    ac = 0
    for i in np.arange(0, n-p):
        ac += ((y[i+p])-y_mean)*(y[i]-y_mean)
    return (1/(n))*ac
    
def splitdata(data,index,i):
    data=data.copy()
    split = data.index.get_loc(index)
    
    df1=data.iloc[i:split+1,:]
    
    return df1


#def plotdata(df):
 #   df = df.set_index("Year")
  #  #plt.rc('xtick', labelsize=70) 
    #plt.rc('ytick', labelsize=70) 
   # x,y = np.meshgrid(df.columns.astype(float), df.index)
   # z = df.values
   # fig = plt.figure(figsize=(120,100))
   # ax = plt.axes(projection='3d')
   # ax.plot_surface(x, y, z, rstride=1, cstride=1,
 #               cmap='viridis', edgecolor='black')
 #   ax.view_init(20, 50)

  #  plt.savefig('3dplot.png')
  #  plt.show()


def integertodate(df):  #dataframe date to date object
    df['Year'] = df['Year'].apply(str)
    df['Year']=[datetime.strptime(el, '%Y%m%d' ) for el in (df['Year'])]    
    return df


    
    
    
def datetointeger(df):    #dataframe dates to integer
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

def AR1(column): # c =0 for no constant , else there is constant 
    y=column.values
    h=1
    p=1
    c=1
    h=h-1
    x=getxar(y,1,1)
    y=y[p+h:]
    n=y.shape[0]
    y=y.reshape(n,1)
    if (p==1 and c==0):
        nn=x.shape[0]
        x=x.reshape(nn,1)
    
    x=x[:x.shape[0]-h]
   
    beta= np.linalg.inv(x.T@ x)  @ x.T @ y
 
    yp=x @ beta
    e=y - yp
    yp=yp.reshape(-1)
    e=e.reshape(-1)
    
    print(yp.shape)
    print(e.shape)
     
    df = pd.DataFrame({'b':yp,'e':e})
    
    return  df


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
    

def makeslopecurve(df_old):  #make Slope and curvature like in diebold article
    df=df_old.copy()

    df['level']=df.iloc[:,[17]]
    df['month3']=df.iloc[:,[1]]
    df['month24']=df.iloc[:,[8]]
    
    df['Slope']=df['level'] - df['month3']
    df['Curvature']=df['month24'] + df['month24'] - df['level'] - df['month3']
    
    df=df.drop(columns=['level', 'month3', 'month24'])
    
    return df


def autocorr(dfcolumn,k):  #p lag, autocorrelation method
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

def rse(residual):
    return np.sqrt((residual ** 2))
                   
def se(residual):
    return (residual ** 2)

def mae(residual):
    return (np.abs(residual)).mean()

def autocorcoll(sd,df,lag,columnname,): #get autocorrelation for each maturity
    for index in sd.index:
        sd.loc[index,columnname]=  autocorr(df[index],lag)
    return sd    

def msecoll(sd,df,methodname,columnname): #get mse for each for each maturity
    for index in sd.index:
        sd.loc[index,columnname]=  methodname(df[index])
    return sd 


def getSummaryStats(df):   #desctiptive statistics, summary of yields
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

    
def getSummaryForecast(df,h): # get dataframe with forecast results
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
    sd=msecoll(sd,df,rmse,'RMSE')
    
    print(sd)

def ADF(y,maxlag,a=1,b=0): #augmented dickey fuller test, a is constant or not, b is trend or not
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
       
def DFUnit(y,p=1,a=1,b=0):  #dickey fuller,a for constant, b for trend , p lag, can be used for augmented dicker fuller test with an informatio criterion
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
    
def BIC(residual,k):  #normal distrebution assumed, k parameters
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
    
def newzerodataframe(df_old):    #returns dataframes with zeros and same structure
    df=df_old.copy()
    for col in df.columns:
        if(col not in ['Year']):
            df[col].values[:] = 0
      
    return df
    
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

def plotaverage(df,meanbeta):
    list = [3,6,9,12,15,18,21,24,30,36,48,60,72,84,96,108,120]
    T=np.array(list)
    t=T.astype(float)
    
    realmean=df.mean()
    
    tt=np.linspace(0,120,120)
    x=getmeanforbeta(tt)
    
    y=x @ meanbeta
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    plt.xlim(0,120)
    plt.plot(tt,y,linestyle='--')
    #plt.plot(vT,np.full((510,), np.mean(r)))
    plt.scatter(t,realmean)
    plt.legend(['Average Predicted yield', 'Average real yield'], loc='upper left')
    
    plt.savefig('averagebetaplot.png')
    plt.show()
    
    
def plot4whole(df,dfbeta):
    
    230 #1989-03-31
    df1=df.loc[230]
    dfb1=dfbeta.loc[230]
    plot4yield(df1,dfb1,'3nsplot1.png')
    
    234 #1989-07-31
    df2=df.loc[234]
    dfb2=dfbeta.loc[234]
    plot4yield(df2,dfb2,'3nsplot2.png')
    
    328 #1997-05-30
    df3=df.loc[328]
    dfb3=dfbeta.loc[328]
    plot4yield(df3,dfb3,'3nsplot3.png')
    
    343 #1998-08-31
    df4=df.loc[343]
    dfb4=dfbeta.loc[343]
    plot4yield(df4,dfb4,'3nsplot4.png')

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
    x=loadingvarnelson(t,labda)
    meanpred=x.astype(float)
    
    return meanpred


    

def Nelsolsiegel(df):
    x=getxforbeta()
    print(x)
    temp_y=df.values
    y=temp_y[:,1:]
    y=y.astype(float)
    y=y.T
    print('shape')
    print(x.shape)
    print(y.shape)
    beta2= np.linalg.inv(x.T@ x)  @ x.T @ y
    Beta=beta2.T
               
    #print(np.mean(Beta[:,1]))
    dfbeta = pd.DataFrame(columns=['Year', 'Beta1','Beta2','Beta3'])
    dfbeta['Year']=df['Year']
    dfbeta['Beta1']=Beta[:,0]
    dfbeta['Beta2']=Beta[:,1]
    dfbeta['Beta3']=Beta[:,2]
       
    df = df.set_index("Year")
   
    dfpred=newzerodataframe(df)
   
    dfres=newzerodataframe(df)
       
    i=0
    for index in df.index:     
        dfpred.loc[index]=x @ Beta[i,:].T
        dfres.loc[index] = df.loc[index].values - dfpred.loc[index].values 
        i=i+1
           
        
    betasum=getSummaryStatsbeta2(dfbeta) # for plot with average real yield and predicted yield based on nelson siegel model
    #meanbeta=betasum["mean"].values

    
    
    

    #makeautocorrplot(dfbeta['Beta3'],60)
    
   
    
    
    print(getSummaryStatsbeta(dfres))   #residuals
    
    return dfbeta

def gridNS(df):
    global labda
    
    grid=np.linspace(0.0001,0.10,100)
    result=np.ones((grid.shape[0],2))
    j=0
    
    for i in np.nditer(grid):
        labda=i
        result[j,0]=i
        result[j,1]=Nelsonsiegelgrid(df)
        j=j+1
        
    plt.plot(result[:,0],result[:,1])
    plt.show()
    
    minvalue=np.argmin(result[:,1])
    bestLabda=result[minvalue,0]
    print('bestlabda')
    print(bestLabda)
    labda=bestLabda
    return bestLabda
        

def Nelsonsiegelgrid(df):
    x=getxforbeta()

        
    temp_y=df.values
    y=temp_y[:,1:]
    y=y.astype(float)
    y=y.T
    beta2= np.linalg.inv(x.T@ x)  @ x.T @ y
    Beta=beta2.T
               
    #print(np.mean(Beta[:,1]))
    dfbeta = pd.DataFrame(columns=['Year', 'Beta1','Beta2','Beta3'])
    dfbeta['Year']=df['Year']
    dfbeta['Beta1']=Beta[:,0]
    dfbeta['Beta2']=Beta[:,1]
    dfbeta['Beta3']=Beta[:,2]
    
    
    
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
    #print(result)
    result=sumrmse[16]

    
    return result


#def macronelsonsiegel(dfbeta,df): # macroeconimcs factors added, want see if macro economic factors help in predicting our latent factor to get better forecasts in DNS
   # excelFilemacro = 'MACRO.xlsx'
    
    
   # dfMacro = pd.read_excel(excelFilemacro)  #the federal funds rate
 
    
    
   # CU=dfMacro['MCUMFN'].values  #manufacturing capacity utilization
   # INFL=dfMacro ['PCEPI'].values  #the annual price inflation
   # FFR=dfMacro ['FEDFUNDS'].values  #  #the federal funds rate
   # dfbeta['CU'], dfbeta['FFR'],dfbeta['INFL'], =CU ,FFR,INFL
   # print(dfbeta)
   # r=dfbeta.values
   # r=r[:,1:]
   # r=r.astype(float)
   # np.set_printoptions(linewidth=260)
   # np.set_printoptions(precision=3)
    
   # res=EstimateVARwithSE(r, 1,1,1)
   # print(res)
   # rs = pd.DataFrame({'Column1':res[:,0],'Column2':res[:,1],'Column3':res[:,2],'Column4':res[:,3],'Column5':res[:,4],'Column6':res[:,5],'Column7':res[:,6]})
   # print(rs.to_latex(float_format="%.3f"))
    
 

def plotbetaslopecurv(df2,dfbeta):
    df=makeslopecurve(df2)
    b1=dfbeta['Beta1'].values
    b2=dfbeta['Beta2'].values
    b3=dfbeta['Beta3'].values
    
    
    dfbeta['level']=df[120]
    dfbeta['Slope']=df['Slope']
    dfbeta['Curvature']= df['Curvature']
    print(dfbeta.corr().to_latex(float_format="%.3f"))
    
    

    
    b1=b1
    b2=b2*-1
    b3=b3*0.3
    
    T=df['Year'].values
    
    level=df[120].values
    curvature=df['Curvature'].values
    slope=df['Slope'].values
    
    fig, ax = plt.subplots(figsize=(10, 8))
    #plt.xlim(0,120)
    
    plt.plot(T,b1,linestyle='--')
    plt.plot(T,level,linestyle='-.')
    plt.legend(['B1', 'Level'], loc='upper left')
    
    plt.savefig('b1level.png')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    #plt.xlim(T[0],T[T.shape[0]])
    
    plt.plot(T,b2,linestyle='--')
    plt.plot(T,slope,linestyle='-.')
    plt.legend(['B2', 'Slope'], loc='upper left')
    
    plt.savefig('b2slope.png')
    plt.show()
    
    
    fig, ax = plt.subplots(figsize=(10, 8))
    #plt.xlim(0,120)
    
    plt.plot(T,b3,linestyle='--')
    plt.plot(T,curvature,linestyle='-.')
    plt.legend(['B3', 'Curvature'], loc='upper left')
    
    plt.savefig('b3curvature.png')
    plt.show()
    
    
    
    
    
def main():
    # Magic numbers
    #plottheoreticalnelsonSven()
    #plottheoreticalnelsonBjork()
    #plottheoreticalnelsonFive()

    # Initialisation
    excelFile = 'Data.xlsx'
    #excelFile = 'Data.xlsx'
    excelFile2 = 'Forward.xlsx'
    #excelFile2 = 'Forwarddata.xlsx'
    
    
    df = pd.read_excel(excelFile)
    dfforwardrates = pd.read_excel(excelFile2)
    
    
    
    

    
    
    df=integertodate(df)
    df = df[(df['Year'].dt.year <= 2000) & (df['Year'].dt.year >= 1985)] 
    df2 = df[(df['Year'].dt.year < 1993)] 

    #df=datetointeger(df)
    
    
    #plotdata(df)
    #df=integertodate(df)
    
    
    
    sd=getSummaryStats(df)
    print(sd.to_latex(float_format="%.3f"))
    
    dfbeta=Nelsolsiegel(df)
    
    
    plottheoreticalnelson()
    plottheoreticalnelsonSven()
    plottheoreticalnelsonFive()
    plottheoreticalnelsonBjork()
    
    b1=AR1(dfbeta['Beta1'])
    print(b1)
    makeautocorrplot(b1['b'],60,'autocorreps1.png')
    
    
    plotbetaslopecurv(df,dfbeta)  #the betas and level, curvature and slop plot #plot for each year
    # print(dfbeta.corr().to_latex(float_format="%.3f")) # correlation between betas
    #gridNS(df2)
    #print(labda)
 
    
    lmtest=Engletestforbeta(dfbeta,10)
    
    print(lmtest)
    #lm = pd.DataFrame({'Column1':lmtest[:,0],'Column2':lmtest[:,1],'Column3':lmtest[:,2]})
    #print(lm.to_latex(float_format="%.3f"))
    
    

    
    
    ###################################################################################################################################
    #out of sample forecasts, the commands are in comments, remove them from comments to use
    
    m1,sm6,sm12,e1rw,e12rw=forecastrandomwalk(df)
    
    sm1,sm6,sm12,e1ns,e12ns=forecastnelsonsiegelar(df,dfbeta,labda)  #0.08284545454545456
    
    forecastlevelVAR(df)
    forecastchangeVAR(df)
    forecastECM1(df)
    forecastECM2(df)
    
    
    
    
    sm1,sm6,sm12,e1sven,e12sven=forecastnelsonsiegelSven(df)
    sm1,sm6,sm12,e1bjork,e12bjork=forecastnelsonsiegelBjork(df)
    sm1,sm6,sm12,e1five,e12five=forecastnelsonsiegelFive(df)
    sm1,sm6,sm12,e1two,e12two=forecastnelsonsiegelTwo(df)

    
    dm1p1=Compare2modelforecasts(e1ns,e1ns,e1sven,e1bjork)
    dm12p1=Compare2modelforecasts(e12ns,e12ns,e12sven,e12bjork)
    
    dm1p2=Compare2modelforecasts(e1ns,e1ns,e1five,e1two)
    dm12p2=Compare2modelforecasts(e12ns,e12ns,e12five,e12two)
    
    
    dm1=pd.concat([dm1p1, dm1p2.reindex(dm1p1.index)], axis=1)
    dm12=pd.concat([dm12p1, dm12p2.reindex(dm12p1.index)], axis=1)
    #print('1-step')
    #print(dm1.to_latex(float_format="%.3f"))
    
    #print('12-step')
    #print(dm12.to_latex(float_format="%.3f"))
    
    plottheoreticalnelson()
    
  
    ###################################################################################################################    
    
    
#
    
    








if __name__ == "__main__":
    main()

