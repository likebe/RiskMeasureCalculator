#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 22:30:36 2019

@author: Qiqi
"""

#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")
# Import Modules
import numpy as np
import pandas as pd 
from scipy.stats import norm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt 
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import scipy.special


from scipy.optimize import minimize

import pandas_datareader.data as web
from datetime import datetime,date
import wallstreet as ws



def get_data_from_Yahoo (stock_tickers = [],start_date = '1980-01-01', end_date = date.today().strftime("%Y-%m-%d")):
    data = web.DataReader(stock_tickers, 'yahoo', start_date, end_date)
    return data["Adj Close"].sort_index(ascending = False)


def get_implied_vol(options_tickers = []):
    now = datetime.now() # current date and time
    year = int(now.strftime("%Y"))
    month = int(now.strftime("%m"))
    day = int( now.strftime("%d") )

    implied_vol = []
    for i in range(len(options_tickers)):
        stock = ws.Stock(options_tickers[i])
        put = ws.Put(options_tickers[i], d=day, m=month, y=(year+1), strike=stock.price)
        #print(put)
        implied_vol.append( put.implied_volatility() )
    implied_vol = np.asarray(implied_vol)
    return implied_vol


def port_build(tickers, invest_date, v0):
    df = get_data_from_Yahoo(tickers)
    stocks = df.dropna()  # to keep both tickers of same length 
    num_stocks = len(tickers)
    num_shares = round((v0/num_stocks)/stocks[stocks.index == invest_date]) #equally weight 
    if num_stocks>1:
        port_val =np.sum( np.multiply(stocks, num_shares), axis =1 )   # if multiple tickers are selcted, append a portfolio value column 
        stocks['port_val'] = port_val
    return stocks,num_shares 


def winEstGBM(price,windowsize): 
   # t0 = time.time()
    dt = 1/252 
    log_ret = -np.diff(np.log(price.iloc[:,-1])) 
    log_ret_sqrd = np.square(log_ret)
    windowLen =  windowsize*252
    period = len(log_ret)- windowLen
    x_bar, x2_bar = [], []   
    for i in range(period):
        log_rtn = log_ret[i:(i+windowLen)]
        log_rtn_sqrd = log_ret_sqrd[i:(i+windowLen)]
        x_bar.append( np.mean(log_rtn)) 
        x2_bar.append(np.mean(log_rtn_sqrd) ) 
    var_bar_lst  = np.array(x2_bar) -  np.square(x_bar)  
    sigma_arr = np.sqrt(var_bar_lst)/np.sqrt(dt)  
    mu_arr = np.array(x_bar)/dt + np.square(sigma_arr)/2     
    params = pd.DataFrame({'mus': mu_arr, 'vols': sigma_arr}, index = price.index[:len(mu_arr)])
  #  print("winEstGBM running time:", time.time()- t0, 's')
    return log_ret, params


def mse_cal(lam, windowsize):
    N = windowsize * 252
    x = 50000
    arr1 = np.power(lam, range(N), dtype=np.longdouble)
    x1 = np.square( (1/N)-(1-lam)*arr1, dtype=np.longdouble)
    arr2 = np.power(lam, range(N,x), dtype=np.longdouble)
    x2 = np.square( (1-lam)*arr2, dtype=np.longdouble )
    return np.sum(x1) + np.sum(x2)



def expEstGBM(price,windowsize): 
   # t0 = time.time()
    dt= 1/252  
    log_ret = -np.diff(np.log(price.iloc[:,-1])) 
    log_ret_sqrd = np.square(log_ret)
    ntrials = len(log_ret) 
    lam =  minimize(mse_cal, 0.95, args=(windowsize,)).x[0] 
     
    windowLen = np.ceil( np.log(0.01)/np.log(lam)).astype(int)
   
    if windowLen> 5000: windowLen = 5000 
    lam_lst = np.power(lam, range(windowLen))
    wgt = lam_lst /sum(lam_lst) 
    period = ntrials-windowLen
    
    x_bar, x2_bar = [], [] 
    
    for i in range(period):  
        log_rtn = log_ret[i:(i+windowLen)]
        sqrd_log_rtn = log_ret_sqrd[i:(i+windowLen)]
        wgt_log_rtn = np.multiply(log_rtn, wgt) 
        wgt_sqrd_log_rtn = np.multiply(sqrd_log_rtn, wgt)
        x_bar.append(np.sum(wgt_log_rtn)) 
        x2_bar.append(np.sum(wgt_sqrd_log_rtn))      
    
    var_bar_lst = np.array(x2_bar) - np.power(x_bar,2)
    sigma_arr = np.sqrt(var_bar_lst)/np.sqrt(dt)   
    mu_arr = np.array(x_bar)/dt + np.square(sigma_arr)/2   
    params = pd.DataFrame({'mus': mu_arr, 'vols': sigma_arr}, index = price.index[:period])
  #  print("expEstGBM running time:", time.time()- t0, 's')
    return log_ret, params


# #### Plot parameters

def plot_parameters(tickers, invest_date, windowsize,v0, equal_wgt=True, parameter = "mu"): #price,windowsize, title1, title2, equal_wgt=True):
    
    price,num_shares =  port_build(tickers,invest_date,v0) # suppose investment date 
    #title1, title2,
    if equal_wgt: 
        title2  = "Equally weighted"
        lr, params = winEstGBM(price, windowsize)     
    else: 
        title2 = "Exponentially weighted"
        lr, params = expEstGBM(price, windowsize)
    
    num = len(tickers)
    if num > 1: title1 = 'Portfolio'    
    else: title1 = ' '.join(price.columns.values) 

    # plot mean and volatility 
 
   # yrs= params.shape[0]
    yrs =min(20*252,params.shape[0]) # suppose investment last for 20 years
 
    time_line = pd.to_datetime(params.index)[:yrs]
    labs = "{} yr window".format(windowsize)
     
    fig,ax= plt.subplots(figsize=(10,8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    #fig1,(ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,5))
    
    # plot mus 
    if parameter == "mu": 

        ax.plot(time_line, params.mus[:yrs], color = "blue",label = labs)
        ax.set_title(title1 +  ' Mus'+'\n' +title2 , size = 10  )
        ax.legend(loc='best')
        fig_name = '{}-{}year-mu.png'.format(title1,windowsize)
                
    else: 
    # plot volatility      
        ax.plot(time_line, params.vols[:yrs],color = "purple", label = labs )
        ax.set_title( title1 +  ' Vols'+'\n' + title2, size = 10 )
        ax.legend(loc='best')
        fig_name = '{}-{}year-vol.png'.format(title1,windowsize)
        #fig2.savefig(fig_vol,bbox_inches = 'tight')
    
    fig.savefig(fig_name,bbox_inches = 'tight')
    
    return fig_name



#plot_parameters(tickers= ['F','XRX'],invest_date = '1997-09-05', windowsize= 5, v0=10000, equal_wgt= True,  parameter = "vols")
# #### GBM VAR & ES 

def gbm(params, VaRPct , ESPct, horizon,v0): 
    dt = horizon/252  
    VaR = v0- v0*np.exp((params.mus - np.square(params.vols)/2)*dt + params.vols*np.sqrt(dt)*norm.ppf(1-VaRPct))
    ES = v0*(1-np.exp(params.mus*dt)/(1-ESPct)*norm.cdf(norm.ppf(1-ESPct)-params.vols*np.sqrt(dt)))
    gbmve = pd.DataFrame({'VAR':VaR, 'ES':ES} , index = params.index)   
    return gbmve


# #### Plot VaR and ES 

def plot_var_es(result, var):   
    
    time_line = pd.to_datetime(result.index)
    yrs =  min(20*252, result.shape[0]) #result.shape[0] #
    fig,ax= plt.subplots(figsize=(10,8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    if var: 
        ax.plot(time_line[:yrs], result.VAR[:yrs], color = "red")
    else: 
        ax.plot(time_line[:yrs], result.ES[:yrs], color ="blue")
    
    return fig, ax    
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    #ax.plot(x_list, y)
   # ax.xaxis.set_ticks(x_list)

    #fig,(ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,5) )
    
    
    #ax1.plot(time_line[:yrs], result.VAR[:yrs], color = "red")#,label = "{}% VaR".format())
    #ax2.plot(time_line[:yrs], result.ES[:yrs], color ="blue") #,label = "ES")
    
   



def plot_gbm(tickers,invest_date, windowsize,VaRPct, ESPct, horizon,v0, equal_wgt=True,var = True):
    #t0 = time.time()
    price,num_shares =  port_build(tickers,invest_date,v0) # suppose investment date 
    
    if equal_wgt: 
        title2  = "Equally weighted"
        lr, params = winEstGBM(price, windowsize)     
    else: 
        title2 = "Exponentially weighted"
        lr, params = expEstGBM(price, windowsize)
    
    num = price.shape[1]   
    if num > 1: title1 = 'Portfolio'    
    else: title1 = ' '.join(price.columns.values) 

    # plot mean and volatility 
 
    #fig1_name = plot_parameters(params,windowsize, title1, title2,parameter )
    
    # plot var and es on the same plot 
    vares = gbm(params, VaRPct, ESPct, horizon,v0)   
    fig, ax = plot_var_es(vares,var)
    
    if num > 1: title3 = title1 + ' with lognormal assumption, '
    else: title3 = title1 + ' Parametric'
     
    if var: 
        ax.legend(['{}% VaR'.format(VaRPct*100)])
        figtitle = 'VAR'+'\n' + title3  + '\n'+ title2
        ax.set_title(figtitle ,  size = 10   )
    else: 
        ax.legend(['{}% ES'.format(ESPct*100)])
        figtitle = 'ES'+ '\n' + title3  + '\n'+ title2
        ax.set_title(figtitle ,  size = 10   )
    
    fig_name =  '{} GBM VARES-{}.png'.format(title1,title2)
    fig.savefig(fig_name,bbox_inches = 'tight')
    return fig_name
   # ax2.get_figure().savefig('{} GBM ES.png'.format(title1))
    #print("running time: {}s".format(time.time() - t0))


#plot_gbm(tickers= ['F','XRX'],invest_date = '1997-09-05', windowsize= 5, VaRPct=0.99, ESPct=0.975, horizon=5, v0=10000, equal_wgt= True, var  = True )
#  Method II Historical VaR&ES  - relative changes 

def historical_rel(price, VaRPct,ESPct, horizon, windowsize ,v0): 
   # t0 = time.time()
    npts = 252* windowsize
    npaths= npts - horizon
    ntrials = len(price) - npts 
    price_log = np.log(price.iloc[:,-1])
    xday_rtn = np.array(price_log[:(len(price_log)-horizon)]) - np.array(price_log[horizon:])
    price_res = v0 * np.exp(xday_rtn)
    price_scenarios = np.zeros(shape=(npaths,ntrials))
    
    for i in range(ntrials):
        price_scenarios[:,i] = price_res[i:i+npaths]
    scenarios_sorted = np.sort(price_scenarios, axis=0)
    HVaR = v0 - scenarios_sorted[int( np.ceil((1-VaRPct)*npaths))-1]  
    HES  = v0 - np.mean(scenarios_sorted[0:int( np.ceil((1-ESPct)*npaths))], axis = 0)
    hisve = pd.DataFrame({'VAR': HVaR, 'ES': HES}, index = price.index[:ntrials]) 
  #  print("running time:", time.time() - t0)
    return hisve


def plot_historical(tickers, invest_date, VaRPct,ESPct, horizon, windowsize , v0, var =True ): 
    #t0 = time.time()
    price,num_shares =  port_build(tickers,invest_date,v0) # suppose investment date 
   # vares_a = historical_abs(price, VaRPct,ESPct, horizon, windowsize, v0)
    vares_r = historical_rel(price, VaRPct,ESPct, horizon, windowsize , v0)
    
    # find number of stocks contained in a portfolio 
    num = price.shape[1]  
    if num > 1:  title1 = 'Portfolio'
    else: title1 = ' '.join(price.columns.values)   
    
    
    vares_r = historical_rel(price, VaRPct,ESPct, horizon, windowsize , v0)
    yrs =  min(20*252, vares_r.shape[0]) #result.shape[0] #
    time_line = pd.to_datetime(vares_r.index[:yrs])
    
    fig,ax= plt.subplots(figsize=(10,8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    if var: 
        ax.plot(time_line, vares_r.VAR[:yrs], linewidth =0.6,color = "blue",label = "{}% VaR".format(VaRPct*100))
        ax.legend(fontsize = 10)
        title2 =  " Historical VaR"
       
        
    else:  
        ax.plot(time_line, vares_r.ES[:yrs], linewidth =0.6,color = "purple",label = "{}% ES".format(ESPct*100))
        ax.legend(fontsize = 10)
        title2 =  " Historical ES"
        
    
    ax.set_title(title1 +title2)
    fig_name = '{}-{}.png'.format(title1, title2)
    fig.savefig(fig_name,bbox_inches = 'tight')
    return fig_name
        
    #fig,(ax1,ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20,5))
   # yrs = min( vares_r.shape[0],20*252)
    #time_line = pd.to_datetime(vares_r.index[:yrs]) 
    #ax1.format_xdata = mdates.DateFormatter('%Y-%m')
    #ax2.format_xdata = mdates.DateFormatter('%Y-%m')

    
#plot_historical(['F','XRX'],'1997-09-05', 0.99, 0.975, 5,5,v0 = 10000, var = True)  
    
   # ax2.legend(fontsize = 10)
    
    
    #fig.tight_layout()

    #print("running time: {}s".format(time.time() - t0))


def gbmsampset(s0, horizon, mu, sigma, npath):
    dt = horizon/252
    dts = np.ones(npath)*dt  
    bm = np.sqrt(dt)*np.random.normal(0.0, 1.0, npath) 
    sample_path = s0*np.exp(sigma * bm + (mu- sigma**2/2) * dts)
    return sample_path



# if choose one stock or gbm portfolio 
def monte_carlo(price, horizon,npath, VaRPct, ESPct, windowsize ,v0):
   # t0 =time.time()
    lr, params = winEstGBM(price, windowsize)  
    ntrials = params.shape[0]
    p1 = np.zeros(shape=(npath,ntrials)) 
    for i in range(ntrials):     
        p1[:,i] = gbmsampset(v0,horizon, params.mus[i], params.vols[i], npath) 
    
    p2 = np.sort(p1, axis=0)   
    mcvar= v0 - p2[int( np.ceil((1-VaRPct)*npath))-1]   
    mces  = v0 - np.mean(p2[0:int( np.ceil((1-ESPct)*npath))], axis = 0) 
    mcve = pd.DataFrame({'VAR': mcvar,'ES': mces}, index = price.index[:ntrials]) 
    #print("monte_carlo running:", time.time()-t0, 's')
    return mcve


# #### Find correlation between multiple stocks 
# * For example, 3 stocks, [Ford, Xerox, Apple], the columns of rho_df/cov_df are in the order of correaltion between [F&X, F&A, X&A] 



def winEstGBM2(port_df,windowsize):    
   
    windowLen=windowsize*252
    num_shares = port_df.shape[1]-1
    log_ret_ll,sigma_ll,mu_ll = [],[],[]   
     
    for i in  range(num_shares):
        log_rtns,  parmas =  winEstGBM(port_df.iloc[:,[i]], windowsize)
        log_ret_ll.append(log_rtns)
        sigma_ll.append(parmas.vols) 
        mu_ll.append(parmas.mus)  
    
    mu_df = pd.concat(mu_ll,axis = 1)
    sigma_df = pd.concat(sigma_ll,axis = 1)
    n_row,n_col  = mu_df.shape 
    n_pair = scipy.special.comb(n_col,2).astype(int)  # find pairs of correlation, if 3 stocks, then n_pair = 3     

    rho_nda,cov_nda = np.zeros((n_row,n_pair)) , np.zeros((n_row,n_pair))     
    for t in range(n_row):    
        X = np.array([x[t:(t+windowLen)] for x in log_ret_ll]) 
        cov_mat = np.cov(X)
        dim = cov_mat.shape[0]         
        cov_nda[t,:]= [cov_mat[i,j] for i in range(dim-1) for j in range(i+1,dim)]     
        rho_nda[t,:] = [252*cov_mat[i,j]/(sigma_df.iloc[t,i]*sigma_df.iloc[t,j]) for i in range(dim-1) for j in range(i+1,dim)]
    return mu_df, sigma_df,rho_nda


# #### corrbmsampset: return multivariate normal random numbers 


def corrbmsampset(s0, mus, sigmas, rhos, horizon, npath):  
    ncol = len(s0)
    dt = horizon/252  
    means  = np.zeros(ncol)
    cov_mat = np.identity(ncol)
    cov_mat[np.triu_indices(ncol, k = 1)] = cov_mat[np.tril_indices(ncol, k = -1)] = rhos 
    bms = np.sqrt(dt)*np.random.multivariate_normal(means,cov_mat ,npath)
    sample_paths = s0*np.exp(bms*sigmas.values +(mus.values - np.square(sigmas.values)/2)*dt)
    return sample_paths


# #### monte_carlo_2: calculate the portfolio MCVAR & MCES  for multiple stocks assuming they follows GBM 


def monte_carlo_2(port_df, n_shares, horizon, windowsize, VaRPct,ESPct, npath,  v0): 
   # t0 = time.time()
    stocks_mus, stocks_sigmas,stocks_rho = winEstGBM2(port_df , windowsize)
    ntrials, num_stocks = stocks_mus.shape 
    mc_es_2, mc_var_2 = [], [] 
    
    for i in range(ntrials):  
        s0 = port_df.iloc[i,:num_stocks].values
        startprice = np.sum((n_shares*s0).values) 
        mu = stocks_mus.iloc[i,:]
        sigma = stocks_sigmas.iloc[i,:]
        rho = stocks_rho[i]
        samppath = corrbmsampset(s0 , mu, sigma, rho,horizon,npath )
        y = v0*np.sum(n_shares.values*samppath, axis = 1)/startprice
        mc_var_2.append(  v0 - np.quantile(y,1-VaRPct))
        thres = np.quantile(y,1-ESPct)
        mc_es_2.append( v0- np.mean( y[np.where(y<thres)]))       
    mcve_2 = pd.DataFrame({'VAR': mc_var_2,'ES': mc_es_2}, index = stocks_mus.index[:ntrials])  
   # print("monte_carlo_2 running time",time.time()- t0, 's')              
    return mcve_2       


# #### Stock + options 

# ####  liquid part of portfolio to buy  one year ATM puts for each of the stocks in portfolio 

def put_price_calc(stock, strike,  sigma, rf, maturity):
    sigrt = 1/(sigma*np.sqrt(maturity))
    sig2 = sigma*sigma/2
    lsk = np.log(stock/strike)
    ert = np.exp(-rf*maturity)
    d1 = sigrt*(lsk+(rf+sig2)*maturity)
    d2 = sigrt*(lsk+(rf-sig2)*maturity)
    pr = norm.cdf(-d2)*strike*ert- norm.cdf(-d1)*stock
    return pr



def option_MC_VaR(port_df, n_shares, horizon, windowsize,  VaRPct,ESPct, npath, imp_vol, pct,  v0): 
  #  t0 = time.time()
    dt = horizon/252
    rf = 0.02 
    mat = 1 
    num_stocks = n_shares.shape[1]  
    put_cal_v = np.vectorize(put_price_calc)
    stockshares = n_shares.values
    
    if num_stocks > 1: 
        mcvar = monte_carlo_2(port_df, n_shares, horizon, windowsize, VaRPct,ESPct, npath,  v0)
        VaR_1 = mcvar.VAR[0] 
        s0 =strike = port_df.iloc[0,:num_stocks].values  
        puts0 = put_cal_v(s0,strike,imp_vol,rf, mat)

        nputs = np.asarray(v0*pct)/np.asarray(puts0)
        port_val_0  = np.sum(n_shares*s0+nputs*puts0,axis =1).values   
        stocks_mus, stocks_sigmas,stocks_rhos = winEstGBM2(port_df , windowsize) 
        mu = stocks_mus.iloc[0,:]
        sigma = stocks_sigmas.iloc[0,:]
        rho = stocks_rhos[0]
        samppath = corrbmsampset(s0 , mu, sigma, rho,horizon,npath )     
        vtstocks = np.sum(samppath*stockshares,axis = 1)
         
        putst = put_cal_v(samppath,strike,imp_vol,rf,mat-dt)     
        vtputs = np.sum(putst*nputs,axis = 1)
         
    else: 
        mcvar = monte_carlo(port_df , horizon, npath, VaRPct, ESPct, windowsize,v0 )
        VaR_1 = mcvar.VAR[0] 
        s0 = strike = port_df.iloc[0,0] 
        stockshares = v0*(1-pct)/s0
        puts0 = put_price_calc(s0,strike,imp_vol,rf,mat)
        nputs = np.asarray(v0*pct)/np.asarray(puts0)  
        port_val_0  = stockshares*s0 + puts0*nputs
        lr, params = winEstGBM( port_df, windowsize) 
        samppath = gbmsampset(s0,horizon, params.mus[0], params.vols[0], npath) 
        vtstocks = samppath*stockshares 
        putst = put_cal_v(samppath,strike,imp_vol,rf,mat-dt)    
        vtputs = putst*nputs  
        
    port_val_t = vtstocks +  vtputs
    loss = port_val_0  -  port_val_t
    VaR_2 = np.quantile(loss,VaRPct )
    reduction = 100*(1-VaR_2/VaR_1)
    printlist =  ['Stock price: ' + ', '.join(str(x) for x in s0.round(1).flatten())  ,
                  'Stock shares: ' + ', '.join(str(x) for x in stockshares.round(0).flatten()) ,
                  'Put price on one share: ' + ', '.join(str(x) for x in puts0.round(2).flatten()) ,
                  'Put shares: ' + ', '.join(str(x) for x in nputs.round(2).flatten()),
                  "VaR without options: %.0f" %  VaR_1, 
                  "VaR with options: %.0f" %  VaR_2, 
                  "VaR reduction (percentage): %.2f" % reduction] 
   # print("option_MC_VaR running time:", time.time() - t0)
    return printlist, mcvar


# #### Plot MC VaR & ES  

def plot_mc(tickers, invest_date, horizon, windowsize, VaRPct,ESPct, npath, v0,  pct=None,  gbmport= True,var = True): 
   # t0 = time.time()
    price, n_shares = port_build(tickers, invest_date, v0)
    num = price.shape[1] 
    
    if num > 1:  title1 = 'Portfolio'
    else: title1 = ' '.join(price.columns.values) 
    
    #if npath > 10000: print("Warning: It may take a while to run")  
        
    # case 1: one stock / gbm portfolio(multiple stocks ) 
    if (num==1 or gbmport) and pct == None:
        case = "case1"
        lr, params = winEstGBM(price, windowsize)  
        #plot_parameters(params,windowsize, title1, title2 = '')
        vares =  monte_carlo(price, horizon,npath, VaRPct, ESPct, windowsize ,v0)
        fig, ax  = plot_var_es(vares,var)
        if var: ax.set_title(title1  + ' MC VAR' ,  size = 10   )
        else: ax.set_title(title1  + ' MC ES' ,  size = 10   )
        to_print = ""
        
    # case 2: gbm underlying - no liquidation 
    #if (gbmport != True and num > 1) and pct == None: 
       # case = "case2"
       # vares = monte_carlo_2(price, n_shares, horizon, windowsize, VaRPct,ESPct, npath, v0)   
       # fig,ax1,ax2 = plot_var_es(vares)
       # ax1.set_title(title1 + ' MC VaR(GBM und)' ,  size = 10   )
       # ax2.set_title(title1 + ' MC ES(GBM und)' ,  size = 10   )
 
  
    # case 3 : gbm underlying - liquidate option 
    if pct != None:
        case = "case2"
        imp_vol = get_implied_vol(tickers)
        to_print, vares_no_opt = option_MC_VaR(price, n_shares, horizon, windowsize,  VaRPct,ESPct, npath, imp_vol, pct,v0)
        fig,ax = plot_var_es(vares_no_opt,var)
        
        if var: 
            ax.set_title(title1+ ' (GBM und) VaR without liquidation', size = 10)
        else: 
            ax.set_title(title1 +' (GBM und) ES without liquidation',size = 10)
        print(*to_print,sep = '\n')      
    
    if var: ax.legend(['{}% VaR'.format(VaRPct*100)])
    else: ax.legend(['{}% ES'.format(ESPct*100)])
    
    fig_name = 'MC VaR & ES-{}.png'.format(case)
    fig.savefig(fig_name,bbox_inches = 'tight')
    #fig2.get_figure().savefig('MC ES.png') 
   # print("running time: {}s".format(time.time() - t0))
    return fig_name, to_print

#plot_mc(['F','XRX'] , '1997-09-05', 5, 5, 0.99, 0.975,10000, gbmport= True,pct = 0.01 v0=1000, var = True )


# measure the subsequent 5 days change in each 1 year window 
def back_test(tickers, invest_date, VaRPct, ESPct, horizon, windowsize,    npath,  method, v0, equal_wgt=True):  #long=True, 
    #price, n_shares = port_build(tickers, invest_date, v0)
   #horizondays=5,
    dataDaysdt=1  
    price, n_shares = port_build(tickers, invest_date, v0)
    num =len(tickers)
    # P: parametric method  H:historical  else : monte carlp 
    if method  == "P": 
        if equal_wgt: lr, params = winEstGBM(price, windowsize)  
        else: lr, params = expEstGBM(price, windowsize) 
        
        gbm_vares = gbm(params, VaRPct, ESPct, horizon,v0)   
        VaR = gbm_vares.VAR
        
    elif method == "H":
         his_vares = historical_rel(price, VaRPct,ESPct, horizon, windowsize ,v0)
         VaR = his_vares.VAR
    
    else:   
        mc_vares = monte_carlo(price, horizon,npath, VaRPct, ESPct, windowsize ,v0)
        VaR = mc_vares.VAR
               
    price_past = price.iloc[horizon:,-1]
    price_current = price.iloc[:-horizon,-1]  
    dataDays = dataDaysdt*252  
    onesharechange = price_current.values/price_past.values
    real_loss = v0 -  onesharechange*v0   
    npts = len(VaR) - horizon   
    VaR_trunc = [VaR[t+horizon] for t in range(npts) ]  
    real_loss_trunc = real_loss[:npts]
    lst = [np.sum(real_loss_trunc[t:(t+dataDays)] > VaR_trunc[t:(t+dataDays)]) for t in range(npts-dataDays)] 
    exceptions = pd.DataFrame({'count':lst}, index = VaR.index[:len(lst)])
   
    num = price.shape[1]  # find number of stocks contained in a portfolio 
    if num > 1: title = 'Portfolio'     #of '  + ' & '.join(price.columns.values[:-1])
    else: title = ' '.join(price.columns.values) 

    # plot exceptions     
    timeline = pd.to_datetime(exceptions.index) 
    fig,(ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,4) )
    ax1.format_xdata = mdates.DateFormatter('%Y-%m')
    ax1.set_title ("{} Exceptions Per Year \n (Horizon={} days, Window={} yrs)".format(title, horizon,windowsize ),size = 10 )
    ax1.plot(timeline, exceptions)
    
    # plot VaR vs Realized Losses 
    ax2.format_xdata = mdates.DateFormatter('%Y-%m')
    ax2.set_title ("{} VaR vs Realized Loss \n (Horizon={} days, Window={} yrs)".format(title, horizon,windowsize ),size = 10 )
    ax2.plot(timeline, real_loss_trunc[:len(lst)])
    
    
    
    ax2.plot(timeline, VaR_trunc[:len(lst)] )
    ax2.legend(['Losses', 'VaR'])
    fig.savefig("{} Backtest.png".format(title),bbox_inches = 'tight')
    figname = "{} Backtest.png".format(title)
   # return real_loss_trunc, VaR_trunc, exceptions
    return figname


#back_test(['F','XRX'], '1997-09-05', 0.99, 0.975, 1, 2, 10,  "P", v0= 10000, equal_wgt=True)



