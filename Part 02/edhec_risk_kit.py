
import pandas as pd
import scipy.stats
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

def get_ffmr_returns():
      
  """
  Load Fama- French Dataset for the returns of the top and the bottom declines by marketcap  
  """
  me_m = pd.read_csv(r'D:\IIT Roorkee DS\CSVs\Portfolio course data\Portfolios_Formed_on_ME_monthly_EW.csv', header= 0, index_col = 0, na_values = -99.99,parse_dates=True)
  rets = me_m[['Lo 10','Hi 10']]
  rets.columns=['Smallcap','Largecap']
  rets = rets/100.
  rets.index = pd.to_datetime(rets.index, format ='%Y%m').to_period('M')
        
  return rets

def get_hfi_returns():
  """
  Load and format the EDHEC hedge funds return  
  """
  me_m = pd.read_csv(r'D:\IIT Roorkee DS\CSVs\Portfolio course data\edhec-hedgefundindices.csv', header= 0, index_col = 0, parse_dates=True)
  hfi = me_m/100.
  hfi.index = pd.to_datetime(hfi.index, format ='%Y%m').to_period('M')
  
  return hfi

def Drawdown(return_series :pd.Series):
  """
  It takes a time series of asset retuens and retuens, wealth index,
  previous peak and percent drawdown 
  """
  wealth_idx = 1000*(1+return_series).cumprod()
  previous_peak = wealth_idx.cummax()
  drawdowns = (wealth_idx - previous_peak)/previous_peak

  return pd.DataFrame({
      'Wealth_index': wealth_idx,
      'Peaks':previous_peak,
      '% Drawdown': drawdowns
  })

def semideviation(r):
    
    """Returns semideviations aka negative semideviation of r
    r must be Series or dataframe
    """
    is_negative = r<0
    return r[is_negative].std(ddof=0)

def skewness(r):
    
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied series or dataframe 
    returns a float or series
    """
    demeaned_r = r - r.mean()
  #use population standard deviation so set degree of freedom = 0
    sigma_r = r.std(ddof = 0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def var_historic(r, level = 5):
    """
    VaR Historic
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be series or Dataframe")

def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    scipy.stats.kurtosis() gives excess kurtosis(kurtosis - 3) not the exact kurtosis
    normal distribution has kurtosis value equal to 3
    Computes the kurtosis of the supplied series or dataframe 
    returns a float or series
    """
    demeaned_r = r - r.mean()
  #use population standard deviation so set degree of freedom = 0
    sigma_r = r.std(ddof = 0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal(r, level = 0.01):
    """
  Applies Jarque_bera test on the series to check it is normal or not
  Test is applied at the 1% level by default
  Returns True if hypothesis of normality is accepted, otherwise false.
  Null hypothesis of JB test is that given series is normally distributed
  """
    statistic,p_val = scipy.stats.jarque_bera(r)
    return p_val > level

def var_gaussian(r, level=5, modified=False):
    """
    Returns Parametric Gaussian VaR of a Series or DataFrame"""
    #computing Z score assuming Gaussian distribution
    z = norm.ppf(level/100)
    s = skewness(r)
    k = kurtosis(r)
    if modified == True:
        z = (z + (s/6)*(z**2-1)+((k-3)/24)*(z**3-(3*z))-(s**2/36)*(2*z**3-5*z))
    return -(r.mean()+z*r.std(ddof=0))
    

def CVaR_historic(r,level=5):
    """
    Compute the historic CVaR on the Series or dataframe"""
    if isinstance(r,pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r,pd.DataFrame):
        return r.aggregate(CVaR_historic,level=level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")

def get_ind_returns():
    """
    Load and format the Ken French 30 industry portfolios value weightes monthly returns"""
    ind = pd.read_csv(r"D:\IIT Roorkee DS\CSVs\Portfolio course data\ind30_m_vw_rets.csv",index_col = 0, parse_dates=True, header=0)/100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    """
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r,riskfree_rate,periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns """
    #convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret,periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol

def annualize_rets(r, periods_per_year):
    """
    annualizes a set of returns
    we should infer the periods per year"""
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def portfolio_return(weights, returns):
    """
    Weights --> returns"""
    return weights.T@returns 

def portfolio_vol(weights, covmat):
    """
    Weights --> vol"""
    return (weights.T @ covmat @ weights)**0.5

def plot_ef2(n_points, er, cov):
    """
    Plots the 2-asset efficient frontier"""
    if er.shape[0]!=2 :
        raise ValueError("plot_ef2 can only plot 2_asset frontiers")
    weights = [np.array([w,1-w]) for w in np.linspace(0,1,n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns":rets,"Volatility":vols})
    
    return ef.plot.line(x="Volatility", y="Returns", style=".-")
                        
        
def minimize_vol(target_return, er, cov):
    """
    target_return --> W"""
    n = er.shape[0]
    init_guess = np.repeat(1/n,n)
    bounds = ((0.0,1.0),)*n
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_1 ={
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    weights_result = minimize(portfolio_vol, init_guess,args = (cov,), method ="SLSQP", options={'disp': False}, 
                       constraints=(return_is_target, weights_sum_to_1), bounds = bounds )
    return weights_result.x

def optimal_weights(n_points, er, cov):
    """
    --> list of weights to run the optimizer on to minimize the vol
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def msr(riskfree_rate, er, cov):
    """
    return the weights of the portfolio that gives you the maximum sharpe ratio
    Gives the riskfree rate and expected returns and a covariance matrix
    risk_free_rate + ER + COV --> W
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n,n)
    bounds = ((0.0,1.0),)*n
    weights_sum_to_1 ={
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    def negative_sharp_ratio(weights, riskfree_rate,er,cov):
        """
        Returns the sharpe_ration given weights
        """
        r = portfolio_return(weights,er)
        vol =portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
        
    weights_result = minimize(negative_sharp_ratio, init_guess,args = (riskfree_rate, er,cov,), method ="SLSQP", options={'disp': False}, 
                       constraints=(weights_sum_to_1), bounds = bounds )
    return weights_result.x

def gmv(cov):
    """
    Returns the Weights of the global Minimum Vol portfolio
    given Cov matrix
    """
    n = cov.shape[0]
    return msr(0,np.repeat(1,n),cov)
               
def plot_ef(n_points, er, cov, show_cml = False, style =".-", riskfree_rate = 0, show_ew=False, show_gmv=False):
    """
    Plots the N-asset efficient frontier"""
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns":rets,"Volatility":vols})
    
    ax = ef.plot.line(x="Volatility", y="Returns", style=style)
    
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n,n)
        r_ew =portfolio_return(w_ew,er)
        vol_ew = portfolio_vol(w_ew,cov)
        #display EW
        ax.plot([vol_ew],[r_ew], color = "goldenrod", marker ="o",markersize = 12)
        
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv =portfolio_return(w_gmv,er)
        vol_gmv = portfolio_vol(w_gmv,cov)
        #display GMV
        ax.plot([vol_gmv],[r_gmv], color = "midnightblue", marker ="o",markersize = 10)
     
    if show_cml:
        ax.set_xlim(left = 0)
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        #add CML
        cml_x = [0,vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x,cml_y, color='green', marker ="o", linestyle="dashed", markersize = 12, linewidth = 2)
    
    return ax 



