
import pandas as pd
import scipy.stats
import numpy as np
from scipy.stats import norm


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

def Drwadown(return_series :pd.Series):
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

        
        
  





