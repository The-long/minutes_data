'''
@Date: 2020-02-17 08:50:46
@LastEditors: baochen
@LastEditTime: 2020-02-17 09:13:21
@FilePath: /FactorRD/FactorAutoBuild/FactorEvaluation/FactorEvaluation.py
@Description: Do not edit
'''
from FactorAutoBuild.FactorEvaluation import FactorTest 
from FactorAutoBuild.FactorEvaluation import FactorFolio
from FactorAutoBuild.FactorCalculate.FunctionUntility import *
import pandas as pd
import numpy as np
import logging
import time

def autocorr(df,interval_day=1):
    lis =[]
    for i in range(interval_day,len(df)):
        lis.append(df.iloc[i].corr(df.iloc[i-1]))
    return np.mean(lis)

def cal_factors_stat(factor_values,trade_price,by_group,groupby= None,show_plot=False, quantiles =5,periods = (1,5,10),period = '1D'):
    '''
    @description: 
    @param {
        factor_values: facotr dataframe
        trade_price: the trade price of the stock
        show_plot: if show the plot of backtest
    } 
    @return: 
    '''
    start_time = time.time()
    ac_1 = autocorr(factor_values)
    ac_2 = autocorr(factor_values,2)
    ac_3 = autocorr(factor_values,3)
    turnover_1d = abs(factor_values.diff(1)).mean().mean()
    turnover_2d = abs(factor_values.diff(2)).mean().mean()
    turnover_3d = abs(factor_values.diff(3)).mean().mean()
    if show_plot==True:
        mean_ic , pf_returns, pf_positions,mean_quant_ret, factor_returns =  factor_eval(factor_values, trade_price,groupby,show_plot,quantiles=quantiles,periods=periods,period=period,by_group=by_group)
        logging.info("Calculation Cost: " + str(time.time() - start_time))
        mean_ic['autocorr_1d'] = ac_1
        mean_ic['autocorr_2d'] = ac_2
        mean_ic['autocorr_3d'] = ac_3
        mean_ic['turnover_1d'] = turnover_1d
        mean_ic['turnover_2d'] = turnover_2d
        mean_ic['turnover_3d'] = turnover_3d
        mean_ic['test_time_duration'] = [str(factor_values.index[0]),str(factor_values.index[-1])]
        return mean_ic , pf_returns, pf_positions,mean_quant_ret, factor_returns
    else:
        mean_ic , pf_returns, pf_positions =  factor_eval(factor_values, trade_price,groupby,show_plot,quantiles = quantiles,periods=periods,period=period,by_group=by_group)
        logging.info("Calculation Cost: " + str(time.time() - start_time))
        mean_ic['autocorr_1d'] = ac_1
        mean_ic['autocorr_2d'] = ac_2
        mean_ic['autocorr_3d'] = ac_3
        mean_ic['turnover_1d'] = turnover_1d
        mean_ic['turnover_2d'] = turnover_2d
        mean_ic['turnover_3d'] = turnover_3d
        mean_ic['test_time_duration'] = [str(factor_values.index[0]),str(factor_values.index[-1])]
        return mean_ic , pf_returns, pf_positions

def factor_eval(alpha_pre,price_data,groupby,show_plot , T1 =False, backtest = False, by_group =False, quantiles=5 ,periods = (1,5,10),  period = '1D' , turnover_analysis = False , long_short = True):
    alpha_pre.index = pd.to_datetime(alpha_pre.index)
    price_data.index = pd.to_datetime(price_data.index)
    factor = pd.DataFrame(alpha_pre.T.unstack(),columns=['alpha'])
    if T1:
        factor = pd.DataFrame(alpha_pre.shift(1).T.unstack(),columns=['alpha'])

    factor_data = FactorTest.utils.get_clean_factor_and_forward_returns(factor,
                                                                        price_data,
                                                                        groupby,
                                                                        periods=periods,
                                                                        quantiles = quantiles,
                                                                        max_loss=1
                                                                        )
    mean_ic =  FactorTest.performance.mean_information_coefficient(factor_data, by_group)
    mean_monthly_ic = FactorTest.performance.mean_information_coefficient(factor_data, by_time='M')

    if turnover_analysis:
        FactorTest.tears.create_turnover_tear_sheet(factor_data)

    pf_returns, pf_positions, pf_benchmark = \
        FactorTest.performance.create_pyfolio_input(factor_data,
                                                    period,
                                                    capital=1000000,
                                                    long_short=True,
                                                    group_neutral=False,
                                                    equal_weight=False,
                                                    quantiles=None,
                                                    groups=None,
                                                    benchmark_period='1D')
    perf_stat = FactorFolio.timeseries.perf_stats(pf_returns)
    mean_ic = mean_ic.append(perf_stat)
    price_returns = price_data.diff(1)/price_data.shift(1)
    try:
        corr_coef = np.corrcoef(rank(alpha_pre.fillna(0)).values.reshape(-1),rank(price_returns.shift(1).fillna(0)).values.reshape(-1))[0,1]
    except:
        corr_coef = None
    price_return_reshaped = price_returns.shift(1).fillna(0).values.reshape(-1)
    price_return_5d = price_data.shift(5)/price_data.shift(5)
    price_return_5d_reshaped = price_return_5d.shift(1).fillna(0).values.reshape(-1)
    alpha_reshaped = alpha_pre.fillna(0).values.reshape(-1)
    mean_ic['factor_corr_with_returns_1d'] = np.corrcoef(price_return_reshaped,alpha_reshaped)[0,1]
    mean_ic['factor_corr_with_returns_5d'] = np.corrcoef(price_return_5d_reshaped,alpha_reshaped)[0,1]
    mean_ic['rankfactor_corr_with_rankreturns_1d'] = corr_coef
    if show_plot:
        mean_quant_ret, factor_returns = FactorTest.tears.create_returns_tear_sheet(factor_data,                                                                 long_short,group_neutral=False,by_group=by_group)
        FactorTest.tears.create_information_tear_sheet(factor_data,
                                  group_neutral=False,
                                  by_group=by_group)
#             FactorTest.tears.create_information_tear_sheet(factor_data , by_group)
#         FactorFolio.tears.create_full_tear_sheet(pf_returns,
#                                         positions=pf_positions,
#                                         #benchmark_rets=pf_benchmark, # optional, default to SPY
#                                         hide_positions=True)
        return mean_ic , pf_returns, pf_positions, mean_quant_ret, factor_returns
    else:
        return mean_ic , pf_returns, pf_positions
    
    
    
    
def factor_eval_min(alpha_pre,price_data,groupby,show_plot , T1 =False, backtest = False, by_group =False, quantiles=5 ,periods = (1,5,10),  period = '1M' , turnover_analysis = False , long_short = True):
    factor = pd.DataFrame(alpha_pre.T.unstack(),columns=['alpha'])
    if T1:
        factor = pd.DataFrame(alpha_pre.shift(1).T.unstack(),columns=['alpha'])

    factor_data = FactorTest.utils.get_clean_factor_and_forward_returns_min(factor,
                                                                        price_data,
                                                                        groupby,
                                                                        periods=periods,
                                                                        quantiles = quantiles,
                                                                        max_loss=1
                                                                        )
    mean_ic =  FactorTest.performance.mean_information_coefficient(factor_data, by_group)
    
    if turnover_analysis:
        FactorTest.tears.create_turnover_tear_sheet(factor_data)

    pf_returns=FactorTest.performance.factor_returns_min(factor_data,
                   demeaned=True,
                   group_adjust=False,
                   equal_weight=False,
                   by_asset=False)
    
    
    price_returns = price_data.diff(1)/price_data.shift(1)
    try:
        corr_coef = np.corrcoef(rank(alpha_pre.fillna(0)).values.reshape(-1),rank(price_returns.shift(1).fillna(0)).values.reshape(-1))[0,1]
    except:
        corr_coef = None
    price_return_reshaped = price_returns.shift(1).fillna(0).values.reshape(-1)
    price_return_5d = price_data.shift(5)/price_data.shift(5)
    price_return_5d_reshaped = price_return_5d.shift(1).fillna(0).values.reshape(-1)
    alpha_reshaped = alpha_pre.fillna(0).values.reshape(-1)
    mean_ic['factor_corr_with_returns_1d'] = np.corrcoef(price_return_reshaped,alpha_reshaped)[0,1]
    mean_ic['factor_corr_with_returns_5d'] = np.corrcoef(price_return_5d_reshaped,alpha_reshaped)[0,1]
    mean_ic['rankfactor_corr_with_rankreturns_1d'] = corr_coef
    if show_plot:
        mean_quant_ret, factor_returns = FactorTest.tears.create_returns_tear_sheet_min(factor_data,                                                                 long_short,group_neutral=False,by_group=by_group)
#             FactorTest.tears.create_information_tear_sheet(factor_data , by_group)
#         FactorFolio.tears.create_full_tear_sheet(pf_returns,
#                                         positions=pf_positions,
#                                         #benchmark_rets=pf_benchmark, # optional, default to SPY
#                                         hide_positions=True)
        return mean_ic , pf_returns, mean_quant_ret, factor_returns
    else:
        return mean_ic , pf_returns