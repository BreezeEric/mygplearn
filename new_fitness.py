"""Metrics to evaluate the fitness of a program.

The :mod:`gplearn.fitness` module contains some metric with which to evaluate
the computer programs created by the :mod:`gplearn.genetic` module.
"""


import numbers

import numpy as np
from joblib import wrap_non_picklable_objects
from scipy.stats import rankdata

__all__ = ['make_fitness']


class _Fitness(object):

    """A metric to measure the fitness of a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting floating point score quantifying the quality of the program's
    representation of the true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    """

    def __init__(self, function, greater_is_better):
        self.function = function
        self.greater_is_better = greater_is_better
        self.sign = 1 if greater_is_better else -1

    def __call__(self, *args):
        return self.function(*args)


def make_fitness(function, greater_is_better, wrap=True):
    """Make a fitness measure, a metric scoring the quality of a program's fit.

    This factory function creates a fitness measure object which measures the
    quality of a program's fit and thus its likelihood to undergo genetic
    operations into the next generation. The resulting object is able to be
    called with NumPy vectorized arguments and return a resulting floating
    point score quantifying the quality of the program's representation of the
    true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom metrics is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    """
    if not isinstance(greater_is_better, bool):
        raise ValueError('greater_is_better must be bool, got %s'
                         % type(greater_is_better))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))
    if function.__code__.co_argcount != 3:
        raise ValueError('function requires 3 arguments (y, y_pred, w),'
                         ' got %d.' % function.__code__.co_argcount)
        '''
    if not isinstance(function(np.array([1, 1]),
                      np.array([2, 2]),
                      np.array([1, 1])), numbers.Number):
        raise ValueError('function must return a numeric.')
        '''

    if wrap:
        return _Fitness(function=wrap_non_picklable_objects(function),
                        greater_is_better=greater_is_better)
    return _Fitness(function=function,
                    greater_is_better=greater_is_better)


def _weighted_pearson(y, y_pred, w=None):
    """Calculate the weighted Pearson correlation coefficient."""
    with np.errstate(divide='ignore', invalid='ignore'):
        y_pred_demean = y_pred - np.average(y_pred, weights=w)
        y_demean = y - np.average(y, weights=w)
        corr = ((np.sum(w * y_pred_demean * y_demean) / np.sum(w)) /
                np.sqrt((np.sum(w * y_pred_demean ** 2) *
                         np.sum(w * y_demean ** 2)) /
                        (np.sum(w) ** 2)))
    if np.isfinite(corr):
        return np.abs(corr)
    return 0.

def _pearson(y,y_pred,w=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        y_pred_demean = y_pred - np.average(y_pred, weights=w)
        y_demean = y - np.average(y, weights=w)
        corr = np.nansum(y_pred_demean * y_demean)/np.sqrt(np.nansum(y_pred_demean**2)*np.nansum(y_demean**2))
    if np.isfinite(corr):
        return np.abs(corr)
    return 0.

def _icir(y,y_pred,w = None,ww=[0.9,0.1,0,0,0]):
    with np.errstate(divide='ignore', invalid='ignore'):
        tempt1 = [np.corrcoef(y[:,i],y_pred[:,i])[0][1] for i in range((y.shape[1]))]
        tempt2 = [np.corrcoef(y[3:,i],y_pred[:-3,i])[0][1] for i in range((y.shape[1]))]
        tempt3 = [np.corrcoef(y[5:,i],y_pred[:-5,i])[0][1] for i in range((y.shape[1]))]
        tempt4 = [np.corrcoef(y[10:,i],y_pred[:-10,i])[0][1] for i in range((y.shape[1]))]
        tempt5 = [np.corrcoef(y[30:,i],y_pred[:-30,i])[0][1] for i in range((y.shape[1]))]
        icir1 = np.nanmean(tempt1)/np.nanstd(tempt1)
        icir2 = np.nanmean(tempt2)/np.nanstd(tempt2)
        icir3 = np.nanmean(tempt3)/np.nanstd(tempt3)
        icir4 = np.nanmean(tempt4)/np.nanstd(tempt4)
        icir5 = np.nanmean(tempt5)/np.nanstd(tempt5)
        icir = ww[0]*icir1 + ww[1]*icir2 + ww[2]*icir3 + ww[3]*icir4 + ww[4]*icir5
    if np.isfinite(icir):
        return np.abs(icir)
    return 0.
    


def _weighted_spearman(y, y_pred, w = None):
    """Calculate the weighted Spearman correlation coefficient."""
    y_pred_ranked = np.apply_along_axis(rankdata, 0, y_pred)
    y_ranked = np.apply_along_axis(rankdata, 0, y)
    return _weighted_pearson(y_pred_ranked, y_ranked, w)


def _mean_absolute_error(y, y_pred, w=None):
    """Calculate the mean absolute error."""
    return np.average(np.abs(y_pred - y), weights=w)


def _mean_square_error(y, y_pred, w=None):
    """Calculate the mean square error."""
    return np.average(((y_pred - y) ** 2), weights=w)


def _root_mean_square_error(y, y_pred, w=None):
    """Calculate the root mean square error."""
    return np.sqrt(np.average(((y_pred - y) ** 2), weights=w))


def _log_loss(y, y_pred, w=None):
    """Calculate the log loss."""
    eps = 1e-15
    inv_y_pred = np.clip(1 - y_pred, eps, 1 - eps)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    score = y * np.log(y_pred) + (1 - y) * np.log(inv_y_pred)
    return np.average(-score, weights=w)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def to_final_position(factor_score, method_func='standard'):
    """
    形成最终的仓位，即shift(1)，并填补缺失
    :param factor_score: factor df
    :param method_func: fill method of NA
    :return: final position
    """
    if method_func == 'simple':
        pos_fin = factor_score.shift(1)
        pos_fin = pos_fin.replace(np.nan, 0)  # fill NA with 0
        return pos_fin
    else:
        pos_fin = factor_score.shift(1)
        pos_fin = pos_fin.ffill()  # fill NA with preceding value
        return pos_fin


def factor_stats(factor_df, rtn_df):
    """
    按照仓位计算收益率，返回最终的仓位与每个时点的收益率
    :param factor_df:   factor df
    :param rtn_df:    df of future rtn
    :return:  final position/final pnl
    """
    factor_sel = factor_df.copy()
    return_df = rtn_df.reindex_like(factor_sel)

    pos_final = to_final_position(factor_sel)
    pnl_final = (pos_final.shift(1) * return_df).sum(axis=1)
    return pos_final, pnl_final


def Evaluation(factor_df, rtn_df, plt_save_str = 'my_plt', freq='1min', time_stamp=True, convert_date=False, t_cost=0.0003, cutoff_date='2021-01-01', need_graph=False):
    """
        进行回测，输入仓位矩阵与收益率矩阵，成本预设万分之三
        :param factor_df:   factor df
        :param rtn_df:    df of stock rtn
        :param time_stamp: whether time_stamp
        :param convert_date:    Whether conversion time is required
        :param t_cost:    trading cost
        :param cutoff_date:   cutoff date for two evaluation
        :param need_graph:   whether graph
        :return: sharpe ratio / pot / annual return
    """
    if time_stamp:

        def standard_data(data):
            result = pd.DataFrame(data[data.columns[1:]])
            result.index = pd.to_datetime(data[data.columns[0]].apply(
                lambda x: str(x)[0:4] + '-' + str(x)[4:6] + '-' + str(x)[6:8] + '-' + str(x)[8:10] + '-'
                          + str(x)[10:12] + '-' + str(x)[12:14] + str(int(str(x)[14:17]) / 1000)[1:]), format='%Y-%m-%d-%H-%M-%S.%f')
            return result

        if convert_date:
            factor_df = standard_data(factor_df)
            rtn_df = standard_data(rtn_df)

        pos_df, dailypnl_gross = factor_stats(factor_df, rtn_df)
        daily_cost = (pos_df.diff().abs() * t_cost).sum(axis=1)
        dailypnl_net = dailypnl_gross - daily_cost

        dailypnl_net_p1 = dailypnl_net.loc[dailypnl_net.index <= cutoff_date]
        dailypnl_gross_p1 = dailypnl_gross.loc[dailypnl_gross.index <= cutoff_date]
        pos_p1 = pos_df.loc[pos_df.index <= cutoff_date]

        dailypnl_net_p2 = dailypnl_net.loc[dailypnl_net.index > cutoff_date]
        dailypnl_gross_p2 = dailypnl_gross.loc[dailypnl_gross.index > cutoff_date]
        pos_p2 = pos_df.loc[pos_df.index > cutoff_date]

        def sharpe(pnl_df):
            if freq == '1min':
                return (np.sqrt(250 * 360) * pnl_df.mean()) / pnl_df.std()
            elif freq == '1day':
                return (np.sqrt(250) * pnl_df.mean()) / pnl_df.std()

        def Pot(pos_df, asset_last):
            """
            计算 pnl/turover*10000的值,衡量cost的影响
            :param pos_df: 仓位信息
            :param asset_last: 最后一天的收益
            :return:
            """
            trade_times = pos_df.diff().abs().sum().sum()
            if trade_times == 0:
                return 0
            else:
                pot = asset_last / trade_times * 10000
                return round(pot, 2)

        def annual_return(pnl_df):
            temp_pnl = pnl_df.sum()
            num = len(pnl_df.resample('1d').last().dropna())

            if num == 0:
                return .0
            else:
                return temp_pnl * 251.0 / num

        sp1 = sharpe(dailypnl_net_p1)
        pot1 = Pot(pos_p1, dailypnl_gross_p1.fillna(0).cumsum().iloc[-1])
        annr1 = annual_return(dailypnl_net_p1)

        sp2 = sharpe(dailypnl_net_p2)
        pot2 = Pot(pos_p2, dailypnl_gross_p2.fillna(0).cumsum().iloc[-1])
        annr2 = annual_return(dailypnl_net_p2)

        # 预设的回测需要达到的标准，可以根据需要调整
        if sp1 <= 2.7 or pot1 <= 32 or annr1 <= 0.18:
            print(f"unsatisfied submission standard, before {cutoff_date}, SP:{sp1}, POT:{pot1}, AnR:{annr1}")

        elif sp2 <= 2.1 or pot2 <= 27 or annr2 <= 0.15:
            print(f"unsatisfied submission standard, after {cutoff_date}, SP:{sp2}, POT:{pot2}, AnR:{annr2}")
        else:
            print(f"satisfied submission standard, before {cutoff_date}, SP:{sp1}, POT:{pot1}, AnR:{annr1}; "
                  f"after {cutoff_date}, SP:{sp2}, POT:{pot2}, AnR:{annr2}")
        if need_graph:
            plt.figure()
            p1 = plt.subplot(211)
            p2 = plt.subplot(212)

            p1.plot(dailypnl_gross_p1.fillna(0).cumsum(), label=f"PNL_Gross")
            p1.plot(dailypnl_net_p1.fillna(0).cumsum(), label=f"PNL_Net, SP:{sp1}, POT:{pot1}, AnR:{annr1}")
            p1.set_title(f'Before {cutoff_date}')
            p1.grid(linestyle='--')
            p1.legend(loc='upper left')
            plt.grid(True)

            p2.plot(dailypnl_gross_p2.fillna(0).cumsum(), label=f"PNL_Gross")
            p2.plot(dailypnl_net_p2.fillna(0).cumsum(), label=f"PNL_Net, SP:{sp2}, POT:{pot2}, AnR:{annr2}")
            p2.set_title(f'After {cutoff_date}')
            p2.grid(linestyle='--')
            p2.legend(loc='upper left')
            plt.grid(True)
            plt.setp(p1.get_xticklabels(), rotation=30, horizontalalignment='right')
            plt.setp(p2.get_xticklabels(), rotation=30, horizontalalignment='right')
            plt.savefig(plt_save_str)
            plt.show()

        return sp1, pot1, annr1, sp2, pot2, annr2
    else:
        pos_df, dailypnl_gross = factor_stats(factor_df, rtn_df)
        daily_cost = (pos_df.diff().abs() * t_cost).sum(axis=1)
        dailypnl_net = dailypnl_gross - daily_cost

        num = int(len(dailypnl_net) / 2)
        dailypnl_net_p1 = dailypnl_net.iloc[:num]
        dailypnl_gross_p1 = dailypnl_gross.iloc[:num]
        pos_p1 = pos_df.iloc[:num]

        dailypnl_net_p2 = dailypnl_net.iloc[num:]
        dailypnl_gross_p2 = dailypnl_gross.iloc[num:]
        pos_p2 = pos_df.iloc[num:]

        def sharpe(pnl_df):
            if freq == '1min':
                return (np.sqrt(250 * 360) * pnl_df.mean()) / pnl_df.std()
            elif freq == '1day':
                return (np.sqrt(250) * pnl_df.mean()) / pnl_df.std()

        def Pot(pos_df, asset_last):
            """
            计算 pnl/turover*10000的值,衡量cost的影响
            :param pos_df: 仓位信息
            :param asset_last: 最后一天的收益
            :return:
            """
            trade_times = pos_df.diff().abs().sum().sum()
            if trade_times == 0:
                return 0
            else:
                pot = asset_last / trade_times * 10000
                return round(pot, 2)

        def annual_return(pnl_df):
            temp_pnl = pnl_df.sum()

            if freq == '1min':
                num = len(pnl_df) / 360
            elif freq == '1day':
                num = len(pnl_df)
            else:
                num = 0
            if num == 0:
                return .0
            else:
                return temp_pnl * 251.0 / num

        sp1 = sharpe(dailypnl_net_p1)
        pot1 = Pot(pos_p1, dailypnl_gross_p1.fillna(0).cumsum().iloc[-1])
        annr1 = annual_return(dailypnl_net_p1)

        sp2 = sharpe(dailypnl_net_p2)
        pot2 = Pot(pos_p2, dailypnl_gross_p2.fillna(0).cumsum().iloc[-1])
        annr2 = annual_return(dailypnl_net_p2)
        
        # 预设的回测需要达到的标准，可以根据需要调整
        if sp1 <= 2.7 or pot1 <= 32 or annr1 <= 0.18:
            print(f"unsatisfied submission standard, before {cutoff_date}, SP:{sp1}, POT:{pot1}, AnR:{annr1}")

        elif sp2 <= 2.1 or pot2 <= 27 or annr2 <= 0.15:
            print(f"unsatisfied submission standard, after {cutoff_date}, SP:{sp2}, POT:{pot2}, AnR:{annr2}")
        else:
            print(f"satisfied submission standard, before {cutoff_date}, SP:{sp1}, POT:{pot1}, AnR:{annr1}; "
                  f"after {cutoff_date}, SP:{sp2}, POT:{pot2}, AnR:{annr2}")
        if need_graph:
            plt.figure()
            p1 = plt.subplot(211)
            p2 = plt.subplot(212)

            p1.plot(dailypnl_gross_p1.fillna(0).cumsum(), label=f"PNL_Gross")
            p1.plot(dailypnl_net_p1.fillna(0).cumsum(), label=f"PNL_Net, SP:{sp1}, POT:{pot1}, AnR:{annr1}")
            p1.set_title(f'First period')
            p1.grid(linestyle='--')
            p1.legend(loc='upper left')
            plt.grid(True)

            p2.plot(dailypnl_gross_p2.fillna(0).cumsum(), label=f"PNL_Gross")
            p2.plot(dailypnl_net_p2.fillna(0).cumsum(), label=f"PNL_Net, SP:{sp2}, POT:{pot2}, AnR:{annr2}")
            p2.set_title(f'Second period')
            p2.grid(linestyle='--')
            p2.legend(loc='upper left')
            plt.grid(True)
            plt.setp(p1.get_xticklabels(), rotation=30, horizontalalignment='right')
            plt.setp(p2.get_xticklabels(), rotation=30, horizontalalignment='right')
            plt.savefig(plt_save_str)
            plt.show()

        return sp1, pot1, annr1, sp2, pot2, annr2

from sklearn.preprocessing import MinMaxScaler
def _evaluation(y,y_pred,w=None):
    signal = y_pred.mean(0)
    hold_label = []
#     model = MinMaxScaler()
#     model.fit(np.array(signal).reshape(-1,1))
#     signal = model.fit_transform(np.array(signal).reshape(-1,1))
    for i in range(len(signal)):
        if signal[i]>np.quantile(signal,0.7):
            hold_label.append(1)
        elif signal[i]<np.quantile(signal,0.3):
            hold_label.append(-1)
        else:
            hold_label.append(0)
    r = pd.DataFrame(y.mean(0))
    r.columns = ['future1']
    hold_label2 = -np.array(hold_label)
    hold_label1 = np.array(hold_label)
    pos1 = pd.DataFrame(hold_label1,columns=['future1'])
    pos2 = pd.DataFrame(hold_label2,columns=['future1'])
    sp1, pot1, annr1, sp2, pot2, annr2 = Evaluation(pos1, r, time_stamp=False, need_graph=False)
    return np.nan_to_num(sp1) + np.nan_to_num(sp2)



# import pandas as pd
# import matplotlib as mpl
# import numba as nb
# import pandas as pd
# import os
import datetime
# import numpy as np
# from dateutil.relativedelta import relativedelta
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn import linear_model
#计算累计收益函数
def accumReturn(Asset_nav):
    return Asset_nav[len(Asset_nav)-1]/Asset_nav[0]-1
#计算年化收益函数
def annReturn(Asset_nav,annual_day=250):
    during_day=len(Asset_nav)
    annual_rate=( Asset_nav[len(Asset_nav)-1]/Asset_nav[0])**(annual_day/during_day)  -1
    return annual_rate
#计算年化波动率
def volatility(Asset_nav,annual_day=250):
    ret=(Asset_nav[1:len(Asset_nav)]-Asset_nav[0:(len(Asset_nav)-1 )])/Asset_nav[0:(len(Asset_nav)-1 )]
    volatility=np.std(ret)*np.sqrt(annual_day)
    return volatility
#计算年化夏普
def sharpRatio(Asset_nav,annual_day=250):
    return annReturn(Asset_nav,annual_day)/volatility(Asset_nav,annual_day)
#计算最大回撤
def max_drawdown(Asset_nav,annual_day=250):
    acc_max=np.maximum.accumulate(Asset_nav)
    max_drawdown=np.max((acc_max-Asset_nav)/acc_max)
    return max_drawdown

#单因子的IC，IR分析
def IC_RIC(data,factor,vReturn):
    date_arr=data['Date'].unique()
    COR_df=pd.DataFrame()
    for dt in date_arr:
        Tmp_cor_pearson=data[data['Date']==dt ][ [factor, vReturn] ].corr('pearson')
        Tmp_cor_spearman=data[data['Date']==dt ][ [factor, vReturn] ].corr('spearman')
        tmp_COR_df=pd.DataFrame([ dt, Tmp_cor_pearson[factor][vReturn], Tmp_cor_spearman[factor][vReturn],factor ]).T
        tmp_COR_df.columns=['TRADE_DT','IC','RIC','fact']
        COR_df=COR_df.append(pd.DataFrame(tmp_COR_df)  )
    return COR_df
#单因子的IC，IR走势图
def IC_RIC_plot(data,factor):
    fig,ax=plt.subplots(figsize=(16,8))
    plt.xticks(rotation=45)
    n=len(data[data['fact']==factor])
    plt.bar(np.arange(n),np.array(data[data['fact']==factor]['IC']),0.35,alpha=0.9)
    plt.bar(np.arange(n)+0.35,np.array(data[data['fact']==factor]['RIC']),0.35,alpha=0.9)
    plt.xticks([])
    x_axis_data=np.array(data['TRADE_DT'].astype(str).unique())
    plt.xticks(range(0,len(x_axis_data),10 ),x_axis_data[np.array(range(0,len(x_axis_data),10 ))])
    plt.legend(['IC','RIC'],loc=1,title='legend')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.show
#回测函数
def Backtest(Hold_df,price_table,vfee=0.0003):
    start_nav=1
    vData=Hold_df
    Date_arr=vData['Date'].unique()
    nav=pd.DataFrame()
    for i_date in Date_arr:
        tmp_vData=vData[vData['Date']==i_date]
        tmp_date=Date_arr[Date_arr>i_date ]
        if len(tmp_date)>0:
            end_date=tmp_date[0]
        else:
            end_date=datetime.datetime.now()
        tmp_price=price_table[(price_table.index>i_date) & (price_table.index<=end_date) ][tmp_vData['S_INFO_WINDCODE']]
        for j_name in tmp_vData['S_INFO_WINDCODE']:
            try:
                tmp_price[j_name]=tmp_price[j_name]/(tmp_price[j_name][0])
            except:
                continue
            else:
                tmp_price[j_name]=tmp_price[j_name]/(tmp_price[j_name][0])
        tmp_price = tmp_price.fillna(0)
        tmp_nav=np.dot(tmp_price.values,tmp_vData['Weight'] )*start_nav*(1-vfee)
        if not len(tmp_nav):
            continue
        start_nav =tmp_nav[len( tmp_nav)-1]
        tmp_nav_df=pd.DataFrame(tmp_nav,index=tmp_price.index)
        tmp_nav_df.columns=['Nav']
        nav=nav.append(tmp_nav_df)
    return nav


#my_columns = pd.read_csv('windcode.csv')['0'].values
#my_index = pd.read_csv('date.csv')['0'].values
def _multievaluation(y,y_pred,w=None):
    ll = y_pred.shape[1]
    data = pd.DataFrame(y_pred)
    data.index= my_columns
    data.columns = my_index[:ll]
    data = data.T.fillna(method = 'ffill').fillna(0)
    pd.DataFrame(data.stack()).to_csv('tempt.csv')
    d1= pd.read_csv('tempt.csv')
    d1.columns = ['TRADE_DATE', 'WINDCODE', '0']
    dd = pd.DataFrame(y)
    dd.index = my_columns
    dd.columns = my_index[:ll]
    dd = dd.T.fillna(method = 'ffill').fillna(0)
    pd.DataFrame(dd.stack()).to_csv('tempt.csv')
    dd = pd.read_csv('tempt.csv')
    dd.columns = ['TRADE_DATE', 'WINDCODE', 'CLOSE']
    dd.index = range(len(dd))
    df4 = dd.fillna(0)
    df4 = df4.set_index(['TRADE_DATE','WINDCODE'])
    df5 = dd.fillna(0)
    vfactor = '0'
    tempt_data = d1
    flag = 'equal'
    ###################分组###################
    Group_df=pd.DataFrame()
    for i_date in tempt_data['TRADE_DATE'].unique():
        tmp_any_data=tempt_data[tempt_data['TRADE_DATE']==i_date ]
        tmp_any_data.drop(tmp_any_data[np.isnan(tmp_any_data[vfactor])].index, inplace=True)
        tmp_any_data['Group'] = ""
        th = [np.quantile(tmp_any_data[vfactor].values,0.2),
             np.quantile(tmp_any_data[vfactor].values,0.4),
             np.quantile(tmp_any_data[vfactor].values,0.6),
             np.quantile(tmp_any_data[vfactor].values,0.8)]
        tmp_any_data.loc[tmp_any_data[vfactor]<th[0],'Group'] = "P1"
        tmp_any_data.loc[(tmp_any_data[vfactor]>=th[0]) & (tmp_any_data[vfactor]<th[1]),'Group'] = "P2"
        tmp_any_data.loc[(tmp_any_data[vfactor]>=th[1]) & (tmp_any_data[vfactor]<th[2]),'Group'] = "P3"
        tmp_any_data.loc[(tmp_any_data[vfactor]>=th[2]) & (tmp_any_data[vfactor]<th[3]),'Group'] = "P4"
        tmp_any_data.loc[tmp_any_data[vfactor]>=th[3],'Group'] = "P5"
        Group_df=Group_df.append(tmp_any_data)

    Hold_df=pd.DataFrame()
    for i_date in Group_df['TRADE_DATE'].unique():
        tmp_any_data=Group_df[Group_df['TRADE_DATE']==i_date]
        for i_group in tmp_any_data['Group'].unique():
            tmp_Group_df=tmp_any_data[tmp_any_data['Group']==i_group ]
            tmp_Group_df['Weight'] = np.nan
            #用异常处理的方式处理nan的值，对于nan的权重取0
            if flag!='equal':
                for j in tmp_Group_df.index:
                    try:
                        tmp_Group_df.loc[j,'NEG_MARKET_VALUE']/np.nansum(tmp_Group_df['NEG_MARKET_VALUE'].values)
                    except:
                        tmp_Group_df.loc[j,'Weight']=0
                    else:
                        num = tmp_Group_df.loc[j,'NEG_MARKET_VALUE']/np.nansum(tmp_Group_df['NEG_MARKET_VALUE'].values)
                        tmp_Group_df.loc[j,'Weight']=np.nan_to_num(num)
            else:
                try:
                    1/len(tmp_Group_df)
                except:
                    tmp_Group_df['Weight'] = 0
                else:
                    tmp_Group_df['Weight']=1/len(tmp_Group_df)
            Hold_df=Hold_df.append(tmp_Group_df[['WINDCODE','TRADE_DATE','Group','Weight'] ])

    ######################################
    Hold_df['price'] = np.nan
    Hold_df['TRADE_DATE']=pd.to_datetime(Hold_df['TRADE_DATE'],format='%Y-%m-%d')
    Hold_df_tempt = Hold_df.set_index(['TRADE_DATE','WINDCODE'])

    Hold_df_tempt.loc[list(set(Hold_df_tempt.index) & set(df4.index)),'price'] = df4.loc[list(set(Hold_df_tempt.index) & set(df4.index)),'CLOSE']
    Hold_df_tempt = Hold_df_tempt.fillna(0)
    Hold_df_tempt.to_csv('tempt.csv')
    Hold_df = pd.read_csv('tempt.csv')
    #################################
    
    price_data = df5.copy()

    Hold_df.columns = ['Date','S_INFO_WINDCODE','Group','Weight','Price']
    price_data.columns = ['Date','S_INFO_WINDCODE','Price']
    price_data['Date'] = pd.to_datetime(price_data['Date'],format='%Y-%m-%d')
    price_table=price_data.pivot(values='Price',columns='S_INFO_WINDCODE',index='Date')
    price_table =price_table.fillna(method='bfill').fillna(0)
    nav_P1=Backtest(Hold_df[Hold_df['Group' ]=="P1" ],price_table,vfee=0.003)
    nav_P5=Backtest(Hold_df[Hold_df['Group' ]=="P5" ],price_table,vfee=0.003)
    nav_P1_new = pd.DataFrame(np.nan,index = pd.to_datetime(my_index),columns= ['Nav'])
    nav_P5_new = pd.DataFrame(np.nan,index = pd.to_datetime(my_index),columns= ['Nav'])
    nav_P1_new.loc[nav_P1.index.values,'Nav'] = nav_P1.Nav.values
    nav_P5_new.loc[nav_P5.index.values,'Nav'] = nav_P5.Nav.values
    nav_P1_new = nav_P1_new.fillna(1)
    nav_P5_new = nav_P5_new.fillna(1)
    
    res = sharpRatio(np.cumprod(1+(nav_P1_new.pct_change().fillna(0).values - nav_P5_new.pct_change().fillna(0).values)))
    print(res)
    
    return res
    





weighted_pearson = _Fitness(function=_weighted_pearson,
                            greater_is_better=True)
weighted_spearman = _Fitness(function=_weighted_spearman,
                             greater_is_better=True)
mean_absolute_error = _Fitness(function=_mean_absolute_error,
                               greater_is_better=False)
mean_square_error = _Fitness(function=_mean_square_error,
                             greater_is_better=False)
root_mean_square_error = _Fitness(function=_root_mean_square_error,
                                  greater_is_better=False)
log_loss = _Fitness(function=_log_loss,
                    greater_is_better=False)

pearson = _Fitness(function=_pearson,
                    greater_is_better=True)

evaluation = _Fitness(function=_evaluation,greater_is_better=True)

icir = _Fitness(function=_icir,greater_is_better=True)

multievaluation = _Fitness(function=_multievaluation,greater_is_better=True)

_fitness_map = {'pearson': weighted_pearson,
                'spearman': weighted_spearman,
                'mean absolute error': mean_absolute_error,
                'mse': mean_square_error,
                'rmse': root_mean_square_error,
                'pearson1':icir,
                'evaluation':multievaluation,
                'log loss': log_loss}
