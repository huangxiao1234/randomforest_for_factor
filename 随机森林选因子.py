
# coding: utf-8

# 一．筛选因子模型介绍 
# 	利用股市波动具有趋势性的特点，由算法得出近段时间的上涨的主要依据因子，推测未来股票上涨依据的因子，从而可参照上涨因子更好地筛选未来更有可能上涨的股票。
# 	举个例子，如果这个月大家都看好绩优股那么下个月大概率还是会看好绩优股，只有当不看好绩优股的人慢慢变多了，市场的趋势才会改变。那么如何知道这个月大家最看重的是什么因素，人工筛选容易疏漏，考虑使用算法完成该功能。
# 	筛选因子的思路概括如下，以2017年8月到10月这段时间为例，对应不同的随机种子生成不同的随机森林模型，将这些模型对8到9月的股票训练集数据进行学习，从而得到不同的可用于测试的随机森林模型，再将这些模型对9月到10月的股票测试集进行测试，统计每个模型真阳率评分，选择真阳率最高的作为最终需要的随机森林模型（真阳率意味着预测涨的真的涨了）。此时该模型所依据的分类标准就是我们最终得到的因子筛选标准，将该模型所依据的因子按重要性进行降序排列，选择前n个重要性高的因子作为模型选出的趋势因子，再由股市波动具有趋势性的特征，将模型选出的因子作为下个月的参考因子，至此模型筛选因子过程结束。

# In[ ]:

import datetime
import time
import pandas as pd
from numpy import linalg as la
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy as np
import random
import matplotlib.pyplot as plt


def dealdata(data):
    '''
    数据预处理
    :param data: 未经处理的初始数据集
    :return: 处理好的数据集
    '''
    col = [c for c in data.columns if c not in ['secID','tradeDate','Return', 'Y_bin']]

    for i in col:
        data[i].fillna(data[i].mean(), inplace=True)
    data['Y_bin'].fillna(0, inplace=True)
    data['Return'].fillna(0, inplace=True)
    
    return data


def dataset(startdate, enddate, window, label_type, stockindex,stockpool_type):
    '''
    获取数据

    :param startdate:需要获取的数据开始日期
    :param enddate:需要获取的数据结束日期
    :param window:涨幅分类的阈值百分比，大于window的为正类，小于为负类
    :param label_type:使用哪种分类算法。1为按涨幅比例分，0为按涨跌与否分
    :param stockindex:哪个股票池的数据
    :param stockpool_type:使用哪种股票池。1为上证指数等大股票池，0为行业股票池
    :return: 列为各因子的值，行为startdate当天的所有股票代码
    '''
    stock_l=set_universe(stockindex)
    # 获取股票池开始日期的因子值
    fdf=DataAPI.MktStockFactorsOneDayGet(tradeDate=startdate,secID=stock_l,field=u"",pandas="1")
    #修改Index（为了后面的merge）
    fdf.index = fdf['ticker']
    fdf.pop('ticker')
    # 获取股票列表中的股票在所设置时间区间的收盘价
    current_close = DataAPI.MktEqudGet(secID=stock_l,tradeDate=startdate,pandas="1",field=['secID','preClosePrice'])
    stock_1=np.array(current_close['secID']).tolist()
    forcast_close = DataAPI.MktEqudGet(secID=stock_l,tradeDate=enddate,pandas="1",field=['secID','preClosePrice'])
    stock_2=np.array(forcast_close['secID']).tolist()
    # 为了让股票池在两个时间段内对齐，设置统一的stock_set
    stock_set=list(set(stock_1)&set(stock_2))
    #计算两个日期内的涨幅
    current_close = DataAPI.MktEqudGet(secID=stock_set,tradeDate=startdate,pandas="1",field=['ticker','preClosePrice'])
    forcast_close = DataAPI.MktEqudGet(secID=stock_set,tradeDate=enddate,pandas="1",field=['ticker','preClosePrice'])
    grow = list((forcast_close.iloc[:, 1] - current_close.iloc[:, 1]) / current_close.iloc[:, 1])
    
    grow = pd.DataFrame(grow, columns=['Return'])
    grow.index=current_close['ticker']
    # 拼接
    df = pd.merge(fdf, grow, left_index=True, right_index=True)
    
    # 选择第一种阈值计算类型（前百分之window的涨幅为1类，之后的为0类）
    if label_type == 1:  
        bound = np.nanpercentile(df['Return'], window)
        # if bound<0:
        #     bound=0
        df.loc[(df['Return'] >= bound ), 'Y_bin'] = 1
        df.loc[(df['Return'] < bound), 'Y_bin'] = 0
    else:  # 涨的为1类，跌的为0类（用于短期阈值比较好）
        bound = 0
        df.loc[(df['Return'] >= bound), 'Y_bin'] = 1
        df.loc[(df['Return'] < bound), 'Y_bin'] = 0
    return df


def print_(random_seed, s, train_data,col,n):
    '''
    作图
    :param random_seed: 最优的模型的随机种子编号
    :param s: 图的位置
    :param train_data: 训练数据集
    :param n:画出排名最前的n个因子
    '''
    rfc = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=random_seed)
    rfc.fit(train_data[col], train_data['Y_bin'])
    importances_result_dict = {}
    importance = rfc.feature_importances_
    std_importance=np.array(importance)
    
    for i in range(len(col)):
        importances_result_dict[col[i]] = importance[i]
    #字典排序
    result = sorted(importances_result_dict.items(), key=lambda item: item[1], reverse=True)
    im = []#因子
    score = []#分数
    for i in range(len(col)):
        im.append(result[i][0])
        score.append(result[i][1])
    
    # plt.figure(figsize=(20,10))
    # plt.subplot(s)
    plt.xticks(list(map(lambda x, y: x + y, list(range(n)), [0.5] * n)), im, rotation=90)
    plt.bar(range(n), score[:n])


def train_max_index(train_data, test_data,col):
    '''
    训练最优模型
    :param train_data: 训练数据集
    :param test_data: 测试数据集
    :return: 最优模型的随机种子编号
    '''
    max_index = 0  # 存储最大真阳率的random_seed
    max_zhenyang = 0
    for i in range(200):

        rfc = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=i)
        rfc.fit(train_data[col], train_data['Y_bin'])
        t1 = rfc.predict(test_data[col])

        zhenyang= ROC(t1, np.array(test_data['Y_bin']))
        if max_zhenyang < zhenyang:
            max_index = i
            max_zhenyang = zhenyang
    return max_index


def ROC(prediction, test, trueclass=1, flaseclass=0): 
    '''
    :param prediction: 预测标志
    :param test: 真实标志
    :param trueclass: 视为真的类别值
    :param flaseclass: 视为假的类别值
    :return: 真阳率
    '''
    TP = 0  # 预测真为真
    FN = 0  # 预测真为假
    FP = 0  # 预测假为真
    TN = 0  # 预测假为假

    for i in range(np.shape(test)[0]):
        if prediction[i] == trueclass and test[i] == trueclass:
            TP = TP + 1
        elif prediction[i] == flaseclass and test[i] == trueclass:
            FN = FN + 1
    return TP / (TP + FN)

def main_(stockindex,train_startday,train_endday,test_endday,label_type,stockpool_type,col,n):

    test_startday=train_endday
    
    train_data= dealdata(dataset(train_startday,train_endday,70,label_type,stockindex,stockpool_type))
    test_data=dealdata(dataset(test_startday,test_endday,70,label_type,stockindex,stockpool_type))
   
    max_index=train_max_index(train_data,test_data,col)  

    plt.title(train_startday+' '+test_endday+' '+stockindex)
    print_(max_index,111,train_data,col,n)


# 一、针对所有A股使用模型
# 1️⃣以去年5月份白马股上涨那段时间为例，模型结束时间定为7月份。可以发现很明显市值因子被区分出来，而其他因子区分不明显。
# 这结果也证明了模型的有效性。选出来的第一位市值因子，第二位收益因子，正好符合了白马股的特征，所以可以说明市场确实可以被算法分析出来。

# In[ ]:

col = ['LFLO','PE','PB','ROE','CTOP','ETOP','PCF','FY12P','TA2EV','CurrentAssetsRatio','DebtsAssetRatio','NPToTOR','OperatingProfitToTOR']
main_(stockindex='A',train_startday='2017-05-03',train_endday='2017-07-03',test_endday='2017-09-20',label_type=1,stockpool_type=1,col=col,n=len(col))


# 2️⃣那么将时间周期拉长，以年为单位（时间还得看着日历去找工作日，比较麻烦）。可以发现时间拉长，因子区分更明显。因此可以这么认为股市长线趋势要比短线明朗。

# In[ ]:

col = ['LFLO','PE','PB','ROE','CTOP','ETOP','PCF','FY12P','TA2EV','CurrentAssetsRatio','DebtsAssetRatio','NPToTOR','OperatingProfitToTOR']
main_(stockindex='A',train_startday='2015-07-03',train_endday='2016-07-04',test_endday='2017-07-04',label_type=1,stockpool_type=1,col=col,n=len(col))


# 3️⃣所以我们可以放入更多的因子，来寻找比市值区分度更高因子。

# In[ ]:

train_data= dealdata(dataset('2015-07-03','2016-07-04',70,1,'A',1))
col = [c for c in train_data.columns if c not in ['secID','tradeDate','Return', 'Y_bin']]
main_(stockindex='A',train_startday='2015-07-03',train_endday='2016-07-04',test_endday='2017-07-04',label_type=1,stockpool_type=1,col=col,n=30)


# 二、模型对于不同板块的使用（就不列出来了）
# 1️⃣创业板
# 2️⃣沪深300
# 3️⃣细分行业（这个会更明显，因此可以说明这个模型更适合用于细分行业后进行因子筛选。优矿输入行业股票池跟输入指数string又不一样，还得去改函数，就先不跑了）

# In[ ]:

三、混沌模型
既然模型可以反映出当前市场的主导因子，那么如果当前市场的因子区分度很高，则说明市场是有主线的，若不高则是无主线的。那么既然如此，市场有无主线和市场未来的走势是否有关系呢？（简单地猜想一下，大盘每次筑顶都是横盘，而横盘则是市场失去主线的过程，如果判断出市场是没主线的横盘，那么下跌的可能性就会加大）。那么现在来验证一下

#下面要重新写一下函数


# In[ ]:

import datetime
import time
import pandas as pd
from numpy import linalg as la
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy as np
import random
import matplotlib.pyplot as plt


def dealdata(data):
    '''
    数据预处理
    :param data: 未经处理的初始数据集
    :return: 处理好的数据集
    '''
    col = [c for c in data.columns if c not in ['secID','tradeDate','Return', 'Y_bin']]

    for i in col:
        data[i].fillna(data[i].mean(), inplace=True)
    data['Y_bin'].fillna(0, inplace=True)
    data['Return'].fillna(0, inplace=True)
    
    return data


def dataset(startdate, enddate, window, label_type, stockindex,stockpool_type):
    '''
    获取数据

    :param startdate:需要获取的数据开始日期
    :param enddate:需要获取的数据结束日期
    :param window:涨幅分类的阈值百分比，大于window的为正类，小于为负类
    :param label_type:使用哪种分类算法。1为按涨幅比例分，0为按涨跌与否分
    :param stockindex:哪个股票池的数据
    :param stockpool_type:使用哪种股票池。1为上证指数等大股票池，0为行业股票池
    :return: 列为各因子的值，行为startdate当天的所有股票代码
    '''
    stock_l=set_universe(stockindex)
    # 获取股票池开始日期的因子值
    fdf=DataAPI.MktStockFactorsOneDayGet(tradeDate=startdate,secID=stock_l,field=u"",pandas="1")
    #修改Index（为了后面的merge）
    fdf.index = fdf['ticker']
    fdf.pop('ticker')
    # 获取股票列表中的股票在所设置时间区间的收盘价
    current_close = DataAPI.MktEqudGet(secID=stock_l,tradeDate=startdate,pandas="1",field=['secID','preClosePrice'])
    stock_1=np.array(current_close['secID']).tolist()
    forcast_close = DataAPI.MktEqudGet(secID=stock_l,tradeDate=enddate,pandas="1",field=['secID','preClosePrice'])
    stock_2=np.array(forcast_close['secID']).tolist()
    # 为了让股票池在两个时间段内对齐，设置统一的stock_set
    stock_set=list(set(stock_1)&set(stock_2))
    #计算两个日期内的涨幅
    current_close = DataAPI.MktEqudGet(secID=stock_set,tradeDate=startdate,pandas="1",field=['ticker','preClosePrice'])
    forcast_close = DataAPI.MktEqudGet(secID=stock_set,tradeDate=enddate,pandas="1",field=['ticker','preClosePrice'])
    grow = list((forcast_close.iloc[:, 1] - current_close.iloc[:, 1]) / current_close.iloc[:, 1])
    
    grow = pd.DataFrame(grow, columns=['Return'])
    grow.index=current_close['ticker']
    # 拼接
    df = pd.merge(fdf, grow, left_index=True, right_index=True)
    
    # 选择第一种阈值计算类型（前百分之window的涨幅为1类，之后的为0类）
    if label_type == 1:  
        bound = np.nanpercentile(df['Return'], window)
        # if bound<0:
        #     bound=0
        df.loc[(df['Return'] >= bound ), 'Y_bin'] = 1
        df.loc[(df['Return'] < bound), 'Y_bin'] = 0
    else:  # 涨的为1类，跌的为0类（用于短期阈值比较好）
        bound = 0
        df.loc[(df['Return'] >= bound), 'Y_bin'] = 1
        df.loc[(df['Return'] < bound), 'Y_bin'] = 0
    return df


def print_(random_seed, s, train_data,col,n):
    '''
    作图
    :param random_seed: 最优的模型的随机种子编号
    :param s: 图的位置
    :param train_data: 训练数据集
    :param n:画出排名最前的n个因子
    '''
    rfc = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=random_seed)
    rfc.fit(train_data[col], train_data['Y_bin'])
    importances_result_dict = {}
    importance = rfc.feature_importances_
    std_importance=np.array(importance)
    return np.std(std_importance)
#     for i in range(len(col)):
#         importances_result_dict[col[i]] = importance[i]
#     #字典排序
#     result = sorted(importances_result_dict.items(), key=lambda item: item[1], reverse=True)
#     im = []#因子
#     score = []#分数
#     for i in range(len(col)):
#         im.append(result[i][0])
#         score.append(result[i][1])
    
#     plt.figure(figsize=(20,10))
#     plt.subplot(s)
#     plt.xticks(list(map(lambda x, y: x + y, list(range(n)), [0.5] * n)), im, rotation=90)
#     plt.bar(range(n), score[:n])


def train_max_index(train_data, test_data,col):
    '''
    训练最优模型
    :param train_data: 训练数据集
    :param test_data: 测试数据集
    :return: 最优模型的随机种子编号
    '''
    max_index = 0  # 存储最大真阳率的random_seed
    max_zhenyang = 0
    for i in range(100):

        rfc = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=i)
        rfc.fit(train_data[col], train_data['Y_bin'])
        t1 = rfc.predict(test_data[col])

        zhenyang= ROC(t1, np.array(test_data['Y_bin']))
        if max_zhenyang < zhenyang:
            max_index = i
            max_zhenyang = zhenyang
    return max_index


def ROC(prediction, test, trueclass=1, flaseclass=0): 
    '''
    :param prediction: 预测标志
    :param test: 真实标志
    :param trueclass: 视为真的类别值
    :param flaseclass: 视为假的类别值
    :return: 真阳率
    '''
    TP = 0  # 预测真为真
    FN = 0  # 预测真为假
    FP = 0  # 预测假为真
    TN = 0  # 预测假为假

    for i in range(np.shape(test)[0]):
        if prediction[i] == trueclass and test[i] == trueclass:
            TP = TP + 1
        elif prediction[i] == flaseclass and test[i] == trueclass:
            FN = FN + 1
    return TP / (TP + FN)

def main_(stockindex,train_startday,train_endday,test_endday,label_type,stockpool_type,col,n):

    test_startday=train_endday
    
    train_data= dealdata(dataset(train_startday,train_endday,70,label_type,stockindex,stockpool_type))
    test_data=dealdata(dataset(test_startday,test_endday,70,label_type,stockindex,stockpool_type))
   
    max_index=train_max_index(train_data,test_data,col)  

    # plt.title(train_startday+' '+test_endday+' '+stockindex)
    std=print_(max_index,111,train_data,col,n)
    return std


# In[ ]:

A_close=DataAPI.MktIdxdGet(indexID=u"000001.ZICN",beginDate=u"20131219",endDate=u"20171229",field=['tradeDate','preCloseIndex'],pandas="1")


# In[ ]:

col = ['LFLO','PE','PB','ROE','CTOP','ETOP','PCF','FY12P','TA2EV','CurrentAssetsRatio','DebtsAssetRatio','NPToTOR','OperatingProfitToTOR']
close=np.array(A_close['preCloseIndex']).tolist()
date_list=np.array(A_close['tradeDate']).tolist()
std_list=[]
close_list=[]
for i in range(60,len(date_list)):
    train_startday=date_list[i-60]
    train_endday=date_list[i-30]
    test_endday=date_list[i]
    std=main_(stockindex='A',train_startday=train_startday,train_endday=train_endday,test_endday=test_endday,label_type=1,stockpool_type=1,col=col,n=30)
    std_list.append(std)
    close_list.append(close[i])
    print(date_list[i])


# In[ ]:

#存一下结果
last_result=pd.DataFrame()
last_result['date']=date_list[60:]
last_result['hundun_value']=std_list
last_result['A_close']=close_list
last_result.to_csv('hundun_2014-2017.csv')


# In[ ]:

D=1
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(111)
p1,=ax1.plot(range(len(std_list[::D])),std_list[::D],color='g')
ax2 = ax1.twinx()  # this is the important function
p2,=ax2.plot(range(len(close_list[::D])),close_list[::D],color='r')


# 上图所示就是以两个月为区间做出来的混沌指标（参数还没有进行优化，在这里只是用目前做过的参数作为例子），可以发现在牛市的后半期，指标显示当时的市场是没有主导因子的，因此有可能那时候就是场外资金疯狂进场推动股市的上涨，后来场外资金越来越来少，加上市场本来就没有主线，大家不懂买啥就出逃了，最后造成股价的下跌。那么假如在牛市后半段参考了我的这个指标，就会意识到当前的市场是没有主线的，要随时做好减仓清仓的准备。（后面3600的那一波下跌就不分析了）。所以我认为，将参数优化后，一定能更准确地去反映当前市场情况，从而可以依次做一些止损策略。
# 
# 
# 不足：
# 现在的模型参数都是没有优化过的，只是说我这么做让我意识到好像确实是可以用机器学习分析、描述当前市场，那这么想的话，以后研究的方向就有很多了。最近课程比较多，也比较忙就没有弄。
# 
# 
# 最后说一下优矿和聚宽的不同
# 因为优矿我也没用多久，所以只能从自身经历简单地谈一下自己的看法
# 
# 1.聚宽获取历史数据的时候如果输入日期没有开市则返回日期之前的最近一次交易日的数据，优矿遇到这种情况还要再处理就比较麻烦，对于一个新手来说，先不会去想聚宽这么处理数据严不严谨，而是会觉得很人性化，对比一下可能就会选择更方便的那个。当然我现在为了确保结果的准确性、严谨性，还是更倾向于优矿提供原始数据。
# 2.优矿因子没有分类，想做针对性训练比较麻烦。
# 3.API的设计上，参数有些冗余，聚宽一个code就表示的，优矿还要secid,ticker等
# 4.优矿数据确实是多，以至于要用很多的函数，但对于一个新手的话其实用不到那么多，新手看到这么多数据应该会觉得很乱，无从下手吧。
# 5.在使用的时候发现优矿的函数存在比较多的bug，而且说明文档不全
# 例如：stock_index_start=set(set_universe(stockindex,startdate))这个函数日期根本不起作用....还是返回最新的股票池
# 6.缺少神经网络库，聚宽有（优矿出于安全考虑确实也能理解）
# 7.跑着跑着就突然开始初始化。。。。
# 
# 最后的最后，还是非常希望成为优矿的量化实习研究员，我知道自己懂得不多，但我非常愿意花大量的时间去学去研究，希望大佬给次机会~
