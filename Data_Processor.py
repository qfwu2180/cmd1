import numpy as np
import pandas as pd
import math
from sklearn import preprocessing



class Stock_Data(object):
    #对每一次股票数据进行处理
    def __init__(self,stock_code,input_size,num_steps,train_ratio,interval,future):
        '''
        一个Stock_Data对象对应一只股票历史数据，在Stock_count中将股票历史交易数据处理成模型需要的训练集和测试集
        :param stock_code: 股票的编码
        :param input_size: 决定target的涨跌幅是几日的平均，例如涨跌幅取一个月后五天内的平均涨跌幅，input_size=5
        :param num_steps: 决定输入LSTM神经网络的时序长度
        :param train_ratio: 训练集占所有样本的比率
        :param interval: interval用于将收盘价处理成interval天内的涨跌幅
        :param future: future表示用来预测多少天之后的涨跌幅
        '''
        self.stock_code=stock_code
        self.input_size=input_size
        self.num_steps=num_steps
        self.train_ratio=train_ratio
        self.interval=interval
        #interval表示将价格换成涨跌率的时间间隔，比如将50天之内的价格换成涨跌率interval=50，就是用50天内的价格去除以50天之前的最后一天的价格
        self.future=future
        #future表示取多久之后的平均值，比如预测30个交易日之后的10日平均值，future=30，input_size=10
        self.data_x=[]
        self.data_y=[]

        df=pd.read_csv('Data\\交易数据\\'+stock_code+'.csv')

        '''
        将价格每interval天计算成涨跌率，
        '''
        close=df['close']
        self.close_changed = []
        for i in range(int(math.ceil(len(close) / interval))):
            if i == 0:
                for j in range(interval):
                    self.close_changed += [(close[i * interval + j] - close[i * interval]) / close[i * interval]]
            elif i != 0 and (len(close) - i * interval) > interval:
                for j in range(interval):
                    self.close_changed += [(close[i * interval + j] - close[i * interval - 1]) / close[
                        i * interval - 1]]  # -1是为了取这interval天之前最后一天的
            elif i != 0 and (len(close) - i * interval) < interval:
                for j in range(len(close) - i * interval):
                    self.close_changed += [(close[i * interval + j] - close[i * interval - 1]) / close[i * interval - 1]]



        '''数据归一化'''
        self.trans_close = (self.close_changed[:] - min(self.close_changed[:])) / (max(self.close_changed[:])-min(self.close_changed[:]))
        self.trans_vol = (df['vol'][:] - min(df['vol'][:])) / (max(df['vol'])-min(df['vol']))

        '''用未归一化的数据来构建模型targets'''
        for i in range(len(self.close_changed) - num_steps - self.input_size - self.future):
            change_y = 0
            for j in range(input_size):
                change_y += self.close_changed[i + num_steps + self.future - j]
            self.data_y += [[change_y / self.input_size]]
        self.data_y = np.array(self.data_y)

        self.data=[[self.trans_close[i],self.trans_vol[i]] for i in range(len(self.trans_close))]
        self.data=preprocessing.MinMaxScaler().fit_transform(self.data)
        self.data_x=np.array([np.array(self.data[i:i+num_steps]) for i in range(len(self.data)-num_steps-self.input_size-self.interval)])
        '''划分训练集测试集'''
        self.train_x,self.test_x = np.split(self.data_x,[int(len(self.data_x)*self.train_ratio)])
        self.train_y,self.test_y = np.split(self.data_y,[int(len(self.data_y)*self.train_ratio)])


    def get_fluctuation(self):
        '''
        获取每一只股票在历史数据上的涨跌幅波动
        '''
        fluctuation=max(self.close_changed[:])-min(self.close_changed[:])
        return fluctuation


def get_stocks(input_size,num_steps,train_ratio,interval,future):
    '''
    获取每只股票对应的Stock_Data对象，读取数据文件中的股票数据
    :return: 每只股票的训练集测试集，股票名，历史数据中涨跌幅的波动
    '''
    #获取文件中所有股票
    stocks_df=pd.DataFrame.from_csv('Data/交易数据/download_stocks_symbol.csv')
    stock_name=stocks_df['code'].values.tolist()
    #处理成Stock_Data对象
    stock_data=[
        Stock_Data(stock_code=code,input_size=input_size,num_steps=num_steps,train_ratio=train_ratio,interval=interval,future=future)
        for code in stock_name
    ]
    #计算所有股票的平均波动
    sum_fluctuation=0
    for code in stock_name:
        sum_fluctuation+=Stock_Data(stock_code=code,input_size=input_size,num_steps=num_steps,train_ratio=train_ratio,interval=interval,future=future).get_fluctuation()
    mean_fluctuation=sum_fluctuation/len(stock_name)

    return stock_name,stock_data,mean_fluctuation
