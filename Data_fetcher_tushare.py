import tushare as ts
import pandas as pd
import random,time

'''使用tushare API需要提供的指令'''
tushare_token='9d318f4f16fc290756e756ddca50bdaabda9d3d98698345f4e505768'

def get_stock_symbol():
    '''获取当日所有上市股票信息，并存于文件中'''
    pro=ts.pro_api(tushare_token)
    stocks_info=pro.query('stock_basic',exchange='',list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    stocks_info.to_csv('Data\\stocks_info.csv')
    return stocks_info


def random_download_stocks(stock_count):
    '''
    该函数用来随机下载stock_count数量的股票
    :param stock_count: 是要下载的股票数量
    '''
    ts.set_token(tushare_token)
    Stocks=pd.DataFrame.from_csv('Data\\stocks_info.csv')
    length=len(Stocks)
    print('共有',length,'只股票，现在随机下载',stock_count,'只！')

    '''处理时间，将开始时间处理成当日的三年前的同一天'''
    now = time.localtime()
    year = now.tm_year - 3
    start = str(year)
    if now.tm_mon < 10:
        start = start + '0' + str(now.tm_mon)
    else:
        start = start + str(now.tm_mon)
    if now.tm_mday < 10:
        start = start + '0' + str(now.tm_mday)
    else:
        start = start + str(now.tm_mday)
    '''下载股票'''
    success_stocks=pd.DataFrame(columns=['code','name'])
    for i in range(stock_count):
        df=pd.DataFrame()
        while df.empty:
            index = random.randint(1, 1000000) % length
            ts_code, name = Stocks.iloc[index][['ts_code', 'name']]
            df = ts.pro_bar(ts_code=ts_code, start_date=start, adj='qfq')
            lastday=time.strptime(df.iloc[[df.shape[0]-1]]['trade_date'].values[0],'%Y%m%d')
            if lastday.tm_year==year and (lastday.tm_mon==now.tm_mon or lastday.tm_mon==now.tm_mon+1):
                #判断获取的股票在三年前有没有交易数据
                pass
            else:
                df=pd.DataFrame()
        df.to_csv('Data\\交易数据\\'+ts_code+'.csv')
        #存于文件中
        success_stocks.loc[success_stocks.shape[0]+1]={'code':ts_code,'name':name}
    success_stocks.to_csv('Data\\交易数据\\download_stocks_symbol.csv')
