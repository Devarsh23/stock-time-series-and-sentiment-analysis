import pytz
import requests
import pandas as pd


def stocktwits_scrap(ticker, base=107048252):
    id = []
    created_at = []
    username = []
    sentiment = []
    body = []
    url = "https://api.stocktwits.com/api/2/streams/symbol/" + ticker + ".json?&since="+str(base)
    response = requests.request("GET", url, headers={}, data={})
    response = response.json()
    if response['response']['status'] == 404:
        print("Not a valid symbol")
    messages = response['messages']
    iter = range(len(messages))
    for idx in iter:
        id.append(messages[idx]['id'])
        username.append(messages[idx]['user']['username'])
        created_at.append(messages[idx]['created_at'])
        if messages[idx]['entities']['sentiment'] is not None:
            sentiment.append(messages[idx]['entities']['sentiment']['basic'])
        else:
            sentiment.append('')
        body.append(messages[idx]['body'])

    df_stocktwits = pd.DataFrame(id, columns=['id'])
    df_stocktwits['username'] = username
    df_stocktwits['created_at'] = created_at
    df_stocktwits['created_at'] = pd.to_datetime(df_stocktwits['created_at'])
    df_stocktwits['sentiment'] = sentiment
    df_stocktwits['body'] = body
    df_stocktwits = df_stocktwits.sort_values(by='created_at')
    df_stocktwits['created_at'] = df_stocktwits['created_at'].apply(lambda x: x.astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z%z'))
    df_stocktwits = df_stocktwits.reset_index(drop=True)
    if len(df_stocktwits) > 0:
        last_id = df_stocktwits['id'].iat[-1]
        return df_stocktwits, last_id
    else:
        return [], 0


def main():
    tickerList = ['AAOI']
    for ticker in tickerList:
        filename = ticker.lower()
        df_stocktwits, last_id = stocktwits_scrap(ticker=ticker)
        df_stocktwits.to_csv('%s_api_data.csv' % filename, index=False)
        for i in range(1000):
            df_stocktwits, last_id = stocktwits_scrap(ticker=ticker, base=last_id)
            if last_id != 0:
                df_stocktwits.to_csv('%s_api_data.csv' % filename, mode='a', index=False, header=False)
                print("Scraped: " + str(i*30) + " twits")
        # df = pd.read_csv('%s_api_data.csv' % filename)
        # df = df.iloc[::-1]
        # df.to_csv('%s_api_data.csv' % filename, index=False)


if __name__ == '__main__':
    main()
