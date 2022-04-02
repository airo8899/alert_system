import telegram
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import pandas as pd
import pandahouse
from scipy.stats import t

connection = {
    'host': 'https://clickhouse.lab.karpov.courses',
    'password': 'dpo_python_2020',
    'user': 'student',
    'database': 'simulator_20220320'
}

def select(q):
    return pandahouse.read_clickhouse(q, connection=connection)

sns.set_style('darkgrid')
sns.set_palette('bright')
sns.set_context('notebook')

def check_anomaly_CI(df, metric, alpha=0.01, n=5):
    df['mean'] = df[metric].shift(1).rolling(n).mean()
    df['std'] = df[metric].shift(1).rolling(n).std()
    df['lower'] = t.ppf((alpha/2, 1 - alpha/2), df=n-1, loc=df[['mean']], scale=df[['std']])[:, 0]
    df['up'] = t.ppf((alpha/2, 1 - alpha/2), df=n-1, loc=df[['mean']], scale=df[['std']])[:, 1]
    df['lower'] = df['lower'].rolling(3, center=True, min_periods=1).mean()
    df['up'] = df['up'].rolling(3, center=True, min_periods=1).mean()
    
    
    if df[metric].iloc[-1] < df['lower'].iloc[-1] or df[metric].iloc[-1] > df['up'].iloc[-1]:
        alert_flag = 1
    else:
        alert_flag = 0
    
    return alert_flag, df

def check_anomaly_IQR(df, metric, a=1.5):
    iqr = df[metric].quantile(0.75) - df[metric].quantile(0.25)
    up = df[metric].quantile(0.75) + a * iqr 
    lower = df[metric].quantile(0.25) - a * iqr
    avg = df[metric].mean()
    
    if df[metric].iloc[-1] > up or df[metric].iloc[-1] < lower:
        lert_flag = 1
    else:
        alert_flag = 0
        
    return alert_flag, df, lower, up, avg

def run_alerts(chat=None):
    # chat_id = chat or 453565850
    chat_id = chat or -1001706798154
    bot = telegram.Bot(token='5167010511:AAETy3cSIsBkRmmrI-4DmhMTVurzlwfVLi4')
    # bot = telegram.Bot(token=os.environ.get("REPORT_BOT_TOKEN"))
    
    data = select("""
    SELECT toStartOfFifteenMinutes(time) ts,
          toDate(time) date,
          formatDateTime(ts, '%R') hm,
          uniqExact(user_id) users_feed,
          countIf(user_id, action='view') views,
          countIf(user_id, action='like') likes
    FROM simulator_20220320.feed_actions
    WHERE time >= today() AND time < toStartOfFifteenMinutes(now())
    GROUP BY ts, date, hm
    ORDER BY ts""")
    
    data_message = select("""
    SELECT uniqExact(user_id) users_message, 
          toStartOfFifteenMinutes(time) ts,
          toDate(time) date,
          formatDateTime(ts, '%R') hm
    FROM simulator_20220320.message_actions
    WHERE time >= today() AND time < toStartOfFifteenMinutes(now())
    GROUP BY ts, date, hm
    ORDER BY ts""")
    
    data['users_message'] = data_message['users_message']
    data.columns = ['ts', 'date', 'hm', 'Пользователи ленты новостей', 'Просмотры', 'Лайки', 'Пользователи сервиса сообщений']
    
    metrics_list = ['Пользователи ленты новостей', 'Просмотры', 'Лайки', 'Пользователи сервиса сообщений']

    for metric in metrics_list:
        df = data[['ts', 'date', 'hm', metric]].copy()
        is_alert, df = check_anomaly_CI(df, metric)
        
        if is_alert or True:
            msg = f'''Метрика {metric}:
    текущее значение - {df[metric].iloc[-1]},
    отклонение от предыдущего значения - {abs(1 - df[metric].iloc[-1]/df[metric].iloc[-2]) * 100:.2f}%,
    <a href="https://superset.lab.karpov.courses/superset/dashboard/589/">Смотреть на дашборде</a>'''
                
            sns.set(rc={'figure.figsize':(15, 10)})
            plt.tight_layout()

            ax = sns.lineplot(x=df['ts'], y=df[metric], label=metric)
            ax = sns.lineplot(x=df['ts'], y=df['lower'], label='lower', linestyle='--')
            ax = sns.lineplot(x=df['ts'], y=df['up'], label='up', linestyle='--')

            ax.set(xlabel='time', ylabel=metric, title=metric, ylim=(0, None));

            fig_object = io.BytesIO()
            plt.savefig(fig_object)
            fig_object.name = 'report.png'
            fig_object.seek(0)
            plt.close()
            

            bot.sendMessage(chat_id=chat_id, text=msg, parse_mode='HTML')
            bot.sendPhoto(chat_id=chat_id, photo=fig_object)
            
            
        
        
        
        
    data = select("""
    SELECT toStartOfFifteenMinutes(time) ts,
          uniqExact(user_id) users_feed,
          countIf(user_id, action='view') views,
          countIf(user_id, action='like') likes
    FROM simulator_20220320.feed_actions
    WHERE time > today()-7 AND time < toStartOfFifteenMinutes(now()) 
          AND formatDateTime(toStartOfFifteenMinutes(time), '%R') >= formatDateTime(toStartOfFifteenMinutes(date_add(minute, -75, now())), '%R') 
          AND formatDateTime(toStartOfFifteenMinutes(time), '%R') < formatDateTime(toStartOfFifteenMinutes(date_add(hour, 1, now())), '%R') 
    GROUP BY ts
    ORDER BY ts""")

    data_message = select("""
    SELECT toStartOfFifteenMinutes(time) ts,
          uniqExact(user_id) users_message
    FROM simulator_20220320.message_actions
    WHERE time > today()-7 AND time < toStartOfFifteenMinutes(now()) 
          AND formatDateTime(toStartOfFifteenMinutes(time), '%R') >= formatDateTime(toStartOfFifteenMinutes(date_add(minute, -75, now())), '%R') 
          AND formatDateTime(toStartOfFifteenMinutes(time), '%R') < formatDateTime(toStartOfFifteenMinutes(date_add(hour, 1, now())), '%R')
    GROUP BY ts
    ORDER BY ts""")

    data['users_message'] = data_message['users_message']
    data.columns = ['ts', 'Пользователи ленты новостей', 'Просмотры', 'Лайки', 'Пользователи сервиса сообщений']

    for metric in metrics_list:
        df = data[['ts', metric]].copy()
        is_alert, df, lower, up, avg = check_anomaly_IQR(df, metric)
        
        if is_alert or True:
            msg = f'''Метрика {metric}:
    текущее значение - {df[metric].iloc[-1]},
    отклонение от среднего значения - {abs(1 - df[metric].iloc[-1]/avg) * 100:.2f}%,
    <a href="https://superset.lab.karpov.courses/superset/dashboard/589/">Смотреть на дашборде</a>'''
            
            
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,12))
            plt.suptitle(metric)

            sns.lineplot(x=df['ts'].iloc[-5:], y=df[metric].iloc[-5:], ax=ax[0], marker='o', label=metric)
            ax[0].set
            ax[0].axhline(lower, color='red', label='lower')
            ax[0].axhline(up, color='green', label='up')
            ax[0].legend()
            ax[0].set_xlabel('time')
            ax[0].set_xticks(df['ts'].iloc[-5:])
            ax[0].set_xticklabels(df['ts'].iloc[-5:].dt.strftime('%d %b %H:%M'))

            sns.kdeplot(df[metric], ax=ax[1], shade=True)
            ax[1].axvline(lower, color='red', label='lower')
            ax[1].axvline(up, color='green', label='up')
            ax[1].axvline(df[metric].iloc[-1], color='black', label=metric)
            ax[1].legend()
            
            plt.tight_layout();
            fig_object = io.BytesIO()
            plt.savefig(fig_object)
            fig_object.name = 'report.png'
            fig_object.seek(0)
            plt.close()
            

            bot.sendMessage(chat_id=chat_id, text=msg, parse_mode='HTML')
            bot.sendPhoto(chat_id=chat_id, photo=fig_object)


try:
    run_alerts()
except Exception as e:
    print(e)

