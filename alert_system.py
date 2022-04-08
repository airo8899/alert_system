import telegram
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import pandas as pd
import pandahouse
from scipy.stats import t
# from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# подключение к бд
connection = {
    'host': 'https://clickhouse.lab.karpov.courses',
    'password': 'dpo_python_2020',
    'user': 'student',
    'database': 'simulator_20220320'
}

# функция запроса к бд
def select(q):
    return pandahouse.read_clickhouse(q, connection=connection)

# установка стилей графиков
sns.set_style('darkgrid')
sns.set_palette('bright')
sns.set_context('notebook')


# функция проверки аномалий с помощью доверительных интервалов
# параметры:
# - df -- таблица со значениями метрик
# - metric -- название метрики
# - alpha -- уровень значимости для вычисления доверительного интервала метрики
# - n -- количество предыдущих промежутков времени для вычисления доверительного интервала
# возвращает явлется ли последние значение метрики выбросом, а также таблицу данных с доверительными интервалами
def check_anomaly_CI(df, metric, alpha=0.01, n=6):
    # расчет скользящего среднего и скользящего среднего отклонения за n периодов
    df['mean'] = df[metric].shift(1).rolling(n).mean()
    df['std'] = df[metric].shift(1).rolling(n).std()
    
    # расчет доверительных интервалов как CI = mean ± T * std, где T - величина смещения нормального распределения для alpha
    df['lower'] = t.ppf((alpha/2, 1 - alpha/2), df=n-1, loc=df[['mean']], scale=df[['std']])[:, 0]
    df['up'] = t.ppf((alpha/2, 1 - alpha/2), df=n-1, loc=df[['mean']], scale=df[['std']])[:, 1]
    
    # сглаживание линий доверительных интервалов за n периодов
    df['lower'] = df['lower'].rolling(n, center=True, min_periods=1).mean()
    df['up'] = df['up'].rolling(n, center=True, min_periods=1).mean()
    
    # если последние значение метрики выходит за доверителный интервал - возвращается 1, если нет - 0
    if df[metric].iloc[-1] < df['lower'].iloc[-1] or df[metric].iloc[-1] > df['up'].iloc[-1]:
        alert_flag = 1
    else:
        alert_flag = 0
    
    return alert_flag, df



# функция проверки аномалий с помощью межквартильного размаха
# параметры:
# - df -- таблица со значениями метрик
# - metric -- название метрики
# - a -- коэффициент настройки чувствительности срабатывания теста
# возвращает 
# -явлется ли последние значение метрики выбросом,
# -таблицу данных 
# - нижнюю и верхнюю границу значений
# - среднее значение метрики
def check_anomaly_IQR(df, metric, a=2):
    # получение межквартильного размаха, нижней и верхней границ значений
    iqr = df[metric].quantile(0.75) - df[metric].quantile(0.25)
    up = df[metric].quantile(0.75) + a * iqr 
    lower = df[metric].quantile(0.25) - a * iqr
    
    # среднее значение метрики
    avg = df[metric].mean()
    
    # если последние значение метрики выходит за значения нижней и верхней границ - возвращается 1, если нет - 0
    if df[metric].iloc[-1] > up or df[metric].iloc[-1] < lower:
        alert_flag = 1
    else:
        alert_flag = 0
        
    return alert_flag, df, lower, up, avg



# функция проверки аномалий с помощью алгоритма DBSCAN
# параметры:
# - df -- таблица со значениями метрик
# - metric -- название метрики
# - a -- коэффициент настройки чувствительности срабатывания теста
# - n -- параметр min_samples для алгоритма DBSCAN
# возвращает 
# - явлется ли последние значение метрики выбросом
# - таблицу данных 
# - нижнюю и верхнюю границу значений
# - среднее значение метрики
def check_anomaly_DBSCAN(df, metric, a=1.8, n=5):
    # получение расстояний между ближащими точками набора данных с помощью NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=2).fit(df[[metric]])
    distances, _ = nbrs.kneighbors(df[[metric]])
    
    # предсказание DBSCAN выбросов на метрике
    # eps берется как максимальное расстояние между двух ближащих точках, умножениое на коэффициент a
    dbscan = DBSCAN(eps = a*distances.max(), min_samples = n)
    pred = dbscan.fit_predict(df[[metric]])
    
    # будем считать, что все кластеры относятся к одному кластеру
    pred[pred != -1] = 0
    
    # для рассчета границ n-ое значение метрики с начала и конца отсортированной выборки 
    # и прибавим максимальное расстояние между двух ближащих точках, умножениое на коэффициент a
    df_temp = df[metric][pred == 0].sort_values() 
    lower = df_temp.iloc[n-1] - a * distances.max()
    up = df_temp.iloc[-1*(n)] + a * distances.max()
    
    avg = df[metric].mean()
    
    # если последние значение метрики является выбросом - возвращается 1, если нет - 0
    if pred[-1] == -1:
        alert_flag = 1
    else:
        alert_flag = 0
    
    return alert_flag, lower, up, avg






def run_alerts(chat=None):
    # chat_id = chat or 453565850
    chat_id = chat or -1001706798154
    
    bot = telegram.Bot(token=os.environ.get("REPORT_BOT_TOKEN"))
    
    # получение времени, даты, DAU, число просмотров, лайков ленты новостей за 15 минутные интервалы сегодняшнего дня
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
    
    # получение времени, даты, DAU сервиса сообщений за 15 минутные интервалы сегодняшнего дня
    data_message = select("""
    SELECT uniqExact(user_id) users_message, 
          toStartOfFifteenMinutes(time) ts,
          toDate(time) date,
          formatDateTime(ts, '%R') hm
    FROM simulator_20220320.message_actions
    WHERE time >= today() AND time < toStartOfFifteenMinutes(now())
    GROUP BY ts, date, hm
    ORDER BY ts""")
    
    # объединим оба запроса в одну таблицу, переименуем названия столбцов
    data['users_message'] = data_message['users_message']
    data.columns = ['ts', 'date', 'hm', 'Пользователи ленты новостей', 'Просмотры', 'Лайки', 'Пользователи сервиса сообщений']
    
    # определение метрик для проверки
    metrics_list = ['Пользователи ленты новостей', 'Просмотры', 'Лайки', 'Пользователи сервиса сообщений']

    # для каждой метрики из таблицы проведем проверку последнего значения на аномалию с помощью функции check_anomaly_CI
    for metric in metrics_list:
        df = data[['ts', 'date', 'hm', metric]].copy()
        is_alert, df = check_anomaly_CI(df, metric)
        
        # если проверка определила выброс, то формируется сообщение и графиик значения метрики и доверительных интервалов
        if is_alert:
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
            
            # отпавка сообщения и графика
            bot.sendMessage(chat_id=chat_id, text=msg, parse_mode='HTML')
            bot.sendPhoto(chat_id=chat_id, photo=fig_object)
            
            
        
        
        
    # получение времени, DAU, число просмотров, лайков ленты новостей за 15 минутные интервалы в диапазоне ±1 часа от текущего времени за неделю  
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

    # получение времени, DAU сервиса сообщений за 15 минутные интервалы в диапазоне ±1 часа от текущего времени за неделю  
    data_message = select("""
    SELECT toStartOfFifteenMinutes(time) ts,
          uniqExact(user_id) users_message
    FROM simulator_20220320.message_actions
    WHERE time > today()-7 AND time < toStartOfFifteenMinutes(now()) 
          AND formatDateTime(toStartOfFifteenMinutes(time), '%R') >= formatDateTime(toStartOfFifteenMinutes(date_add(minute, -75, now())), '%R') 
          AND formatDateTime(toStartOfFifteenMinutes(time), '%R') < formatDateTime(toStartOfFifteenMinutes(date_add(hour, 1, now())), '%R')
    GROUP BY ts
    ORDER BY ts""")

    # объединим оба запроса в одну таблицу, переименуем названия столбцов
    data['users_message'] = data_message['users_message']
    data.columns = ['ts', 'Пользователи ленты новостей', 'Просмотры', 'Лайки', 'Пользователи сервиса сообщений']

    # для каждой метрики из таблицы проведем проверку последнего значения на аномалию с помощью функции check_anomaly_IQR
    for metric in metrics_list:
        df = data[['ts', metric]].copy()
        is_alert, df, lower, up, avg = check_anomaly_IQR(df, metric)
        
        # если проверка определила выброс, то формируется сообщение и графиик 
        if is_alert:
            msg = f'''Метрика {metric}:
    текущее значение - {df[metric].iloc[-1]},
    отклонение от среднего значения - {abs(1 - df[metric].iloc[-1]/avg) * 100:.2f}%,
    <a href="https://superset.lab.karpov.courses/superset/dashboard/589/">Смотреть на дашборде</a>'''
            
            
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,12))
            plt.suptitle(metric)

            # линейный график с последними 5 значениями метрик и верхней и нижней границой
            sns.lineplot(x=df['ts'].iloc[-5:], y=df[metric].iloc[-5:], ax=ax[0], marker='o', label=metric)
            ax[0].set
            ax[0].axhline(lower, color='red', label='lower')
            ax[0].axhline(up, color='green', label='up')
            ax[0].legend()
            ax[0].set_xlabel('time')
            ax[0].set_xticks(df['ts'].iloc[-5:])
            ax[0].set_xticklabels(df['ts'].iloc[-5:].dt.strftime('%d %b %H:%M'))

            # график плотности распределения значений метрик и верхней и нижней границой 
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

            
            
            
    # для каждой метрики из таблицы проведем проверку последнего значения на аномалию с помощью функции check_anomaly_DBSCAN        
    for metric in metrics_list:
        df = data[['ts', metric]].copy()           
        is_alert, lower, up, avg = check_anomaly_DBSCAN(df, metric)

        # если проверка определила выброс, то формируется сообщение и графиик 
        if is_alert:
            msg = f'''DBSCAN
        Метрика {metric}:
          текущее значение - {df[metric].iloc[-1]},
          отклонение от среднего значения - {abs(1 - df[metric].iloc[-1]/avg) * 100:.2f}%,
          <a href="https://superset.lab.karpov.courses/superset/dashboard/589/">Смотреть на дашборде</a>'''


            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,12))
            plt.suptitle(metric)
    
            # линейный график с последними 5 значениями метрик и верхней и нижней границой
            sns.lineplot(x=df['ts'].iloc[-5:], y=df[metric].iloc[-5:], ax=ax[0], marker='o', label=metric)
            ax[0].set
            ax[0].axhline(lower, color='red', label='Верхняя граница')
            ax[0].axhline(up, color='green', label='Нижняя граница')
            ax[0].legend()
            ax[0].set_xlabel('time')
            ax[0].set_xticks(df['ts'].iloc[-5:])
            ax[0].set_xticklabels(df['ts'].iloc[-5:].dt.strftime('%d %b %H:%M'))

            # sns.kdeplot(df[metric], ax=ax[1], shade=True)
            
            # график разброса значений метрик с верхней и нижней границой 
            sns.rugplot(df[metric], ax=ax[1])
            sns.swarmplot(x=df[metric], ax=ax[1], size=12)
            ax[1].axvline(lower, color='red', label='Верхняя граница')
            ax[1].axvline(up, color='green', label='Нижняя граница')
            ax[1].axvline(df[metric].iloc[-1], color='black', label='Текущее значение')
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

