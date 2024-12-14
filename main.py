from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import requests
from bs4 import BeautifulSoup


def get_weather():
    url = 'https://world-weather.ru/pogoda/russia/moscow/month/'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        titles = soup.select('.foreacast span')

        temperatures = [int(temp.get_text().replace("°", "")) for temp in titles]
        return temperatures
    else:
        print(f'Ошибка при запросе: {response.status_code}')
        return

df = pd.read_excel('data.xlsx')

if not pd.api.types.is_datetime64_any_dtype(df['day']):
    df['day'] = pd.to_datetime(df['day'], errors='coerce')

df['day_number'] = df['day'].dt.day
df['month'] = df['day'].dt.month
df['year'] = df['day'].dt.year

X = df[['day_number', 'month', 'year', 'temp']]
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

new_dates = pd.date_range(start=datetime.now().strftime('%Y-%m-%d'), periods=len(get_weather()))
new_temps = get_weather()

new_data = pd.DataFrame({
    'day_number': new_dates.day,
    'month': new_dates.month,
    'year': new_dates.year,
    'temp': new_temps
})

predictions = regressor.predict(new_data)
predictions = np.maximum(predictions, 1)

predictions_df = pd.DataFrame()
predictions_df['day'] = new_data['day_number'].astype(str) + '.' + new_data['month'].astype(str) + '.' + new_data['year'].astype(str)
predictions_df['temp'] = new_data['temp']
predictions_df['sales'] = np.round(predictions)

if (predictions_df['sales'] < 0).any():
    print("Обнаружены отрицательные значения в предсказаниях продаж.")

predictions_df.to_excel('predictions_next_week.xlsx', index=False)