# Демонстрация применения линейных моделей для исследования взаимосвязи двух количественных переменных

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# Загрузка данных, выбор нужных столбцов, отсечение строк с отсутствующими значениями
"""
Настоящий датасет доступен для загрузки со страницы по адресу:
https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india?select=city_day.csv
"""
aqi_ds_city_day = pd.read_csv("F:/DATASETS/kaggle/air_quality_india/city_day.csv")
df = aqi_ds_city_day[aqi_ds_city_day["City"] == "Delhi"][["NOx","NO"]]
df = df.dropna()
df = df.loc[11350:]
print(df.info())

# Рассчет линейной регрессии методом наименьших квадратов
slope, intercept, r, p, std_err = stats.linregress(df["NOx"], df["NO"])

def myfunc(x):
  return slope * x + intercept

model = list(map(myfunc, df["NOx"]))

# Отображение выборки, линии регрессии и вывод R-квадрат
plt.scatter(df["NOx"], df["NO"], c='#00ffcc', marker='.')
plt.plot(df["NOx"], model)
plt.axis([0, 200, 0, 170])
plt.ylabel('NOx')
plt.xlabel('NO')
plt.show()
print(slope, intercept, r, p, std_err)
print("R-square =", r**2)

# Логарифмическая трансформация переменных
log_y = np.log(df["NOx"].to_numpy())
log_x = np.log(df["NO"].to_numpy())

# Рассчет регрессии после трансформации переменных и вывод R-квадрат
slope, intercept, r, p, std_err = stats.linregress(log_y, log_x)

model2 = list(map(myfunc, log_y))
plt.scatter(log_y, intercept + log_x, c='#9900ff', marker='.')
plt.plot(log_y,intercept + model2)
plt.show()
print("R-square after log-transform =", r**2)
