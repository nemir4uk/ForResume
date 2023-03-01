import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

aqi_ds_city_day = pd.read_csv("F:/DATASETS/kaggle/air_quality_india/city_day.csv")
df = aqi_ds_city_day[aqi_ds_city_day["City"] == "Delhi"][["NOx","NO"]]
df = df.dropna()
df = df.loc[11350:]
print(df.info())

slope, intercept, r, p, std_err = stats.linregress(df["NOx"], df["NO"])

def myfunc(x):
  return slope * x + intercept

model = list(map(myfunc, df["NOx"]))

plt.scatter(df["NOx"], df["NO"])
plt.plot(df["NOx"], model, "oc")
plt.show()