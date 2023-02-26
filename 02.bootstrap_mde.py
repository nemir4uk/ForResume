# Демонстрация подготовки данных к анализу и рассчета MDE на бутстраппированных выборках

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import anderson, norm
from scipy import stats

# Загрузка данных, выбор нужных столбцов NO, отсечение строк с отсутствующими значениями
aqi_ds_city_day = pd.read_csv("F:/DATASETS/kaggle/air_quality_india/city_day.csv")
df = aqi_ds_city_day[aqi_ds_city_day["City"] == "Delhi"][["NO"]]
df = df.dropna()

# Разделение выборки на 2 части
a_df = df.loc[:11229]
b_df = df.loc[11230:]
print("A-subsample mean: ", a_df["NO"].mean())
print("B-subsample mean: ", b_df["NO"].mean())

# Гистограммы распределения изначальных выборок
sns_plot = sns.pairplot(a_df)
plt.show()
sns_plot = sns.pairplot(b_df)
plt.show()

# Проверка распределений на нормальность
# Критерий Андерсона-Дарлинга
norm_check = anderson(a_df["NO"], dist='norm')
output = "Statistic of Anderson-Darling test A-subsample is: {}\nCritical_values for significance_levels 15%, 10%, 5%, 2.5%, 1% is: {}"
print(output.format(norm_check[0], norm_check[1]))
norm_check = anderson(b_df["NO"], dist='norm')
output = "Statistic of Anderson-Darling test B-subsample is: {}\nCritical_values for significance_levels 15%, 10%, 5%, 2.5%, 1% is: {}\n"
print(output.format(norm_check[0], norm_check[1]))

# Тест Шапиро-Вилка
norm_check = stats.shapiro(a_df)
output = "Statistic of Shapiro-Wilk test A-subsample is: {}\nP-value : {}"
print(output.format(norm_check[0], norm_check[1]))
norm_check = stats.shapiro(b_df)
output = "Statistic of Shapiro-Wilk test B-subsample is: {}\nP-value : {}\n"
print(output.format(norm_check[0], norm_check[1]))

# Квантиль-квантиль график (QQplot)
plot_check = stats.probplot(a_df["NO"], plot=plt)
plt.show()
plot_check = stats.probplot(b_df["NO"], plot=plt)
plt.show()

# Тест Колмогорова-Смирнова
print("Kolmogorov-Smirnov test A-subsample: ", stats.kstest(a_df["NO"], "norm"))
print("Kolmogorov-Smirnov test B-subsample: ", stats.kstest(b_df["NO"], "norm"), "\n")

# Тест Крамера фон Мизеса
norm_check = stats.cramervonmises(a_df["NO"], "norm")
print("Cramér-von Mises test A-subsample: ", norm_check)
norm_check = stats.cramervonmises(b_df["NO"], "norm")
print("Cramér-von Mises test B-subsample: ", norm_check, "\n")

# Бутстрап выборочных средних

B = 1000  # количество итераций генерации подвыборок
n = 1000  # количество случайных элементов в каждой подвыборке
# А-подвыборка
bootstrap_values = []
for _ in range(B):
    values = np.random.choice(a_df["NO"], n, True)
    bootstrap_values.append(values.mean())

norm_check = stats.shapiro(bootstrap_values)
output = "Statistic of Shapiro-Wilk test bootstrapped means of A-subsample is: {}\nP-value : {}"
print(output.format(norm_check[0], norm_check[1]))  # Шапиро-Вилк тест на бутстрапированных выборочных средних
sns_plot = sns.histplot(bootstrap_values)
plt.show()  # Гистограмма распределения бутстрапированных выборочных средних

# B-подвыборка
bootstrap_values2 = []
for _ in range(B):
    values = np.random.choice(a_df["NO"], n, True)
    bootstrap_values2.append(values.mean())

norm_check = stats.shapiro(bootstrap_values2)
output = "Statistic of Shapiro-Wilk test bootstrapped means of B-subsample is: {}\nP-value : {}\n"
print(output.format(norm_check[0], norm_check[1]))  # Шапиро-Вилк тест на бутстрапированных выборочных средних
sns_plot = sns.histplot(bootstrap_values2)
plt.show()  # Гистограмма распределения бутстрапированных выборочных средних

# Т-статистика
print("T-test of bootstrapped  values: ", stats.ttest_ind(bootstrap_values,bootstrap_values2))

# MDE для соответствующих уровней значимости и мощности
sd_a = np.std(bootstrap_values)
sd_b = np.std(bootstrap_values2)
alpha = 0.05
beta = 0.8
def estimate_effect_size(sd1, sd2, n, alpha, power):
    S = np.sqrt((sd1**2 / n) + (sd2**2 / n))
    M = norm.ppf(q=1-alpha/2) + norm.ppf(q=power)
    return M * S

output = "MDE for alpha={} and power={}: "
print(output.format(alpha, beta), estimate_effect_size(sd_a, sd_b, n, alpha, beta))
