# Конкурс "Турникеты"
# https://ods.ai/tracks/linear-models-spring23/competitions/gates
# Использована логистическая регрессия
# Решение в таком виде дало 0,1639528355

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

# Две функции для извлечения даты и времени из Timestamp
def date(string):
    lst = string.split(" ")
    return lst[0]

def time(string):
    lst = string.split(" ")
    return lst[1]

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# print(sorted(train_data.user_id.unique())) # 58

# Далее две Почти идентичные функции для извлечения фичей отдельно из train и из test
# Все комментарии к извлеченным фичам во второй функции
def preprocess_test(data):
    data["date"] = data["timestamp"].apply(date)
    data["time"] = data["timestamp"].apply(time)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["1day"] = data["timestamp"].dt.is_month_start.apply(lambda x: 1 if x == True else 0)
    data["last_day"] = data["timestamp"].dt.is_month_end.apply(lambda x: 1 if x == True else 0)
    data["DofW"] = data["timestamp"].dt.dayofweek
    data["DofY1"] = (data["timestamp"].dt.dayofyear % 4).apply(lambda x: 1 if x in (1, 2) else 0)
    # data["DofY2"] = (data["timestamp"].dt.dayofyear % 6).apply(lambda x: 1 if x in (0, 1, 2) else 0)
    data["month"] = data["timestamp"].dt.month
    data["day"] = data["timestamp"].dt.day
    data["is_weekend"] = data["timestamp"].apply(lambda x: 1 if x.date().weekday() in (5, 6) else 0)
    data.loc[data["month"].eq(1) & data["day"].eq(16), "is_weekend"] = 1  #16.01
    data.loc[data["month"].eq(2) & data["day"].ge(4), "is_weekend"] = 1  #after 02.03
    data.loc[data["month"].eq(9) & data["day"].eq(12), "is_weekend"] = 1  # 12.09
    data.loc[data["month"].eq(11) & data["day"].eq(11), "is_weekend"] = 1  #11.11
    data.loc[data["month"].eq(11) & data["day"].eq(16), "is_weekend"] = 1  #16.11
    data["hour"] = data["timestamp"].dt.hour
    data["m1"] = data["hour"].apply(lambda x: 1 if x == 7 else 0)
    data["m2"] = data["hour"].apply(lambda x: 1 if x == 8 else 0)
    data["m3"] = data["hour"].apply(lambda x: 1 if x == 9 else 0)
    data["m4"] = data["hour"].apply(lambda x: 1 if x == 10 else 0)
    data["d1"] = data["hour"].apply(lambda x: 1 if x == 11 else 0)
    data["d2"] = data["hour"].apply(lambda x: 1 if x == 12 else 0)
    data["d3"] = data["hour"].apply(lambda x: 1 if x == 14 else 0)
    data["d4"] = data["hour"].apply(lambda x: 1 if x == 15 else 0)
    data["e1"] = data["hour"].apply(lambda x: 1 if x == 18 else 0)
    data["e2"] = data["hour"].apply(lambda x: 1 if x == 19 else 0)
    data["e3"] = data["hour"].apply(lambda x: 1 if x == 20 else 0)
    data["e4"] = data["hour"].apply(lambda x: 1 if x == 21 else 0)
    data["e5"] = data["hour"].apply(lambda x: 1 if x == 22 else 0)
    data["e6"] = data["hour"].apply(lambda x: 1 if x == 23 else 0)
    data["min"] = data["timestamp"].dt.minute
    data["sec"] = data["timestamp"].dt.second
    data["shift"] = data["timestamp"].shift(1)
    data["shift"] = pd.to_datetime(data["shift"])
    data["lag"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["l3s"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["lag1"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["lag2"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["lag3"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["lag4"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["lag5"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["lag6"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["lag7"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["lag8"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["lag9"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["lag0"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["lagx"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["gs1"] = data["gate_id"].shift(1)
    data["gs2"] = data["gate_id"].shift(2)
    data["gs3"] = data["gate_id"].shift(3)
    data["gs4"] = data["gate_id"].shift(4)
    data["gs5"] = data["gate_id"].shift(5)
    data["gs-1"] = data["gate_id"].shift(-1)
    data["gs-2"] = data["gate_id"].shift(-2)
    data["gs-3"] = data["gate_id"].shift(-3)
    data["gs-4"] = data["gate_id"].shift(-4)
    data["gs-5"] = data["gate_id"].shift(-5)
    data["lag"] = data["lag"].apply(lambda x: 1 if x < 2 else 0)
    data["l3s"] = data["l3s"].apply(lambda x: 1 if x <= 3 else 0)
    data["l-1"] = data["l3s"].shift(-1)
    data.loc[data["l3s"].eq(1) | data["l-1"].eq(1), "ls1"] = 1
    data["lag1"] = data["lag1"].apply(lambda x: 1 if 6 > x > 2 else 0)
    data["lag2"] = data["lag2"].apply(lambda x: 1 if 15 > x >= 6 else 0)
    data["lag3"] = data["lag3"].apply(lambda x: 1 if 22 > x >= 15 else 0)
    data["lag4"] = data["lag4"].apply(lambda x: 1 if 32 > x >= 22 else 0)
    data["lag5"] = data["lag5"].apply(lambda x: 1 if 42 > x > 32 else 0)
    data["lag6"] = data["lag6"].apply(lambda x: 1 if 58 > x >= 42 else 0)
    data["lag7"] = data["lag7"].apply(lambda x: 1 if 69 > x >= 58 else 0)
    data["lag8"] = data["lag8"].apply(lambda x: 1 if 76 > x >= 69 else 0)
    data["lag9"] = data["lag9"].apply(lambda x: 1 if 130 >= x > 127 else 0)
    data["lag0"] = data["lag0"].apply(lambda x: 0 if x > 69 else x)
    data["lagx"] = data["lagx"].apply(lambda x: 1 if x > 130 else 0)
    data["double"] = data["gate_id"][data["gate_id"].eq(data["gs1"]) & data["ls1"].eq(1) | data["gate_id"].eq(data["gs-1"])].apply(lambda x: 1 if x >= 1 else 0)
    data["double2"] = data["gate_id"][data["gate_id"].eq(data["gs1"]) & data["lag"].eq(0) | data["gate_id"].eq(data["gs-1"])].apply(lambda x: 1 if x >= 1 else 0)
    data["double3"] = data["gate_id"][data["gate_id"].eq(data["gs1"]) & data["lag1"].eq(1) | data["gate_id"].eq(data["gs-1"])].apply(lambda x: 1 if x >= 1 else 0)
    data["double4"] = data["gate_id"][data["gate_id"].eq(data["gs1"]) & data["lag2"].eq(1) | data["gate_id"].eq(data["gs-1"])].apply(lambda x: 1 if x >= 1 else 0)
    data["double5"] = data["gate_id"][data["gate_id"].eq(data["gs1"]) & data["lag3"].eq(1) | data["gate_id"].eq(data["gs-1"])].apply(lambda x: 1 if x >= 1 else 0)
    data["double6"] = data["gate_id"][data["gate_id"].eq(data["gs1"]) & data["lag4"].eq(1) | data["gate_id"].eq(data["gs-1"])].apply(lambda x: 1 if x >= 1 else 0)
    data["double7"] = data["gate_id"][data["gate_id"].eq(data["gs1"]) & data["lag7"].eq(1) | data["gate_id"].eq(data["gs-1"])].apply(lambda x: 1 if x >= 1 else 0)
    data["double8"] = data["gate_id"][data["gate_id"].eq(data["gs1"]) & data["lagx"].eq(1) | data["gate_id"].eq(data["gs-1"])].apply(lambda x: 1 if x >= 1 else 0)
    data.loc[(data["gate_id"].eq(3) & data["gs-1"].eq(3) & data["gs-2"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(3) & data["gs1"].eq(3) & data["gs-1"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(10) & data["gs1"].eq(3) & data["gs2"].eq(3)) |
             (data["gate_id"].eq(3) & data["gs-2"].eq(3) & data["gs-3"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(3) & data["gs2"].eq(3) & data["gs-1"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(10) & data["gs1"].eq(3) & data["gs3"].eq(3)) |
             (data["gate_id"].eq(3) & data["gs-1"].eq(3) & data["gs-3"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(3) & data["gs1"].eq(3) & data["gs-2"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(10) & data["gs2"].eq(3) & data["gs3"].eq(3)) |
             (data["gate_id"].eq(3) & data["gs-3"].eq(3) & data["gs-4"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(3) & data["gs3"].eq(3) & data["gs-1"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(10) & data["gs1"].eq(3) & data["gs4"].eq(3)) |
             (data["gate_id"].eq(3) & data["gs-2"].eq(3) & data["gs-4"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(3) & data["gs2"].eq(3) & data["gs-2"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(10) & data["gs2"].eq(3) & data["gs4"].eq(3)) |
             (data["gate_id"].eq(3) & data["gs-1"].eq(3) & data["gs-4"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(3) & data["gs1"].eq(3) & data["gs-3"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(10) & data["gs3"].eq(3) & data["gs4"].eq(3)), "xxy"] = 1
    # data["3310"] = data["gate_id"][(data["gate_id"].eq(3) & data["gs-1"].eq(3) & data["gs-2"].eq(10)) |
    #                                (data["gate_id"].eq(3) & data["gs1"].eq(3) & data["gs-1"].eq(10)) |
    #                                (data["gate_id"].eq(10) & data["gs1"].eq(3) & data["gs2"].eq(3)) |
    #                                (data["gate_id"].eq(10) & data["gs1"].eq(3) & data["gs2"].eq(3))].apply(lambda x: 1 if x >= 1 else 0)
    data.loc[(data["gate_id"].eq(11) & data["gs-1"].eq(4) & data["gs-2"].eq(4)) |
             (data["gate_id"].eq(4) & data["gs1"].eq(11) & data["gs-1"].eq(4) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(4) & data["gs1"].eq(4) & data["gs2"].eq(11) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(11) & data["gs-2"].eq(4) & data["gs-3"].eq(4)) |
             (data["gate_id"].eq(4) & data["gs2"].eq(11) & data["gs-1"].eq(4) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(4) & data["gs1"].eq(4) & data["gs3"].eq(11) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(11) & data["gs-1"].eq(4) & data["gs-3"].eq(4)) |
             (data["gate_id"].eq(4) & data["gs1"].eq(11) & data["gs-2"].eq(4) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(4) & data["gs2"].eq(4) & data["gs3"].eq(11) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(11) & data["gs-3"].eq(4) & data["gs-4"].eq(4)) |
             (data["gate_id"].eq(4) & data["gs3"].eq(11) & data["gs-1"].eq(4) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(4) & data["gs1"].eq(4) & data["gs4"].eq(11) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(11) & data["gs-2"].eq(4) & data["gs-4"].eq(4)) |
             (data["gate_id"].eq(4) & data["gs2"].eq(11) & data["gs-2"].eq(4) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(4) & data["gs2"].eq(4) & data["gs4"].eq(11) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(11) & data["gs-1"].eq(4) & data["gs-4"].eq(4)) |
             (data["gate_id"].eq(4) & data["gs1"].eq(11) & data["gs-3"].eq(4) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(4) & data["gs3"].eq(4) & data["gs4"].eq(11) & data["ls1"].eq(1)), "1144"] = 1
    # data["1144"] = data["gate_id"][(data["gate_id"].eq(11) & data["gs-1"].eq(4) & data["gs-2"].eq(4)) |
    #                                (data["gate_id"].eq(4) & data["gs1"].eq(11) & data["gs-1"].eq(4)) |
    #                                (data["gate_id"].eq(4) & data["gs1"].eq(4) & data["gs2"].eq(11))].apply(lambda x: 1 if x >= 1 else 0)
    data["73310"] = data["gate_id"][(data["gate_id"].eq(7) & data["gs-1"].eq(3) & data["gs-2"].eq(3) & data["gs-3"].eq(10)) |
                                   (data["gate_id"].eq(3) & data["gs1"].eq(7) & data["gs-1"].eq(3) & data["gs-2"].eq(10)) |
                                   (data["gate_id"].eq(3) & data["gs1"].eq(3) & data["gs2"].eq(7) & data["gs-1"].eq(10)) |
                                   (data["gate_id"].eq(10) & data["gs1"].eq(3) & data["gs2"].eq(3) & data["gs3"].eq(7))].apply(lambda x: 1 if x >= 1 else 0)
    data["6row"] = data["gate_id"][(data["gate_id"].eq(7) & data["gs-1"].eq(9) & data["gs-2"].eq(9) & data["gs-3"].eq(5) & data["gs-4"].eq(5) & data["gs-5"].eq(10)) |
                                   (data["gate_id"].eq(9) & data["gs-1"].eq(9) & data["gs-2"].eq(5) & data["gs-3"].eq(5) & data["gs-4"].eq(10) & data["gs1"].eq(7)) |
                                   (data["gate_id"].eq(9) & data["gs-1"].eq(5) & data["gs-2"].eq(5) & data["gs-3"].eq(10) & data["gs1"].eq(9) & data["gs2"].eq(7)) |
                                   (data["gate_id"].eq(5) & data["gs-1"].eq(5) & data["gs-2"].eq(10) & data["gs1"].eq(9) & data["gs2"].eq(9) & data["gs3"].eq(7)) |
                                   (data["gate_id"].eq(5) & data["gs-1"].eq(10) & data["gs1"].eq(5) & data["gs2"].eq(9) & data["gs3"].eq(9) & data["gs4"].eq(7)) |
                                    (data["gate_id"].eq(10) & data["gs1"].eq(5) & data["gs2"].eq(5) & data["gs3"].eq(9) & data["gs4"].eq(9) & data["gs5"].eq(7))]\
                                    .apply(lambda x: 1 if x >= 1 else 0)
    data["5row"] = data["gate_id"][(data["gate_id"].eq(9) & data["gs-1"].eq(9) & data["gs-2"].eq(5) & data["gs-3"].eq(5) & data["gs-4"].eq(10)) |
                                   (data["gate_id"].eq(9) & data["gs1"].eq(9) & data["gs-1"].eq(5) & data["gs-2"].eq(5) & data["gs-3"].eq(10)) |
                                   (data["gate_id"].eq(5) & data["gs1"].eq(9) & data["gs2"].eq(9) & data["gs-1"].eq(5) & data["gs-2"].eq(10)) |
                                   (data["gate_id"].eq(5) & data["gs1"].eq(5) & data["gs2"].eq(9) & data["gs3"].eq(9) & data["gs-1"].eq(10)) |
                                   (data["gate_id"].eq(10) & data["gs1"].eq(5) & data["gs2"].eq(5) & data["gs3"].eq(9) & data["gs4"].eq(9))]\
                                    .apply(lambda x: 1 if x >= 1 else 0)
    data = data.replace(np.nan, 0)
    data["beep_count"] = data.groupby("date").gate_id.transform("count")
    data = data.astype({"row_id": "int32", "gate_id": "int8", "1day": "int8", "last_day": "int8",
                        "DofW": "int8", "DofY1": "int16", "month": "int8", "day": "int8", "is_weekend": "int8",
                        "hour": "int8", "m1": "int8", "m2": "int8", "m3": "int8", "m4": "int8", "d1": "int8",
                        "d2": "int8", "d3": "int8", "d4": "int8", "e1": "int8", "e2": "int8", "e3": "int8",
                        "e4": "int8", "e5": "int8", "e6": "int8", "min": "int8", "sec": "int8", "lag": "int8",
                        "lag1": "int8", "lag2": "int8", "lag3": "int8", "lag4": "int8", "lag5": "int8", "lag6": "int8",
                        "lag7": "int8", "lag8": "int8", "lag9": "int8", "lag0": "int8", "lagx": "int8", "gs1": "int8",
                        "gs2": "int8", "gs3": "int8", "gs4": "int8", "gs5": "int8", "gs-1": "int8", "gs-2": "int8",
                        "gs-3": "int8", "gs-4": "int8", "gs-5": "int8", "double": "int8", "double2": "int8",
                        "double3": "int8", "double4": "int8", "double5": "int8", "double6": "int8", "double7": "int8",
                        "double8": "int8", '6row': 'int8', '5row': 'int8', "beep_count": "int16",
                        "1144": "int8", "73310": "int8", "l-1": "int8", "ls1": "int8", "xxy": "int8"})
    # data = data.astype({"row_id": "int32", "gate_id": "category", "1day": "category", "last_day": "category",
    #                     "DofW": "category", "DofY1": "int16", "month": "category", "day": "category", "is_weekend": "category",
    #                     "hour": "category", "m1": "category", "m2": "category", "m3": "category", "m4": "category", "d1": "category",
    #                     "d2": "category", "d3": "category", "d4": "category", "e1": "category", "e2": "category", "e3": "category",
    #                     "e4": "category", "e5": "category", "e6": "category", "min": "category", "sec": "int8", "lag": "category",
    #                     "lag1": "category", "lag2": "category", "lag3": "category", "lag4": "category", "lag5": "category", "lag6": "category",
    #                     "lag7": "category", "lag8": "category", "lag9": "category", "lag0": "int8", "lagx": "category", "gs1": "int8",
    #                     "gs2": "int8", "gs3": "int8", "gs4": "int8", "gs5": "int8", "gs-1": "int8", "gs-2": "int8",
    #                     "gs-3": "int8", "gs-4": "int8", "gs-5": "int8", "double": "category", "double2": "category",
    #                     "double3": "category", "double4": "category", "double5": "category", "double6": "category", "double7": "category",
    #                     "double8": "category", '6row': 'category', '5row': 'category', "beep_count": "int16"})
    return data
test_data = preprocess_test(test_data)

def preprocess(data):
    data["date"] = data["timestamp"].apply(date)
    data["time"] = data["timestamp"].apply(time)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["1day"] = data["timestamp"].dt.is_month_start.apply(lambda x: 1 if x == True else 0) # 1й день месяца
    data["last_day"] = data["timestamp"].dt.is_month_end.apply(lambda x: 1 if x == True else 0) # Последний день месяца
    data["DofW"] = data["timestamp"].dt.dayofweek # День недели от 0 до 6
    data["DofY1"] = (data["timestamp"].dt.dayofyear % 4).apply(lambda x: 1 if x in (1, 2) else 0) # Попытка вывести метку "график 2 через 2" пробовал иные вариации - работают хуже
    # data["DofY2"] = (data["timestamp"].dt.dayofyear % 6).apply(lambda x: 1 if x in (0, 1, 2) else 0) # "3 через 3" не прокатило
    data["month"] = data["timestamp"].dt.month
    data["day"] = data["timestamp"].dt.day
    data["is_weekend"] = data["timestamp"].apply(lambda x: 1 if x.date().weekday() in (5, 6) else 0) # Метка выходного дня
    data.loc[data["month"].eq(1) & data["day"].eq(16), "is_weekend"] = 1  #16.01 16 Января по количеству записей - праздник
    data.loc[data["month"].eq(2) & data["day"].ge(4), "is_weekend"] = 1  #after 02.03 - всё что после 3 февраля также выглядело в тестовой выборке как праздничные дни
    data.loc[data["month"].eq(9) & data["day"].eq(12), "is_weekend"] = 1  # 12.09
    data.loc[data["month"].eq(11) & data["day"].eq(11), "is_weekend"] = 1  #11.11
    data.loc[data["month"].eq(11) & data["day"].eq(16), "is_weekend"] = 1  #16.11
    data["hour"] = data["timestamp"].dt.hour
    data["m1"] = data["hour"].apply(lambda x: 1 if x == 7 else 0) # отдельно вывести некоторые часы позволило слегка повысить точность
    data["m2"] = data["hour"].apply(lambda x: 1 if x == 8 else 0)
    data["m3"] = data["hour"].apply(lambda x: 1 if x == 9 else 0)
    data["m4"] = data["hour"].apply(lambda x: 1 if x == 10 else 0)
    data["d1"] = data["hour"].apply(lambda x: 1 if x == 11 else 0)
    data["d2"] = data["hour"].apply(lambda x: 1 if x == 12 else 0)
    data["d3"] = data["hour"].apply(lambda x: 1 if x == 14 else 0)
    data["d4"] = data["hour"].apply(lambda x: 1 if x == 15 else 0)
    data["e1"] = data["hour"].apply(lambda x: 1 if x == 18 else 0)
    data["e2"] = data["hour"].apply(lambda x: 1 if x == 19 else 0)
    data["e3"] = data["hour"].apply(lambda x: 1 if x == 20 else 0)
    data["e4"] = data["hour"].apply(lambda x: 1 if x == 21 else 0)
    data["e5"] = data["hour"].apply(lambda x: 1 if x == 22 else 0)
    data["e6"] = data["hour"].apply(lambda x: 1 if x == 23 else 0)
    data["min"] = data["timestamp"].dt.minute
    data["sec"] = data["timestamp"].dt.second
    data["shift"] = data["timestamp"].shift(1) # Далее разположен блок фичей по разнице во времени срабатываний (разница в секундах)
    data["shift"] = pd.to_datetime(data["shift"])
    data["lag"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s") # Сначала подготовительный блок
    data["l3s"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s") # Отдельно выделено 13 фичей, некоторые с перекрытием
    data["lag1"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["lag2"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["lag3"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["lag4"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["lag5"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["lag6"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["lag7"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["lag8"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["lag9"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["lag0"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["lagx"] = (data["timestamp"] - data["shift"]) / np.timedelta64(1, "s")
    data["gs1"] = data["gate_id"].shift(1) # Также понадобилось выделить для каждой записи gate_id со смещением на 5 строк в обе стороны
    data["gs2"] = data["gate_id"].shift(2) # Далее объясню для чего
    data["gs3"] = data["gate_id"].shift(3)
    data["gs4"] = data["gate_id"].shift(4)
    data["gs5"] = data["gate_id"].shift(5)
    data["gs-1"] = data["gate_id"].shift(-1)
    data["gs-2"] = data["gate_id"].shift(-2)
    data["gs-3"] = data["gate_id"].shift(-3)
    data["gs-4"] = data["gate_id"].shift(-4)
    data["gs-5"] = data["gate_id"].shift(-5)
    data["lag"] = data["lag"].apply(lambda x: 1 if x < 2 else 0) # Перевод фичей с разницей срабатываний в нули и единицы
    data["l3s"] = data["l3s"].apply(lambda x: 1 if x <= 3 else 0)
    data["l-1"] = data["l3s"].shift(-1)
    data.loc[data["l3s"].eq(1) | data["l-1"].eq(1), "ls1"] = 1 # Отдельно выделен диапазон <=3 сек
    data["lag1"] = data["lag1"].apply(lambda x: 1 if 6 > x > 2 else 0)
    data["lag2"] = data["lag2"].apply(lambda x: 1 if 15 > x >= 6 else 0)
    data["lag3"] = data["lag3"].apply(lambda x: 1 if 22 > x >= 15 else 0)
    data["lag4"] = data["lag4"].apply(lambda x: 1 if 32 > x >= 22 else 0)
    data["lag5"] = data["lag5"].apply(lambda x: 1 if 42 > x > 32 else 0)
    data["lag6"] = data["lag6"].apply(lambda x: 1 if 58 > x >= 42 else 0)
    data["lag7"] = data["lag7"].apply(lambda x: 1 if 69 > x >= 58 else 0)
    data["lag8"] = data["lag8"].apply(lambda x: 1 if 76 > x >= 69 else 0)
    data["lag9"] = data["lag9"].apply(lambda x: 1 if 130 >= x > 127 else 0)
    data["lag0"] = data["lag0"].apply(lambda x: 0 if x > 69 else x)
    data["lagx"] = data["lagx"].apply(lambda x: 1 if x > 130 else 0)
    # Блок фичей с использованием разницы срабатываний и сравнением gate_id соседних записей
    data["double"] = data["gate_id"][data["gate_id"].eq(data["gs1"]) & data["ls1"].eq(1) | data["gate_id"].eq(data["gs-1"])].apply(lambda x: 1 if x >= 1 else 0)
    data["double2"] = data["gate_id"][data["gate_id"].eq(data["gs1"]) & data["lag"].eq(0) | data["gate_id"].eq(data["gs-1"])].apply(lambda x: 1 if x >= 1 else 0)
    data["double3"] = data["gate_id"][data["gate_id"].eq(data["gs1"]) & data["lag1"].eq(1) | data["gate_id"].eq(data["gs-1"])].apply(lambda x: 1 if x >= 1 else 0)
    data["double4"] = data["gate_id"][data["gate_id"].eq(data["gs1"]) & data["lag2"].eq(1) | data["gate_id"].eq(data["gs-1"])].apply(lambda x: 1 if x >= 1 else 0)
    data["double5"] = data["gate_id"][data["gate_id"].eq(data["gs1"]) & data["lag3"].eq(1) | data["gate_id"].eq(data["gs-1"])].apply(lambda x: 1 if x >= 1 else 0)
    data["double6"] = data["gate_id"][data["gate_id"].eq(data["gs1"]) & data["lag4"].eq(1) | data["gate_id"].eq(data["gs-1"])].apply(lambda x: 1 if x >= 1 else 0)
    data["double7"] = data["gate_id"][data["gate_id"].eq(data["gs1"]) & data["lag7"].eq(1) | data["gate_id"].eq(data["gs-1"])].apply(lambda x: 1 if x >= 1 else 0)
    data["double8"] = data["gate_id"][data["gate_id"].eq(data["gs1"]) & data["lagx"].eq(1) | data["gate_id"].eq(data["gs-1"])].apply(lambda x: 1 if x >= 1 else 0)
    # Далее интересное: при детальном рассмотрении трейн и тестовых данных было замечен паттерн последовательности
    # проходов одним юзером через гейты 3-3-10. Также в эту последовательность может вклиниться другой юзер с одним или двумя проходами
    # Данная реализация содержит часть ложных срабатываний, как это пофиксить пока не придумал.
    # То место где понадобятся смещения!
    data.loc[(data["gate_id"].eq(3) & data["gs-1"].eq(3) & data["gs-2"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(3) & data["gs1"].eq(3) & data["gs-1"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(10) & data["gs1"].eq(3) & data["gs2"].eq(3)) |
             (data["gate_id"].eq(3) & data["gs-2"].eq(3) & data["gs-3"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(3) & data["gs2"].eq(3) & data["gs-1"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(10) & data["gs1"].eq(3) & data["gs3"].eq(3)) |
             (data["gate_id"].eq(3) & data["gs-1"].eq(3) & data["gs-3"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(3) & data["gs1"].eq(3) & data["gs-2"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(10) & data["gs2"].eq(3) & data["gs3"].eq(3)) |
             (data["gate_id"].eq(3) & data["gs-3"].eq(3) & data["gs-4"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(3) & data["gs3"].eq(3) & data["gs-1"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(10) & data["gs1"].eq(3) & data["gs4"].eq(3)) |
             (data["gate_id"].eq(3) & data["gs-2"].eq(3) & data["gs-4"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(3) & data["gs2"].eq(3) & data["gs-2"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(10) & data["gs2"].eq(3) & data["gs4"].eq(3)) |
             (data["gate_id"].eq(3) & data["gs-1"].eq(3) & data["gs-4"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(3) & data["gs1"].eq(3) & data["gs-3"].eq(10) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(10) & data["gs3"].eq(3) & data["gs4"].eq(3)), "xxy"] = 1
    # похожий паттерн, только через гейты 11-4-4
    data.loc[(data["gate_id"].eq(11) & data["gs-1"].eq(4) & data["gs-2"].eq(4)) |
             (data["gate_id"].eq(4) & data["gs1"].eq(11) & data["gs-1"].eq(4) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(4) & data["gs1"].eq(4) & data["gs2"].eq(11) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(11) & data["gs-2"].eq(4) & data["gs-3"].eq(4)) |
             (data["gate_id"].eq(4) & data["gs2"].eq(11) & data["gs-1"].eq(4) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(4) & data["gs1"].eq(4) & data["gs3"].eq(11) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(11) & data["gs-1"].eq(4) & data["gs-3"].eq(4)) |
             (data["gate_id"].eq(4) & data["gs1"].eq(11) & data["gs-2"].eq(4) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(4) & data["gs2"].eq(4) & data["gs3"].eq(11) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(11) & data["gs-3"].eq(4) & data["gs-4"].eq(4)) |
             (data["gate_id"].eq(4) & data["gs3"].eq(11) & data["gs-1"].eq(4) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(4) & data["gs1"].eq(4) & data["gs4"].eq(11) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(11) & data["gs-2"].eq(4) & data["gs-4"].eq(4)) |
             (data["gate_id"].eq(4) & data["gs2"].eq(11) & data["gs-2"].eq(4) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(4) & data["gs2"].eq(4) & data["gs4"].eq(11) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(11) & data["gs-1"].eq(4) & data["gs-4"].eq(4)) |
             (data["gate_id"].eq(4) & data["gs1"].eq(11) & data["gs-3"].eq(4) & data["ls1"].eq(1)) |
             (data["gate_id"].eq(4) & data["gs3"].eq(4) & data["gs4"].eq(11) & data["ls1"].eq(1)), "1144"] = 1
    # начальная(не очень умная) реализация паттерна 7-3-3-10
    data["73310"] = data["gate_id"][(data["gate_id"].eq(7) & data["gs-1"].eq(3) & data["gs-2"].eq(3) & data["gs-3"].eq(10)) |
                                   (data["gate_id"].eq(3) & data["gs1"].eq(7) & data["gs-1"].eq(3) & data["gs-2"].eq(10)) |
                                   (data["gate_id"].eq(3) & data["gs1"].eq(3) & data["gs2"].eq(7) & data["gs-1"].eq(10)) |
                                   (data["gate_id"].eq(10) & data["gs1"].eq(3) & data["gs2"].eq(3) & data["gs3"].eq(7))].apply(lambda x: 1 if x >= 1 else 0)
    # паттерн 7-9-9-5-5-10
    data["6row"] = data["gate_id"][(data["gate_id"].eq(7) & data["gs-1"].eq(9) & data["gs-2"].eq(9) & data["gs-3"].eq(5) & data["gs-4"].eq(5) & data["gs-5"].eq(10)) |
                                   (data["gate_id"].eq(9) & data["gs-1"].eq(9) & data["gs-2"].eq(5) & data["gs-3"].eq(5) & data["gs-4"].eq(10) & data["gs1"].eq(7)) |
                                   (data["gate_id"].eq(9) & data["gs-1"].eq(5) & data["gs-2"].eq(5) & data["gs-3"].eq(10) & data["gs1"].eq(9) & data["gs2"].eq(7)) |
                                   (data["gate_id"].eq(5) & data["gs-1"].eq(5) & data["gs-2"].eq(10) & data["gs1"].eq(9) & data["gs2"].eq(9) & data["gs3"].eq(7)) |
                                   (data["gate_id"].eq(5) & data["gs-1"].eq(10) & data["gs1"].eq(5) & data["gs2"].eq(9) & data["gs3"].eq(9) & data["gs4"].eq(7)) |
                                    (data["gate_id"].eq(10) & data["gs1"].eq(5) & data["gs2"].eq(5) & data["gs3"].eq(9) & data["gs4"].eq(9) & data["gs5"].eq(7))]\
                                    .apply(lambda x: 1 if x >= 1 else 0)
    # паттерн 9-9-5-5-10
    data["5row"] = data["gate_id"][(data["gate_id"].eq(9) & data["gs-1"].eq(9) & data["gs-2"].eq(5) & data["gs-3"].eq(5) & data["gs-4"].eq(10)) |
                                   (data["gate_id"].eq(9) & data["gs1"].eq(9) & data["gs-1"].eq(5) & data["gs-2"].eq(5) & data["gs-3"].eq(10)) |
                                   (data["gate_id"].eq(5) & data["gs1"].eq(9) & data["gs2"].eq(9) & data["gs-1"].eq(5) & data["gs-2"].eq(10)) |
                                   (data["gate_id"].eq(5) & data["gs1"].eq(5) & data["gs2"].eq(9) & data["gs3"].eq(9) & data["gs-1"].eq(10)) |
                                   (data["gate_id"].eq(10) & data["gs1"].eq(5) & data["gs2"].eq(5) & data["gs3"].eq(9) & data["gs4"].eq(9))]\
                                    .apply(lambda x: 1 if x >= 1 else 0)
    data = data.replace(np.nan, 0) # Замена NaN из колонок со смещением и не только
    data["beep_count"] = data.groupby("date").gate_id.transform("count") # Подсчет количества срабатываний за день
    # Переводим типы данных в минимально допустимые - экономим ресурсы
    data = data.astype({"row_id": "int32", "user_id": "int8", "gate_id": "int8", "1day": "int8", "last_day": "int8",
                        "DofW": "int8", "DofY1": "int16", "month": "int8", "day": "int8", "is_weekend": "int8",
                        "hour": "int8", "m1": "int8", "m2": "int8", "m3": "int8", "m4": "int8", "d1": "int8",
                        "d2": "int8", "d3": "int8", "d4": "int8", "e1": "int8", "e2": "int8", "e3": "int8",
                        "e4": "int8", "e5": "int8", "e6": "int8", "min": "int8", "sec": "int8", "lag": "int8",
                        "lag1": "int8", "lag2": "int8", "lag3": "int8", "lag4": "int8", "lag5": "int8", "lag6": "int8",
                        "lag7": "int8", "lag8": "int8", "lag9": "int8", "lag0": "int8", "lagx": "int8", "gs1": "int8",
                        "gs2": "int8", "gs3": "int8", "gs4": "int8", "gs5": "int8", "gs-1": "int8", "gs-2": "int8",
                        "gs-3": "int8", "gs-4": "int8", "gs-5": "int8", "double": "int8", "double2": "int8",
                        "double3": "int8", "double4": "int8", "double5": "int8", "double6": "int8", "double7": "int8",
                        "double8": "int8", '6row': 'int8', '5row': 'int8', "beep_count": "int16",
                        "1144": "int8", "73310": "int8", "l-1": "int8", "ls1": "int8", "xxy": "int8"})
    # data = data.astype({"row_id": "int32", "user_id": "category", "gate_id": "category", "1day": "category", "last_day": "category",
    #                     "DofW": "category", "DofY1": "int16", "month": "category", "day": "category", "is_weekend": "category",
    #                     "hour": "category", "m1": "category", "m2": "category", "m3": "category", "m4": "category", "d1": "category",
    #                     "d2": "category", "d3": "category", "d4": "category", "e1": "category", "e2": "category", "e3": "category",
    #                     "e4": "category", "e5": "category", "e6": "category", "min": "category", "sec": "int8", "lag": "category",
    #                     "lag1": "category", "lag2": "category", "lag3": "category", "lag4": "category", "lag5": "category", "lag6": "category",
    #                     "lag7": "category", "lag8": "category", "lag9": "category", "lag0": "int8", "lagx": "category", "gs1": "int8",
    #                     "gs2": "int8", "gs3": "int8", "gs4": "int8", "gs5": "int8", "gs-1": "int8", "gs-2": "int8",
    #                     "gs-3": "int8", "gs-4": "int8", "gs-5": "int8", "double": "category", "double2": "category",
    #                     "double3": "category", "double4": "category", "double5": "category", "double6": "category", "double7": "category",
    #                     "double8": "category", '6row': 'category', '5row': 'category', "beep_count": "int16"})

    return data

train_data = preprocess(train_data)

# Список всех Используемых фичей здесь
df_x = train_data[["gate_id", "DofW", "is_weekend", "1day", "last_day", "hour", "min", "sec", "lag", "lag1", "lag2", "lag3", "lag4", "lag5", "lag6", "lag7", "lag8", "lag9", "lag0",
                   "lagx", "DofY1","m1","m2","m3","m4","d1","d2","d3","d4","e1","e2","e3","e4","e5","e6", "6row", "5row", "beep_count", "1144", "73310", "xxy",
                   "double", "double2", "double3", "double4", "double5", "double6", "double7", "double8"]]

df_y = train_data["user_id"]
data_x = np.array(df_x)
# print(data_x)
# Здесь были попытки преобразований. Результат SCORE был меньше, но отправлять решение регрессии обученной на таких данных я не пробовал
# scaled_x = StandardScaler().fit_transform(data_x, y=np.array(df_y))
# scaled_x = MinMaxScaler().fit_transform(data_x)

data_y = np.array(df_y)
# scaled_y = StandardScaler().fit_transform(data_y)
# scaled_y = MinMaxScaler().fit_transform((data_y))
# print(scaled_x)
# print(scaled_y)

######model = LogisticRegression(random_state=None, multi_class="ovr", solver="newton-cholesky", penalty='none', n_jobs=8).fit(data_x, data_y) ## 0.179273
model = LogisticRegression(random_state=None, multi_class="ovr", solver="newton-cholesky", penalty='none', n_jobs=16, max_iter=100).fit(data_x, data_y)
print("SCORE", model.score(data_x, data_y))

df_test = test_data[["gate_id", "DofW", "is_weekend", "1day", "last_day", "hour", "min", "sec", "lag", "lag1", "lag2", "lag3", "lag4", "lag5", "lag6", "lag7", "lag8", "lag9", "lag0",
                   "lagx", "DofY1","m1","m2","m3","m4","d1","d2","d3","d4","e1","e2","e3","e4","e5","e6", "6row", "5row", "beep_count", "1144", "73310", "xxy",
                   "double", "double2", "double3", "double4", "double5", "double6", "double7", "double8"]]
test = np.array(df_test)
arr = model.predict(test)
sub = pd.DataFrame()
sub["row_id"] = test_data["row_id"]
sub["target"] = arr
sub.to_csv("sub11.csv", index=False)