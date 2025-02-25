###################################################
# Rating Products
###################################################

# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating


############################################
# Uygulama: Kullanıcı ve Zaman Ağırlıklı Kurs Puanı Hesaplama
############################################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None) # tüm sütun tek satır - 1
pd.set_option('display.max_rows', None) # tüm sütun tek satır - 2
pd.set_option('display.width', 500) # tüm sütun tek satır / alternatif
pd.set_option('display.expand_frame_repr', False) # tüm sütun tek satır - 3
pd.set_option('display.float_format', lambda x: '%.5f' % x) # ondalık olarak sıfırdan sonra 5 basamak

# (50+ Saat) Python A-Z™: Veri Bilimi ve Machine Learning
# Puan: 4.8 (4.764925)
# Toplam Puan: 4611
# Puan Yüzdeleri: 75, 20, 4, 1, <1
# Yaklaşık Sayısal Karşılıkları: 3458, 922, 184, 46, 6

df = pd.read_csv("datasets/course_reviews.csv")
df.head()
df.shape

# rating dagılımı
df["Rating"].value_counts()

# sorulan sorular
df["Questions Asked"].value_counts()

# ortalama puan
df.groupby("Questions Asked").agg({"Questions Asked": "count",
                                   "Rating": "mean"})


df.head()

####################
# Average
####################

# Ortalama Puan
df["Rating"].mean()

####################
# Time-Based Weighted Average
####################
# Puan Zamanlarına Göre Ağırlıklı Ortalama

df.head()
df.info()

df["Timestamp"] = pd.to_datetime(df["Timestamp"]) # type i datetime yap

current_date = pd.to_datetime('2021-02-10 0:0:0') # guncel kontrol tarihi belirleyip o tarih baz alinarak islem yapma

df["days"] = (current_date - df["Timestamp"]).dt.days # tarihleri gun cinsinden ifade etme

df[df["days"] <= 30].count() # son 30 gun yorum sayisi
df.loc[df["days"] <= 30, "Rating"].mean() # 30 gun ve daha yakin gunlerin ortalamasi

df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() # 30 dan fazla 90 dan az gun ortalamasi

df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() # 90 dan fazla 180 den az gun ortalamasi

df.loc[(df["days"] > 180), "Rating"].mean() # 180 gunden fazla gunler ortalamasi

# zamana gore agirlikli ortalamayi hesaplama / gun ortalamalarina agirliklar ekleme
# \ asagida kod yazmaya devam icin
# agirliklar toplami 100 olmali. yakin tarihe verilen agirlik ve geri gidildikce agirligi azalttik cunku son zamanlarda
# egitime olan ilgi azalmis durumda
df.loc[df["days"] <= 30, "Rating"].mean() * 28/100 + \
    df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26/100 + \
    df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24/100 + \
    df.loc[(df["days"] > 180), "Rating"].mean() * 22/100

# pratik fonk yazma
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > 180), "Rating"].mean() * w4 / 100

time_based_weighted_average(df)

time_based_weighted_average(df, 30, 26, 22, 22) # yeni agirlik degerleri deneme



####################
# User-Based Weighted Average
####################
# Kullanıcı İzleme Oranlarına Göre Ağırlıklı Ortalama

df.head()

# Progress: Kursun ne kadar izlendiği
df.groupby("Progress").agg({"Rating": "mean"}) # izleme oranlari / puan ortalamalari

# izleme oranlarina gore puan ortalamalarini agirliklandirarak hesaplama
df.loc[df["Progress"] <= 10, "Rating"].mean() * 22 / 100 + \
    df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24 / 100 + \
    df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26 / 100 + \
    df.loc[(df["Progress"] > 75), "Rating"].mean() * 28 / 100


# pratik fonk yazma/ agirliklar toplami 100 olmali / izleme oranlarina gore puan ortalamalarini agirliklandirarak hesaplama
def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 75), "Rating"].mean() * w4 / 100

user_based_weighted_average(df)
user_based_weighted_average(df, 20, 24, 26, 30)


####################
# Weighted Rating
####################

def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w/100 + user_based_weighted_average(dataframe)*user_w/100

course_weighted_rating(df)

course_weighted_rating(df, time_w=40, user_w=60)

# on tanimli degiskenleri uyarlanabilir yaptim
# def course_weighted_rating_detailed(dataframe, time_w=50, tw1=28, tw2=26, tw3=24, tw4=22, user_w=50, uw1=22, uw2=24, uw3=26, uw4=28):
#     return time_based_weighted_average(dataframe, tw1, tw2, tw3, tw4) * time_w/100 + user_based_weighted_average(dataframe, uw1, uw2, uw3, uw4) * user_w/100
#
# course_weighted_rating_detailed(df)
# course_weighted_rating_detailed(df, time_w=40, tw1=30, tw2=26, tw3=22, tw4=22, user_w=60, uw1=20, uw2=24, uw3=26, uw4=30)
#






