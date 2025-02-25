########################################################################################################################
# MEASUREMENT PROBLEMS
########################################################################################################################

# 1 - Rating Products
# 2 - Sorting Products
# 3 - Sorting Reviews
# 4 - AB Testing

########################################################################################################################
# 1 - Rating Products
########################################################################################################################

# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating

# df.groupby("Questions Asked").agg({"Questions Asked": "count",
#                                    "Rating": "mean"})

########################################
# Average
########################################

# df["Timestamp"] = pd.to_datetime(df["Timestamp"])
#
# current_date = pd.to_datetime('2021-02-10 0:0:0')
#
# df["days"] = (current_date - df["Timestamp"]).dt.days



########################################
# Time-Based Weighted Average
########################################

# def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
#     return dataframe.loc[df["days"] <= 30, "Rating"].mean() * w1 / 100 + \
#            dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
#            dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
#            dataframe.loc[(dataframe["days"] > 180), "Rating"].mean() * w4 / 100
#
# time_based_weighted_average(df)
#
# time_based_weighted_average(df, 30, 26, 22, 22)



########################################
# User-Based Weighted Average
########################################

# df.groupby("Progress").agg({"Rating": "mean"})
#
# def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
#     return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
#            dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
#            dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
#            dataframe.loc[(dataframe["Progress"] > 75), "Rating"].mean() * w4 / 100
#
# user_based_weighted_average(df)
# user_based_weighted_average(df, 20, 24, 26, 30)



########################################
# Weighted Rating
########################################

# def course_weighted_rating(dataframe, time_w=50, user_w=50):
#     return time_based_weighted_average(dataframe) * time_w/100 + user_based_weighted_average(dataframe)*user_w/100
#
# course_weighted_rating(df)
#
# course_weighted_rating(df, time_w=40, user_w=60)

# Ön Tanımlı Değişkenler Uyarlanabilir:
# def course_weighted_rating_detailed(dataframe, time_w=50, tw1=28, tw2=26, tw3=24, tw4=22, user_w=50, uw1=22, uw2=24, uw3=26, uw4=28):
#     return time_based_weighted_average(dataframe, tw1, tw2, tw3, tw4) * time_w/100 + user_based_weighted_average(dataframe, uw1, uw2, uw3, uw4) * user_w/100
#
# course_weighted_rating_detailed(df)
# course_weighted_rating_detailed(df, time_w=40, tw1=30, tw2=26, tw3=22, tw4=22, user_w=60, uw1=20, uw2=24, uw3=26, uw4=30)





########################################################################################################################
# 2 - Sorting Products
########################################################################################################################

########################################
# 1 - Weighted Sorting Score
########################################

# df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
#     fit(df[["purchase_count"]]). \
#     transform(df[["purchase_count"]])
# df.head(10)
#
# df.describe().T
#
# df["comment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
#     fit(df[["commment_count"]]). \
#     transform(df[["commment_count"]])
# df.head(10)
#
# def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
#     return (dataframe["comment_count_scaled"] * w1 / 100 +
#             dataframe["purchase_count_scaled"] * w2 / 100 +
#             dataframe["rating"] * w3 / 100)
#
# df["weighted_sorting_score"] = weighted_sorting_score(df)
#
# df["weighted_sorting_score"] = weighted_sorting_score(df, w1=30, w2=30, w3=40)
#
# df.sort_values("weighted_sorting_score", ascending=False).head(20)
#
# df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weighted_sorting_score", ascending=False).head(20)


########################################
# 2 - Bayesian Average Rating Score
########################################
#
# Sorting Products with 5 Star Rated
# Sorting Products According to Distribution of 5 Star Rating
#
# def bayesian_average_rating(n, confidence=0.95):
#     if sum(n) == 0:
#         return 0
#     K = len(n)
#     z = st.norm.ppf(1 - (1 - confidence) / 2)
#     N = sum(n)
#     first_part = 0.0
#     second_part = 0.0
#     for k, n_k in enumerate(n):
#         first_part += (k + 1) * (n[k] + 1) / (N + K)
#         second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
#     score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
#     return score
#
#
# df.head()
#
# df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
#                                                                 "2_point",
#                                                                 "3_point",
#                                                                 "4_point",
#                                                                 "5_point"]]), axis=1)
#
# df.sort_values("weighted_sorting_score", ascending=False).head(20)
# df.sort_values("bar_score", ascending=False).head(20)
#
# df[df["course_name"].index.isin([5, 1])].sort_values("bar_score", ascending=False)


########################################
# 3 - Hybrid Sorting: BAR Score + Diğer Faktorler
########################################
#
# def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
#     bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
#                                                                      "2_point",
#                                                                      "3_point",
#                                                                      "4_point",
#                                                                      "5_point"]]), axis=1)
#     wss_score = weighted_sorting_score(dataframe)
#
#     return bar_score*bar_w/100 + wss_score*wss_w/100
#
#
# df["hybrid_sorting_score"] = hybrid_sorting_score(df)
#
# df.sort_values("hybrid_sorting_score", ascending=False).head(20)
#
# df[df["course_name"].str.contains("Veri Bilimi")].sort_values("hybrid_sorting_score", ascending=False).head(20)


########################################
# 4 - Uygulama : IMDB Movie Scoring & Sorting
########################################
#
# import pandas as pd
# import math
# import scipy.stats as st
# pd.set_option('display.max_columns', None)
# pd.set_option('display.expand_frame_repr', False)
# pd.set_option('display.float_format', lambda x: '%.5f' % x)
#
# df = pd.read_csv("datasets/movies_metadata.csv",
#                  low_memory=False)  # DtypeWarning kapamak icin
#
# df = df[["title", "vote_average", "vote_count"]]
#
# df.head()
# df.shape
#
#
########################################
# 4.1 - Vote Average'a Göre Sıralama
########################################
#
# df.sort_values("vote_average", ascending=False).head(20)
#
# df["vote_count"].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T
#
# df[df["vote_count"] > 400].sort_values("vote_average", ascending=False).head(20)
#
# from sklearn.preprocessing import MinMaxScaler
#
# df["vote_count_score"] = MinMaxScaler(feature_range=(1, 10)). \
#     fit(df[["vote_count"]]). \
#     transform(df[["vote_count"]])
#
#
########################################
# 4.2 - vote_average * vote_count
########################################
#
# df["average_count_score"] = df["vote_average"] * df["vote_count_score"]
#
# df.sort_values("average_count_score", ascending=False).head(20)
#
#
########################################
# 4.3 - IMDB Weighted Rating
########################################
#
# weighted_rating = (v/(v+M) * r) + (M/(v+M) * C)
#
# r = vote average
# v = vote count
# M = minimum votes required to be listed in the Top 250
# C = the mean vote across the whole report (currently 7.0)
#
# Film 1:
# r = 8
# M = 500
# v = 1000
#
# (1000 / (1000+500))*8 = 5.33
#
#
# Film 2:
# r = 8
# M = 500
# v = 3000
#
# (3000 / (3000+500))*8 = 6.85
#
#
# Film 3:
# r = 9.5
# M = 500
# v = 1000
#
# (1000 / (1000+500))*9.5 = 6.33
#
#
# Film 1:
# r = 8
# M = 500
# v = 1000
#
# Birinci bölüm:
# (1000 / (1000+500))*8 = 5.33
#
# İkinci bölüm:
# 500/(1000+500) * 7 = 2.33
#
# Toplam = 5.33 + 2.33 = 7.66
#
#
# Film 2:
# r = 8
# M = 500
# v = 3000
#
# Birinci bölüm:
# (3000 / (3000+500))*8 = 6.85
#
# İkinci bölüm:
# 500/(3000+500) * 7 = 1
#
# Toplam = 7.85
#
#
# Film 3:
# r = 9.5
# M = 500
# v = 1000
#
# Birinci bölüm:
# (1000 / (1000+500))*9.5 = 6.33
#
# İkinci bölüm:
# 500/(1000+500) * 7 = 2.33
#
# Toplam = 8.66
#
#
# M = 2500
# C = df['vote_average'].mean()
#
# def weighted_rating(r, v, M, C):
#     return (v / (v + M) * r) + (M / (v + M) * C)
#
# df.sort_values("average_count_score", ascending=False).head(10)
#
# weighted_rating(7.40000, 11444.00000, M, C)
#
# weighted_rating(8.10000, 14075.00000, M, C)
#
# weighted_rating(8.50000, 8358.00000, M, C)
#
# df["weighted_rating"] = weighted_rating(df["vote_average"],
#                                         df["vote_count"], M, C)
#
# df.sort_values("weighted_rating", ascending=False).head(10)
#
#
########################################
# 4.4 - Bayesian Average Rating Score
########################################
#
# def bayesian_average_rating(n, confidence=0.95):
#     if sum(n) == 0:
#         return 0
#     K = len(n)
#     z = st.norm.ppf(1 - (1 - confidence) / 2)
#     N = sum(n)
#     first_part = 0.0
#     second_part = 0.0
#     for k, n_k in enumerate(n):
#         first_part += (k + 1) * (n[k] + 1) / (N + K)
#         second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
#     score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
#     return score
#
# bayesian_average_rating([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351])
#
# bayesian_average_rating([37128, 5879, 6268, 8419, 16603, 30016, 78538, 199430, 402518, 837905])
#
# df = pd.read_csv("datasets/imdb_ratings.csv")
# df = df.iloc[0:, 1:]
#
#
# df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five",
#                                                                 "six", "seven", "eight", "nine", "ten"]]), axis=1)
# df.sort_values("bar_score", ascending=False).head(20)





########################################################################################################################
# 3 - Sorting Reviews
########################################################################################################################

########################################
# 1 - Up-Down Diff Score = (up ratings) − (down ratings)
########################################
#
# Review 1: 600 up 400 down total 1000
# Review 2: 5500 up 4500 down total 10000
#
# def score_up_down_diff(up, down):
#     return up - down
#
# # Review 1 Score:
# score_up_down_diff(600, 400)
#
# # Review 2 Score
# score_up_down_diff(5500, 4500)
#
#
########################################
# 2 - Score = Average rating = (up ratings) / (all ratings)
########################################
#
# def score_average_rating(up, down):
#     if up + down == 0:
#         return 0
#     return up / (up + down)
#
# score_average_rating(600, 400)
# score_average_rating(5500, 4500)
#
# # Review 1: 2 up 0 down total 2
# # Review 2: 100 up 1 down total 101
#
# score_average_rating(2, 0)
# score_average_rating(100, 1)
#
#
########################################
# 3 - Wilson Lower Bound Score
########################################
#
# 600-400
# 0.6
# 0.5 0.7
# 0.5
#
#
# def wilson_lower_bound(up, down, confidence=0.95):
#     """
#     Wilson Lower Bound Score hesapla
#
#     - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
#     - Hesaplanacak skor ürün sıralaması için kullanılır.
#     - Not:
#     Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
#     Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.
#
#     Parameters
#     ----------
#     up: int
#         up count
#     down: int
#         down count
#     confidence: float
#         confidence
#
#     Returns
#     -------
#     wilson score: float
#
#     """
#     n = up + down
#     if n == 0:
#         return 0
#     z = st.norm.ppf(1 - (1 - confidence) / 2)
#     phat = 1.0 * up / n
#     return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)
#
#
# wilson_lower_bound(600, 400)
# wilson_lower_bound(5500, 4500)
#
# wilson_lower_bound(2, 0)
# wilson_lower_bound(100, 1)
#
#
########################################
# Case Study
########################################
#
# up = [15, 70, 14, 4, 2, 5, 8, 37, 21, 52, 28, 147, 61, 30, 23, 40, 37, 61, 54, 18, 12, 68]
# down = [0, 2, 2, 2, 15, 2, 6, 5, 23, 8, 12, 2, 1, 1, 5, 1, 2, 6, 2, 0, 2, 2]
# comments = pd.DataFrame({"up": up, "down": down})
#
#
# # score_pos_neg_diff
# comments["score_pos_neg_diff"] = comments.apply(lambda x: score_up_down_diff(x["up"], x["down"]), axis=1)
#
# # score_average_rating
# comments["score_average_rating"] = comments.apply(lambda x: score_average_rating(x["up"], x["down"]), axis=1)
#
# # wilson_lower_bound
# comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"], x["down"]), axis=1)
#
#
# comments.sort_values("wilson_lower_bound", ascending=False)





########################################################################################################################
# 4 - AB Testing
########################################################################################################################

########################################
# Temel İstatistik Kavramları
########################################

########################################
# Sampling (Örnekleme)
########################################

# populasyon = np.random.randint(0, 80, 10000)
# populasyon.mean()
#
# np.random.seed(115)
#
# orneklem = np.random.choice(a=populasyon, size=100)
# orneklem.mean()


########################################
# Descriptive Statistics (Betimsel İstatistikler)
########################################

# df = sns.load_dataset("tips")
# df.describe().T


########################################
# Confidence Intervals (Güven Aralıkları)
########################################

# # Tips Veri Setindeki Sayısal Değişkenler için Güven Aralığı Hesabı
# df = sns.load_dataset("tips")
# df.describe().T
#
# df.head()
#
# sms.DescrStatsW(df["total_bill"]).tconfint_mean()
# sms.DescrStatsW(df["tip"]).tconfint_mean()
#
# # Titanic Veri Setindeki Sayısal Değişkenler için Güven Aralığı Hesabı
# df = sns.load_dataset("titanic")
# df.describe().T
# sms.DescrStatsW(df["age"].dropna()).tconfint_mean()
#
# sms.DescrStatsW(df["fare"].dropna()).tconfint_mean()


########################################
# Correlation (Korelasyon)
########################################

# Bahşiş veri seti:
# total_bill: yemeğin toplam fiyatı (bahşiş ve vergi dahil)
# tip: bahşiş
# sex: ücreti ödeyen kişinin cinsiyeti (0=male, 1=female)
# smoker: grupta sigara içen var mı? (0=No, 1=Yes)
# day: gün (3=Thur, 4=Fri, 5=Sat, 6=Sun)
# time: ne zaman? (0=Day, 1=Night)
# size: grupta kaç kişi var?
#
# df = sns.load_dataset('tips')
# df.head()
#
# df["total_bill"] = df["total_bill"] - df["tip"]
#
# df.plot.scatter("tip", "total_bill")
# plt.show()
#
# df["tip"].corr(df["total_bill"])


########################################
# AB Testing (Bağımsız İki Örneklem T Testi)
########################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı
#   - 2. Varyans Homojenliği
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direk 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.



########################################
# Uygulama 1: Sigara İçenler ile İçmeyenlerin Hesap Ortalamaları Arasında İst Ol An Fark var mı?
########################################

# df = sns.load_dataset("tips")
# df.head()
#
# df.groupby("smoker").agg({"total_bill": "mean"})


########################################
# 1. Hipotezi Kur
########################################

# H0: M1 = M2
# H1: M1 != M2


########################################
# 2. Varsayım Kontrolü
########################################

# Normallik Varsayımı
# Varyans Homojenliği


########################################
# Normallik Varsayımı
########################################

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.
#
# test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])
# print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#
# # p-value < ise 0.05'ten HO RED.
# # p-value < değilse 0.05 H0 REDDEDILEMEZ.
#
#
# test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
# print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


########################################
# Varyans Homojenligi Varsayımı
########################################

# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir
#
# test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
#                            df.loc[df["smoker"] == "No", "total_bill"])
# print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#
# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.


########################################
# 3 ve 4. Hipotezin Uygulanması
########################################

# 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)


########################################
# 1.1 Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
########################################

# test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
#                               df.loc[df["smoker"] == "No", "total_bill"],
#                               equal_var=True)
#
# print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#
# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.


########################################
# 1.2 Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
########################################

# test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
#                                  df.loc[df["smoker"] == "No", "total_bill"])
#
# print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))



########################################
# Uygulama 2: Titanic Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. var mıdır?
########################################

# df = sns.load_dataset("titanic")
# df.head()
#
# df.groupby("sex").agg({"age": "mean"})
#
#
# 1. Hipotezleri kur:
# H0: M1  = M2 (Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. Yoktur)
# H1: M1! = M2 (... vardır)
#
#
# 2. Varsayımları İncele
#
# Normallik varsayımı
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır
#
#
# test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
# print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#
# test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
# print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#
#
# Varyans homojenliği
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir
#
# test_stat, pvalue = levene(df.loc[df["sex"] == "female", "age"].dropna(),
#                            df.loc[df["sex"] == "male", "age"].dropna())
#
# print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#
# # Varsayımlar sağlanmadığı için nonparametrik
#
# test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
#                                  df.loc[df["sex"] == "male", "age"].dropna())
#
# print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#
# 90 280



########################################
# Uygulama 3: Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark var mıdır?
########################################

# df = pd.read_csv("datasets/diabetes.csv")
# df.head()
#
# df.groupby("Outcome").agg({"Age": "mean"})
#
#
# 1. Hipotezleri kur
# H0: M1 = M2
# Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark Yoktur
# H1: M1 != M2
# .... vardır.
#
#
# 2. Varsayımları İncele
#
# Normallik Varsayımı (H0: Normal dağılım varsayımı sağlanmaktadır.)
# test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
# print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#
# test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
# print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#
#
# Normallik varsayımı sağlanmadığı için nonparametrik.
#
# Hipotez (H0: M1 = M2)
# test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
#                                  df.loc[df["Outcome"] == 0, "Age"].dropna())
# print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))



########################################
# İş Problemi: Kursun Büyük Çoğunluğunu İzleyenler ile İzlemeyenlerin Puanları Birbirinden Farklı mı?
########################################

# H0: M1 = M2 (... iki grup ortalamaları arasında ist ol.anl.fark yoktur.)
# H1: M1 != M2 (...vardır)
#
# df = pd.read_csv("datasets/course_reviews.csv")
# df.head()
#
# df[(df["Progress"] > 75)]["Rating"].mean()
#
# df[(df["Progress"] < 25)]["Rating"].mean()
#
#
# test_stat, pvalue = shapiro(df[(df["Progress"] > 75)]["Rating"])
# print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#
#
# test_stat, pvalue = shapiro(df[(df["Progress"] < 25)]["Rating"])
# print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#
# test_stat, pvalue = mannwhitneyu(df[(df["Progress"] > 75)]["Rating"],
#                                  df[(df["Progress"] < 25)]["Rating"])
#
# print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))



########################################
# AB Testing (İki Örneklem Oran Testi)
########################################

# H0: p1 = p2
# Yeni Tasarımın Dönüşüm Oranı ile Eski Tasarımın Dönüşüm Oranı Arasında İst. Ol. Anlamlı Farklılık Yoktur.
# H1: p1 != p2
# ... vardır
#
# basari_sayisi = np.array([300, 250])
# gozlem_sayilari = np.array([1000, 1100])
#
# proportions_ztest(count=basari_sayisi, nobs=gozlem_sayilari)
#
#
# basari_sayisi / gozlem_sayilari



########################################
# Uygulama: Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Farklılık var mıdır?
########################################

# H0: p1 = p2
# Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Fark yoktur
#
# H1: p1 != p2
# .. vardır
#
# df = sns.load_dataset("titanic")
# df.head()
#
# df.loc[df["sex"] == "female", "survived"].mean()
#
# df.loc[df["sex"] == "male", "survived"].mean()
#
# female_succ_count = df.loc[df["sex"] == "female", "survived"].sum()
# male_succ_count = df.loc[df["sex"] == "male", "survived"].sum()
#
# test_stat, pvalue = proportions_ztest(count=[female_succ_count, male_succ_count],
#                                       nobs=[df.loc[df["sex"] == "female", "survived"].shape[0],
#                                             df.loc[df["sex"] == "male", "survived"].shape[0]])
# print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))



########################################
# ANOVA (Analysis of Variance)
########################################

# İkiden fazla grup ortalamasını karşılaştırmak için kullanılır.
#
# df = sns.load_dataset("tips")
# df.head()
#
# df.groupby("day")["total_bill"].mean()
#
#
# 1. Hipotezleri kur
#
# HO: m1 = m2 = m3 = m4
# Grup ortalamaları arasında fark yoktur.
#
# H1: .. fark vardır
#
#
# 2. Varsayım kontrolü
#
# Normallik varsayımı
# Varyans homojenliği varsayımı
#
# Varsayım sağlanıyorsa one way anova
# Varsayım sağlanmıyorsa kruskal
#
#
# H0: Normal dağılım varsayımı sağlanmaktadır.
#
# for group in list(df["day"].unique()):
#     pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]
#     print(group, 'p-value: %.4f' % pvalue)
#
#
# H0: Varyans homojenliği varsayımı sağlanmaktadır.

# test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
#                            df.loc[df["day"] == "Sat", "total_bill"],
#                            df.loc[df["day"] == "Thur", "total_bill"],
#                            df.loc[df["day"] == "Fri", "total_bill"])
# print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#
#
# 3. Hipotez testi ve p-value yorumu
#
# Hiç biri sağlamıyor.
# df.groupby("day").agg({"total_bill": ["mean", "median"]})
#
#
# HO: Grup ortalamaları arasında ist ol anl fark yoktur
#
# parametrik anova testi:
# f_oneway(df.loc[df["day"] == "Thur", "total_bill"],
#          df.loc[df["day"] == "Fri", "total_bill"],
#          df.loc[df["day"] == "Sat", "total_bill"],
#          df.loc[df["day"] == "Sun", "total_bill"])
#
# Nonparametrik anova testi:
# kruskal(df.loc[df["day"] == "Thur", "total_bill"],
#         df.loc[df["day"] == "Fri", "total_bill"],
#         df.loc[df["day"] == "Sat", "total_bill"],
#         df.loc[df["day"] == "Sun", "total_bill"])
#
# from statsmodels.stats.multicomp import MultiComparison
# comparison = MultiComparison(df['total_bill'], df['day'])
# tukey = comparison.tukeyhsd(0.05)
# print(tukey.summary())





