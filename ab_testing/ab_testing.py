######################################################
# Temel İstatistik Kavramları
######################################################

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


############################
# Sampling (Örnekleme)
############################

populasyon = np.random.randint(0, 80, 10000)
populasyon.mean()

np.random.seed(115)

orneklem = np.random.choice(a=populasyon, size=100)
orneklem.mean()


np.random.seed(10)
orneklem1 = np.random.choice(a=populasyon, size=100)
orneklem2 = np.random.choice(a=populasyon, size=100)
orneklem3 = np.random.choice(a=populasyon, size=100)
orneklem4 = np.random.choice(a=populasyon, size=100)
orneklem5 = np.random.choice(a=populasyon, size=100)
orneklem6 = np.random.choice(a=populasyon, size=100)
orneklem7 = np.random.choice(a=populasyon, size=100)
orneklem8 = np.random.choice(a=populasyon, size=100)
orneklem9 = np.random.choice(a=populasyon, size=100)
orneklem10 = np.random.choice(a=populasyon, size=100)

(orneklem1.mean() + orneklem2.mean() + orneklem3.mean() + orneklem4.mean() + orneklem5.mean()
 + orneklem6.mean() + orneklem7.mean() + orneklem8.mean() + orneklem9.mean() + orneklem10.mean()) / 10


############################
# Descriptive Statistics (Betimsel İstatistikler)
############################

df = sns.load_dataset("tips")
df.describe().T

############################
# Confidence Intervals (Güven Aralıkları)
############################

# Tips Veri Setindeki Sayısal Değişkenler için Güven Aralığı Hesabı
df = sns.load_dataset("tips")
df.describe().T

df.head()

# DescrStatsW ->
sms.DescrStatsW(df["total_bill"]).tconfint_mean() # tconfint -> Guven araligi hesabi
# ilgili restoran sahibi olarak müşterilerinin ödedikleri hesap ortalama ücretleri istatistiki olarak % 95 güven oranı % 5
# hata payı ile 18.66 ile 20.90 arasındadır.

sms.DescrStatsW(df["tip"]).tconfint_mean()

# Titanic Veri Setindeki Sayısal Değişkenler için Güven Aralığı Hesabı
df = sns.load_dataset("titanic")
df.describe().T
sms.DescrStatsW(df["age"].dropna()).tconfint_mean() # önce eksik değerler dropna çıkar sonra tconfint_mean al

sms.DescrStatsW(df["fare"].dropna()).tconfint_mean()


######################################################
# Correlation (Korelasyon)
######################################################

# Bahşiş veri seti:
# total_bill: yemeğin toplam fiyatı (bahşiş ve vergi dahil)
# tip: bahşiş
# sex: ücreti ödeyen kişinin cinsiyeti (0=male, 1=female)
# smoker: grupta sigara içen var mı? (0=No, 1=Yes)
# day: gün (3=Thur, 4=Fri, 5=Sat, 6=Sun)
# time: ne zaman? (0=Day, 1=Night)
# size: grupta kaç kişi var?

df = sns.load_dataset('tips')
df.head()

df["total_bill"] = df["total_bill"] - df["tip"]

df.plot.scatter("tip", "total_bill") # saçılım grafiği
plt.show()

df["tip"].corr(df["total_bill"]) # tip ile total bill korelasyon
# toplam hesap ile bırakılan tips ler arasında pozitif yönlü orta şiddetli(biraz üstü) bir ilişki vardır
# ödenen hesap miktarı arttıkça bahşişinde birlikte artacağını söyleyebiliriz


######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

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


############################
# Uygulama 1: Sigara İçenler ile İçmeyenlerin Hesap Ortalamaları Arasında İst Ol An Fark var mı?
############################

df = sns.load_dataset("tips")
df.head()

# genel matematiksel ortalama hesabi
df.groupby("smoker").agg({"total_bill": "mean"})


############################
# 1. Hipotezi Kur
############################

# H0: M1 = M2 -> (Sigara icenler ile icmeyenlerin odedikleri hesap ortalamalari arasinda fark yoktur)
# H1: M1 != M2


############################
# 2. Varsayım Kontrolü
############################

# Normallik Varsayımı -> Hipotez testi (Bir degiskenin dagilimi standart normal dagilima benzer mi?)
# Varyans Homojenliği ->


############################
# Normallik Varsayımı
############################

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.

test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"]) # shapiro -> normal dagilim testi
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

# 0.0002 oldugundan normallik varsayimi saglanmamaktadir

test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value = 0.0000 yani < 0.05 bu sebeple H0 RED. Yani normal dagilim varsayimi saglanmamaktadir diyip non-parametric testi uygula


############################
# Varyans Homojenligi Varsayımı
############################

# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"] == "No", "total_bill"]) # varyans homojenligi testi&homojen dagilim
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

# p-value = 0.0452 < 0.05 yani H0 RED, varyanslar homoen degilmis.


############################
# 3 ve 4. Hipotezin Uygulanması
############################

# 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)


############################
# 1.1 Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
############################

test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"],
                              equal_var=True) # T testi
# ihtimal 1 = varsayim saglaniyor normallik varsayimi saglaniyor olsaydi equal_var=True
# ihtimal 2 = varsayim saglaniyor olsaydi hem normallik hem de varyans homojenligi saglaniyor olsaydi equal_var=True
# ihtimal 3 = varsayim saglaniyor olsaydi normallik saglaniyor vasyans homojenligi saglanmiyor olsaydi equal_var=False
# ve ihtimal 3 sonucunda "welch testi" yani "parametrik olmayan tek değişkenli/t testi alternatifi/tek varyansli" testi yapar.


print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

# p-value = 0.18 > 0.05 yani HO REDDEDILEMEZ. "Sigara icen ile icmeyenlerin odedigi ortalama hesap arasinda fark yoktur" yorumu yapilir.


############################
# 1.2 Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
############################

test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"]) # non-parametrik 2 grup karsilastirma testi

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# HO i reddederiz ya da reddedemeyiz. Farkli bir yorum yapilamaz. H1 kabul veya red yorumlari yapilamaz.

# Bu ornekte p-value = 0.34 > 0.05 yani H0 REDDEDILEMEZ yorumu gecerlidir. Fark yoktur.


############################
# Uygulama 2: Titanic Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. var mıdır?
############################

df = sns.load_dataset("titanic")
df.head()

df.groupby("sex").agg({"age": "mean"})


# 1. Hipotezleri kur:
# H0: M1  = M2 (Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. Yoktur)
# H1: M1! = M2 (... vardır)


# 2. Varsayımları İncele

# Normallik varsayımı
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır


test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Varyans homojenliği
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(df.loc[df["sex"] == "female", "age"].dropna(),
                           df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Varsayımlar sağlanmadığı için nonparametrik

test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                                 df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value 0.0261 < 0.05 H0 REDDEDILIR, "Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. vardır!"

# 90 280 gibi ort sahip durumlarda cokta ihtiyac duyulmaz bu testlere cunku sonuc bariz bir sekilde farklidir


############################
# Uygulama 3: Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark var mıdır?
############################

df = pd.read_csv("datasets/diabetes.csv")
df.head()

df.groupby("Outcome").agg({"Age": "mean"}) # outcome(diyabet) olma-olmama durumu yaslara gore ort

# 1. Hipotezleri kur
# H0: M1 = M2
# Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark Yoktur
# H1: M1 != M2
# .... vardır.

# 2. Varsayımları İncele

# Normallik Varsayımı (H0: Normal dağılım varsayımı sağlanmaktadır.)
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value 0.0000 < 0.05 H0 RED


# Normallik varsayımı sağlanmadığı için nonparametrik. / medyan kiyasi / siralama kiyasi

# Hipotez (H0: M1 = M2)
test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                 df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value 0.0000 < 0.05 H0 RED, "Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark vardır."


###################################################
# İş Problemi: Kursun Büyük Çoğunluğunu İzleyenler ile İzlemeyenlerin Puanları Birbirinden Farklı mı?
###################################################

# H0: M1 = M2 (... iki grup ortalamaları arasında ist ol.anl.fark yoktur.)
# H1: M1 != M2 (...vardır)

df = pd.read_csv("datasets/course_reviews.csv")
df.head()

df[(df["Progress"] > 75)]["Rating"].mean()

df[(df["Progress"] < 25)]["Rating"].mean()

# df[(df["Progress"] < 10)]["Rating"].mean() # deger azaldikca fark degismeye devam ediyor


test_stat, pvalue = shapiro(df[(df["Progress"] > 75)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


test_stat, pvalue = shapiro(df[(df["Progress"] < 25)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value 0.0000 < 0.05 H0 REDDEDILIR

test_stat, pvalue = mannwhitneyu(df[(df["Progress"] > 75)]["Rating"],
                                 df[(df["Progress"] < 25)]["Rating"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value 0.0000 < 0.05 H0 REDDEDILIR, "iki grup ortalamaları arasında ist ol.anl.fark vardır"


######################################################
# AB Testing (İki Örneklem Oran Testi)
######################################################

# GOZLEM SAYISI > 30 OLMALI

# H0: p1 = p2
# Yeni Tasarımın Dönüşüm Oranı ile Eski Tasarımın Dönüşüm Oranı Arasında İst. Ol. Anlamlı Farklılık Yoktur.
# H1: p1 != p2
# ... vardır

#                           Birinci Grup             Ikinci Grup

# Kullanici Sayisi          1000                      1100

# Carpan-Kaydolan Sayisi    300                       250

basari_sayisi = np.array([300, 250])
gozlem_sayilari = np.array([1000, 1100])

proportions_ztest(count=basari_sayisi, nobs=gozlem_sayilari) # Z Testi

# p-value 0.0000 < 0.05 H0 REDDEDILIR, "P1 P2 Oranlari arasinda /
# Yeni Tasarımın Dönüşüm Oranı ile Eski Tasarımın Dönüşüm Oranı Arasında İst. Ol. Anlamlı Farklılık vardır"

basari_sayisi / gozlem_sayilari # bu sekilde basitce hangi oranin daha iyi oldugunu gozlemleyelim
# P1 Orani daha iyi


############################
# Uygulama: Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Farklılık var mıdır?
############################

# H0: p1 = p2 / p1 - p2 = 0
# Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Fark yoktur

# H1: p1 != p2
# .. vardır

df = sns.load_dataset("titanic")
df.head()

# hizli degerlendirme : kadinlarin hayatta kalma oranlari erkeklere oranla bariz belli
df.loc[df["sex"] == "female", "survived"].mean()

# hizli degerlendirme
df.loc[df["sex"] == "male", "survived"].mean()

# basari sayisi : hayatta kalan kadin/erkek sayisini bulma
female_succ_count = df.loc[df["sex"] == "female", "survived"].sum()
male_succ_count = df.loc[df["sex"] == "male", "survived"].sum()

# gozlem sayisi : df.loc[df["sex"] == "male", "survived"].shape[0] direk kod icinde

test_stat, pvalue = proportions_ztest(count=[female_succ_count, male_succ_count],
                                      nobs=[df.loc[df["sex"] == "female", "survived"].shape[0],
                                            df.loc[df["sex"] == "male", "survived"].shape[0]])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value 0.0000 < 0.05 H0 REDDEDILIR, "Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Fark vardır"


######################################################
# ANOVA (Analysis of Variance)
######################################################

# İkiden fazla grup ortalamasını karşılaştırmak için kullanılır.

df = sns.load_dataset("tips")
df.head()

# hizli degerlendirme: gunler ort arasinda fark varmi
df.groupby("day")["total_bill"].mean()

# 1. Hipotezleri kur
list(df["day"].unique()) # gun sayisi 4 =  grup sayisi
# HO: m1 = m2 = m3 = m4
# Grup ortalamaları arasında fark yoktur.

# H1: .. fark vardır

# 2. Varsayım kontrolü

# Normallik varsayımı
# Varyans homojenliği varsayımı

# Varsayım sağlanıyorsa one way anova
# Varsayım sağlanmıyorsa kruskal

# H0: Normal dağılım varsayımı sağlanmaktadır.

# degiskeni liste icine attigimiz grup degerleri icinde dondur + shapiro testi ile ayni normallik degerleri bul + 1. index yazdir
for group in list(df["day"].unique()):
    pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]
    print(group, 'p-value: %.4f' % pvalue)

# p-value en az biri icin bile < 0.05 yani H0 REDDEDILIR. Hic bir grup icin normallik varsayimi saglanmamaktadir


# non-parametric testi yapilmali ama yine de inceleyelim
# H0: Varyans homojenliği varsayımı sağlanmaktadır.

test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                           df.loc[df["day"] == "Sat", "total_bill"],
                           df.loc[df["day"] == "Thur", "total_bill"],
                           df.loc[df["day"] == "Fri", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value 0.5741 > 0.05 H0 REDDEDILEMEZ, Varyans homojenliği varsayımı sağlanmaktadır(AMA NORMALLIK SAGLANMIYOR ZATEN)
# HER TURLU NON-PARAMETRIC TESTE GIDECEZ



# 3. Hipotez testi ve p-value yorumu

# !!!!!!****** DIKKAT ******!!!!!!
# gruplarin ort ve medyanlari arasinda farklar var ama kayda deger olup olmadigini degerlendiremeyiz
# bu ort arasinda fark var mi yokmu sorusunu genelden sormak ile iki den fazla grup testi ile gruplarin arasinda fark var mi yokmu test etmek ile
# gruplarin ayri ayri ort arasinda fark varmi yokmu test etmek farkli seylerdir.
# grup ici ve gruplar arasi degiskenlik goz onunde bulundurulur bu bir parametredir, hesaplanacak olan test istatistigi/p-value hesabi ile/ yapilacak yorum ile
# gruplar kendi icinde 2 li karsilastirildiginda ortaya cikacak sonuc farkli olacaktir


# Hiç biri sağlamıyor.

# hizli degerlendirme:
df.groupby("day").agg({"total_bill": ["mean", "median"]})


# Varsayim Homojenligi: (Yine de gozlemleyelim)
# HO: Grup ortalamaları arasında ist ol anl fark yoktur

# Varsayim saglaniyor ise;
# parametrik anova testi: Tek yonlu parametrik test/f istatistigi ve p-value degeri dondurur
f_oneway(df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Sun", "total_bill"])

# p-value degeri 0.042 < 0.05 H0 REDDEDILIR, gruplar arasinda anlamli bir fark vardir(AMA NORMALLIK SAGLANMIYOR ZATEN)


# Varsayim saglanmiyor ise;
# Nonparametrik anova testi: kruskal testi ve p-value degeri dondurur
kruskal(df.loc[df["day"] == "Thur", "total_bill"],
        df.loc[df["day"] == "Fri", "total_bill"],
        df.loc[df["day"] == "Sat", "total_bill"],
        df.loc[df["day"] == "Sun", "total_bill"])

# p-value degeri 0.015 < 0.05 H0 REDDEDILIR, gruplar arasinda anlamli bir fark vardir

# DIKKAT BURADA BASKA BIR PROBLEM VAR, ONEMLI BIR NOKTA
# Burada fark istatistiksel olarak fark oldugu kanaatine vardik fakat hangi gruptan kaynaklaniyor

# farkin kimden kaynaklandigini verir
from statsmodels.stats.multicomp import MultiComparison # coklu karsilastirma fonksiyonu import
comparison = MultiComparison(df['total_bill'], df['day'])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary()) # tukey testi, print zorunlu ipy desteklemiyor

# Multiple Comparison of Means - Tukey HSD, FWER=0.05
# ====================================================
# group1 group2 meandiff p-adj   lower   upper  reject
# ----------------------------------------------------
#    Fri    Sat   3.2898 0.4541 -2.4799  9.0595  False
#    Fri    Sun   4.2584 0.2371 -1.5856 10.1025  False
#    Fri   Thur   0.5312 0.9957 -5.4434  6.5057  False
#    Sat    Sun   0.9686 0.8968 -2.6088   4.546  False
#    Sat   Thur  -2.7586 0.2374 -6.5455  1.0282  False
#    Sun   Thur  -3.7273 0.0668 -7.6264  0.1719  False
# ----------------------------------------------------

# p-adj: ikili karsilastirma p-value degeri
#
# burada hicbir ikili arasinda H0 REDDEDILEMEZ sonucu cikiyor ama nasil?
# bu durumda ikili karsilastirma sonuclarina bakilarak burada fark yokmus gibi dusunulebilir(alfa degeri 0.05 ile oynanabilir, farkli deger cikabilir)
# burada hangi grubun agir bastigini gormemiz gerekirdi ya da bunun icin alfa degeri ile oynayarak/tolerans gostererek farkli sonuclar arayalim
#
# alfa 0.01 ise;
# from statsmodels.stats.multicomp import MultiComparison
# comparison = MultiComparison(df['total_bill'], df['day'])
# tukey = comparison.tukeyhsd(0.01)
# print(tukey.summary())
#
# sonuc ayni hepsi false/ H0 REDDEDILEMEZ, cunku zaten girilen deger sonucu uc noktalara yine pozitif tarafa cekti
#
#
# alfa 0.10 ise;
# from statsmodels.stats.multicomp import MultiComparison
# comparison = MultiComparison(df['total_bill'], df['day'])
# tukey = comparison.tukeyhsd(0.05)
# print(tukey.summary())
#
# burada Sun Thur ikili karsilastirmasinin p-value degerinin uc noktasi/upper degeri < 0.05 H0 REDDEDILIR sonucuna vardir.
# Bu ikilinin kruskal testi sonucunu dogurdugunu / diger gruplara nazaran agir bastigini ifade edebiliriz.
#
# ilgili dagilim yapilarinin(z, t, f) karar noktalari bu dagilim uc noktalaridir. bu uc noktalara gore bir degerlendirme yapildiginda
# uc noktanin tanimi 0.05 tir yayginca kullanilan. Dolayisiyla bunu referans alinca biz birde kendi calismamizdan bir test istatistigi hesapliyoruz ya
# o test ist olasiliksal karsiligidir p-value degeri. Dolayisiyla calisma basindaki alfa 0.05 ile red bolgesinin sinirini belirleyen deger ile
# bizim belirledigimiz degeri kiyasladigimizda 0.05 ten kucuk ise H0 Reddederiz. Cunku belirledigimiz o sinir degeri olan alfa dan daha asagida
# demek ki iyice bir farklilik var. Bu deger degistirilemez mi evet degistirilir.
# O halde nasil ilerlemeliyiz??? Yaygin deger 0.05 olarak kabul etmeliyiz.
#
# Sadece o agir basan grubu gormek icin yaptik. Ama zaten p-adj ye baktigimizda da en dusuk degere sahip grup Sun-Thur idi.
#