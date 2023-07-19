import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 50)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak..

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak
# yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

###############################################################
# GÖREVLER
###############################################################

# GÖREV 1: Veriyi Anlama (Data Understanding) ve Hazırlama
# 1. flo_data_20K.csv verisini okuyunuz.

df_ = pd.read_csv(r"C:\Users\90538\Desktop\Miuul\CRMAnalytics\Case_Study\Dataset\flo_data_20k.csv")
df = df_.copy()

# 2. Veri setinde
# a. İlk 10 gözlem,

df.head()

# b. Değişken isimleri,

df.columns

# c. Betimsel istatistik,

df.describe().T

# d. Boş değer,

df.isnull().sum()

# e. Değişken tipleri, incelemesi yapınız.

df.info()


# 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df["total_shopping"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_price"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df.head()

# 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

df.info()

date_cols = df.columns[df.columns.str.contains("date")]
df[date_cols] = df[date_cols].apply(pd.to_datetime)


# 5. Alışveriş kanallarındaki müşteri sayısının, ortalama alınan ürün sayısının ve ortalama harcamaların dağılımına
# bakınız.

df.groupby("order_channel").agg({"master_id": "count",
                                 "order_num_total_ever_online": "mean",
                                 "customer_value_total_ever_online": "mean"})

df.groupby("order_channel").agg({"master_id": "count",
                                 "total_shopping": "mean",
                                 "total_price": "mean"})


# 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.

df.sort_values("total_price", ascending=False)[["master_id", "total_price"]].head(10)

# 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.

df.sort_values("total_shopping", ascending=False)[["master_id", "total_shopping"]].head(10)
df.sort_values("order_num_total_ever_online", ascending=False)[["master_id", "order_num_total_ever_online"]].head(10)

# 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.


def data_prep(dataframe):
    dataframe["total_shopping"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["total_price"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    date_cols = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_cols] = dataframe[date_cols].apply(pd.to_datetime)
    return df

# GÖREV 2: RFM Metriklerinin Hesaplanması

# Recency = Yenilik = Bizden en son ne zaman alışveriş yaptığı
# Frequency = Sıklık = Müşterinin işlem sayısı
# Monetary = Parasal Değer = Müşterinin bize bıraktığı para

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)

rfm = df.groupby("master_id").agg({"last_order_date": lambda d: (today_date - d),
                                   "total_shopping": lambda b: b.sum(),
                                   "total_price": lambda tp: tp.sum()})

rfm.columns = ["recency", "frequency", "monetary"]

rfm.head()

# GÖREV 3: RF ve RFM Skorlarının Hesaplanması

rfm['recency_score'] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm['frequency_score'] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm['monetary_score'] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])
rfm["rf_score"] = rfm["recency_score"].astype(str) + rfm["monetary_score"].astype(str)
rfm.head()

# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['rf_score'].replace(seg_map, regex=True)
rfm.head()


# GÖREV 5: Aksiyon zamanı!
# 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

rfm.groupby("segment").agg({"recency": "mean",
                           "frequency": "mean",
                           "monetary": "mean"})

# 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv ye kaydediniz.
# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri
# tercihlerinin üstünde. Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel
# olarak iletişime geçeilmek isteniliyor. Sadık müşterilerinden(champions,loyal_customers), ortalama 250 TL üzeri ve
# kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kuralacak müşteriler. Bu müşterilerin id
# numaralarını csv dosyasına yeni_marka_hedef_müşteri_id.cvs olarak kaydediniz.
rfm.reset_index(inplace=True)
df = pd.merge(df, rfm, on="master_id")
df_a = df[((df["interested_in_categories_12"].str.contains("KADIN")) & (df["monetary"].mean() > 250)) & ((df["segment"] == "champions") | (df["segment"] == "loyal_customers"))]["master_id"]
df.head()
df_a.to_csv("yeni_marka_hedef_müşteri_id.cvs")
# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen
# geçmişte iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve
# yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına
# indirim_hedef_müşteri_ids.csv olarak kaydediniz.

df_b = df[((df["interested_in_categories_12"].str.contains("ERKEK")) | (df["interested_in_categories_12"].str.contains("COCUK"))) & ((df["segment"] == "about_to_sleep") | (df["segment"] == "new_customers"))]["master_id"]
df_b.to_csv("indirim_hedef_müşteri_ids.cvs")

