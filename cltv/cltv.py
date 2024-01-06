import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_= pd.read_excel("datasets/online_retail_II.xlsx", sheet_name ="Year 2009-2010")
df = df_.copy()
df.head()
df.isnull().sum()

df = df[~df["Invoice"].str.contains("C", na=False)]
df.describe().T

df = df[(df["Quantity"]> 0)]
df.dropna(inplace = True)
df["TotalPrice"] = df["Quantity"] * df["Price"]


cltv_c =df.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                    'Quantity': lambda x: x.sum(),
                                    'TotalPrice': lambda x: x.sum()})


cltv_c.columns = ['total_transaction','total_unit','total_price']

#######################################################
# 2. Average Order Value ###
#average_order_value = total_price/ total_transaction
cltv_c.head()

cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]

# 3.Purchase Frequency: total_transaction / total_number_of_customer
##############################################
cltv_c.head()
cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]
cltv_c.shape[0]


###########################################
#4.Repeat Rate & Churn Rate

repeat_rate = cltv_c[cltv_c["total_transaction"]>1].shape[0] / cltv_c.shape[0]

churn_rate = 1- repeat_rate

# 5. Profit Margin (profit_margin = total_price * 0.10)

cltv_c['profit_margin'] = cltv_c['total_price'] *0.10

# 6. Customer Value :customer_value = average_order_value * pruchase_frequency
###################################

cltv_c['customer_value']= cltv_c['average_order_value']* cltv_c['purchase_frequency']

# 7. Customer Lifetime Value
##################################
# CLTV= (customer_value / churn_rate) * profit_margin
cltv_c["cltv"] = (cltv_c["customer_value"]/ churn_rate) * cltv_c["profit_margin"]

cltv_c.sort_values(by="cltv", ascending= False).head()

# 8 .Creating Segments :

cltv_c.sort_values(by="cltv", ascending= False).tail()

cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4 , labels= ["D","C","B","A"])

cltv_c.sort_values(by="cltv", ascending= False).head()

cltv_c.groupby("segment").agg({"count","mean","sum"})

cltv_c.to_csv("cltv_c.csv")

#9.  Functionalization
###################################

def create_cltv_c(dataframe, profit=0.10):

    #Veriyi Hazırlama
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[(dataframe['Quantity']> 0)]
    dataframe.dropna(inplace= True)
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    cltv_c = dataframe.groupby('Customer ID').agg({'Invoice': lambda x:x.nunique(),
                                                   'Quantity':lambda  x:x.sum(),
                                                   'TotalPrice': lambda x:x.sum()})

    cltv_c.columns = ['total_transaction','total_unit','total_price']
    #avg_order_value
    cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]
    #pruchase_frequency
    cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]
    #repeat rate& chrun rate
    repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]
    churn_rate = 1 - repeat_rate
    #profit_margin
    cltv_c['profit_margin'] = cltv_c['total_price'] * profit
    #customer value
    cltv_c['customer_value'] = cltv_c['average_order_value'] * cltv_c['purchase_frequency']
    #Customer Lifetime Value
    cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]
    #Segment
    cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])
    return cltv_c

df = df_.copy()

clv =create_cltv_c(df)


#BG-NBD ve Gamma Gamma ile CLTV Tahmini ( CLTV Prediction with BG-NBD & Gamma Gamma )
#########################################################

#1.Data Preperation###

#pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import  MinMaxScaler

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    #dataframe.loc[(dataframe[variable]< low_limit), variable]
    dataframe.loc[(dataframe[variable]> up_limit), variable] = up_limit


df_= pd.read_excel("datasets/online_retail_II.xlsx", sheet_name ="Year 2010-2011")

df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()

#Veri Ön işleme:
###################
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C",na = False)]
df = df[df["Quantity"]> 0]
df = df[df["Price"]> 0 ]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"]= df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

######################################
# Preparation of Lifetime Data Structure
#######################################

cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda invoice_date:(invoice_date.max()- invoice_date.min()).days,
                                                         lambda invoice_date:(today_date - invoice_date.min()).days],
                                         'Invoice': lambda invoice: invoice.nunique(),
                                         'TotalPrice': lambda  total_price: total_price.sum()})

cltv_df.columns= cltv_df.columns.droplevel(0)

cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df["monetary"]= cltv_df["monetary"] / cltv_df["frequency"]

cltv_df.describe().T
cltv_df = cltv_df[(cltv_df['frequency'] >1)]

cltv_df["recency"]= cltv_df["recency"] / 7

cltv_df["T"]= cltv_df["T"] / 7

#################################
#2.  Establishment of BG-NBD Model
##################################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)

bgf.predict(1,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)

cltv_df["expected_purc_1_week"]= bgf.predict(1,
                                             cltv_df['frequency'],
                                             cltv_df['recency'],
                                             cltv_df['T'])

##################
#3 ayda Tüm şirketin beklenen Satış Sayısı Nedir?
##########################

bgf.predict(4 * 3,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()

cltv_df["expected_purc_3_month"]=bgf.predict(4 * 3,
                                             cltv_df['frequency'],
                                             cltv_df['recency'],
                                             cltv_df['T'])

###################
# Tahmin sonuçlarını değerlendirilmesi:
###########################

plot_period_transactions(bgf)
plt.show()

################################
#3. Establishing the Gamma-Gamma Model
####################################

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'],cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)
cltv_df["expected_average_profit"] =ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                            cltv_df['monetary'])

cltv_df.sort_values(["expected_average_profit"], ascending=False).head(10)


######################################
# 4 . Calculation of CLTV with BG-NBD and GG Model
#####################################

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time= 3,#3 aylık
                                   freq="W",# T'nin frekans bilgisi
                                   discount_rate=0.01)
cltv.head()

cltv = cltv.reset_index()

cltv_final = cltv_df.merge(cltv, on ="Customer ID", how ="left")
cltv_final.sort_values(by="clv",ascending=False).head(10)


#############################################
#5. Creating the Customer Segment

cltv_final

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D","C","B","A"])

cltv_final.sort_values(by="clv",ascending=False).head(50)
cltv_final.groupby("segment").agg({
    "count","mean","sum"
})

#6.Functionalization
##################################

def create_cltv_p(dataframe, month=3):

    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe =dataframe[dataframe["Quantity"] >0]
    dataframe = dataframe[dataframe["Price"]>0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    today_date = dt.datetime(2011,12,11)

    cltv_df = dataframe.groupby('Customer ID').agg(
        {'InvoiceDate': [lambda invoice_date:(invoice_date.max()- invoice_date.min()).days,
                         lambda invoice_date:(today_date - invoice_date.min()).days],
         'Invoice': lambda invoice: invoice.nunique(),
         'TotalPrice': lambda  total_price: total_price.sum()})

    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7


    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

    cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])
    cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])
    cltv_df["expected_purc_3_month"] = bgf.predict(12,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])


    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=month,
                                       freq="W",
                                       discount_rate=0.01)
    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

    return cltv_final

df = df_.copy()

cltv_final2= create_cltv_p(df)

cltv_final2.to_csv("cltv_prediction.csv")