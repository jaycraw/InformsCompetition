#------------------------Imports---------------------------#
from __future__ import division
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta,date
import pandas as pd
import numpy as np
import plotly.offline as pyoff
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report,confusion_matrix
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split

def order_cluster(cluster_field_name, target_field_name,df,ascending):
    #Supporting function used to help cluster in feature generation
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

#-----------------------Imports & Data Cleaning-----------------------------#
#Import Settings File and convert to dictionary
settings_df = pd.read_csv("Settings.csv")
settings_dict = settings_df.set_index('Field').to_dict('index')
partition_dt = datetime.strptime(settings_dict["date_training_cutoff"]["Value"],'%m/%d/%Y')
#Reading in source
df = pd.read_parquet('sample_txn_data_2022.parquet', engine='pyarrow')
#Dropping and renaming columns
df = df.drop(["fiscal_week_end","fiscal_year","lineitem_sequence",\
      "lineitem_seq_parent","lineitem_description","dw_gc_header_2","acttax","fiscal_week","parent_product_code"\
             ,"child_product_code","modification_product_code","actgrosssales","actnetsales","actdiscsales"\
             ,"actpromosales"],1)
df = df.drop(["dw_gc_header","daypart_name","parent_product_description","child_product_description","modification_description","actpromoqty","actmodqty","actdiscqty"],1)
df = df.rename(columns={'parent_product_description': 'Description',"actqtysold":"Quantity","business_date"\
                       :"Date","actprodprice":"Price"})
#Cleaning Data types & NAs
df['Date'] = pd.to_datetime(df['Date'])
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df = df.loc[~df['Quantity'].isna()]
print("Imports and Cleaning Complete")
#-----------------------------Partition------------------------------------#
#Partition based on user defined date
df['Date'] = df['Date'].astype("datetime64[ns]")
tx_hist = df[(df.Date <= partition_dt)].reset_index(drop=True)
tx_next = df[(df.Date > partition_dt)].reset_index(drop=True)

#-----------------------------Feature Generation------------------------------------#
#To be used for ultimate testing, this calculates the number of days between the last purchase before our threshold and first purchase after that threshold
#Table of distinct token_primary_account_identifiers that we will build all of our features onto
tx_user = pd.DataFrame(tx_hist['token_primary_account_identifier'].unique())
tx_user.columns = ['token_primary_account_identifier'] 

#first purchase
tx_next_first_purchase = tx_next.groupby('token_primary_account_identifier').Date.min().reset_index()
tx_next_first_purchase.columns = ['token_primary_account_identifier','MinPurchaseDate']
#last purchase date in test dataset
tx_last_purchase = tx_hist.groupby('token_primary_account_identifier').Date.max().reset_index()
tx_last_purchase.columns = ['token_primary_account_identifier','MaxPurchaseDate']
tx_purchase_dates = pd.merge(tx_last_purchase,tx_next_first_purchase,on='token_primary_account_identifier',how='left')
#Day difference
tx_purchase_dates['NextPurchaseDay'] = (tx_purchase_dates['MinPurchaseDate'] - tx_purchase_dates['MaxPurchaseDate']).dt.days
tx_user = pd.merge(tx_user, tx_purchase_dates[['token_primary_account_identifier','NextPurchaseDay']],on='token_primary_account_identifier',how='left')
tx_user = tx_user.fillna(999)

tx_max_purchase = tx_hist.groupby('token_primary_account_identifier').Date.max().reset_index()
tx_max_purchase.columns = ['token_primary_account_identifier','MaxPurchaseDate']

#RECENCY
tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days
tx_user = pd.merge(tx_user, tx_max_purchase[['token_primary_account_identifier','Recency']], on='token_primary_account_identifier')
#K-Means Clustering users based on how recently they last purchased an item
recency = tx_user['Recency'].values.reshape(-1, 1)
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(recency)
tx_user['RecencyCluster'] = kmeans.labels_
tx_user = order_cluster('RecencyCluster', 'Recency',tx_user,False)

#REVENUE
tx_hist["Quantity"] = tx_hist["Quantity"].astype("float64")
tx_hist['Revenue'] = tx_hist['Price'] * tx_hist['Quantity']
tx_revenue = tx_hist.groupby('token_primary_account_identifier').Revenue.sum().reset_index()
tx_user = pd.merge(tx_user, tx_revenue, on='token_primary_account_identifier')
#K Means clustering users by the revenue generated over their lifetime
Revenue = tx_user['Revenue'].values.reshape(-1, 1)
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(Revenue)
tx_user['RevenueCluster'] = kmeans.labels_
tx_user = order_cluster('RevenueCluster', 'Revenue',tx_user,True)

#FREQUENCY
tx_frequency = tx_hist.groupby('token_primary_account_identifier').Date.count().reset_index()
tx_frequency.columns = ['token_primary_account_identifier','Frequency']
tx_user = pd.merge(tx_user, tx_frequency, on='token_primary_account_identifier')
Frequency = tx_user['Frequency'].values.reshape(-1, 1)
#Cluster by Frequency of Orders
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(Frequency)
tx_user['FrequencyCluster'] = kmeans.labels_
tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)
#Calculate overall score based on three previous clusters
tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']
#Classify into value segments
tx_user['Segment'] = 'Low-Value'
tx_user.loc[tx_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
tx_user.loc[tx_user['OverallScore']>4,'Segment'] = 'High-Value'
#create a dataframe with token_primary_account_identifier and Date
tx_day_order = tx_hist[['token_primary_account_identifier','Date']]
#convert Invoice Datetime to day
tx_day_order['InvoiceDay'] = tx_hist['Date']
tx_day_order = tx_day_order.sort_values(['token_primary_account_identifier','Date'])
#drop duplicates
tx_day_order = tx_day_order.drop_duplicates(subset=['token_primary_account_identifier','InvoiceDay'],keep='first')

#3 MOST RECENT PURCHASES
#shifting last 3 purchase dates
tx_day_order['PrevInvoiceDate'] = tx_day_order.groupby('token_primary_account_identifier')['InvoiceDay'].shift(1)
tx_day_order['T2InvoiceDate'] = tx_day_order.groupby('token_primary_account_identifier')['InvoiceDay'].shift(2)
tx_day_order['T3InvoiceDate'] = tx_day_order.groupby('token_primary_account_identifier')['InvoiceDay'].shift(3)
tx_day_order['DayDiff'] = (tx_day_order['InvoiceDay'] - tx_day_order['PrevInvoiceDate']).dt.days
tx_day_order['DayDiff2'] = (tx_day_order['InvoiceDay'] - tx_day_order['T2InvoiceDate']).dt.days
tx_day_order['DayDiff3'] = (tx_day_order['InvoiceDay'] - tx_day_order['T3InvoiceDate']).dt.days
#calculate mean and standard deviation of date difference between last 3 transactions
tx_day_diff = tx_day_order.groupby('token_primary_account_identifier').agg({'DayDiff': ['mean','std']}).reset_index()
tx_day_diff.columns = ['token_primary_account_identifier', 'DayDiffMean','DayDiffStd']
tx_day_order_last = tx_day_order.drop_duplicates(subset = ['token_primary_account_identifier'],keep ='last')
print("Feature Generation Complete")
#-----------------Putting it all together----------------------#
tx_day_order_last = tx_day_order_last.dropna()
tx_day_order_last = pd.merge(tx_day_order_last, tx_day_diff, on='token_primary_account_identifier')
tx_user = pd.merge(tx_user, tx_day_order_last[['token_primary_account_identifier','DayDiff','DayDiff2','DayDiff3','DayDiffMean','DayDiffStd']], on='token_primary_account_identifier')
#copy data frame, drop token_primary_account_identifier, and convert categorical variables into dummies
tx_class = tx_user.copy()
tx_class = tx_class.drop(["token_primary_account_identifier"],1)
tx_class = pd.get_dummies(tx_class)

#-----------------------Model Training-------------------------#
#calculating the final cluster labels for purposes of training 
tx_class['NextPurchaseDayRange'] = 2
tx_class.loc[tx_class.NextPurchaseDay>20,'NextPurchaseDayRange'] = 1
tx_class.loc[tx_class.NextPurchaseDay>50,'NextPurchaseDayRange'] = 0

#train & test split
tx_class = tx_class.drop('NextPurchaseDay',axis=1)
X, y = tx_class.drop('NextPurchaseDayRange',axis=1), tx_class.NextPurchaseDayRange
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

#fit the model
xgb_model = xgb.XGBClassifier(max_depth=5,min_child_weight = 3,gamma=5).fit(X_train, y_train)
print('Accuracy of XGB classifier on training set: {:.2f}'.format(xgb_model.score(X_train, y_train)))
print('Accuracy of XGB classifier on test set: {:.2f}'.format(xgb_model.score(X_test[X_train.columns], y_test)))

#save the final model
xgb_model.save_model("date_XGB_model.txt")
print("Model Training Complete")