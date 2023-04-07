#-----------------Imports------------------#

import pandas as pd
import numpy as np
from surprise import Dataset
from surprise import Reader
from surprise import CoClustering
from collections import defaultdict
from sklearn.cluster import KMeans
from datetime import datetime

#---------------Generates combined list of predicited order frequencies-----------------#
def get_top_n(predictions, n, contentratings, prodclusters,contentfilteringratio = 0.5, sp = ""):
    #Return the top-N recommendation for each user from a set of predictions.
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        c = prodclusters[iid]["ClusterID"]
        content_rat = contentratings[uid + ' - ' + str(c)]["N Weighted Cluster % Orders"]
        predrat = (est*(1-contentfilteringratio))+(contentfilteringratio*content_rat)
        top_n[uid].append((iid, predrat))
        
    # Then sort the predictions for each user and retrieve the k highest ones.
    if sp == "":
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]
    else:
    #if only looking for specific product, filter out everything else and return all ratings for this item regardless of how high or low they are
        for uid, user_ratings in top_n.items():
             si = []
             for i in user_ratings:
                if i[0] == sp:
                    si.append(i)
             if len(si) != 0:
                top_n[uid] = si  
    return top_n

#--------------- Settings-------------# 
#Import Settings File and convert to dictionary
settings_df = pd.read_csv("Settings.csv")
settings_dict = settings_df.set_index('Field').to_dict('index')
#Product universe size
prodnum = int(settings_dict["product_universe_size"]["Value"])
#Times of day to include
daypart = settings_dict["daypart_name"]["Value"]
#factor to decay based on how far in the past order is
decayfactor = float(settings_dict["time_decay_factor"]["Value"])
#run date
asofdate = datetime.strptime(settings_dict["run_date"]["Value"],'%m/%d/%Y')
#measure accuracy?
measure_accuracy = int(settings_dict["measure_accuracy"]["Value"])
#what weighting to give content filtering (as compared to collaborative filtering)
contentfilteringratio = float(settings_dict["content_filtering_ratio"]["Value"])
#only recommend new items?
new_only = int(settings_dict["new_items_only"]["Value"])
#only pull top users for certain items?
spec_prod = settings_dict["specific_product"]["Value"]
#only look at users from a given time cluster?
spec_cluster = int(settings_dict["specific_cluster"]["Value"])

#----------------Product Traits + Clustering for Content-Based Filtering-------------# 
#pull spreadsheet with inherent traits of each product
prod_traits = pd.read_csv("Product Traits.csv")

#Labels for each column
lbl = ["Food", "Combo", "Beverage", "Has Meat", "Beef", "Chicken", "Beans", "Cheese", "Potatoes", "Pork", "Egg", "Price < 2", "Price 2-4", "Price 4-6", "Price 6-8", "Price 8-10", "Price >= 10"]
cluster_output_df = pd.DataFrame(columns = ['Cluster Num']+lbl)
#Perform KMeans Clustering based on traits
kmeans = KMeans(n_clusters=10, n_init='auto').fit(prod_traits.loc[:, prod_traits.columns != 'child_product_code'])
#obtain count of each cluster
count = np.zeros(10)
for x in kmeans.labels_:
    count[x]+=1

#Output stats for each cluster to csv
cluster_output_df = pd.DataFrame(columns = lbl,data=kmeans.cluster_centers_)
cluster_output_df["Cluster Size"] = count
cluster_output_df.to_csv("content_clusters.csv")

cluster_id_df = pd.DataFrame(kmeans.labels_,columns = ["ClusterID"])
#Add cluster number to traits
prod_traits = prod_traits.merge(cluster_id_df, left_index=True, right_index=True)


#---------------------------Transaction History Import & Cleaning-------------------------------#
#Import transaction history
df = pd.read_parquet("sample_txn_data_2022.parquet")
df.drop_duplicates(inplace=True)

#Cleaning
#ignore modifications
df = df[df["modification_product_code"]=="N/A"]
#type conversions
df["actgrosssales"] = df["actgrosssales"].astype("float")
df["actprodprice"] = df["actprodprice"].astype("float")
df["business_date"] = df["business_date"].astype("datetime64[ns]")
#calculate months since each transaction for the purposes of time weighting
df["months_since_transactions"] = (pd.Timestamp("2022-09-30") - df["business_date"]) / np.timedelta64(1, 'M')
df["time_weighting"] = 1 / ((1+decayfactor)**df["months_since_transactions"])
print("Imports Complete")

#------------------------Day Part + Train/Test Split as relevant----------------------------#
#if only using specific day part, filter out here
if daypart != "All":
    df = df[df["daypart_name"]==daypart]

df = df.merge(prod_traits,how='inner', on = 'child_product_code')

#split as of given date
train_df = df[df["business_date"]<=asofdate]
df_val = df[df["business_date"]>asofdate]

#-------------------------Generate universe of products---------------------------------#
prod_by_rev = train_df.groupby("child_product_code").agg({"actgrosssales":"sum"})
prod_by_rev.reset_index(inplace=True)
prod_by_rev.sort_values(by="actgrosssales",inplace=True, ascending=False)
prod_universe = prod_by_rev.head(prodnum)

#-------------------------Generate actual user/product probabilities---------------------------------#
#Calculate the total orders for each user
tot_orders_df = train_df.drop_duplicates(subset=["token_primary_account_identifier","dw_gc_header"]).groupby("token_primary_account_identifier").agg({"time_weighting":"sum"})
tot_orders_df.reset_index(inplace=True)
tot_orders_df.rename(columns={"time_weighting":"# of Orders"},inplace=True)

#Calculate the number of orders each item is included in per user
orders_by_user = train_df.drop_duplicates(subset=["token_primary_account_identifier","dw_gc_header","child_product_code"]).groupby(["token_primary_account_identifier", "child_product_code"]).agg({"time_weighting":"sum"})
orders_by_user.reset_index(inplace=True)
orders_by_user.rename(columns={"time_weighting":"# Order Inclusions"},inplace=True)

#build dataframe showing user, item, & % of time item is in order
#get list of users
products_by_user = df.drop_duplicates(subset="token_primary_account_identifier")["token_primary_account_identifier"].to_frame()
#cross join to generate 1 row per customer + product pairing
products_by_user = products_by_user.merge(prod_universe["child_product_code"],how='cross')

#join in total orders and # orders for each item
products_by_user = products_by_user.merge(tot_orders_df, on = "token_primary_account_identifier", how="inner")
products_by_user = products_by_user.merge(orders_by_user, on = ["token_primary_account_identifier","child_product_code"], how = "inner")

products_by_user.fillna(value=0,inplace=True)
products_by_user["% Inclusion"] = products_by_user["# Order Inclusions"] / products_by_user["# of Orders"]
#tag item as actual rather than predicted
products_by_user["isActual"] = 1 
print("Data Cleaning + Prep Complete")


#--------------------------------------Content Based Filtering --------------------------------------#
#create dataframe for all users + clusters
user_cluster_df = train_df.drop_duplicates(subset=["token_primary_account_identifier"]).merge(pd.DataFrame(range(0,10),columns=["ClusterId"]),how="cross")
user_cluster_df = user_cluster_df[["token_primary_account_identifier","ClusterId"]]
user_cluster_df.rename(columns={"ClusterId":"ClusterID"},inplace=True)

#Calculate % of orders that contains an item from each cluster (per user) 
content_cluster_df = train_df.drop_duplicates(subset=["token_primary_account_identifier","dw_gc_header","ClusterID"]).groupby(["token_primary_account_identifier","ClusterID"]).agg({"time_weighting":"sum"})
content_cluster_df.reset_index(inplace=True)
content_cluster_df = content_cluster_df.merge(tot_orders_df, on = "token_primary_account_identifier", how="inner")
content_cluster_df["Cluster % Orders"] = content_cluster_df["time_weighting"] / content_cluster_df["# of Orders"]

user_cluster_df = user_cluster_df.merge(content_cluster_df,how="left", on=["ClusterID","token_primary_account_identifier"])
user_cluster_df["Cluster % Orders"] = user_cluster_df["Cluster % Orders"].fillna(0)
user_cluster_df["# of Orders"] = user_cluster_df["# of Orders"].fillna(0)
#curve that discounts probabilities more harshly for users with fewer orders
user_cluster_df["N Weighted Cluster % Orders"] = abs(user_cluster_df["Cluster % Orders"] * (1-(user_cluster_df["# of Orders"]+.25)**-.6))

#Dictionaries to be used for final predicted probabilities
user_cluster_df["id"] = user_cluster_df["token_primary_account_identifier"] + " - " + user_cluster_df["ClusterID"].astype(str)
user_cluster_dict = user_cluster_df.set_index('id').to_dict("index")
prod_traits_dict = prod_traits.set_index('child_product_code').to_dict("index")

print("Content Filtering Complete")

#---------------------------------------Collaborative Filtering---------------------------------------#

reader = Reader(rating_scale=(0,1))
data = Dataset.load_from_df(products_by_user[["token_primary_account_identifier","child_product_code","% Inclusion"]],reader)
algo = CoClustering()
trainSet = data.build_full_trainset()
algo.fit(trainSet)
#anti_testset are all items that each user has never ordered before
testset = trainSet.build_anti_testset()

predictions = algo.test(testset)
print("Collaborative Filtering Complete")

#------------------------------------Generate Final Recommendations-----------------------------------#
top_n = get_top_n(predictions,10,user_cluster_dict,prod_traits_dict,contentfilteringratio, spec_prod)
#Generate dataframe of predicted items
predictions_df = pd.DataFrame(columns=["token_primary_account_identifier", "child_product_code"	, "# of Orders"	, "# Order Inclusions",	"% Inclusion",	"isActual"])
prediction_list = []
for uid, user_ratings in top_n.items():
    for pred_item in user_ratings:
        prediction_list.append([uid, pred_item[0],np.nan,np.nan,pred_item[1],0])
predictions_df = pd.DataFrame(columns=["token_primary_account_identifier", "child_product_code"	, "# of Orders"	, "# Order Inclusions",	"% Inclusion",	"isActual"], data=prediction_list)
final_rec_df = pd.concat([products_by_user, predictions_df])
#Optional feature to restrict to only predictions
if new_only == 1:
    final_rec_df = final_rec_df[final_rec_df["isActual"] == 0]

if spec_prod == "":
    #if not filtering for single product, sort by user and % inclusion
    final_rec_df.sort_values(by=["token_primary_account_identifier","% Inclusion"],ascending=[True,False],inplace=True)
else:
    #if just looking for a single product, filter down to that and sort exclusively by % inclusion
    final_rec_df = final_rec_df[final_rec_df["child_product_code"]==spec_prod]
    final_rec_df.sort_values(by=["% Inclusion"],ascending=[False],inplace=True)
#output recs to csv
final_rec_df.to_csv("recommendations.csv")

#Create Dataframe showing user, actual items ordered, and our top predictions/recommendations
#Pull Items Ordered
df_val = pd.merge(df_val,prod_universe["child_product_code"],how="inner",on="child_product_code").drop_duplicates(subset=["dw_gc_header","child_product_code"])

next_transaction = df_val[df_val["business_date"]>'2022-09-30'].drop_duplicates(subset="dw_gc_header").sort_values("business_date",ascending=True).groupby("token_primary_account_identifier").head(1)[["token_primary_account_identifier","dw_gc_header"]]
next_trans_items =df_val.groupby('dw_gc_header')["child_product_code"].apply(lambda x: "%s" % ', '.join(x)).to_frame()
next_trans_items.reset_index(inplace=True)
next_trans_items.rename(columns={"child_product_code":"items_ordered"},inplace=True)
next_transaction = next_transaction.merge(next_trans_items, how='inner',on='dw_gc_header')

#Pull top 10 predictions
toppred = final_rec_df.groupby('token_primary_account_identifier')[["token_primary_account_identifier","child_product_code"]].head(10)
toppred = toppred.groupby('token_primary_account_identifier')["child_product_code"].apply(lambda x: "%s" % ', '.join(x)).to_frame()
toppred.reset_index(inplace=True)
toppred.rename(columns={"child_product_code":"Recommendation Basket"},inplace=True)

#Add next transaction
next_transaction = next_transaction.merge(toppred, how='inner', on='token_primary_account_identifier')
print("Predictions Complete")
#----------------------------------------Evaluate Accuracy---------------------------------------#
if measure_accuracy == 1:
    ordered_dict = next_transaction[['token_primary_account_identifier','items_ordered']].set_index('token_primary_account_identifier').to_dict('index')
    user_rec_dict = next_transaction[['token_primary_account_identifier','Recommendation Basket']].set_index('token_primary_account_identifier').to_dict('index')

    #Check if each item in the recommendation dictionary was in the users next order
    in_order_dict = {}
    for k, v in user_rec_dict.items():
        in_order_dict[k] = np.zeros(10)
        for i in range (0,10):
            if v['Recommendation Basket'].split(", ")[i] in ordered_dict[k]['items_ordered'].split(", "):
                in_order_dict[k][i] = 1
    #map results of above to next_transaction dataframe
    next_transaction["Rec in Order"] = next_transaction["token_primary_account_identifier"].map(in_order_dict)
    #determine success of baskets of size 1 to n
    for n in range(1,11):
        next_transaction["Basket Size " + str(n) + " Success"] = next_transaction['Rec in Order'].apply(lambda x:x[0:n].max())

    #Final table that returns % of users that had item from recommendation basket in next order at each basket size
    final_tally = next_transaction.agg({"token_primary_account_identifier":"count","Basket Size 1 Success":"sum","Basket Size 2 Success":"sum","Basket Size 3 Success":"sum","Basket Size 4 Success":"sum","Basket Size 5 Success":"sum","Basket Size 6 Success":"sum",
                                    "Basket Size 7 Success":"sum","Basket Size 8 Success":"sum","Basket Size 9 Success":"sum","Basket Size 10 Success":"sum"})
    final_tally = final_tally.to_frame().reset_index()#.transpose()
    final_tally.rename(columns={0:"value"},inplace=True)
    tot = final_tally[final_tally["index"]=="token_primary_account_identifier"]["value"][0]
    final_tally["% Accuracy"] = final_tally["value"] / tot
    #output final model accuracy to csv
    final_tally.to_csv('rec_accuracy.csv')
    print("Accuracy Measurements Complete")