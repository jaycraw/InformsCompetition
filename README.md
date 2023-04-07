# InformsCompetition
For this project, we helped a national quick-service food brand build a model to predict the next action of their customers.
This includes predicting approximately when they will make their next order, as well as what sorts of items that order may contain.

#Date Prediction Approach
Our date prediction model uses XG Boosting to classify users into one of 3 date clusters: within the next 20 days, in the next 20-60 days,
or 60+ days. This is not intended to be an exact science, just a shorthand to allow us to target campaigns to users expected to make an order fairly soon

#Product Recommendation
This hybrid recommender algorithm utilizes both collaborative filtering and content-based filtering to predict a basket of n items (from 1 to 10) that the
user is likely to order next. For items they have actually ordered before, we use their actual order inclusion rate (# of orders item is in for user / total number of orders for that user) as its rating. However, for unseen items, we utilize colaborative/content filtering
 
#Collaborative Filtering
We have opted to use a co-clustering approach taken from the scikitSurprise package here. Users are clustered based on their transaction histories, and order inclusion
rates for items a user has never ordered before are calculated based on the behavior of others in that cluster

#Content Based Filtering
To handle cold start issues and explicitly account for the inherent traits of an item, we also use content based filtering in this model. We manually inputted the
attributes of each item in our 125 product universe including things like ingredients, price, and type of item (food, beverage, etc). These attributes are then used
to cluster products into 1 of 10 groups. For items a user has never ordered before, we predict an inclusion rate based on their frequency of ordering other items from
that cluster, discounting those probabilites by a factor so as to not make the predictions overconfident. That factor applies steeper discounts to users with fewer
orders in their history, as we know less about their preferences

#Combining the models
These predicted order inclusions are then combined at a user defined ratio to form a single predicted %. Predictions and actual order inclusions are intermingled and
sorted from high to low in order to form our recommendation basket of n items

#Model Settings
In the settings file, we allow the user to tune the following options in our models:
  daypart_name: only make recommendations for a certain time of day. Use "All" to make recommendations regardless of time
  time_decay_factor: When >0, weight more recent transactions more heavily in determining user preferences. Weighs very aggressively above .05 (particularly >.1)
  product_universe_size: can optionally limit to the top n items. Default is 125
  run_date: date this model is being run as of. Only looks at transaction on or before this date
  measure_accuracy: only to be used historically for testing. If 1, will look at next transaction for each user and return the % of orders that included an item from
    recommendation basket
  content_filtering_ratio: what % of prediction order inclusion is based on content based. 1-cfr = collaborative filtering ratio
  new_items_only: when 1, only return recommendations of items a user has never ordered before
  specific_product: when blank, build a basket of n recommended items. When not blank, return only the inclusion rates for the item specified
  specific_cluster: filter to only users from a given cluster in our date clustering
