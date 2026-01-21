# task2_2.py
# usage: spark-submit task2_2.py <folder_path> <test_file_name> <output_file_name>
#case 2: using xgboost to train a model 
#aim for: rmse < 1 and time 400s

import sys, csv, os, json
from math import log, floor, sqrt
from pyspark import SparkConf, SparkContext
#import xgboost as xgb

#testing functions 
def load_val_map(sc, val_path):
    # user_id,business_id,stars -> dict[(user_id,business_id)] = rating
    rdd = load_csv_rdd(sc, val_path, expect_label=True)
    return {(u, b): r for (u, b, r) in rdd.collect()}

def compute_rmse(predictions, val_map):
    # predictions: list of (user_id, business_id, pred)
    mse_sum = 0.0
    n = 0
    for u, b, p in predictions:
        y = val_map.get((u, b))
        if y is None:
            # if a pair isn't in validation, skip it (shouldn't happen if files align)
            continue

        #rmse formula
        diff = p - y
        mse_sum += diff * diff
        n += 1
    if n == 0:
        return None
    
    #get rmse for self testing 
    return sqrt(mse_sum / n)

# =========================
#adjustables 
# =========================
#bayesian shrinking, also used in task2_1 (for cases where there are not enough ratings from user/business)
A_U, A_B = 20.0, 110.0

#rounding for testing
rounding = False

#xgb xgb_params
xgb_params = {
        #linear is the 0.72 version
        "objective": "reg:linear",  
        #eval on rmse
        "eval_metric": "rmse",
        #"learning rate"
        "eta": 0.01,
        "max_depth": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.6,
        "colsample_bylevel": 0.5,
        "min_child_weight": 2,
        "base_score": 0.0,
        "gamma": 0.20,
        "lambda": 3.0,           
        "alpha": 0.0,            
        "silent": 1,
        "seed": 42
    }

#train rounds
num_rounds = 385

# =========================
#reading/writing/etc. to read input
# =========================

#had some trouble reading, using this func to ensure proper reading
def to_file_path(path):
    #abs path for spark
    path = os.path.abspath(path).replace("\\", "/")
    if not path.startswith("file:/"):
        path = "file:///" + path
    return path

#past input checker, modified for this task
def check_inputs():
    if len(sys.argv) != 4:
        print("Usage: spark-submit task2_2.py <folder_path> <test_file_name> <output_file_name>", file=sys.stderr)
        sys.exit(1)
    #taking folder path inputted and normalizing it before output
    folder = os.path.abspath(os.path.normpath(sys.argv[1]))
    return folder, sys.argv[2], sys.argv[3]

#taken from my older assignments 
def read_train_rows(lines_iter):
    reader = csv.reader(lines_iter)
    for row in reader:
        if not row or len(row) < 3:
            continue
        #was having trouble reading with first, trying this approach instead
        if row[0].strip().lower() == "user_id" and row[1].strip().lower() == "business_id":
            continue

        #strip to ensure clean reading of the csv 
        user_id = row[0].strip()
        business_id = row[1].strip()
        stars = row[2].strip()
        if not user_id or not business_id:
            continue
        try:
            rating = float(stars)
        except:
            continue
        yield (user_id, business_id, rating)

#same as train, but without stars field 
def read_test_rows(lines_iter):
    reader = csv.reader(lines_iter)
    for row in reader:
        if not row or len(row) < 2:
            continue
        #same as above but without ratings as we should be predicting the ratings of this file 
        if row[0].strip().lower() == "user_id" and row[1].strip().lower() == "business_id":
            continue
        user_id = row[0].strip()
        business_id = row[1].strip()
        if not user_id or not business_id:
            continue
        yield (user_id, business_id)

#csv to rdd, can take both train/test/val based on label
def load_csv_rdd(sc, path, expect_label):
    path = to_file_path(path)
    rdd = sc.textFile(path)

    #if labels, it is train, if not, test 
    if expect_label:
        return rdd.mapPartitions(read_train_rows)
    else:
        return rdd.mapPartitions(read_test_rows)

#turn jsons (our additional files) to rdds 
def load_json_rdd(sc, path):
    if not os.path.exists(path):
        return None
    path = to_file_path(path)
    return sc.textFile(path).map(lambda s: json.loads(s))

def write_output(out_csv, rows):
    #rows: list of (user_id, business_id, prediction)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "business_id", "prediction"])
        for user_id, business_id, prediction in rows:
            #do not need to reduce the decimals
            w.writerow([user_id, business_id, prediction])

# =========================
#helper functions, used to do calculations etc. 
# =========================

#build helpers for jsons and math values
#turn things into floats
def make_float(x, default=0.0):
    #if number, easy to make a float
    try:
        return float(x)
    #if we cannot, then make it into default (0.0)
    except:
        return default

#need this for stabilizing ranges for poppular users/businesses 
#suggested online
def log1p_plus(x):
    try:
        x = float(x)
        #log(1+x) helps to shrink outlier/large/small 
        return log(1.0 + x) if x > 0 else 0.0
    except:
        return 0.0
    
#changed from task2_1, but for ensuring that the rating stays within 1-5 range of yelp
def clip_rating(x, min=1.0, max=5.0):
    if x < min: return min
    if x > max: return max

    #testing rounding for rmse performance 
    if rounding:
        #clip and round 
        x = round(x)
        if x < 1: x = 1
        if x > 5: x = 5

    #ensure x is in 1 to 5
    return x

#parse bool values like 1/0 or yes/nos
def parse_bool(v):
    if v is None:
        return 0.0
    
    #take bool str and strip/lower so easier to read, 1 for yes, 0 for no
    s = str(v).strip().lower()
    return 1.0 if s in {"true","yes","y","1"} else 0.0

#noise level of the establishment
def parse_noise_level(v):
    if v is None:
        return 1.0
    s = str(v).strip().lower()
    #various ways they can be rated and mapped for text to number value 
    mapping = {
        "quiet": 0.0,
        "average": 1.0,
        "loud": 2.0,
        "very_loud": 3.0,
        "very loud": 3.0,
    }
    return mapping.get(s, 1.0)

#yelp elite years of users
def count_elite_years(elite_field):
    #years of being an elite 
    if not elite_field or str(elite_field).strip().lower() == "none":
        return 0.0
    years = [y.strip() for y in str(elite_field).split(",") if y.strip()]

    #years of elite, maybe trusworthiness
    return float(len(years))


# =========================
#math and calculation functions
#and also building out rdds 
# =========================

#for use getting top % of people
def percentile_from_counts(count_map, q):
    values = sorted(count_map.values())
    if not values: return 0

    #get % out of total
    index = int(q * (len(values) - 1))
    return float(values[index])



def compute_global_and_aggregates(train_rdd):
    #global mean for the whole dataset 
    global_sum, global_count = (
        train_rdd.map(lambda x: ("*", (x[2], 1)))
                 .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
                 .map(lambda kv: kv[1])
                 .first()
    )
    global_mean = global_sum / global_count if global_count > 0 else 3.5

    #user aggregate values mean/count/variance
    user_agg = (train_rdd
                .map(lambda x: (x[0], (x[2], x[2]*x[2], 1)))
                .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1], a[2]+b[2])))
    user_mean = user_agg.mapValues(lambda s: s[0]/s[2]).collectAsMap()
    user_count = user_agg.mapValues(lambda s: s[2]).collectAsMap()
    user_variance = user_agg.mapValues(lambda s: max(0.0, s[1]/s[2] - (s[0]/s[2])**2)).collectAsMap()

    #business version of above 
    business_agg = (train_rdd
                    .map(lambda x: (x[1], (x[2], x[2]*x[2], 1)))
                    .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1], a[2]+b[2])))
    business_mean = business_agg.mapValues(lambda s: s[0]/s[2]).collectAsMap()
    business_count = business_agg.mapValues(lambda s: s[2]).collectAsMap()
    business_variance = business_agg.mapValues(lambda s: max(0.0, s[1]/s[2] - (s[0]/s[2])**2)).collectAsMap()

    #return these 3 aggregations (global, user, business )
    return (global_mean,
            user_mean, user_count, user_variance,
            business_mean, business_count, business_variance)

#shrinking means as a way to reduce impact of extreme cases
#testing to work if shrinking helps
def compute_shrunk_means(global_mean, user_mean, user_count, business_mean, business_count):
    user_mean_shrunk = {}
    for user_id, user_avg in user_mean.items():

        #count of users 
        n_user = float(user_count.get(user_id, 0))
        #shrink user average, or use global if not possible 
        user_mean_shrunk[user_id] = (user_avg*n_user + global_mean*A_U) / (n_user + A_U) if (n_user + A_U) > 0 else global_mean

    business_mean_shrunk = {}
    for business_id, business_avg in business_mean.items():

        #same process as above, but for business as well 
        n_business = float(business_count.get(business_id, 0))
        business_mean_shrunk[business_id] = (business_avg*n_business + global_mean*A_B) / (n_business + A_B) if (n_business + A_B) > 0 else global_mean

    #shrunken means 
    return user_mean_shrunk, business_mean_shrunk

#use user json_rdd to build out user feature mapping to trai model on 
def build_user_features(user_json_rdd):
    #check json first 
    if not user_json_rdd:
        return {}
    
    #get details by uid 
    return (user_json_rdd.map(lambda record: (
                    record.get("user_id"),
                    {
                        "u_review_count_log": log1p_plus(make_float(record.get("review_count"), 0)),
                        "u_average_stars": make_float(record.get("average_stars"), 0.0),
                        "u_fans_log": log1p_plus(make_float(record.get("fans"), 0)),
                        "u_useful_log": log1p_plus(make_float(record.get("useful"), 0)),
                        "u_elite_count": count_elite_years(record.get("elite")),
                        "u_elite_count_log": log1p_plus(count_elite_years(record.get("elite"))),
                    }
                ))
                .filter(lambda t: t[0] is not None)
                .collectAsMap())

#business feature mapping 
#businesses have a lot more stats on yelp than users so a lot more computation needed lol 
def build_business_features(business_json_rdd):
    #store features 
    business_features = {}
    city_pop = {}
    state_pop = {}

    #validate exists before anything
    if not business_json_rdd:
        return business_features

    #make business records into individual data columns 
    def map_business(record):
        business_id = record.get("business_id")
        if not business_id:
            return None
        
        #basic business features 
        categories_field = (record.get("categories") or "").lower()
        categories_count = 0 if not categories_field else len([category for category in categories_field.split(",") if category.strip()])
        attributes = record.get("attributes") or {}


        price_range = make_float(attributes.get("RestaurantsPriceRange2"), 0.0)
        hours = record.get("hours") or {}
        hours_count = len(hours) if isinstance(hours, dict) else 0
        city = (record.get("city") or "").strip()
        state = (record.get("state") or "").strip()
        latitude = make_float(record.get("latitude"), 0.0)
        longitude = make_float(record.get("longitude"), 0.0)

        #experimental ones 
        accepts_credit_cards = parse_bool(attributes.get("BusinessAcceptsCreditCards"))
        noise_level = parse_noise_level(attributes.get("NoiseLevel"))
        takeout = parse_bool(attributes.get("RestaurantsTakeOut"))
        delivery = parse_bool(attributes.get("RestaurantsDelivery"))

        #build out data features for each business 
        row = {
            "b_stars": make_float(record.get("stars"), 0.0),
            "b_review_count_log": log1p_plus(make_float(record.get("review_count"), 0)),
            "b_is_open": 1.0 if make_float(record.get("is_open"), 0) > 0 else 0.0,
            "b_categories_n": float(categories_count),
            "b_price": price_range,
            "b_hours_n": float(hours_count),
            "b_lat_bin": floor(latitude*2)/2.0,
            "b_lon_bin": floor(longitude*2)/2.0,
            "city": city,
            "state": state,
            "b_city_pop_log": 0.0,
            "b_state_pop_log": 0.0,
            "b_checkins_total": 0.0,
            "b_photo_count": 0.0,
            "b_tip_count": 0.0,
            "b_accepts_credit_cards": accepts_credit_cards,
            "b_noise_level": noise_level,
            "b_takeout": takeout,
            "b_delivery": delivery,
        }
        return (business_id, row)
    
    #materialize it for spark
    business_rows = (business_json_rdd.map(map_business)
                     .filter(lambda t: t is not None)
                     .collect())

    #count for popularity 
    for business_id, row in business_rows:
        city = row["city"]; state = row["state"]
        if city: city_pop[city] = city_pop.get(city, 0) + 1
        if state: state_pop[state] = state_pop.get(state, 0) + 1

    #log the popularities for normalizing against strong tail ends 
    #remove some later, removed some from final map
    for business_id, row in business_rows:
        city = row.pop("city"); state = row.pop("state")
        row["b_city_pop_log"]  = log1p_plus(float(city_pop.get(city, 0)))
        row["b_state_pop_log"] = log1p_plus(float(state_pop.get(state, 0)))
        business_features[business_id] = row

    return business_features

#additional rdds from othe rjsons to validate and check for popularity of business 
def extra_business_features(business_features, checkin_rdd, photo_rdd, tip_rdd):

    #these 3 are all rdds that we're counting 
    #proxies for "popularity" of the business
    if checkin_rdd:
        total_checkins_by_business = (checkin_rdd.map(lambda rec: (rec.get("business_id"),
                                                                   sum([make_float(v, 0.0) for v in (rec.get("time") or {}).values()])))
                                               .filter(lambda t: t[0] is not None)
                                               .collectAsMap())
        for business_id, total in total_checkins_by_business.items():
            if business_id in business_features:
                business_features[business_id]["b_checkins_total"] = float(total)

    if photo_rdd:
        photo_count_by_business = (photo_rdd.map(lambda rec: (rec.get("business_id"), 1.0))
                                   .reduceByKey(lambda a, b: a + b)
                                   .collectAsMap())
        for business_id, count in photo_count_by_business.items():
            if business_id in business_features:
                business_features[business_id]["b_photo_count"] = float(count)

    if tip_rdd:
        tip_count_by_business = (tip_rdd.map(lambda rec: (rec.get("business_id"), 1.0))
                                 .reduceByKey(lambda a, b: a + b)
                                 .collectAsMap())
        for business_id, count in tip_count_by_business.items():
            if business_id in business_features:
                business_features[business_id]["b_tip_count"] = float(count)

#build out business features 
def make_feature_builder(global_mean,
                         user_mean, user_count, user_variance, user_mean_shrunk,
                         business_mean, business_count, business_variance, business_mean_shrunk,
                         user_features, business_features):
    
    #for each feature row of user and business together 
    def build_feature_row(user_id, business_id):
        user_feat = user_features.get(user_id, {})
        business_feat = business_features.get(business_id, {})

        user_mean_value = user_mean.get(user_id, global_mean)
        business_mean_value = business_mean.get(business_id, global_mean)
        user_mean_shrunk_value = user_mean_shrunk.get(user_id, global_mean)
        business_mean_shrunk_value = business_mean_shrunk.get(business_id, global_mean)

        user_n = float(user_count.get(user_id, 0))
        business_n = float(business_count.get(business_id, 0))

        user_var = user_variance.get(user_id, 0.0)
        business_var = business_variance.get(business_id, 0.0)

        check_ins = sqrt(max(0.0, business_feat.get("b_checkins_total", 0.0)))
        photos = log1p_plus(business_feat.get("b_photo_count", 0.0))
        tips = 0.8*log1p_plus(business_feat.get("b_tip_count", 0.0))

        base_rating = global_mean + (user_mean_shrunk_value - global_mean) + (business_mean_shrunk_value - global_mean)

        diff_raw_means = abs(user_mean_value - business_mean_value)
        diff_shrunk_means = abs(user_mean_shrunk_value - business_mean_shrunk_value)
        diff_shrunk_means_squared = diff_shrunk_means * diff_shrunk_means

        interaction_centered = (user_mean_value - global_mean) * (business_mean_value - global_mean)
        user_vs_business_avg = (user_feat.get("u_average_stars", 0.0) - business_feat.get("b_stars", 0.0))

        user_avg_center = user_feat.get("u_average_stars", 0.0) - global_mean
        business_stars_center = business_feat.get("b_stars", 0.0) - global_mean

        categories_log = log1p_plus(business_feat.get("b_categories_n", 0.0))
        hours_log = log1p_plus(business_feat.get("b_hours_n", 0.0))
        price = business_feat.get("b_price", 0.0)
        price_squared = price * price

        is_new_user = 1.0 if user_n == 0.0 else 0.0
        is_new_business = 1.0 if business_n == 0.0 else 0.0

        accepts_credit_cards = business_feat.get("b_accepts_credit_cards", 0.0)
        noise_level = business_feat.get("b_noise_level", 1.0)
        has_takeout = business_feat.get("b_takeout", 0.0)
        has_delivery = business_feat.get("b_delivery", 0.0)
        user_elite_years = user_feat.get("u_elite_count", 0.0)
        user_elite_years_log = user_feat.get("u_elite_count_log", 0.0)

        features = [
            #activity counts / basic stats, log to normalie and reduce impact of the larger ones 
            log1p_plus(user_n), 
            log1p_plus(business_n),
            user_mean_value, business_mean_value,
            #user_mean_value - global_mean, business_mean_value - global_mean, diff_raw_means,

            #user features
            user_feat.get("u_average_stars", 0.0),
            user_feat.get("u_review_count_log", 0.0),
            user_feat.get("u_fans_log", 0.0),
            user_feat.get("u_useful_log", 0.0),
            #user_avg_center,

            #business features
            business_feat.get("b_stars", 0.0),
            business_feat.get("b_review_count_log", 0.0),
            business_feat.get("b_is_open", 0.0),
            #business_feat.get("b_categories_n", 0.0),
            business_feat.get("b_price", 0.0),
            business_feat.get("b_hours_n", 0.0),
            business_stars_center,
            #categories_log,
            hours_log,
            price_squared,

            #shrunken mean interactions
            #user_mean_shrunk_value, 
            #business_mean_shrunk_value,
            #user_mean_shrunk_value - global_mean, business_mean_shrunk_value - global_mean,
            #diff_shrunk_means, diff_shrunk_means_squared, interaction_centered,
            user_vs_business_avg,

            #region/popularity
            #business_feat.get("b_city_pop_log", 0.0),
            business_feat.get("b_state_pop_log", 0.0),
            check_ins,
            #photos,
            tips,

            #locations (might not matter much?)
            #business_feat.get("b_lat_bin", 0.0),
            #business_feat.get("b_lon_bin", 0.0),

            #stability / cold start bits
            #log1p_plus(user_var),
            #log1p_plus(business_var),
            #is_new_user,
            #is_new_business,

            #diff between user and business ranking 
            #(user_mean_shrunk_value - business_mean_shrunk_value),

            #extra binary or misc features that i added to test 
            #accepts_credit_cards,
            noise_level,
            #has_delivery,
            has_takeout,
            user_elite_years,
            #user_elite_years_log,
        ]
        return features, base_rating, user_n, business_n
    return build_feature_row

#build rdd into x and ys for training of our xgbregressor
def prepare_training_matrix(train_rdd, build_feature_row):
    import xgboost as xgb
    X_train = []
    y_train = []

    #put train rdd into matrix for training model 
    for user_id, business_id, rating in train_rdd.collect():
        features, base_rating, _, _ = build_feature_row(user_id, business_id)
        X_train.append(features)
        y_train.append(rating - base_rating)
    return xgb.DMatrix(X_train, label=y_train)

#to put together xgbregressor model with params and the train matrix and rounds to train 
def train_booster(dtrain):
    import xgboost as xgb

    return xgb.train(xgb_params, dtrain, num_boost_round=num_rounds)

#building up matrix with features testing/predicting on 
def prepare_pairs_matrix(pairs, build_feature_row):
    import xgboost as xgb
    X, bases, n_users, n_businesses, user_ids, business_ids = [], [], [], [], [], []
    for user_id, business_id in pairs:
        features, base_rating, user_n, business_n = build_feature_row(user_id, business_id)
        X.append(features); bases.append(base_rating)
        n_users.append(user_n); n_businesses.append(business_n)
        user_ids.append(user_id); business_ids.append(business_id)
    return xgb.DMatrix(X), bases, n_users, n_businesses, user_ids, business_ids

#predicting ratings column based on booster built
def predict_pairs(booster, dmatrix, bases, n_users, n_businesses, user_ids, business_ids, u90=None, b90=None, u9999=None, b9999=None):
    import xgboost as xgb

    #using residuals to predict because yelp data is very very centered around 3-4 (not as much variation from this)
    #formula is yhat = base (y) + reliance*residual 
    residual_predictions = booster.predict(dmatrix)
    predictions = []

    #iterate through user_ids to regularize some of their predictions 
    for i in range(len(user_ids)):

        #calculate r valus (residuals)
        #helps early reviewiers count more than later ones (diminishing returns )
        r_user = min(1.0, log1p_plus(n_users[i]) / 6.0)
        r_business = min(1.0, log1p_plus(n_businesses[i]) / 6.0)

        #try to implement both sides having decent observability, if either side is poorly reviewed, reduce reliance 
        harmonic_mean = (2.0 * r_user * r_business) / (r_user + r_business)

        #tweaked this amount until something good for rmse was fond 
        reliance = 0.38 + 0.62 * harmonic_mean

        #for cold starts, reduce how much impact the model has 
        if n_users[i] < 2 or n_businesses[i] < 3:
            #rely on cold start cases less 
            reliance = reliance * 0.80
        
        #for very high counts (top 90% of users/businesses) increase how much they impact the final prediction
        if u90 is not None and b90 is not None:
            if (n_users[i] >= u90 and n_users[i] < u9999) or (n_businesses[i] >= b90 and n_businesses[i] < b9999):
                reliance = min(0.98, reliance * 1.06)
        if u9999 is not None and b9999 is not None:
            if (n_users[i] >= u9999) or (n_businesses[i] >= b9999):
                reliance = reliance * 0.75

        #puting together the final prediction value
        y_hat = bases[i] + reliance * float(residual_predictions[i])
        
        #clip and add into the predictions (make sure its 1-5)
        predictions.append((user_ids[i], business_ids[i], clip_rating(y_hat)))
    return predictions

# =========================
#main to put everything together
# =========================

def main():
    folder_path, test_file_name, output_file_name = check_inputs()

    #paths for the files we have (train/additional datasets)
    train_csv = os.path.join(folder_path, "yelp_train.csv")
    user_json = os.path.join(folder_path, "user.json")
    business_json = os.path.join(folder_path, "business.json")
    checkin_json = os.path.join(folder_path, "checkin.json")
    photo_json = os.path.join(folder_path, "photo.json")
    tip_json = os.path.join(folder_path, "tip.json")

    #validation file for testing purposes 
    val_path = os.path.join(folder_path, "yelp_val.csv")

    #build spark
    conf = SparkConf().setAppName("task2_2")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    #csv to rdd 
    #expect label for train vs test/val sets 
    train_rdd = load_csv_rdd(sc, train_csv, expect_label=True).persist()

    #get values from train_rdd (ended up not using all of this lol )
    (global_mean,
     user_mean, user_count, user_variance,
     business_mean, business_count, business_variance) = compute_global_and_aggregates(train_rdd)

    #also didnt end up using a good chunk of this...
    user_mean_shrunk, business_mean_shrunk = compute_shrunk_means(
        global_mean, user_mean, user_count, business_mean, business_count
    )

    #top percentile of users/busineses
    u90 = percentile_from_counts(user_count, 0.90)
    b90 = percentile_from_counts(business_count, 0.90)
    u9999 = percentile_from_counts(user_count, 0.9999)
    b9999 = percentile_from_counts(business_count, 0.9999)

    #building all features across various jsons 
    user_features = build_user_features(load_json_rdd(sc, user_json))
    business_features = build_business_features(load_json_rdd(sc, business_json))
    extra_business_features(
        business_features,
        load_json_rdd(sc, checkin_json),
        load_json_rdd(sc, photo_json),
        load_json_rdd(sc, tip_json),
    )
    
    #feature builder 
    #extra features should be built into this too 
    build_feature_row = make_feature_builder(
        global_mean,
        user_mean, user_count, user_variance, user_mean_shrunk,
        business_mean, business_count, business_variance, business_mean_shrunk,
        user_features, business_features
    )

    #build train matrix for xgbregressor
    dtrain = prepare_training_matrix(train_rdd, build_feature_row)
    booster = train_booster(dtrain)

    #put together test prediction by using input csv and our xgbregressor prediction
    test_pairs_rdd = load_csv_rdd(sc, os.path.join(folder_path, test_file_name), expect_label=False)
    test_pairs = test_pairs_rdd.collect()
    dtest, bases, n_users, n_businesses, user_ids, business_ids = prepare_pairs_matrix(test_pairs, build_feature_row)
    test_predictions = predict_pairs(booster, dtest, bases, n_users, n_businesses, user_ids, business_ids, u90, b90, u9999, b9999)
    write_output(output_file_name, test_predictions)


    #validation
    val_map = load_val_map(sc, val_path)
    rmse = compute_rmse(test_predictions, val_map)
    if rmse is None:
        print("Validation RMSE: N/A (no overlapping pairs)")
    else:
        print(f"Validation RMSE: {rmse:.6f}")
    #end validation 

    #stop spark
    sc.stop()

if __name__ == "__main__":
    main()
