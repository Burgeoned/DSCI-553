# task2_3.py
# usage: spark-submit task2_3.py <folder_path> <test_file_name> <output_file_name>
# case 3: hybrid meta model = cf + xgb into meta model 
# goal: beat both solo models on rmse, sub 1800s runtime, and <0.99 rmse

#notes to self: maybe do a proper data preprocessing step for final project?
#build proper set
#consider model into models


#experimented with meta models
#uhh i did not do as well as id like lol
import sys, os, csv, json, math, heapq, hashlib, re
from math import sqrt, log
from pyspark import SparkConf, SparkContext

# =========================
#adjustables (tweak here)
# =========================

#inner-validation split (for training the meta layer combining the two scripts (cf+xgb))
val_set_size = 0.35

#i tried too many different iterations and nothing is working
#i probably overcomplicated this
#simplify on the final proejct part
val_salt = "im_going_to_cry_why_is_this_so_hard"

#L2 penalty on meta ridge
#cant believe im using 552 lol
L2_penalty = 0.14

#winsor caps for meta features (keep outliers from moving blend too much)
#winsor was suggestion from john
residual_clipping = 1.05 
difference_clipping  = 1.60 

#tail guard (guard for when we don't have strong reliance on either model)
low_neighbor_cf = 2
low_reliance_cf = 0.35
meta_model_tail_shrinking = 0.8

#cf adjustables
user_beta, business_beta = 5.0, 15.0
cf_k_neighbors = 65
cf_min_coraters = 3
cf_min_sim = 0.40
cf_sim_power = 1.25
cf_dead = 0.0035
min_rating, max_rating = 1.0, 5.0

#xgb params (2_2)
xgb_params = {
    "objective": "reg:linear", 
    "eval_metric": "rmse",
    "eta": 0.005,
    "max_depth": 5,
    "subsample": 0.75,
    "colsample_bytree": 0.43,
    "colsample_bylevel": 0.43,
    "min_child_weight": 3,
    "base_score": 0.0,
    "gamma": 0.20,
    "lambda": 3.7,
    "alpha": 0.0,
    "silent": 1,
    "seed": 42
}
xgb_num_rounds = 135   

#priors for shrunk means (same idea as task2_2)
A_U = 20.0
A_B = 125.0

#xgb residual reliance schedule (how hard to lean on residual based on user/business counts)
trust_model = 0.67
trust_harmonic_mean = 0.33
reliance_user = 9.0
reliance_business = 9.0

#feature toggles
#made some levers because i got tired of adjusting code
use_statepoplog = False
use_geobins = True
use_interaction_centered = True
base_features_meta = False
use_review_text_features = True
use_user_trust_feature = True

#tiny sentiment review text (maybe expand this for the final project?)
positive_words = {
    "good","great","amazing","awesome","excellent","love","loved","like","liked",
    "perfect","friendly","nice","tasty","delicious","favorite","fantastic","best",
    "fresh","clean","quick","fast","happy","recommend","recommended"
}
negative_words = {
    "bad","terrible","awful","hate","hated","rude","dirty","overpriced",
    "worst","disappointing","disappointed","bland","stale",
    "unfriendly","noisy","loud","gross","poor", "unprofessional", "yikes"
}

#find sequences of words
token = re.compile(r"[A-Za-z]+")

# =========================
#reading/writing/etc, mostly taken from 2_1 and 2_2
# =========================
def to_file_path(path):
    path = os.path.abspath(path).replace("\\", "/")
    if not path.startswith("file:/"):
        path = "file:///" + path
    return path

def check_inputs():
    if len(sys.argv) != 4:
        print("Usage: spark-submit task2_3.py <folder_path> <test_file_name> <output_file_name>", file=sys.stderr)
        sys.exit(1)
    folder = os.path.abspath(os.path.normpath(sys.argv[1]))
    return folder, sys.argv[2], sys.argv[3]

def read_train_rows(lines_iter):
    reader = csv.reader(lines_iter)
    for row in reader:
        if not row or len(row) < 3: continue
        if row[0].strip().lower()=="user_id" and row[1].strip().lower()=="business_id": 
            continue
        uid = row[0].strip(); bid = row[1].strip(); stars = row[2].strip()
        if not uid or not bid: 
            continue
        try: 
            rating = float(stars)
        except: 
            continue
        yield (uid, bid, rating)

def read_test_rows(lines_iter):
    reader = csv.reader(lines_iter)
    for row in reader:
        if not row or len(row) < 2: 
            continue
        if row[0].strip().lower()=="user_id" and row[1].strip().lower()=="business_id": 
            continue
        uid = row[0].strip(); bid = row[1].strip()
        if not uid or not bid: 
            continue
        yield (uid, bid)

def load_csv_rdd(sc, path, expect_label):
    rdd = sc.textFile(to_file_path(path))
    return rdd.mapPartitions(read_train_rows if expect_label else read_test_rows)

def load_json_rdd(sc, path):
    if not os.path.exists(path): return None
    return sc.textFile(to_file_path(path)).map(lambda s: json.loads(s))

def write_output(out_csv, rows):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "business_id", "prediction"])
        for uid, bid, prediction in rows:
            w.writerow([uid, bid, prediction])

#test function for rmse testing 
def compute_rmse(predictions, val_map):
    mse_sum, n = 0.0, 0
    for user, business, p in predictions:
        y = val_map.get((user, business))
        if y is None: continue
        d = p - y
        mse_sum += d*d
        n += 1
    return None if n==0 else sqrt(mse_sum/n)

# =========================
#misc helpers mostly copied from 2_1/2_2
# =========================
def clip_rating(x, min=min_rating, max=max_rating):
    if x < min: return min
    if x > max: return max
    return x

def make_float(x, default=0.0):
    try: 
        return float(x)
    except: 
        return default

def log1p_plus(x):
    try:
        x = float(x)
        return log(1.0 + x) if x > 0 else 0.0
    except:
        return 0.0

def parse_bool(v):
    if v is None: return 0.0
    s = str(v).strip().lower()
    return 1.0 if s in {"true","yes","y","1"} else 0.0

def parse_noise_level(v):
    if v is None: return 1.0
    s = str(v).strip().lower()
    return {
        "quiet":0.0,
        "average":1.0,
        "loud":2.0,
        "very_loud":3.0,
        "very loud":3.0}.get(s,1.0)

#elite years for user
def count_elite_years(elite_field):
    if not elite_field or str(elite_field).strip().lower()=="none": return 0.0
    years = [y.strip() for y in str(elite_field).split(",") if y.strip()]
    return float(len(years))

#cap crazy residual directions (winsor tech)
def winsor(x, a):
    if x >  a: return a
    if x < -a: return -a
    return x

# =========================
#splitting our initial train set 
# =========================

#tbh just a hashing function i found online that was deterministic and good lol 
def stable_pair_bucket(uid, bid, salt):
    key = (uid + "||" + bid + "||" + salt).encode("utf-8")
    h = hashlib.md5(key).hexdigest()
    val = int(h[:8], 16)
    return val % 1000


def split_train_inner_val(train_rdd, frac=val_set_size, salt=val_salt):
    # % of our set for val
    threshold = int(frac * 1000)

    #tagging and building train/val sets 
    def tag_row(t):
        uid, bid, rating = t
        business = stable_pair_bucket(uid, bid, salt)
        return ("validation" if business < threshold else "train", (uid, bid, rating))
    #tagged rows split accordingly 
    tagged = train_rdd.map(tag_row)
    train_inner = tagged.filter(lambda x: x[0]=="train").map(lambda x: x[1]).persist()
    val_inner = tagged.filter(lambda x: x[0]=="validation").map(lambda x: x[1]).persist()

    #train and val set 
    return train_inner, val_inner

# =========================
#aggregate, base, features, mostly from 2_1 2_2, added some others though
# =========================
def compute_global_and_aggregates(train_rdd):
    #global mean
    global_sum, global_count = (
        train_rdd.map(lambda x: ("*", (x[2], 1)))
                 .reduceByKey(lambda a,business: (a[0]+business[0], a[1]+business[1]))
                 .map(lambda kv: kv[1]).first()
    )
    global_mean = global_sum/global_count if global_count>0 else 3.5

    #user mean + count
    user_agg = (train_rdd.map(lambda x: (x[0], (x[2], 1)))
                        .reduceByKey(lambda a,business: (a[0]+business[0], a[1]+business[1])))
    user_mean  = user_agg.mapValues(lambda s: s[0]/s[1]).collectAsMap()
    user_count = user_agg.mapValues(lambda s: s[1]).collectAsMap()

    #business mean + count
    business_agg = (train_rdd.map(lambda x: (x[1], (x[2], 1)))
                       .reduceByKey(lambda a,business: (a[0]+business[0], a[1]+business[1])))
    business_mean  = business_agg.mapValues(lambda s: s[0]/s[1]).collectAsMap()
    business_count = business_agg.mapValues(lambda s: s[1]).collectAsMap()

    return global_mean, user_mean, user_count, business_mean, business_count


#reuse of shrunk 
#shrink supposed to help with values but i think im just overthinking and overcomplicating this at this point sigh
def compute_shrunk_means(global_mean, user_mean, user_count, business_mean, business_count):
    u_shrunk = {}
    for uid, mu in user_mean.items():
        n = float(user_count.get(uid, 0))
        u_shrunk[uid] = (mu*n + global_mean*A_U)/(n + A_U) if (n + A_U)>0 else global_mean
    b_shrunk = {}
    for bid, mb in business_mean.items():
        n = float(business_count.get(bid, 0))
        b_shrunk[bid] = (mb*n + global_mean*A_B)/(n + A_B) if (n + A_B)>0 else global_mean
    return u_shrunk, b_shrunk

#user features from user json and reviewjson
def build_user_features(user_json_rdd):
    if not user_json_rdd: return {}
    return (user_json_rdd.map(lambda record:(
                record.get("user_id"),
                {
                    "u_review_count_log": log1p_plus(make_float(record.get("review_count"), 0)),
                    "u_average_stars": make_float(record.get("average_stars"), 0.0),
                    "u_fans_log": log1p_plus(make_float(record.get("fans"), 0)),
                    "u_useful_log": log1p_plus(make_float(record.get("useful"), 0)),
                    "u_elite_count": count_elite_years(record.get("elite")),
                    #added reviews/elite status, maybe userful?
                    "u_reviews_raw": make_float(record.get("review_count"), 0.0),
                    "u_useful_raw": make_float(record.get("useful"), 0.0),
                    "u_elite_status": 0.0 if (str(record.get("elite") or "").strip().lower() in {"", "none"}) else 1.0,
                }))).filter(lambda t: t[0] is not None).collectAsMap()

#buisness features from our multiple json rdd
def build_business_features(business_json_rdd, checkin_rdd, photo_rdd, tip_rdd):
    if not business_json_rdd: return {}
    business_rows, state_pop = [], {}

    #taken mostly from 2_2, modified for this 2_3 with some model changes
    def map_business(record):
        bid = record.get("business_id")
        if not bid: return None
        attrs = record.get("attributes") or {}
        categories_field = (record.get("categories") or "").lower()
        categories_count = 0 if not categories_field else len([category for category in categories_field.split(",") if category.strip()])
        price = make_float(attrs.get("RestaurantsPriceRange2"), 0.0)
        hours = record.get("hours") or {}; hours_n = len(hours) if isinstance(hours, dict) else 0
        state = (record.get("state") or "").strip()
        lat = make_float(record.get("latitude"),  0.0)
        lon = make_float(record.get("longitude"), 0.0)

        #recommended bin size for lat/lon online 
        lat_bin = math.floor(lat*2)/2.0
        lon_bin = math.floor(lon*2)/2.0

        row = {
            "b_stars": make_float(record.get("stars"), 0.0),
            "b_review_count_log": log1p_plus(make_float(record.get("review_count"), 0)),
            "b_is_open": 1.0 if make_float(record.get("is_open"), 0)>0 else 0.0,
            "b_categories_n": float(categories_count),
            "b_price": price,
            "b_hours_n": float(hours_n),
            "state": state,
            "b_checkins_total": 0.0,
            "b_photo_count": 0.0,
            "b_tip_count": 0.0,
            "b_noise_level": parse_noise_level(attrs.get("NoiseLevel")),
            "b_takeout":  parse_bool(attrs.get("RestaurantsTakeOut")),
            "b_lat_bin": lat_bin,
            "b_lon_bin": lon_bin,

            #review text added in 
            "rev_text_count": 0.0,
            "rev_text_avg_len": 0.0,
            "rev_pos": 0.0,
            "rev_neg": 0.0,
            "rev_neg_dom": 0.0,
        }
        return (bid, row)

    #get states and add for populatoin
    for bid, row in (business_json_rdd.map(map_business).filter(lambda x: x is not None).collect()):
        business_rows.append((bid, row))
        s = row["state"]
        if s: state_pop[s] = state_pop.get(s, 0) + 1

    if checkin_rdd:
        check_map = (checkin_rdd.map(lambda record: (record.get("business_id"),
                                                     sum([make_float(v,0.0) for v in (record.get("time") or {}).values()])))
                              .filter(lambda t: t[0] is not None)
                              .collectAsMap())
        for bid, row in business_rows:
            if bid in check_map: row["b_checkins_total"] = float(check_map[bid])

    if photo_rdd:
        photo_map = (photo_rdd.map(lambda record: (record.get("business_id"), 1.0))
                             .reduceByKey(lambda a,business: a+business).collectAsMap())
        for bid, row in business_rows:
            if bid in photo_map: row["b_photo_count"] = float(photo_map[bid])

    if tip_rdd:
        tip_map = (tip_rdd.map(lambda record: (record.get("business_id"), 1.0))
                           .reduceByKey(lambda a,business: a+business).collectAsMap())
        for bid, row in business_rows:
            if bid in tip_map: row["b_tip_count"] = float(tip_map[bid])

    bf = {}
    for bid, row in business_rows:
        row["b_state_pop_log"] = log1p_plus(float(state_pop.get(row.pop("state"), 0))) if use_statepoplog else 0.0
        bf[bid] = row
    return bf

# =========================
#additional portion i added for review_trai.json
# =========================
#count positive/negative words in a review,and length of review
def count_review_sentiment(text):
    if not text: 
        return 0, 0, 0
    tokens = token.findall(text.lower())
    pos = sum(1 for w in tokens if w in positive_words)
    neg = sum(1 for w in tokens if w in negative_words)
    return pos, neg, len(tokens)

#add to user features
def add_review_text_features_to_users(user_features, review_rdd):
    if (not use_review_text_features) or (not review_rdd): return
    #should be shape for ref: (user_id, (count, total_len, pos, neg))
    agg = (review_rdd
           .map(lambda record: (record.get("user_id"), record.get("text") or ""))
           .filter(lambda t: t[0] is not None)
           .map(lambda t: (t[0], count_review_sentiment(t[1])))
           .mapValues(lambda x: (1.0, x[2], x[0], x[1]))
           .reduceByKey(lambda a,business: (a[0]+business[0], a[1]+business[1], a[2]+business[2], a[3]+business[3]))
           .collectAsMap())
    
    #aggregates for user within text (reviews)
    for uid, (count, total_len, pos, neg) in agg.items():
        #calculate user average length of reviews
        avg_len = (total_len / count) if count > 0 else 0.0

        #influence negative more, i was thinking maybe for sarcastic users?
        neg_dom = float(neg) * (1.0 + 0.5 * min(1.0, (float(pos) / (float(neg)+1.0))))
        uf = user_features.get(uid, {})
        uf["u_rev_text_count"] = float(count)
        uf["u_rev_text_avg_len"] = float(avg_len)
        uf["u_rev_pos"] = float(pos)
        uf["u_rev_neg"] = float(neg)

        #seemed to not perform well, maybe trim?
        uf["u_rev_neg_dom"] = float(neg_dom)

        #user_trust on? 
        if use_user_trust_feature:
            u_reviews_json = uf.get("u_reviews_raw", 0.0)
            u_useful       = uf.get("u_useful_raw", 0.0)
            u_elite_status = uf.get("u_elite_status", 0.0)

            #scale accordingly to decide if trust user 
            uf["u_trust"] = (0.3*log1p_plus(u_reviews_json) +
                             0.35*log1p_plus(u_useful) +
                             0.1*u_elite_status +
                             0.25*log1p_plus(avg_len))
        user_features[uid] = uf

#adding review portions also to business features
def add_review_text_features_to_businesses(business_features, review_rdd):
    #check for exist features and rdd
    if (not use_review_text_features) or (not review_rdd): 
        return
    agg = (review_rdd
           .map(lambda record: (record.get("business_id"), record.get("text") or ""))
           .filter(lambda t: t[0] is not None)
           .map(lambda t: (t[0], count_review_sentiment(t[1])))
           .mapValues(lambda x: (1.0, x[2], x[0], x[1]))
           .reduceByKey(lambda a,business: (a[0]+business[0], a[1]+business[1], a[2]+business[2], a[3]+business[3]))
           .collectAsMap())
    
    #same logic as user side
    for bid, (count, total_len, pos, neg) in agg.items():
        avg_len = (total_len / count) if count > 0 else 0.0
        neg_dom = float(neg) * (1.0 + 0.5 * min(1.0, (float(pos) / (float(neg)+1.0))))
        bf = business_features.get(bid, {})
        bf["rev_text_count"] = float(count)
        bf["rev_text_avg_len"] = float(avg_len)
        bf["rev_pos"] = float(pos)
        bf["rev_neg"] = float(neg)
        bf["rev_neg_dom"] = float(neg_dom)
        business_features[bid] = bf

# =========================
#xgb residual model, copied mostly from 2_2 with some slight modifications
# =========================

def compute_reliance(num_users, num_businesses):

    #reliance, based on how many we have of user/business
    r_u = min(1.0, log1p_plus(num_users)/reliance_user)
    r_b = min(1.0, log1p_plus(num_businesses)/reliance_business)

    #same harmonic mean func as before 
    harmonic_mean = (2.0*r_u*r_b)/(r_u + r_b) if (r_u + r_b) > 0 else 0.0
    return trust_model + trust_harmonic_mean*harmonic_mean

#mostly from 2_2
def make_feature_builder(global_mean,
                         user_mean, user_count, user_mean_shrunk,
                         business_mean, business_count,  business_mean_shrunk,
                         user_features, business_features):
    def build_row(uid, bid):
        uf = user_features.get(uid, {})
        bf = business_features.get(bid, {})

        #shrunk for normalizing
        #seemed to help, but overcomplicating things i think...
        mu = user_mean.get(uid, global_mean)
        mb = business_mean.get(bid, global_mean)
        mu_shrunk = user_mean_shrunk.get(uid, global_mean)
        mb_shrunk = business_mean_shrunk.get(bid, global_mean)
        num_users = float(user_count.get(uid, 0))
        num_businesses = float(business_count.get(bid, 0))

        #base is to look at bias, used in 2_2 too 
        base = global_mean + (mu_shrunk - global_mean) + (mb_shrunk - global_mean)
        inter_center = (mu - global_mean) * (mb - global_mean) if use_interaction_centered else 0.0

        #user trust from review json 
        u_trust = uf.get("u_trust", 0.0) if use_user_trust_feature else 0.0

        features = [
            #activity and means
            log1p_plus(num_users), log1p_plus(num_businesses),
            mu, mb,

            #user feature
            uf.get("u_average_stars", 0.0),
            uf.get("u_review_count_log", 0.0),
            uf.get("u_fans_log", 0.0),
            uf.get("u_useful_log", 0.0),
            u_trust,
            uf.get("u_rev_text_count", 0.0),
            log1p_plus(uf.get("u_rev_text_avg_len", 0.0)),
            uf.get("u_rev_pos", 0.0),
            uf.get("u_rev_neg", 0.0),
            #uf.get("u_rev_neg_dom", 0.0),

            #business features 
            bf.get("b_stars", 0.0),
            bf.get("b_review_count_log", 0.0),
            bf.get("b_is_open", 0.0),
            bf.get("b_price", 0.0),
            bf.get("b_hours_n", 0.0),
            (bf.get("b_stars", 0.0) - global_mean),
            log1p_plus(bf.get("b_hours_n", 0.0)),
            (bf.get("b_price", 0.0) ** 2),
            (uf.get("u_average_stars", 0.0) - bf.get("b_stars", 0.0)),

            #poppularity measures
            #bf.get("b_state_pop_log", 0.0),
            math.sqrt(max(0.0, bf.get("b_checkins_total", 0.0))),
            #0.8*log1p_plus(bf.get("b_tip_count", 0.0)),

            #geo bins
            (bf.get("b_lat_bin", 0.0) if use_geobins else 0.0),
            (bf.get("b_lon_bin", 0.0) if use_geobins else 0.0),

            #ambience and logistics
            bf.get("b_noise_level", 1.0),
            bf.get("b_takeout", 0.0),

            #elite years, dunno if matters 
            uf.get("u_elite_count", 0.0),

            #centered interaction
            inter_center,

            #business text section
            bf.get("rev_text_count", 0.0),
            log1p_plus(bf.get("rev_text_avg_len", 0.0)),
            bf.get("rev_pos", 0.0),
            bf.get("rev_neg", 0.0),
            #bf.get("rev_neg_dom", 0.0),
        ]
        return features, base, num_users, num_businesses
    return build_row

#xgb modified for this new hybrid model 2_3
def xgb_train_and_predict_with_reliance(train_rdd, pairs, build_row):
    import xgboost as xgb
    X_train, y_train = [], []
    for user, business, y in train_rdd.collect():
        features, base, _, _ = build_row(user, business)
        X_train.append(features); y_train.append(y - base)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    booster = xgb.train(xgb_params, dtrain, num_boost_round=xgb_num_rounds)

    #build matrix for xgb 
    X, bases, nus, nbs, u, b = [], [], [], [], [], []
    for user, business in pairs:
        features, base, num_users, num_businesses = build_row(user, business)
        X.append(features); bases.append(base); nus.append(num_users); nbs.append(num_businesses)
        u.append(user); b.append(business)
    dtest = xgb.DMatrix(X)

    #predicting residuals 
    resid = booster.predict(dtest)

    #get to output for meta model (combining both models)
    out = []
    for i in range(len(u)):
        reliance = compute_reliance(nus[i], nbs[i])
        prediction = bases[i] + reliance * float(resid[i])
        out.append((u[i], b[i], clip_rating(prediction), reliance))
    return out

# =========================
#CF section, updated for this part too
# =========================
#rewrote a lot for practice and to make a lil cleaner lol 

#direction of CF 
def cf_weight_signed(s):
    a = abs(s)
    if a <= cf_dead: 
        return 0.0
    a -= cf_dead
    return (a ** cf_sim_power) * (1.0 if s >= 0 else -1.0)

#still getting means 
def cf_get_means(train_rdd):
    user_record = (train_rdd.map(lambda t: (t[0], (t[2], 1)))
                      .reduceByKey(lambda a,business: (a[0]+business[0], a[1]+business[1])))
    business_record  = (train_rdd.map(lambda t: (t[1], (t[2], 1)))
                      .reduceByKey(lambda a,business: (a[0]+business[0], a[1]+business[1])))
    s, c = train_rdd.map(lambda t: t[2]).aggregate(
        (0.0, 0),
        lambda acc, x: (acc[0]+x, acc[1]+1),
        lambda a, b: (a[0]+b[0], a[1]+b[1])
    )
    avg = s/c if c else 3.0
    
    user_avg, business_avg = {}, {}
    for uid, (sum_ratings, count_ratings) in user_record.collect():
        user_avg[uid] = (sum_ratings + user_beta*avg)/(count_ratings + user_beta)
    for bid, (sum_ratings, count_ratings) in business_record.collect():
        business_avg[bid] = (sum_ratings + business_beta*avg)/(count_ratings + business_beta)
    return {"global_mean": avg, "user_avg": user_avg, "business_avg": business_avg}

#impute portion of cf model 
def cf_impute(uid, bid, means):
    global_mean = means["global_mean"]
    user_mean = means["user_avg"].get(uid)
    business_mean = means["business_avg"].get(bid)
    if user_mean and business_mean: 
        return max(min(user_mean + business_mean - global_mean, max_rating), min_rating)
    if user_mean: 
        return user_mean
    if business_mean: 
        return business_mean
    return global_mean

#reuse of 2_1
def cf_build_user_index(train_rdd):
    return train_rdd.map(lambda t: t[0]).distinct().zipWithIndex().collectAsMap()

#also reuse of 2_1
def cf_make_pearson_indices(train_rdd, user_index):
    user_items = (train_rdd.map(lambda t: (t[0], (t[1], t[2]))).groupByKey().mapValues(list).collectAsMap())
    def to_user_index(t):
        user, business, rating = t
        ui = user_index.get(user)
        return (business, (ui, rating)) if ui is not None else None
    business_users = (train_rdd.map(to_user_index).filter(lambda x: x is not None)
                               .groupByKey().mapValues(list).collectAsMap())
    return user_items, business_users


#comparison with pearson (CF), mostly copy of my 2_1
def cf_pearson(bi, bj, business_users, business_avg, global_mean):

    users_for_i = business_users.get(bi)
    users_for_j = business_users.get(bj)
    if not users_for_i or not users_for_j:
        return 0.0, 0

    if len(users_for_i) <= len(users_for_j):
        ratings_i_by_user = {user_index:rating for (user_index, rating) in users_for_i}
        other_side = users_for_j
    else:
        ratings_i_by_user = {user_index:rating for (user_index, rating) in users_for_j}
        other_side = users_for_i

    mean_i = business_avg.get(bi, global_mean)
    mean_j = business_avg.get(bj, global_mean)

    #running totals for pearson pieces
    numer = 0.0
    denom_i = 0.0
    denom_j = 0.0 
    co = 0     

    for user_index, rj in other_side:
        ri = ratings_i_by_user.get(user_index)
        if ri is None:
            continue
        co += 1
        di = ri - mean_i
        dj = rj - mean_j
        numer += di * dj
        denom_i += di * di
        denom_j += dj * dj

    #copying logic from 2_1
    if co < cf_min_coraters or numer == 0.0 or denom_i == 0.0 or denom_j == 0.0:
        return 0.0, co

    sim = numer / math.sqrt(denom_i * denom_j)

    #get rid of small noises
    if abs(sim) < cf_min_sim:
        return 0.0, co
    return sim, co


def cf_build_context(train_rdd):
    #prep:compute shrunk means + the index structures we need for item-pearson
    means = cf_get_means(train_rdd)
    user_index = cf_build_user_index(train_rdd)
    user_items, business_users = cf_make_pearson_indices(train_rdd, user_index)

    return {
        "user_items": user_items,             
        "business_users": business_users,       
        "business_avg": means["business_avg"],   
        "means": means                        
    }


def cf_predict_one(uid, bid, context):
    #predict one pair (uid,bid) via item-CF residuals around a mean-centered base
    user_items = context["user_items"]
    business_users = context["business_users"]
    business_avg = context["business_avg"]
    means = context["means"]

    #cold start cases
    if uid not in user_items or bid not in business_users:
        p = cf_impute(uid, bid, means)
        return (uid, bid, clip_rating(p), 0, 0.0)

    neighbors = []
    for neighbor_bid, r_user_neighbor in user_items[uid]:
        if neighbor_bid == bid:
            continue
        sim, _ = cf_pearson(bid, neighbor_bid, business_users, business_avg, means["global_mean"])
        if sim != 0.0:
            neighbors.append((abs(sim), sim, neighbor_bid, r_user_neighbor))

    #no usable neighbors, impute
    if not neighbors:
        p = cf_impute(uid, bid, means)
        return (uid, bid, clip_rating(p), 0, 0.0)

    #top-k by abs sims 
    top = heapq.nlargest(cf_k_neighbors, neighbors, key=lambda x: x[0])

    #mean-centering baseline pieces
    g = means["global_mean"]
    u_center = means["user_avg"].get(uid, g) - g

    #aggregate weighted residuals from neighbors (apply deadzone/power inside cf_weight_signed)
    numer = 0.0
    denom = 0.0
    abs_sim_sum = 0.0
    for _abs_s, sim, neighbor_bid, r_user_neighbor in top:
        j_center = business_avg.get(neighbor_bid, g) - g
        base_user_neighbor = g + u_center + j_center
        resid = r_user_neighbor - base_user_neighbor
        w = cf_weight_signed(sim)
        numer += w * resid
        aw = abs(w)
        denom += aw
        abs_sim_sum += aw

    #final blend around base for the target item
    base_ui = g + u_center + (business_avg.get(bid, g) - g)
    pred = cf_impute(uid, bid, means) if denom < 1e-9 else base_ui + (numer / denom)

    k_used = len(top)
    sim_strength = (abs_sim_sum / k_used) if k_used > 0 else 0.0
    return (uid, bid, clip_rating(pred), k_used, sim_strength)

#predict and use this later for meta model 
def cf_predict_pairs_with_conf(train_rdd, pairs):
    ctx = cf_build_context(train_rdd)
    return [cf_predict_one(u, b, ctx) for (u, b) in pairs]

# =========================
#meta-model, tiny ridge
# =========================
#matrix_mult_ata
#honestly probalby has a lot of mistakes.... first time trying to use this 
#supposedly runs better on numpy lol 

def matrix_mult_ata(X):
    #matrix multiplication of features 
    n_features = len(X[0])
    gram_matrix = [[0.0] * n_features for _ in range(n_features)]

    #for each row, add transposed x in as well 
    for row in X:
        for i in range(n_features):
            xi = row[i]

            #recommended for sparse row speed 
            if xi == 0.0:
                continue 
            for j in range(n_features):
                gram_matrix[i][j] += xi * row[j]

    return gram_matrix

def matrix_vector_aty(X, y):
    #builds aty to use later
    n_features = len(X[0])

    #scale to features
    aty_vector = [0.0] * n_features

    for row_index, row in enumerate(X):
        target_value = y[row_index]
        if target_value == 0.0:
            continue
        for i in range(n_features):
            aty_vector[i] += row[i] * target_value

    #get vector to use in gaussian solve
    return aty_vector


#linear system matrix of matrix*x = vector
def gauss_solve(matrix_a, vector_b):

    #partial pivoting technique 
    #len to match them up 
    n = len(matrix_a)

    #i really had to do a lot of research on this bit-- review on own time
    augmented_matrix = [matrix_a[i][:] + [vector_b[i]] for i in range(n)]

    #partial pivoting?
    #pivoting is anchoring a column to use to eliminate rest of column
    for col in range(n):
        #maximum abs value to use as best option
        pivot_row = col
        pivot_abs_max = abs(augmented_matrix[col][col])
        for r in range(col + 1, n):
            cand = abs(augmented_matrix[r][col])
            if cand > pivot_abs_max:
                pivot_abs_max = cand
                pivot_row = r

        #column near empty, can skip
        if pivot_abs_max < 1e-12:
            continue

        #pivot row to current position 
        #review linear alg for understanding everything here 
        if pivot_row != col:
            augmented_matrix[col], augmented_matrix[pivot_row] = augmented_matrix[pivot_row], augmented_matrix[col]

        pivot_value = augmented_matrix[col][col]

        #get rid of the chunks below the pivot 
        #basically why pivoted 
        for r in range(col + 1, n):
            factor = augmented_matrix[r][col] / pivot_value if pivot_value != 0.0 else 0.0
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                augmented_matrix[r][c] -= factor * augmented_matrix[col][c]

    #substitute back in 
    solution = [0.0] * n
    for i in range(n - 1, -1, -1):
        rhs = augmented_matrix[i][n]
        pivot_value = augmented_matrix[i][i]

        #0.0 if pivot is almost 0 
        if abs(pivot_value) < 1e-12:
            solution[i] = 0.0
            continue

        for j in range(i + 1, n):
            rhs -= augmented_matrix[i][j] * solution[j]

        solution[i] = rhs / pivot_value

    #should give us the weights for our meta model??
    return solution


#fit_ridge_with_bias
#use L2 penalty (omg from 552)
#x is the feature matrix 
#y is target residual (truth - base)
#(ata + penalty)*x = aty
def fit_ridge_with_bias(X_raw, y, l2=L2_penalty):
    #add bias column=1.0 at index 0
    X = [[1.0] + row[:] for row in X_raw]
    d = len(X[0])

    #normal-equation pieces
    ata = matrix_mult_ata(X)
    aty = matrix_vector_aty(X, y)

    #ridge on non-bias weights only (don’t shrink the intercept)
    for i in range(1, d):
        #penalize l2
        ata[i][i] += l2

    #solve for w (includes bias at index 0)
    weights = gauss_solve(ata, aty)
    return weights


#apply_linear to turn into regression problem kinda 
def apply_linear(weights, features):

    #bias score 
    score = weights[0]                 

    #iterates through for value        
    for i, val in enumerate(features, start=1):
        score += weights[i] * val
    
    #should be our actual predictions
    return score

# =========================
# main to combine everything 
# =========================
def main():
    folder_path, test_file_name, output_file_name = check_inputs()

    train_csv = os.path.join(folder_path, "yelp_train.csv")
    test_csv  = os.path.join(folder_path, test_file_name)

    #using val set to validate and compute rmse at end to check how wellits performing 
    val_csv = os.path.join(folder_path, "yelp_val.csv")

    #all the file paths we have 
    user_json = os.path.join(folder_path, "user.json")
    business_json = os.path.join(folder_path, "business.json")
    checkin_json = os.path.join(folder_path, "checkin.json")
    photo_json = os.path.join(folder_path, "photo.json")
    tip_json = os.path.join(folder_path, "tip.json")
    review_json = os.path.join(folder_path, "review_train.json")

    #building spark 
    conf = SparkConf().setAppName("task2_3")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    #load data
    full_train_rdd = load_csv_rdd(sc, train_csv, expect_label=True).persist()
    test_pairs_rdd = load_csv_rdd(sc, test_csv,  expect_label=False)
    test_pairs = test_pairs_rdd.collect()

    #inner split for meta model to train on 
    train_inner, val_inner = split_train_inner_val(full_train_rdd, val_set_size, val_salt)

    #just inner model 
    (gm_in, umean_in, ucount_in, bmean_in, bcount_in) = compute_global_and_aggregates(train_inner)
    u_sh_in, b_sh_in = compute_shrunk_means(gm_in, umean_in, ucount_in, bmean_in, bcount_in)

    uf_in = build_user_features(load_json_rdd(sc, user_json))
    bf_in = build_business_features(
        load_json_rdd(sc, business_json),
        load_json_rdd(sc, checkin_json),
        load_json_rdd(sc, photo_json),
        load_json_rdd(sc, tip_json)
    )
    #review rdd add 
    review_rdd_in = load_json_rdd(sc, review_json)
    add_review_text_features_to_users(uf_in, review_rdd_in)
    add_review_text_features_to_businesses(bf_in, review_rdd_in)

    build_row_in = make_feature_builder(
        gm_in, umean_in, ucount_in, u_sh_in, bmean_in, bcount_in, b_sh_in, uf_in, bf_in
    )

    #inner validation
    val_pairs = [(user,business) for (user,business,_) in val_inner.collect()]
    xgb_val = xgb_train_and_predict_with_reliance(train_inner, val_pairs, build_row_in)
    cf_val = cf_predict_pairs_with_conf(train_inner, val_pairs)

    xg_map = {(user,business):(p,rel) for (user,business,p,rel) in xgb_val}
    cf_map = {(user,business):(p,k,sum_ratings) for (user,business,p,k,sum_ratings) in cf_val}
    val_truth = {(user,business):rating for (user,business,rating) in val_inner.collect()}

    #meta features winsor on residuals for directions 
    X_meta, y_meta = [], []
    for (user,business) in val_pairs:
        xgb_value = xg_map.get((user,business)); cf_value = cf_map.get((user,business))
        if xgb_value is None or cf_value is None: continue
        (xg_p, rel)   = xgb_value
        (cf_p, k, sum_ratings) = cf_value
        _, base, num_users, num_businesses = build_row_in(user,business)


        #clip residuals with value at top of code
        #technique found in book 
        cf_resid = winsor(cf_p - base, residual_clipping)
        xg_resid = winsor(xg_p - base, residual_clipping)

        #compare cf and xgb predictions 
        diff = winsor(cf_p - xg_p,  difference_clipping)
        agree = 1.0 if (cf_resid * xg_resid) > 0 else 0.0

        #building together new feature for model based on cf+xgb
        features = []
        if base_features_meta: 
            features.append(base)
        features.extend([
            cf_p, xg_p,
            log1p_plus(num_users), log1p_plus(num_businesses),
            abs(diff),
            diff*diff,
            log1p_plus(k),
            sum_ratings,
            rel,
            cf_resid,
            xg_resid,
            agree,
        ])
        X_meta.append(features)
        #true values 
        y_meta.append(val_truth[(user,business)] - base)

    #meta model! 
    w_meta = fit_ridge_with_bias(X_meta, y_meta, L2_penalty)

    #all full datasets to do the final train on 
    (gm_full, umean_full, ucount_full, bmean_full, bcount_full) = compute_global_and_aggregates(full_train_rdd)
    u_sh_full, b_sh_full = compute_shrunk_means(gm_full, umean_full, ucount_full, bmean_full, bcount_full)

    #build features to use , full features sets
    uf_full = build_user_features(load_json_rdd(sc, user_json))
    bf_full = build_business_features(
        load_json_rdd(sc, business_json),
        load_json_rdd(sc, checkin_json),
        load_json_rdd(sc, photo_json),
        load_json_rdd(sc, tip_json)
    )
    review_rdd_full = load_json_rdd(sc, review_json)
    add_review_text_features_to_users(uf_full, review_rdd_full)
    add_review_text_features_to_businesses(bf_full, review_rdd_full)

    build_row_full = make_feature_builder(
        gm_full, umean_full, ucount_full, u_sh_full, bmean_full, bcount_full, b_sh_full, uf_full, bf_full
    )

    #predict test using both bases + meta
    xgb_test = xgb_train_and_predict_with_reliance(full_train_rdd, test_pairs, build_row_full)
    cf_test = cf_predict_pairs_with_conf(full_train_rdd, test_pairs)

    #map out the xgb and cf results for the final prediction
    xg_test_map = {(user,business):(p,rel) for (user,business,p,rel) in xgb_test}
    cf_test_map = {(user,business):(p,k,sum_ratings) for (user,business,p,k,sum_ratings) in cf_test}

    final_rows = []
    for (user,business) in test_pairs:
        xgb_value = xg_test_map.get((user,business)); cf_value = cf_test_map.get((user,business))
        if xgb_value is None or cf_value is None:
            _, base, _, _ = build_row_full(user,business)
            final_rows.append((user,business,clip_rating(base)))
            continue

        (xg_p, rel) = xgb_value
        (cf_p, k, sum_ratings) = cf_value
        _, base, num_users, num_businesses = build_row_full(user,business)

        #winsor residuals to normalize, and clip
        #use base to get directiona nd compare how the two rate things separately
        cf_resid = winsor(cf_p - base, residual_clipping)
        xg_resid = winsor(xg_p - base, residual_clipping)
        diff = winsor(cf_p - xg_p,  difference_clipping)
        agree = 1.0 if (cf_resid * xg_resid) > 0 else 0.0

        features = []
        if base_features_meta: 
            features.append(base)

        features.extend([
            #model features for meta model 
            cf_p, xg_p,
            log1p_plus(num_users), log1p_plus(num_businesses),
            abs(diff),
            diff*diff,
            log1p_plus(k),
            sum_ratings,
            rel,
            cf_resid,
            xg_resid,
            agree,
        ])
        meta_resid = apply_linear(w_meta, features)

        #tail guard for extreme cases 
        if (k < low_neighbor_cf) and (rel < low_reliance_cf):
            meta_resid *= meta_model_tail_shrinking

        prediction = clip_rating(base + meta_resid)
        final_rows.append((user, business, prediction))

    #final output
    write_output(output_file_name, final_rows)

    test_with_labels = load_csv_rdd(sc, val_csv, expect_label=True).collect()
    test_label_map = {(user,business): rating for (user,business,rating) in test_with_labels} if test_with_labels else {}

    if test_label_map:
        rmse_test = compute_rmse(final_rows, test_label_map)
        if rmse_test is not None:
            print(f"Test RMSE (stacked): {rmse_test:.6f}")

    #stop spark
    sc.stop()

if __name__ == "__main__":
    main()
