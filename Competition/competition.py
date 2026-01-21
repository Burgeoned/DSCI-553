#Model Evaluation and Improvement/Features

#Validation RMSE: 0.976939

# ERROR DISTRIBUTION (absolute error, overlapping pairs) 
#>=0 and <1  :   102421  (72.11%)
#>=1 and <2  :    32629  (22.97%)
#>=2 and <3  :     6156  ( 4.33%)
#>=3 and <4  :      835  ( 0.59%)
#>=4         :        3  ( 0.00%)

#Total elapsed time: 268.19 seconds (can fluctuate)

#Model specs/changes from HW3:
#competition.py
#usage: spark-submit competition.py <folder_path> <test_file_name> <output_file_name>
#relies solely on xgboost (dropped CF, seemed to give me too much noise and was helpful in very small segments only and built a baseline)
#
# For this competition, my biggest improvements came from stepping back and really re-evaluating how I was building features. Instead of adding 
# more complexity, I focused on simplifying, removing noisy signals, and strengthening the core features that actually mattered. I spent more 
# time doing EDA this round—checking correlations, looking at distributions, and comparing before/after RMSE impacts of individual features. This 
# helped me spot a lot of unnecessary clutter from earlier versions of the model.A major change was trimming down overly granular or weak features. 
# In earlier attempts I normalized too aggressively and included too many small behavioral stats, which made the model noisy. Cutting those out 
# made XGBoost split much more cleanly and improved generalization on the validation set.I also added lightweight tip-sentiment features. Even 
# without any NLP libraries, simple word and phrase scoring (positive vs negative) gave a useful signal for scenarios where businesses didn’t have 
# many reviews. It wasn’t huge, but it consistently improved the model.Another improvement came from grouping the 1300+ Yelp categories into a smaller 
# set of semantic groups. This reduced dimensionality and removed a lot of the randomness from one-off category names. It helped the model understand 
# the “type” of business better rather than chasing sparse one-hot features.Throughout the process, I ran multiple controlled tests—checking error 
# distributions, comparing feature importance, and doing small train/test splits inside the training set to sanity-check improvements before committing. 
# I also experimented with a light CF model, but after evaluating its effect, the biggest gains clearly came from better-engineered structured features.
# Overall, the main lesson was that stronger features and cleaner structure mattered more than layering on additional complexity.

import sys
import os
import csv
import json
import math
import datetime
import re
import hashlib
import time
from math import log, sqrt
from pyspark import SparkConf, SparkContext

# =========================
# category grouping
# =========================

#this took way too long to map i want to kms :) 

group_to_categories = {
    "Restaurants & Dining": [
        "Afghan", "African", "American (New)", "American (Traditional)",
        "Arabian", "Argentine", "Armenian", "Asian Fusion", "Australian",
        "Austrian", "Baguettes", "Bangladeshi", "Barbeque", "Basque",
        "Bavarian", "Belgian", "Bistros", "Brasseries", "Brazilian",
        "Breakfast & Brunch", "British", "Buffets", "Bulgarian", "Burmese",
        "Cajun/Creole", "Cambodian", "Canadian (New)", "Cantonese",
        "Caribbean", "Cheesesteaks", "Chilean", "Chinese", "Colombian",
        "Comfort Food", "Conveyor Belt Sushi", "Cuban", "Czech",
        "Czech/Slovakian", "Delicatessen", "Dim Sum", "Diners", "Dominican",
        "Eastern European", "Eatertainment", "Egyptian", "Empanadas",
        "Ethiopian", "Falafel", "Filipino", "Fish & Chips", "Fondue", "Food",
        "French", "Gastropubs", "German", "Gluten-Free", "Greek", "Guamanian",
        "Hainan", "Haitian", "Hakka", "Halal", "Hawaiian",
        "Himalayan/Nepalese", "Honduran", "Hong Kong Style Cafe", "Hot Pot",
        "Hungarian", "Iberian", "Indian", "Indonesian", "Irish", "Irish Pub",
        "Italian", "Izakaya", "Japanese", "Japanese Curry",
        "Japanese Sweets", "Kebab", "Korean", "Kosher", "Laotian",
        "Latin American", "Lebanese", "Local Flavor", "Malaysian",
        "Mediterranean", "Mexican", "Middle Eastern", "Modern European",
        "Mongolian", "Moroccan", "New Mexican Cuisine", "Nicaraguan", "Noodles",
        "Northern German", "Pakistani", "Pan Asian", "Pasta Shops", "Persian/Iranian",
        "Peruvian", "Pita", "Pizza", "Poke", "Polish", "Pop-Up Restaurants",
        "Portuguese", "Poutineries", "Pub Food", "Puerto Rican", "Ramen",
        "Restaurant Supplies", "Restaurants", "Rotisserie Chicken",
        "Russian", "Salad", "Salvadoran", "Scandinavian", "Scottish", "Seafood",
        "Seafood Markets", "Senegalese", "Serbo Croatian", "Shanghainese",
        "Sicilian", "Signature Cuisine", "Singaporean", "Slovakian",
        "Smokehouse", "Soba", "Soul Food", "Soup", "South African", "Southern",
        "Spanish", "Sri Lankan", "Steakhouses", "Sushi Bars", "Swiss Food",
        "Syrian", "Szechuan", "Tacos", "Taiwanese", "Tapas/Small Plates",
        "Tempura", "Teppanyaki", "Tex-Mex", "Thai", "Tonkatsu", "Trinidadian",
        "Turkish", "Udon", "Ukrainian", "Uzbek", "Vegan", "Vegetarian",
        "Venezuelan", "Vietnamese", "Wraps"
    ],

    "Cafes & Coffee Shops": [
        "Cafes", "Coffee & Tea", "Coffeeshops", "Tea Rooms",
        "Bubble Tea", "Juice Bars & Smoothies"
    ],

    "Fast Food & Quick Service": [
        "Fast Food", "Sandwiches", "Burgers", "Chicken Wings",
        "Hot Dogs", "Food Trucks", "Food Stands", "Delis"
    ],

    "Bakeries & Desserts": [
        "Bakeries", "Desserts", "Donuts", "Patisserie/Cake Shop",
        "Cupcakes", "Ice Cream & Frozen Yogurt", "Gelato", "Waffles"
    ],

    "Bars & Nightlife": [
        "Bars", "Pubs", "Lounges", "Cocktail Bars", "Dive Bars",
        "Beer Gardens", "Wine Bars", "Nightlife", "Karaoke"
    ],

    "Grocery & Specialty Food": [
        "Grocery", "Convenience Stores", "Ethnic Grocery", "Farmers Market",
        "Health Markets", "Butcher", "Cheese Shops", "Specialty Food"
    ],

    "Hotels & Lodging": [
        "Hotels", "Resorts", "Hostels", "Vacation Rentals",
        "Bed & Breakfast"
    ],

    "Travel & Tourism": [
        "Tours", "Travel Services", "Visitor Centers", "Boat Tours",
        "Airport Lounges"
    ],

    "Car Rental & Transportation": [
        "Car Rental", "Taxis", "Limos", "Transportation",
        "Public Transportation", "Train Stations", "Bus Stations"
    ],

    "Fitness & Gyms": [
        "Gyms", "Yoga", "Pilates", "Trainers", "Martial Arts",
        "Boxing", "Kickboxing"
    ],

    "Active Life": [
        "Active Life", "Hiking", "Golf", "Parks",
        "Skating Rinks", "Climbing", "Bike Rentals",
        "Trampoline Parks"
    ],

    "Beauty & Personal Care": [
        "Hair Salons", "Nail Salons", "Skin Care",
        "Barbers", "Hair Stylists", "Massage",
        "Cosmetics & Beauty Supply"
    ],

    "Medical & Dental Services": [
        "Health & Medical", "Doctors", "Dentists", "Chiropractors",
        "Optometrists", "Urgent Care", "Medical Centers"
    ],

    "Health & Wellness": [
        "Nutritionists", "Acupuncture", "Naturopathic/Holistic",
        "Meditation Centers", "IV Hydration"
    ],

    "Home Services": [
        "Contractors", "Electricians", "Plumbing",
        "Heating & Air Conditioning/HVAC", "Landscaping",
        "Home Cleaning"
    ],

    "Home Improvement & Decor": [
        "Home Decor", "Furniture Stores", "Interior Design",
        "Kitchen & Bath", "Flooring"
    ],

    "Automotive Services": [
        "Auto Repair", "Auto Detailing", "Auto Parts & Supplies",
        "Car Wash", "Smog Check Stations", "Oil Change Stations"
    ],

    "Pets & Animal Care": [
        "Pet Services", "Pet Boarding", "Veterinarians",
        "Pet Groomers"
    ],

    "Retail": [
        "Shopping", "Outlet Stores", "Department Stores",
        "Gift Shops", "Toy Stores", "Bookstores"
    ],

    "Electronics & Tech": [
        "Electronics", "Computers", "Mobile Phones",
        "IT Services & Computer Repair"
    ],

    "Financial Services": [
        "Banks & Credit Unions", "Tax Services", "Insurance",
        "Financial Advising", "Payroll Services"
    ],

    "Professional Services (General)": [
        "Business Consulting", "Marketing", "Advertising",
        "Graphic Design", "Photography"
    ],

    "Legal Services": [
        "Lawyers", "Legal Services", "Bankruptcy Law",
        "Criminal Defense Law", "Real Estate Law"
    ],

    "Government & Public Services": [
        "Post Offices", "Departments of Motor Vehicles",
        "Courthouses", "Police Departments"
    ],

    "Nonprofits & Social Services": [
        "Community Service/Non-Profit", "Donation Center",
        "Food Banks"
    ],

    "Religious & Community": [
        "Churches", "Temples", "Mosques", "Synagogues"
    ],

    "Cannabis & Smoke Shops": [
        "Cannabis Dispensaries", "Head Shops", "Vape Shops"
    ],

    "Musty": [
        "Adult Entertainment", "Strip Clubs", "Casinos"
    ],

    "Other": []
}

#make cateogry to group mapping 
category_to_group = {}
for group_name, categories in group_to_categories.items():
    for category in categories:
        category_to_group[category] = group_name

category_groups = sorted(group_to_categories.keys())
group_feature_keys = ["grp::" + group_name for group_name in category_groups]

# =========================
# xgboost params + other adjustables
# =========================

xgb_params = {
    "objective": "reg:linear",
    "eval_metric": "rmse",
    "eta": 0.06,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "colsample_bylevel": 0.8,
    "min_child_weight": 1,
    "lambda": 1.0,
    "alpha": 0.0,
    "silent": 1,
    "seed": 42,
}

num_rounds = 850

inner_validation_fraction = 0.15
inner_validation_salt = "553_competition_inner_split_seed"

min_rating = 1.0
max_rating = 5.0

# =========================
# sentiment words for tips 
# =========================

#found words via EDA of the tips dataset and common sense HAHAHAH
positive_tip_words = [
    "good", "great", "love", "amazing", "awesome", "excellent",
    "fantastic", "favorite", "perfect", "delicious", "tasty", "crispy",
    "fresh", "friendly", "nice", "clean", "fast", "attentive",
    "happy", "recommend", "best", "enjoyed", "yummy",
    "try", "always", "get", "new", "ask", "free", "must",
    "chicken", "everything", "chocolate", "day",
]

negative_tip_words = [
    "bad", "terrible", "awful", "disgusting", "poor", "worst", "yuck",
    "horrible", "rude", "disappointing", "bland", "cold", "slow",
    "overpriced", "mediocre", "nasty", "gross", "soggy",
    "waste", "not", "minutes", "somewhere", "sucks", "don't", "money",
    "told", "reviews", "unless", "management", "service",
    "waited", "slowest", "attitude", "dont", "shit", "manager",
]

positive_tip_phrases = [
    "must try",
    "great place",
    "happy hour",
    "love place",
    "one best",
    "great food",
    "new favorite",
    "highly recommend",
    "great service",
    "go wrong",
    "best pizza",
    "one favorite",
    #funny how best sushi was correlated with scores
    "best sushi",
    "best place",
    "love love",
    "worth it",
    "can't go",
    "new location",
    "don't forget",
    "food great",
    "love it",
    "always great",
    "still best",
    "great customer",
    "ever had",
    "love everything",
    "las vegas",
    "ice cream",
]

negative_tip_phrases = [
    "worst service",
    "don't waste",
    "not worth",
    "horrible service",
    "waste time",
    "go somewhere",
    "worst customer",
    "bad service",
    "not good",
    "horrible customer",
    "somewhere else",
    "don't go",
    "don't come",
    "minutes get",
    "slow service",
    "stay away",
    "double check",
    "no one",
    "customer service",
    "don't eat",
    "don't bother",
    "15 minutes",
    "go here",
    "never come",
    "unless want",
    "don't get",
    "not come",
]

positive_tip_words_set = set(positive_tip_words)
negative_tip_words_set = set(negative_tip_words)
positive_tip_phrases_set = set(positive_tip_phrases)
negative_tip_phrases_set = set(negative_tip_phrases)

# =========================
# io helpers (rdd-based)
# =========================

#most of this is copied from hw3 

def to_file_path(path):
    # align with style that works on vocareum (same as task2_3)
    path = os.path.abspath(path).replace("\\", "/")
    if not path.startswith("file:/"):
        path = "file:///" + path
    return path


def check_inputs():
    if len(sys.argv) != 4:
        print(
            "Usage: spark-submit competition.py <folder_path> <test_file_name> <output_file_name>",
            file=sys.stderr,
        )
        sys.exit(1)
    folder = os.path.abspath(os.path.normpath(sys.argv[1]))
    return folder, sys.argv[2], sys.argv[3]


def read_train_rows(lines_iterator):
    reader = csv.reader(lines_iterator)
    for row in reader:
        if not row or len(row) < 3:
            continue
        if row[0].strip().lower() == "user_id" and row[1].strip().lower() == "business_id":
            continue
        user_id = row[0].strip()
        business_id = row[1].strip()
        stars = row[2].strip()
        if not user_id or not business_id:
            continue
        try:
            rating = float(stars)
        except Exception:
            continue
        yield (user_id, business_id, rating)


def read_test_rows(lines_iterator):
    reader = csv.reader(lines_iterator)
    for row in reader:
        if not row or len(row) < 2:
            continue
        if row[0].strip().lower() == "user_id" and row[1].strip().lower() == "business_id":
            continue
        user_id = row[0].strip()
        business_id = row[1].strip()
        if not user_id or not business_id:
            continue
        yield (user_id, business_id)


def load_csv_rdd(spark_context, path, expect_label):
    # same pattern as task2_3.py (vocareum-safe)
    rdd = spark_context.textFile(to_file_path(path))
    return rdd.mapPartitions(read_train_rows if expect_label else read_test_rows)


def load_json_rdd(spark_context, path):
    if not os.path.exists(path):
        return None
    return spark_context.textFile(to_file_path(path)).map(lambda s: json.loads(s))


def write_output(output_csv_path, rows):
    with open(output_csv_path, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["user_id", "business_id", "prediction"])
        for user_id, business_id, prediction in rows:
            writer.writerow([user_id, business_id, prediction])


def make_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def log1p_plus(value):
    try:
        value = float(value)
        return log(1.0 + value) if value > 0 else 0.0
    except Exception:
        return 0.0


def clip_rating(value, min_val=min_rating, max_val=max_rating):
    if value < min_val:
        value = min_val
    if value > max_val:
        value = max_val
    return value


def parse_bool(value):
    if value is None:
        return 0.0
    s = str(value).strip().lower()
    return 1.0 if s in {"true", "yes", "y", "1"} else 0.0

#get date ordinal from ymd string for things like decay calculations and other date calcs i tried this time around 
def parse_date_ymd(date_string):
    try:
        year, month, day = [int(x) for x in date_string.split("-")]
        return datetime.date(year, month, day).toordinal()
    except Exception:
        return None

#common decay function, will be used for looking at time recency of tips 
def decay(days, half_life=180.0):
    if days is None or days < 0:
        return 0.0
    lam = math.log(2.0) / half_life
    return math.exp(-lam * days)


def text_stats_and_sentiment(text):
    if not text:
        return 0.0, 0.0, 0.0

    text_string = str(text)
    num_chars = len(text_string)
    text_lower = text_string.lower()

    cleaned = re.sub(r"[^a-z0-9']+", " ", text_lower)
    tokens = [token for token in cleaned.split() if token]
    num_words = float(len(tokens))

    if not tokens:
        return float(num_chars), 0.0, 0.0

    positive_score = 0.0
    negative_score = 0.0

    token_count = len(tokens)
    used_indices = set()

    for i in range(token_count - 1):
        phrase = tokens[i] + " " + tokens[i + 1]
        if phrase in positive_tip_phrases_set:
            positive_score += 2.0
            used_indices.add(i)
            used_indices.add(i + 1)
        elif phrase in negative_tip_phrases_set:
            negative_score += 2.0
            used_indices.add(i)
            used_indices.add(i + 1)

    for index, token in enumerate(tokens):
        if index in used_indices:
            continue
        if token in positive_tip_words_set:
            positive_score += 1.0
        if token in negative_tip_words_set:
            negative_score += 1.0

    sentiment = (positive_score - negative_score) / (positive_score + negative_score + 1.0)
    return float(num_chars), num_words, float(sentiment)

#val file for rmse check 
def load_validation_map(spark_context, validation_path):
    validation_rdd = load_csv_rdd(spark_context, validation_path, expect_label=True)
    return {(user_id, business_id): rating for (user_id, business_id, rating) in validation_rdd.collect()}

#for looking at rmse 
def compute_rmse(predictions, validation_map):
    mse_sum = 0.0
    n = 0
    for user_id, business_id, predicted_rating in predictions:
        actual_rating = validation_map.get((user_id, business_id))
        if actual_rating is None:
            continue
        diff = predicted_rating - actual_rating
        mse_sum += diff * diff
        n += 1
    if n == 0:
        return None
    return sqrt(mse_sum / n)

# =========================
# error distribution
# =========================

#building MAE bins shown in the example
def compute_error_distribution(predictions, validation_map):
    error_bins = {
        ">=0 and <1": 0,
        ">=1 and <2": 0,
        ">=2 and <3": 0,
        ">=3 and <4": 0,
        ">=4": 0,
    }
    total = 0

    #look through each prediction
    for user_id, business_id, predicted_rating in predictions:
        actual_rating = validation_map.get((user_id, business_id))
        if actual_rating is None:
            continue
        error = abs(predicted_rating - actual_rating)
        total += 1

        #check error and put into the bin 
        if error < 1.0:
            error_bins[">=0 and <1"] += 1
        elif error < 2.0:
            error_bins[">=1 and <2"] += 1
        elif error < 3.0:
            error_bins[">=2 and <3"] += 1
        elif error < 4.0:
            error_bins[">=3 and <4"] += 1
        else:
            error_bins[">=4"] += 1

    return error_bins, total

# =========================
# category one-hot
# =========================

#building one hots for the category groups at the top of page
def one_hot_category_groups(categories_field):
    features = {}
    raw_count = 0

    #build category group and make it = 1.0 if present for that business 
    if categories_field:
        for category in [x.strip() for x in categories_field.split(",") if x.strip()]:
            raw_count += 1
            group_name = category_to_group.get(category)
            if group_name:
                key = "grp::" + group_name
                features[key] = 1.0

    #also count number of categories (multi category businesses might be different)
    features["b_num_categories"] = float(raw_count)
    return features

# =========================
# business features
# =========================

#reuse a lot of hw3 to add/build business features
def build_business_features(business_json_rdd):
    if not business_json_rdd:
        return {}

    def map_business(record):
        business_id = record.get("business_id")
        if not business_id:
            return None

        attributes = record.get("attributes") or {}
        categories_field = (record.get("categories") or "")

        row = {}

        #simplified business features 
        row["b_stars"] = make_float(record.get("stars"), 0.0)
        row["b_review_count"] = make_float(record.get("review_count"), 0.0)
        row["b_latitude"] = make_float(record.get("latitude"), 0.0)
        row["b_longitude"] = make_float(record.get("longitude"), 0.0)
        row["b_price_range"] = make_float(attributes.get("RestaurantsPriceRange2"), 0.0)

        row["b_credit_card"] = parse_bool(attributes.get("BusinessAcceptsCreditCards"))
        row["b_appointment_only"] = parse_bool(attributes.get("ByAppointmentOnly"))
        row["b_reservations"] = parse_bool(attributes.get("RestaurantsReservations"))
        row["b_table_service"] = parse_bool(attributes.get("RestaurantsTableService"))
        row["b_wheelchair"] = parse_bool(attributes.get("WheelchairAccessible"))

        group_features = one_hot_category_groups(categories_field)
        row.update(group_features)

        return (business_id, row)

    #features dict for use of final model 
    return dict(
        filter(
            lambda item: item is not None,
            business_json_rdd.map(map_business).collect(),
        )
    )

# =========================
# user features
# =========================

#buildingo ut user features, reuse a lot of hw3 code again
def user_elite_status(elite_status):
    if not elite_status or elite_status == "None":
        return 0.0
    return float(len([elite for elite in elite_status.split(",") if elite.strip()]))

def user_friends_count(friends_string):
    if not friends_string or friends_string == "None":
        return 0.0
    return float(len([friend for friend in friends_string.split(",") if friend.strip()]))

#took a lot of the diff user featuers from all the other jsons and tried to simplify
#user features i felt were more important than business in general
def build_user_features(user_json_rdd):
    if not user_json_rdd:
        return {}

    def map_user(record):
        user_id = record.get("user_id")
        if not user_id:
            return None

        row = {
            "u_review_count": make_float(record.get("review_count"), 0.0),
            "u_friends": user_friends_count(record.get("friends")),
            "u_useful": make_float(record.get("useful"), 0.0),
            "u_funny": make_float(record.get("funny"), 0.0),
            "u_cool": make_float(record.get("cool"), 0.0),
            "u_fans": make_float(record.get("fans"), 0.0),
            "u_elite": user_elite_status(record.get("elite") or "None"),
            "u_average_stars": make_float(record.get("average_stars"), 0.0),
            "u_compliment_hot": make_float(record.get("compliment_hot"), 0.0),
            "u_compliment_profile": make_float(record.get("compliment_profile"), 0.0),
            "u_compliment_list": make_float(record.get("compliment_list"), 0.0),
            "u_compliment_note": make_float(record.get("compliment_note"), 0.0),
            "u_compliment_plain": make_float(record.get("compliment_plain"), 0.0),
            "u_compliment_cool": make_float(record.get("compliment_cool"), 0.0),
            "u_compliment_funny": make_float(record.get("compliment_funny"), 0.0),
            "u_compliment_writer": make_float(record.get("compliment_writer"), 0.0),
            "u_compliment_photos": make_float(record.get("compliment_photos"), 0.0),
        }

        return (user_id, row)

    return dict(
        filter(
            lambda item: item is not None,
            user_json_rdd.map(map_user).collect(),
        )
    )

# =========================
# tip aggregates
# =========================

#tip features and also building sentiment of the tips 
#used tips for sentiment since cant do the review of test file so review_train.json kinda unnecessary? 
def build_tip_aggregates(tip_rdd):
    if not tip_rdd:
        return {}, {}, datetime.date(2019, 1, 1).toordinal()

    dates = tip_rdd.map(lambda record: record.get("date")).filter(lambda d: d is not None).collect()
    ordinals = [parse_date_ymd(d) for d in dates if parse_date_ymd(d) is not None]
    current_ordinal = max(ordinals) if ordinals else datetime.date(2019, 1, 1).toordinal()

    #get business id user id date and actual text of each individual tip 
    def per_tip(record):
        business_id = record.get("business_id")
        user_id = record.get("user_id")
        date_string = record.get("date")
        text = record.get("text") or ""

        ordinal_value = parse_date_ymd(date_string) if date_string else None
        if business_id is None or user_id is None or ordinal_value is None:
            return []

        days = max(0, current_ordinal - ordinal_value)
        decay_180 = decay(days, 180.0)
        _, _, sentiment = text_stats_and_sentiment(text)

        #decay of both sides
        return [
            (
                "B",
                business_id,
                (1.0, sentiment, sentiment * decay_180, 1.0 if days <= 365 else 0.0),
            ),
            (
                "U",
                user_id,
                (1.0, sentiment, sentiment * decay_180, 0.0),
            ),
        ]

    aggregated = (
        tip_rdd
        .flatMap(per_tip)
        .map(lambda t: ((t[0], t[1]), t[2]))
        .reduceByKey(
            lambda a, b: tuple(a[i] + b[i] for i in range(len(a)))
        )
        .collect()
    )

    tip_business_features = {}
    tip_user_features = {}

    for (kind, key), values in aggregated:
        count, sentiment_sum, sentiment_decayed_sum, recent_365_sum = values
        count = float(count)
        average_sentiment = sentiment_sum / count if count > 0 else 0.0

        #store business or user tip features accordingly based on kind of tip 
        if kind == "B":
            tip_business_features[key] = {
                "b_tip_count": count,
                "b_tip_avg_sentiment": float(average_sentiment),
                "b_tip_sentiment_decay": float(sentiment_decayed_sum),
                "b_tip_recent365": float(recent_365_sum),
            }
        else:
            tip_user_features[key] = {
                "u_tip_count": count,
                "u_tip_avg_sentiment": float(average_sentiment),
                "u_tip_sentiment_decay": float(sentiment_decayed_sum),
            }

    return tip_business_features, tip_user_features, current_ordinal

# =========================
# feature builder
# =========================

#final feature builder for the xgb model 
def make_feature_builder(business_features, user_features, tip_business_features, tip_user_features):

    #a loto f this was just reuse of hw3 as well 
    def build_feature_row(user_id, business_id):
        business_row = business_features.get(business_id, {})
        user_row = user_features.get(user_id, {})
        tip_business_row = tip_business_features.get(business_id, {})
        tip_user_row = tip_user_features.get(user_id, {})

        features = {}

        features["b_stars"] = business_row.get("b_stars", 0.0)
        features["b_review_count"] = business_row.get("b_review_count", 0.0)
        features["b_latitude"] = business_row.get("b_latitude", 0.0)
        features["b_longitude"] = business_row.get("b_longitude", 0.0)
        features["b_price_range"] = business_row.get("b_price_range", 0.0)
        features["b_credit_card"] = business_row.get("b_credit_card", 0.0)
        features["b_appointment_only"] = business_row.get("b_appointment_only", 0.0)
        features["b_reservations"] = business_row.get("b_reservations", 0.0)
        features["b_table_service"] = business_row.get("b_table_service", 0.0)
        features["b_wheelchair"] = business_row.get("b_wheelchair", 0.0)

        features["u_review_count"] = user_row.get("u_review_count", 0.0)
        features["u_friends"] = user_row.get("u_friends", 0.0)
        features["u_useful"] = user_row.get("u_useful", 0.0)
        features["u_funny"] = user_row.get("u_funny", 0.0)
        features["u_cool"] = user_row.get("u_cool", 0.0)
        features["u_fans"] = user_row.get("u_fans", 0.0)
        features["u_elite"] = user_row.get("u_elite", 0.0)
        features["u_average_stars"] = user_row.get("u_average_stars", 0.0)

        features["u_compliment_hot"] = user_row.get("u_compliment_hot", 0.0)
        features["u_compliment_profile"] = user_row.get("u_compliment_profile", 0.0)
        features["u_compliment_list"] = user_row.get("u_compliment_list", 0.0)
        features["u_compliment_note"] = user_row.get("u_compliment_note", 0.0)
        features["u_compliment_plain"] = user_row.get("u_compliment_plain", 0.0)
        features["u_compliment_cool"] = user_row.get("u_compliment_cool", 0.0)
        features["u_compliment_funny"] = user_row.get("u_compliment_funny", 0.0)
        features["u_compliment_writer"] = user_row.get("u_compliment_writer", 0.0)
        features["u_compliment_photos"] = user_row.get("u_compliment_photos", 0.0)

        features["b_tip_count"] = tip_business_row.get("b_tip_count", 0.0)
        features["b_tip_avg_sentiment"] = tip_business_row.get("b_tip_avg_sentiment", 0.0)
        features["b_tip_sentiment_decay"] = tip_business_row.get("b_tip_sentiment_decay", 0.0)
        features["b_tip_recent365"] = tip_business_row.get("b_tip_recent365", 0.0)

        features["u_tip_count"] = tip_user_row.get("u_tip_count", 0.0)
        features["u_tip_avg_sentiment"] = tip_user_row.get("u_tip_avg_sentiment", 0.0)
        features["u_tip_sentiment_decay"] = tip_user_row.get("u_tip_sentiment_decay", 0.0)

        features["b_num_categories"] = business_row.get("b_num_categories", 0.0)
        for group_key in group_feature_keys:
            features[group_key] = business_row.get(group_key, 0.0)

        return features

    return build_feature_row

#ensuring all features are consistent and same order for xgb esp with the one hot encoding 
def to_dense_vector(feature_dict, feature_order):
    return [float(feature_dict.get(key, 0.0)) for key in feature_order]

# =========================
# xgboost stuff
# =========================

#copied xgboost training and prediction code from hw3 again
def prepare_training_matrix(train_rows, build_feature_row, feature_order):
    import xgboost as xgb  # local import like task2_3
    x_train = []
    y_train = []
    for user_id, business_id, rating in train_rows:
        features = build_feature_row(user_id, business_id)
        x_train.append(to_dense_vector(features, feature_order))
        y_train.append(rating)
    return xgb.DMatrix(x_train, label=y_train)


def train_booster(training_matrix):
    import xgboost as xgb  # local import
    booster = xgb.train(xgb_params, training_matrix, num_boost_round=num_rounds)
    return booster


def prepare_pairs_matrix(pairs, build_feature_row, feature_order):
    import xgboost as xgb  # local import
    x_matrix = []
    user_ids = []
    business_ids = []
    for user_id, business_id in pairs:
        features = build_feature_row(user_id, business_id)
        x_matrix.append(to_dense_vector(features, feature_order))
        user_ids.append(user_id)
        business_ids.append(business_id)
    return xgb.DMatrix(x_matrix), user_ids, business_ids


def predict_pairs(booster, dmatrix, user_ids, business_ids):
    predictions = booster.predict(dmatrix)
    output = []
    for i in range(len(user_ids)):
        predicted_rating = clip_rating(float(predictions[i]))
        output.append((user_ids[i], business_ids[i], predicted_rating))
    return output

#test feature importance and gain printing (trouble shooting to test best things )
def print_full_feature_importance(booster, feature_order, top_n=None):
    raw_importance = booster.get_score(importance_type="gain")
    mapped_importance = []

    #go through each feature and get gain to help do eda on best features to use
    for feature_index, gain in raw_importance.items():
        index = int(feature_index[1:])
        feature_name = feature_order[index] if index < len(feature_order) else f"??({index})"
        mapped_importance.append((feature_name, gain))
    mapped_importance.sort(key=lambda x: -x[1])
    limit = len(mapped_importance) if top_n is None else min(top_n, len(mapped_importance))
    print("\n=== FULL FEATURE IMPORTANCE (by gain) ===")
    for i in range(limit):
        feature_name, gain = mapped_importance[i]
        print(f"{feature_name:40s}  {gain:.6f}")
    print("=== END FEATURE IMPORTANCE ===\n")


# =========================
# main
# =========================

#main sawce to put it all together 
def main():

    #mostly repeat of hw3 again but with the new features and tip sentiment added in
    start_time = time.time()

    folder_path, test_file_name, output_file_name = check_inputs()

    train_csv_path = os.path.join(folder_path, "yelp_train.csv")
    user_json_path = os.path.join(folder_path, "user.json")
    business_json_path = os.path.join(folder_path, "business.json")
    tip_json_path = os.path.join(folder_path, "tip.json")
    validation_path = os.path.join(folder_path, "yelp_val.csv")

    spark_conf = SparkConf().setAppName("competition")
    spark_context = SparkContext(conf=spark_conf)
    spark_context.setLogLevel("ERROR")

    train_rdd = load_csv_rdd(spark_context, train_csv_path, expect_label=True).persist()
    train_rows = train_rdd.collect()

    business_rdd = load_json_rdd(spark_context, business_json_path)
    user_rdd = load_json_rdd(spark_context, user_json_path)
    tip_rdd = load_json_rdd(spark_context, tip_json_path)

    business_features = build_business_features(business_rdd)
    user_features = build_user_features(user_rdd)
    tip_business_features, tip_user_features, current_ordinal = build_tip_aggregates(tip_rdd)

    build_feature_row = make_feature_builder(
        business_features, user_features, tip_business_features, tip_user_features
    )

    core_feature_keys = [
        "b_stars",
        "b_review_count",
        "b_latitude",
        "b_longitude",
        "b_price_range",
        "b_credit_card",
        "b_appointment_only",
        "b_reservations",
        "b_table_service",
        "b_wheelchair",
        "u_review_count",
        "u_friends",
        "u_useful",
        "u_funny",
        "u_cool",
        "u_fans",
        "u_elite",
        "u_average_stars",
        "u_compliment_hot",
        "u_compliment_profile",
        "u_compliment_list",
        "u_compliment_note",
        "u_compliment_plain",
        "u_compliment_cool",
        "u_compliment_funny",
        "u_compliment_writer",
        "u_compliment_photos",
        "b_tip_count",
        "b_tip_avg_sentiment",
        "b_tip_sentiment_decay",
        "b_tip_recent365",
        "u_tip_count",
        "u_tip_avg_sentiment",
        "u_tip_sentiment_decay",
        "b_num_categories",
    ]

    feature_order = core_feature_keys + group_feature_keys

    #build test pairs
    test_pairs_rdd = load_csv_rdd(
        spark_context,
        os.path.join(folder_path, test_file_name),
        expect_label=False,
    )
    test_pairs = test_pairs_rdd.collect()

    #main xgboost path
    training_matrix = prepare_training_matrix(train_rows, build_feature_row, feature_order)
    booster = train_booster(training_matrix)

    test_dmatrix, test_user_ids, test_business_ids = prepare_pairs_matrix(
        test_pairs, build_feature_row, feature_order
    )
    test_predictions = predict_pairs(booster, test_dmatrix, test_user_ids, test_business_ids)

    write_output(output_file_name, test_predictions)

    # validation / error distribution only for local debugging (commented out for competition runs)
    # validation_map = load_validation_map(spark_context, validation_path)
    # rmse = compute_rmse(test_predictions, validation_map)
    # if rmse is None:
    #     print("Validation RMSE: N/A (no overlapping pairs)")
    # else:
    #     print(f"Validation RMSE: {rmse:.6f}")
    # error_bins, total = compute_error_distribution(test_predictions, validation_map)
    # if total > 0:
    #     print("\n=== Error Distribution (MAE) ===")
    #     for label in [">=0 and <1", ">=1 and <2", ">=2 and <3", ">=3 and <4", ">=4"]:
    #         count = error_bins[label]
    #         fraction = count / float(total)
    #         print(f"{label:12s}: {count:8d}  ({fraction:6.2%})")
    #     print("=============================================================\n")
    # else:
    #     print("No overlapping pairs for error distribution.")

    elapsed = time.time() - start_time
    print(f"Total elapsed time: {elapsed:.2f} seconds")

    spark_context.stop()


if __name__ == "__main__":
    main()
