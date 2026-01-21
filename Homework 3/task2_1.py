#task 2_1
#submission: spark-submit task2_1.py <train_file_name> <test_file_name> <output_file_name>
#using the yelp_train.csv to predict the star ratings for given user ids and business ids
#case 1: Item-based Collaborative Filtering (CF) recommendation system with Pearson similarity (2 points)
#use yelp_train.csv input
#RMSE as error
#has cold start issue, find a way to deal with it 
#output format is: user_id, business_id, prediction

import sys, csv, heapq, math
from pyspark import SparkConf, SparkContext 

#building some baseline parameters to use throughout script
k_neighbors = 60
minimum_co_raters = 5
minimum_similarity = 0.0
min_rating, max_rating = 1.0, 5.0
similarity_power = 1.25
dead = 0.005

#taken from past assignmnt, modified to fit this one
def check_inputs():
    #check for 3 inputs (py, train, test, output)
    if len(sys.argv) != 4:
        print("Usage: spark-submit task2_1.py <train_file_name> <test_file_name> <output_file_name>", file=sys.stderr)
        sys.exit(1)

    #dont need to check other bits for anything
    return sys.argv[1], sys.argv[2], sys.argv[3]


#csv reading/parsing for the training data
def read_train_rows(rows):
    #read rows in the csv, can map later to do many
    for row in csv.reader(rows):

        #check size of row first and valid row
        if not row or len(row) < 3:
            continue        
        user_id = row[0].strip()
        business_id = row[1].strip()
        stars = row[2].strip()
        if user_id == "user_id" or business_id == "business_id" or stars == "stars":
            continue
        try:
            r = float(stars)
        except:
            continue

        #get these values from the csv
        yield (user_id, business_id, r)

#will not read stars (this is where we will be filling/predicting)
def read_test_rows(rows):
    #same as above, adjust for 2 instead of 3
    for row in csv.reader(rows):
        if not row or len(row) < 2:
            continue
        user_id = row[0].strip()
        business_id = row[1].strip()
        if user_id == "user_id" or business_id == "business_id":
            continue
        if not user_id or not business_id:
            continue

        #copy above, jus t without the stars now
        yield (user_id, business_id, None)


#use these for reading to rdd from initial csv (also use prior funcs)
def read_train_csv_to_rdd(sc, train_path):
    rdd = sc.textFile(train_path)
    body = rdd.mapPartitions(read_train_rows)
    return body.cache()

def read_test_csv_to_rdd(sc, test_path):
    rdd = sc.textFile(test_path)
    body = rdd.mapPartitions(read_test_rows)
    return body

#added in to dela with more minor noise in the data to reduce rmse more
def weight_signed(s, power):
    #look at similarity (aside from the sign)
    absolute_sim = abs(s)

    #dead assigned at the top
    #if the sim is small enough, turn it to 0 (essentially remove)
    if absolute_sim <= dead:
        return 0.0
    
    #further reduces those close to the dead similarity levels
    absolute_sim -= dead

    #put the sign back on for similarity, and the power (to punish or help)
    return (absolute_sim ** power) * (1.0 if s >= 0 else -1.0)

#for imputing missing values
#strategy here: simple mean found online "plain ladder mean", better than global average by a little, simpler than other ML optoins
def get_means(train_rdd):

    #building ratings by user
    user_ratings = (train_rdd
        #should get user, (rating, 1) so we can get rating and count
        .map(lambda t: (t[0], (t[2], 1)))
        #aggregate by key so we can sum both the ratings and the counts (to avg later)
        .reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1])))
    
    #building ratings by business
    business_ratings  = (train_rdd
        #applying same logic as above for business averaging later too
        .map(lambda t: (t[1], (t[2], 1)))
        .reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1])))

    #getting total, aggregate on tuple t[2] = the rating
    #aggregate syntax = (zero, seqop, comboop)
    #
    total_sum, total_count = train_rdd.map(lambda t: t[2]).aggregate(
        #starter (0 values)
        (0.0,0),
        #accumulated sum + x, while other side is counter +1 each time, counts in each partition
        lambda acc,x: (acc[0]+x, acc[1]+1),
        #aaggregate of sum and countsw
        lambda s,c: (s[0]+c[0], s[1]+c[1])
    )


    #calculating average if possible-- (if theres any ratings at all), if no ratings, set at 3.0 avg (avg of min+max so midpoint)
    avg = total_sum / total_count if total_count else 3.0

    #means dampeners (technique found online after failing rmse multiple times lmfao pleasekill me)
    #basically, # of "pretend" ratings that the user or business has prior to adding anymore
    #dampens extremes 
    #we are more skeptical of businesses than users in this case-- businesses have fewer ratings than users per entity 
    user_beta = 5.0
    business_beta = 12.0

    #initiate to capture and store sum, count, and averages 
    user_sum, user_count, user_avg = {}, {}, {}
    business_sum, business_count, business_avg = {}, {}, {}

    #user/business ratings should be in format of ID, (sum, count)
    for uid, (s, c) in user_ratings.collect():
        #putting into the {} made for each uid, avg is sum/count
        user_sum[uid] = s; user_count[uid] = c; user_avg[uid] = (s+user_beta*avg) / (c+user_beta)
    
    for bid, (s, c) in business_ratings.collect():
        business_sum[bid] = s; business_count[bid] = c; business_avg[bid] = (s+business_beta*avg) / (c+business_beta)

    #get global average, users, and businesses totals
    #global avg for absentee values in all
    #user and business values for when those are the more rated for that specific case
    return {
        "global_mean": avg,
        "user_sum": user_sum, "user_count": user_count, "user_avg": user_avg,
        "business_sum": business_sum,   "business_count": business_count,   "business_avg": business_avg
    }

#use the above calculated mean and impute it as needed into the dataset 
def impute_means(uid, bid, means):
    #if both uid and bid and present, we can do both means - global mean
    #if only uid present, use user mean
    #if only bid present, use business mea
    #if no means, just use the base mean
    global_mean = means["global_mean"]
    user_mean = means["user_avg"].get(uid)
    business_mean = means["business_avg"].get(bid)

    if user_mean and business_mean:
        prediction = user_mean + business_mean - global_mean

        #in case prediction value differs 
        if prediction < 1.0:
            prediction = 1.0
        elif prediction > 5.0:
            prediction = 5.0
        return prediction
    
    #means for if we don't have both user and business means 
    elif user_mean:
        return user_mean
    elif business_mean:
        return business_mean
    
    #global mean if nothing 
    else: 
        return global_mean


#building the indices we need  
def build_user_index(train_rdd):
    return train_rdd.map(lambda t: t[0]).distinct().zipWithIndex().collectAsMap()

def build_business_index(train_rdd):
    return train_rdd.map(lambda t: t[1]).distinct().zipWithIndex().collectAsMap()

#feeds into pearson
def make_pearson_indices(train_rdd, user_index):

    #get user str, (business str, rating)
    user_items = (train_rdd
                  .map(lambda t: (t[0], (t[1], t[2]))) 
                  .groupByKey()
                  .mapValues(list)
                  .collectAsMap())

    #business to user
    #form: (business id, (user index, rating))
    def to_userindex(t):
        user_str, business_str, r = t
        u_index = user_index.get(user_str)
        # filtering for none just in case 
        return (business_str, (u_index, r)) if u_index is not None else None

    #build into businesses, and then (users + the rating)
    #should be used later to compare users (bi)
    business_users = (train_rdd
                 .map(to_userindex)
                 .filter(lambda x: x is not None)
                 .groupByKey()
                 .mapValues(list)
                 .collectAsMap())

    return user_items, business_users

#pearson correlation calculator
#bi/bj = business id / business (neighbor rated by same users), comparing the two 
def pearson_correlation(bi, bj, business_users, business_avg, global_mean, min_co=minimum_co_raters, sim_floor=minimum_similarity):
    
    #get the two businesses to compare 
    business1 = business_users.get(bi)
    business2 = business_users.get(bj)

    #if no rates, no similarity
    if not business1 or not business2:
        return 0.0, 0
    
    #found online for reference: 
    # Pearson correlation coefficient formula:
    # r = Σ[(x_i - mean_x) * (y_i - mean_y)] 
    #     / sqrt( Σ(x_i - mean_x)^2 * Σ(y_i - mean_y)^2 )
    #
    # Where:
    # - r = Pearson correlation coefficient
    # - x_i, y_i = individual sample points
    # - mean_x = average of all x values
    # - mean_y = average of all y values
    # - Σ = sum over all data points
    #match on businesses based on list size (flip for faster iteration and comparison)
    if len(business1) <= len(business2):
        d = {user_business_index: r for (user_business_index, r) in business1}

        #building the numerators and denonminators
        other = business2
        business1_mean = business_avg.get(bi, global_mean)
        business2_mean = business_avg.get(bj, global_mean)
        #initializing 0's to store values in 
        num = den_i = den_j = 0.0
        co = 0

        #compare each ri and rj (rating) rating 1/2 in hindsight maybe better naming too 
        for user_business_index, rj in other:
            ri = d.get(user_business_index)
            if ri is None:
                continue

            #building denominators for the formula 
            co += 1
            di = ri - business1_mean
            dj = rj - business2_mean
            num  += di * dj
            den_i += di * di
            den_j += dj * dj
    else:

        #copy of above but reversed for the other way of iterate j vs i 
        d = {user_business_index: r for (user_business_index, r) in business2}
        other = business1
        business1_mean = business_avg.get(bi, global_mean)
        business2_mean = business_avg.get(bj, global_mean)
        num = den_i = den_j = 0.0
        co = 0
        for user_business_index, ri in other:
            rj = d.get(user_business_index)
            if rj is None:
                continue
            co += 1
            di = ri - business1_mean
            dj = rj - business2_mean

            #numerator/denominators of pearson (can be reused )
            num  += di * dj
            den_i += di * di
            den_j += dj * dj

    #edge cases
    if co < min_co or num == 0.0 or den_i == 0.0 or den_j == 0.0:
        return 0.0, co

    #numerator divided by sqrt denominator, pearson formula 
    sim = num / math.sqrt(den_i * den_j)

    #0.0 if cases of similarity is lower than the floor, we can neglect it
    #case i found online as a way to reduce some noise in the data 
    if abs(sim) < sim_floor:
        return 0.0, co
    return sim, co

#predict values (reuse to predit all the values in the train/val sets )
def predict_one(uid, bid, context):
    user_items = context["user_items"].value
    business_users = context["business_users"].value
    business_avg = context["business_avg"].value
    means = context["means"]

    #for "cold start" case when not enough related ratings, impute (impute should also have laddering from the function before to deal with different mean cases)
    if uid not in user_items or bid not in business_users:
        return (uid, bid, impute_means(uid, bid, means))

    #hold neighbors (similar ratings)
    neighbors = []

    #iterate through all user index for businesses and ratings and check if matching 
    for bj, r_uj in user_items[uid]:
        if bj == bid:
            continue

        #pearson calculation, and getting the similarity/co
        s, co = pearson_correlation(
            bid, bj, business_users, business_avg, means["global_mean"],
            min_co=minimum_co_raters, sim_floor=minimum_similarity
            )

        #only add if similarity is valid/exists
        if s != 0.0:
            neighbors.append((abs(s), s, bj, r_uj))

    #for imputing means if no neighbors 
    if not neighbors:
        return (uid, bid, impute_means(uid, bid, means))

    #using heap for choosing mosst similar neighbor 
    top = heapq.nlargest(k_neighbors, neighbors, key=lambda x: x[0])
    
    #baseline setting 
    mean = means["global_mean"]
    u_center = means["user_avg"].get(uid, mean) - mean

    #store for calculations numerator/denominators
    num = 0.0
    den = 0.0

    #accumulation loop, top is the heap above
    for _, s, bj, r_uj in top:
        #j_center is the buiness mean of bj, minus global mean
        j_center = business_avg.get(bj, mean) - mean
        #user bias calculation
        base_uj = mean + u_center + j_center

        #function for punishing weaker noises, this was created prior
        similarity_eff = weight_signed(s, similarity_power)
        #how much neighbor deviates rom baseline
        res = r_uj - base_uj
        #accumulate the residual and power for residual
        num += similarity_eff * res
        #accumulate denominator 
        den += abs(similarity_eff)

    base_ui = mean + u_center + (business_avg.get(bid, mean) - mean)
    pred = impute_means(uid, bid, means) if den < 1e-9 else base_ui + (num / den)

    #checking if value is within the min/max
    if   pred < min_rating: pred = min_rating
    elif pred > max_rating: pred = max_rating
    return (uid, bid, pred)


#write to csv
def write_output(path, predictions):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "business_id", "prediction"])

        #iterate through the columns
        for user, business, prediction in predictions:
            #write/update 
            w.writerow([user, business, prediction])

def main():
    #validate inputs
    train_path, test_path, out_path = check_inputs()

    conf = SparkConf().setAppName("task2_1")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    #format of (user, biz, rating)
    train_rdd = read_train_csv_to_rdd(sc, train_path) 
    #format of (user, biz, None)
    test_rdd  = read_test_csv_to_rdd(sc, test_path)   

    #getting means for the train set
    means = get_means(train_rdd)

    # building indices out of the train set
    user_idx_map = build_user_index(train_rdd)
    user_items_map, business_users_map = make_pearson_indices(train_rdd, user_idx_map)

    #broadcast spark 
    user_items_b = sc.broadcast(user_items_map)
    business_users_b = sc.broadcast(business_users_map)
    business_avg_b = sc.broadcast(means["business_avg"])

    #pass into predictor (b) so we can use to predict/impute 
    context = {
        "user_items": user_items_b,
        "business_users": business_users_b,
        "business_avg": business_avg_b,
        "means": means
    }

    #build prediction rdd using our predictor and context built
    predicted_rdd = test_rdd.mapPartitions(
        lambda part: (predict_one(u, b, context) for (u, b, _) in part)
    )

    #write the output
    predictions = predicted_rdd.collect()
    write_output(out_path, predictions)

    sc.stop()

if __name__ == "__main__":
    main()
