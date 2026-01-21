#task 3
#includes: A. average stars of each city, B. 2 methods of top 10 cities with highest avg stars, sort using python (m1) and sort suing spark (m2)

import sys, json, time
from pyspark import SparkConf, SparkContext 

def avg_stars_city(sc, review_filepath, business_filepath):
    #get stars from the reviews file and business id to join on
    reviews_stars_rdd = (sc.textFile(review_filepath)
                     .map(lambda s: json.loads(s))
                     .map(lambda r: (r.get("business_id"), float(r.get("stars", 0.0)))))
    
    #get business id and city names, dont remove missing id
    business_locations_rdd = (sc.textFile(business_filepath)
                          .map(lambda s: json.loads(s))
                          .map(lambda b: (b.get("business_id"), b.get("city", ""))))
    
    #join for final rdd, join on matching "business_id"
    business_reviews_locations_rdd = ((reviews_stars_rdd.join(business_locations_rdd))
                                      .map(lambda kv: (kv[1][1] if kv[1][1] is not None else "", (kv[1][0],1))))
    
    #calc avg
    avg_stars_city_rdd = (business_reviews_locations_rdd
                          .reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1]))
                          .mapValues(lambda s: s[0]/s[1]))

    #output avg
    return avg_stars_city_rdd

#python sort
def top_10_m1(sc, review_file_path, business_file_path):
    #timer start
    t0 = time.perf_counter()
    #get list
    avg_stars_city_list = avg_stars_city(sc, review_file_path, business_file_path).collect()
    #sort list
    top_10 = sorted(avg_stars_city_list, key=lambda kv: (-kv[1], kv[0]))[:10]
    #time taken
    elapsed_time = time.perf_counter()-t0

    return top_10, elapsed_time

#spark sort
def top_10_m2(sc, review_file_path, business_file_path):
    #timer start
    t0 = time.perf_counter()
    #get rdd
    avgs_rdd = avg_stars_city(sc, review_file_path, business_file_path)
    #sort rdd
    top_10 = avgs_rdd.takeOrdered(10, key=lambda kv: (-kv[1],kv[0]))
    #time taken
    elapsed_time = time.perf_counter()-t0

    return top_10, elapsed_time


def main(): 
    #checking for 5 inputs, task.py, review filepath, output filepath, based on task1
    if len(sys.argv)!=5: 
        print("Usage: task3.py <review_filepath> <business_filepath> <output_filepath_question_a> <output_filepath_question_b>", file=sys.stderr) 
        sys.exit(1) 

    #paths for script inputs
    review_filepath = sys.argv[1]
    business_filepath = sys.argv[2]
    output_a = sys.argv[3]
    output_b = sys.argv[4]

    #build spark config
    spark_config = SparkConf().setAppName("DSCI553_HW1_Task3")
    sc = SparkContext(conf=spark_config) 
    sc.setLogLevel("ERROR") 

    #error catching spark
    try:
        #part A rdd and sort
        avgs_rdd = avg_stars_city(sc, review_filepath, business_filepath)
        avgs_sorted = (avgs_rdd
                       .sortBy(lambda kv: (-kv[1], kv[0]))
                       .collect())
        
        #write part a as text file, format from hw pdf
        with open(output_a, "w", encoding="utf-8") as f:
            f.write("city,stars\n")
            for city, stars in avgs_sorted:
                f.write(f"{city},{stars}\n")
        
        #part B 
        #python sort
        _, m1 = top_10_m1(sc, review_filepath, business_filepath) 
        #spark sort
        _, m2 = top_10_m2(sc, review_filepath, business_filepath)  

        #results from m1/m2 and hard coded text explanation
        results_b = {
            "m1": m1,
            "m2": m2,
            "reason": (
                "M1 works on Python and entirely in driver, limiting it to whether it can fit. " 
                "It works fine for smaller item counts. "
                "M2 works on Spark and thus distributed. "
                "This gives it an edge in speed when we have many items to sort. "
                )
        }
        with open(output_b, "w", encoding="utf-8") as f:
            json.dump(results_b, f)

    finally:
        sc.stop()

if __name__ == "__main__":
    main()