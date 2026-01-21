#task 1 
#includes: A. total reviews, B. reviews in 2018, C. distcint user who wrote reviews, D. top 10 users who wrote reviews and number of reviews wrote, E. distinct businesses reviewed 
#F. top 10 businesses with largest number of reviews and review count 

import sys, json
from pyspark import SparkConf, SparkContext 

def main(): 
    #checking for 3 inputs, task.py, review filepath, output filepath 
    if len(sys.argv) != 3: 
        print("Input syntax: task1.py <review_filepath> <output_filepath>", file=sys.stderr) 
        sys.exit(1) 
        
    #get paths based on input to run the script 
    review_filepath = sys.argv[1] 
    output_filepath = sys.argv[2] 
        
    #build spark config 
    spark_config = SparkConf().setAppName("DSCI553_HW1_Task1")
    sc = SparkContext(conf=spark_config) 
    sc.setLogLevel("ERROR") 
    
    #to catch errors and stop spark
    try: 
        #review file (input) to read 
        lines = sc.textFile(review_filepath) 

        #getting only the necessary fields based on task1 and cache to reduce time 
        records = (lines 
                    .map(lambda s: json.loads(s)) 
                    .map(lambda r: (r.get("user_id"), r.get("business_id"), r.get("date",""))) 
                    .cache()) 
            
        #for reuse cache records.count() 
        # #A total number of reviews 
        n_review = records.count() 
        #B total number of reviews in 2018 
        n_review_2018 = records.filter(lambda m: str(m[2]).startswith("2018-")).count() 
        #C total number of users 
        n_user = records.map(lambda m: m[0]).distinct().count() 
        #D top 10 users and counts #order by review count, then user id lexicographically 
        top10_user = records.map(lambda j: (j[0], 1)).reduceByKey(lambda a,b: a+b).takeOrdered(10, key=lambda kv: (-kv[1], kv[0])) 
        #E total number of businesses 
        n_business = records.map(lambda m: m[1]).distinct().count() 
        #F top 10 businesses and counts 
        top10_business = records.map(lambda j: (j[1], 1)).reduceByKey(lambda a,b: a+b).takeOrdered(10, key=lambda kv: (-kv[1], kv[0])) 
            
        #store results from above 
        result = { "n_review": n_review, 
                    "n_review_2018": n_review_2018, 
                    "n_user": n_user, "top10_user": top10_user, 
                    "n_business": n_business, 
                    "top10_business": top10_business 
                    } 
            
        #write results into output file 
        with open(output_filepath, "w", encoding="utf-8") as f: json.dump(result, f) 
            
        #free cache and spark 
        records.unpersist() 
    
    #stop spark
    finally:
        sc.stop() 
        
if __name__ == "__main__":
    main()