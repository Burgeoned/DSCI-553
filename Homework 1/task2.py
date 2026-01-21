#task 2 
#includes: A.Task 1 Question F and the number of items per partition
#customized partition and show speed

import sys, json, zlib, time
from pyspark import SparkConf, SparkContext 

#for custom partitions each key, hashing them
def custom_partitioner(n):
    #crc32 over md5/sha1 for speed and reproducibility 
    #calls for each partition provided (n) to hash, encoded for bytes
    return lambda k: 0 if k is None else zlib.crc32(str(k).encode("utf-8"))%n

#get partition item count
def items_per_partition(rdd):
    #run on each partition, counts 1 for each item, iterator consumed after use, collect to get a count
    return rdd.mapPartitions(lambda iterator: [sum(1 for record in iterator)]).collect()

#calculate time and part f
def part_f_with_timer(pair_rdd):
    #time start
    t0 = time.perf_counter()
    #sums up review counts by business id (key)
    counts = pair_rdd.reduceByKey(lambda a,b: a+b)
    #top10 of counts above, ordered by descending count, then lexicographically of id
    top10  = counts.takeOrdered(10, key=lambda kv: (-kv[1], kv[0]))
    #return the top10 of f, then time end minus time start (total time taken)
    return top10, (time.perf_counter()-t0)


def main():
    #checking for 4 inputs, task.py, review filepath, output filepath, based on task1
    if len(sys.argv) != 4: 
        print("Usage: task2.py <review_filepath> <output_filepath> <n_partition>", file=sys.stderr) 
        sys.exit(1) 
        
    #get paths based on input to run the script 
    review_filepath = sys.argv[1] 
    output_filepath = sys.argv[2] 

    #in case of non positive partitions
    try:
        partition_count = int(sys.argv[3])
        if partition_count < 1:
            raise ValueError
    except ValueError:
        print("n_partition must be a positive integer", file=sys.stderr)
        sys.exit(1)
        
    #build spark config 
    spark_config = SparkConf().setAppName("DSCI553_HW1_Task2")
    sc = SparkContext(conf=spark_config) 
    sc.setLogLevel("ERROR") 

    #try to catch errors and stop spark
    try: 
        #review file (input) to read 
        lines = sc.textFile(review_filepath)

        #getting just business id and 1 counter (part f of task 1)
        business_id_pairs = (lines
                            .map(lambda s: json.loads(s))
                            .map(lambda r: (r.get("business_id"), 1))
                            #for missing id 
                            .filter(lambda pair: pair[0] is not None)
                            .cache())
        
        #default run for part f and time
        n_partition_default  = business_id_pairs.getNumPartitions()
        n_items_default = items_per_partition(business_id_pairs)
        _, exe_time_default = part_f_with_timer(business_id_pairs)

        #custom run for part f and time
        custom_run = custom_partitioner(partition_count)
        business_id_pairs_custom = business_id_pairs.partitionBy(partition_count, custom_run).cache()
        n_partition_custom    = business_id_pairs_custom.getNumPartitions()
        n_items_custom   = items_per_partition(business_id_pairs_custom)
        _, exe_time_custom  = part_f_with_timer(business_id_pairs_custom)


        result = {
            "default": {
                "n_partition": n_partition_default,
                "n_items": n_items_default,
                "exe_time": exe_time_default
            },
            "customized": {
                "n_partition": n_partition_custom,
                "n_items": n_items_custom,
                "exe_time": exe_time_custom
            }
        }

        #write results into output file 
        with open(output_filepath, "w", encoding="utf-8") as f: json.dump(result, f) 
            
        #free cache and spark 
        business_id_pairs.unpersist()
        business_id_pairs_custom.unpersist()

    #stop spark
    finally:
        sc.stop() 

if __name__ == "__main__":
    main()