#task 2
#tafeng dataset, but only case1 (no need to flip kv)
#data format: date, customer_id, age group, pin code, product subclass, product id, amount, asset, sales price
#only need date, customer_id, product id (preprocessing step)
#idea: combine date and customer_id for unique transaction basket id
#date_customers that bought more than k items (threshold)
#execution: Python: spark-submit task2.py <filter threshold> <support> <input_file_path> <output_file_path>
#looks like mostly similar to task1, but need to preprocess data 

#csv data formatting: TRANSACTION_DT	CUSTOMER_ID	AGE_GROUP	PIN_CODE	PRODUCT_SUBCLASS	PRODUCT_ID	AMOUNT	ASSET	SALES_PRICE

import sys, csv, math, time
from pyspark import SparkConf, SparkContext 
from itertools import combinations
from functools import partial

#taken mostly from task1.py and modified for task2
def check_inputs():
    #check for 5 inputs (py, case number, support, input, output)
    if len(sys.argv) != 5:
        print("Usage: spark-submit task2.py <filter threshold> <support> <input_file> <output_file>", file=sys.stderr)
        sys.exit(1)
    try:
        #coerce to int if need
        filter_threshold = int(sys.argv[1]); support = int(sys.argv[2])

    #error messagefor invalid input (not int)
    except ValueError:
        print("Error: <filter threshold> and <support> must be integers.", file=sys.stderr)
        sys.exit(1)

    #validate filter threshold and support number 
    if filter_threshold < 0 or support < 1:
        print("Error: filter threshold must be >= 0, and support must be >= 1.", file=sys.stderr)
        sys.exit(1)

    return filter_threshold, support, sys.argv[3], sys.argv[4] 


#helper functions for basket, taken from task1.py
#gets key from k
def get_combiner(k):
    return {k}

#adds value of b to set a for building set of of values like {} for value used later in combinebykey
def add_values(a, b):
    a.add(b)
    return a

#union of the two sets (for reducing)
def union_key(set1, set2):
    return set1.union(set2)

#sorts set and return it as tuple (order lexicographically)
def set_to_tuple(values_set):
    #sort value inside, then turn to tuple
    return tuple(sorted(values_set))


#build preprocessing to build baskets 
def csv_to_baskets(sc, input_path, k_basket_size):

    #read in the data from csv
    data = sc.textFile(input_path)

    #parse the rows to get (date-customer_id, product_id) suggested fom the assignment pdf
    def parse_rows(lines):
        #check for row validity and amount of rows and iterate through rows
        for row in csv.reader(lines):
            if not row or len(row) < 6: 
                continue

            #check for header row
            if row[0].strip().upper() == "TRANSACTION_DT":
                continue

            #relevant fields (we just need date, customer_id, product_id)
            date_str = row[0].strip()
            cust_id  = row[1].strip()
            product_id = row[5].strip() 

            #skip if any of the fields are missing
            if not date_str or not cust_id or not product_id:
                continue

            #coerce product_id to int if possible (to remove leading zeros) and then str
            try:
                product_id = str(int(product_id))

            #use original if not we get errors
            except ValueError:
                product_id = product_id

            #key from date + customer_id
            key = f"{date_str}-{cust_id}"

            #final yield of key (date+customer_id) and product_id
            yield (key, product_id)

    #build our baskets rdd using mapPartitions to parse rows
    date_customer_product = data.mapPartitions(parse_rows)

    baskets_rdd = (date_customer_product
                   #get combiner (finds value to combine on), aggregates value, and merges values for same key
                   .combineByKey(get_combiner, add_values, union_key)
                   #converts values to tuples
                   .mapValues(set_to_tuple)
                   #filter out basekts less than the basket size threshold (provided by user)
                   .filter(lambda kv: len(kv[1]) > k_basket_size))
    
    return baskets_rdd


#apriori again, mostly taken from task1.py for reuse, directly copy pasted
#global support is the input support value
def calc_local_support(global_support, partition_size, total_baskets):
    #if partition size 0 or basket count 0 (or less), put them to 0
    if partition_size <= 0 or total_baskets <= 0:
        return 0
    #calculate local support value using global support value
    #coerce all to float (divisions will have decimals) and then ceil to round up to whole value
    local_support_value = int(math.ceil(float(global_support)*float(partition_size)/float(total_baskets)))
    #ensure it is positive and at least 1 before returning
    return local_support_value if local_support_value >= 1 else 1

#count frequency in baskets (to compare to support), basically counts apriori L1
def frequency_count(baskets, local_support):
    #hold counts of items
    counts = {}
    for basket in baskets:
        #counted items to prevent double counting
        counted_items = set(basket)
        #go through each item and count
        for item in counted_items:
            counts[item] = counts.get(item, 0)+1

    #make L1 a set to hold candidates
    L1 = set()
    #for each item and its count, add to L1 if it >= local support
    for item, count in counts.items():
        if count>=local_support:
            #frequent items as tuples
            L1.add((item,))

    return L1

#compare candidates we got from the prior pass to the baskets, self join for frequency, (c,k) pairs
def generate_candidates(previous_pass, set_size):

    #check if exists or size is 1 or less (no candidates)
    if not previous_pass or set_size <= 1:
        return set()   
    
    #sort previouspass for ordering and set
    previous_pass = sorted(previous_pass)
    previous_set = set(previous_pass)
    #hold candidates
    candidates = set()
    #count items in the previous pass (to know sizes for new pass)
    total_items = len(previous_pass)
    #for each item in previous pass
    for index in range(total_items):
        itemset1 = previous_pass[index]
        #compare to items after it
        for index_plus in range(index+1, total_items):
            itemset2 = previous_pass[index_plus]

            #if first items are the same (except last) we can build new candidate
            if itemset1[:-1] == itemset2[:-1]:
                #new candidate is combine of the prior two (same parts + last different part)
                new_candidate = tuple(sorted(set(itemset1).union(set(itemset2))))
                #check size correct before adding
                if len(new_candidate) != set_size:
                    continue

                #set keep flag, check for all subsets of new candidate in previous set before continuing     
                keep = True
                for subset in combinations(new_candidate, set_size-1):
                    if subset not in previous_set:
                        keep = False
                        break
                #if still keep, add all to candidates
                if keep:
                    candidates.add(new_candidate)
            else:
                #break loop if not match (not candidate)
                break
    return candidates

#count candidates in baskets and return those meeting the support threshold
def count_candidates(baskets, candidates, local_support):
    #if no candidates, return empty
    if not candidates:
        return set()
    
    #initialize counts plus size for each candidate
    counts = {count:0 for count in candidates}
    size = len(next(iter(candidates)))
    candidate_set = set(candidates)

    #iterate over all the baskets 
    for basket in baskets:
        #check size of baskets for candidate size
        if len(basket) < size:
            continue

        #check through combinations of the basket of the candidate size
        for combination in combinations(basket, size):
            #if in candidates, add to count
            if combination in candidate_set:
                counts[combination] += 1

    #return if the count is higher than the support value needed 
    return set(count for count, count_value in counts.items() if count_value >= local_support)

#run apriori on partition and get candidates, feeds into mappartitions
def apriori(partition, global_support, total_baskets):
    partition_size = len(partition)
    #calc function from earlier, s*P/D
    local_support = calc_local_support(global_support, partition_size, total_baskets)
    #frequency function from above 
    frequency = frequency_count(partition, local_support)
    candidates = set(frequency)
    if not frequency:
        return candidates
    
    itemsets = {x[0] for x in candidates}
    filtered_itemsets = []
    for itemset in partition:
        #get rid of items not in L1 to increase speed (dont need to check again)
        pruned = tuple(sorted(set(itemset).intersection(itemsets)))
        if len(pruned) >= 2:
            filtered_itemsets.append(pruned)

    #start with 2 item itemsets (increase each loop)
    counter = 2
    while frequency:
        #generate new candidates from prior pass
        candidate_counter = generate_candidates(frequency, counter)

        #if no new candidates, stop
        if not candidate_counter:
            break

        #for new candidates, count them in baskets and see if they meet local support
        frequency = count_candidates(filtered_itemsets, candidate_counter, local_support)

        #if they don't meet local support, stop
        if not frequency:
            break

        #new frequent itemset list, add to candidates and increase counter (to go up another itemset in loop)
        candidates = candidates.union(frequency)
        counter += 1

    return candidates

#apriori on each partition, combine later
def apriori_on_partition(iter_baskets, support, total_baskets):
    part_baskets = [items for _, items in iter_baskets]

    #apriori on the partition to get total frequents locally
    local_frequents = apriori(part_baskets, global_support=support, total_baskets=total_baskets)

    #get candidates from the local frequents
    for candidate in local_frequents:
        yield candidate

#SON pass 1, candidates of all partitions
def son_pass1(baskets_rdd, support, total_baskets):
    pass1_candidates = partial(apriori_on_partition, support=support, total_baskets=total_baskets)

    # runs apriori on each partition and gets distinct candidates
    return baskets_rdd.mapPartitions(pass1_candidates).distinct()

#building SON pass 2 and onwards
#also copy pasted a lot from task1.py
#group candidates by size to process out by size
def group_candidates_by_size(candidates):
    #store by size
    by_size = {}
    #iterate over candidates
    for candidate in candidates:
        #get size of candidate (basically number of items in the tuple)
        size = len(candidate)
        #if size doesn't exist, make new set
        if size not in by_size:
            by_size[size] = set()
        by_size[size].add(candidate)
    #should build like 1:{a,}, 2:{(a,b),(a,c),(b,c)}, etc, check later in print 
    return by_size

def possible_candidates_size(by_size):
    #hold potential candidates by size
    possible = {}
    #check size and candidate sets
    for sets, candidate_sets in by_size.items():
        possibles = set()
        for candidate in candidate_sets:
            possibles.update(candidate)
        
        #store possible based on possibles
        possible[sets] = possibles
    
    return possible

#counts candidates in each 
def counts_by_record(record, candidates_by_size, possible_candidates_by_size):
    #gets just the items from the record 
    _, items = record
    #count occurrences of each candidate in the record, sort and get rid of duplicates
    items = set(items)

    #store candidates
    candidate_counter = []
    by_size = candidates_by_size.value
    possible_candidates = possible_candidates_by_size.value

    #for all sizes of candidates and candidates 
    for size, candidates in by_size.items():
        #skip if size of items less than candidate size
        if len(items) < size:
            continue
        
        #check possible candidates by size to reduce search 
        trimmed_set = items.intersection(possible_candidates.get(size, set()))
        if len(trimmed_set) < size:
            continue
        trimmed = tuple(sorted(trimmed_set))

        #check combinations of the correct size
        for combination in combinations(trimmed, size):
            #if found, add to candidate counter
            if combination in candidates:
                candidate_counter.append((combination, 1))
    #return iterator of the candidate counter (reduce later)
    return iter(candidate_counter)

#function above, but emitted, use for with map
def emit_counts(record, candidates_by_size, possible_candidates_by_size):
    return counts_by_record(record, candidates_by_size, possible_candidates_by_size)

#used in reducer, could just use lambda, but trying to try making more clear/modular code (minimizing lambda usage)
def add_counts(a, b):
    return a + b

#filter for >= support
def filter_support_kv(kv, support):
    return kv[1] >= support

#son pass 2, get frequent itemsets from candidates
def son_pass2(sc, baskets_rdd, candidates_rdd, support):
    #actualize values from rdd
    candidates = candidates_rdd.collect()
    by_record = group_candidates_by_size(candidates)
    by_record_by_size = sc.broadcast(by_record)

    possible_by_size = possible_candidates_size(by_record)
    possible_by_size_b = sc.broadcast(possible_by_size)

    #partial functions for map/reduce/filter
    emitter = partial(emit_counts, candidates_by_size=by_record_by_size, possible_candidates_by_size=possible_by_size_b)
    support_filter = partial(filter_support_kv, support=support)

    #map, reduce, filter, and get keys (frequent itemsets)
    counts = baskets_rdd.flatMap(emitter).reduceByKey(add_counts)
    frequents = counts.filter(support_filter).keys()

    #frequent items after pass 2 
    return frequents



#formatting and building output
#taken from task 1
#group itemsets by size
def group_by_size(itemsets):
    #store groups by size
    by_size = {}
    #iterate through itemsets and check size (length of tuple)
    for sets in itemsets:
        size = len(sets)
        #if the size doesn't exist, make new list
        if size not in by_size:
            by_size[size] = []
        #add to list of that size 
        by_size[size].append(sets)

    #hold the groups
    groups = []
    #sort by size and then by lexicographically inside the size
    for size in sorted(by_size.keys()):
        groups.append((size, sorted(by_size[size])))
    
    return groups
#build format for the write to output file
def format_groups(groups):

    #lines to join in json
    lines = []

    #for each size group, format the array
    for _, array in groups:
        if not array:
            #add "" if empty
            lines.append("")
            continue
        
        #hold parts of array
        parts = []
        
        for entry in array:
            #format each entry as ('a','b','c') for json formatting
            parts.append("(" + ",".join("'" + x + "'" for x in entry) + ")")
        #join parts with comma and space
        lines.append(", ".join(parts))
        #line after each size for spacing and better visibility/formatting/viewing
        lines.append("") 
    
    #remove last empty line 
    while lines and lines[-1] == "":
        lines.pop()

    #join lines with new line
    return "\n".join(lines)

#make output file
def make_output(output_path, candidates, frequent_itemsets):
    #use prior functions to group and format 
    candidate_groups = group_by_size(candidates)
    frequent_groups = group_by_size(frequent_itemsets)

    #write to output
    with open(output_path, "w") as f:
        f.write("Candidates:\n")
        f.write(format_groups(candidate_groups))
        f.write("\n\n")
        f.write("Frequent Itemsets:\n")
        f.write(format_groups(frequent_groups))



#timers for output in main (time tracking for parts)
#timer start
def start_timer():
    return time.perf_counter()

#timer end and print duration oftime 
def end_timer(t0, label="Duration"):
    secs = int(time.perf_counter() - t0)
    #adds to show Duration: whatever seconds
    print(f"{label}: {secs}")
    return secs

#main to put everything together
#lol cant copy this one from task1.py :( 
def main():

    #inputs from user following task2.py
    filter_threshold, support, input_path, output_path = check_inputs()

    conf = SparkConf().setAppName("task2")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    #start timer for duration
    t0 = start_timer()
    
    #preprocess the data to intermediate rdd
    baskets_rdd = csv_to_baskets(sc, input_path, filter_threshold).cache()
    total_baskets = baskets_rdd.count()

    #SON time baybeee
    candidates_rdd = son_pass1(baskets_rdd, support, total_baskets).cache()
    frequents_rdd = son_pass2(sc, baskets_rdd, candidates_rdd, support).cache()

    #collect and create our output filea
    candidates = candidates_rdd.collect()
    frequents  = frequents_rdd.collect()
    make_output(output_path, candidates, frequents)

    end_timer(t0, label="Duration")
    sc.stop()    

if __name__ == "__main__":
    main()