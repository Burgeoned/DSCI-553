#homework 5
#task1.py
#In the first two tasks, you will generate a simulated data stream with the Yelp dataset and implement the Bloom Filtering and Flajolet-Martin algorithm. 
# In the third task, you will do some analysis using Fixed Size Sample (Reservoir Sampling).

#If we cannot call myhashs(s) in task1 and task2 in your script to get the hash value list, there will be a 50% penalty.
# Bloom Filtering algorithm to estimate whether the user_id in the data stream has shown before
#keep a a global filter bit array and the length is 69997
#hash for bloom should be: independent, uniform distribution

#To calculate the false positive rate (FPR), you need to maintain a previous user set.
# The size of a single data stream will be 100 (stream_size). And we will test your code for more than 30
# times (num_of_asks), and your FPRs are only allowed to be larger than 0.5 at most once.
# The run time should be within 100s for 30 data streams.

#CSV file with the header “Time,FPR”

#encapsulate your hash functions into a function called myhashs
#The input of myhashs function is a user_id (string) and the output is a list of hash values

import sys
import csv
import binascii
from blackbox import BlackBox

# =========================
# global adjustables/variables
# =========================

#large prime for hashing
prime_number = 2147483647

#array given by the pdf 
global_filter = 69997
bloom_filter = [0] * global_filter

#f(x)= (ax + b) % m or f(x) = ((ax + b) % p) % m
#the a, b for the formula ^
hash_params = [
    (3,    7),
    (5,   11),
    (7,   13),
    (11,  17),
    (13,  19),
    (17,  23),
    (19,  29),
    (23,  31),
    (29,  37),
    (31,  41),
]

# =========================
# typical generic easy utilities
# =========================

#old checkinputs, modified for this assignment
def check_inputs():
    if len(sys.argv) != 5:
        print("Usage: task1.py <input_filename> stream_size num_of_asks <output_filename>", file=sys.stderr)
        sys.exit(1)

    #filter threshold, input file name, output file name
    return sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]

#outputs are Time and FPR (false positive rate)
def write_output(output_filename, rows):
     with open(output_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "FPR"])
        for time, fpr in rows:
            writer.writerow([time, fpr])


# =========================
# math/bloom filter funcs
# =========================

#turns uid into int
def string_to_int(uid_string):
    return int(binascii.hexlify(uid_string.encode("utf8")), 16)

#outputs hash values 
def myhashs(uid_string):
    x = string_to_int(uid_string)
    indices = []
    for a, b in hash_params:
        hashed = (a*x + b) % prime_number
        index = hashed % global_filter
        indices.append(index)
    return indices

#calculator for fpr
def calc_fpr(fp, tn):
    #fpr = fp/(fp+tn)
    return fp/(fp+tn)

#takes users and checks for fp
#fp as in the ones who have been seen (marked in bloom filter)
def process_stream(users, seen):
    fp = 0
    tn = 0  
    for uid in users:
        indices = myhashs(uid)

        #bloom filter, the hashed positions are = 1
        #no false negatives, only false positives
        in_filter = all(bloom_filter[i] == 1 for i in indices)
        actually_seen = uid in seen

        #in filter, but not actually seen user, = fp
        if in_filter and not actually_seen:
            fp += 1
        
        #if not in filter, must be a true negative
        elif not in_filter:
            tn += 1

        #add users into seen if they are all 1 in the filter
        for i in indices: 
            bloom_filter[i] = 1
        seen.add(uid)

    #fpr rate 
    return calc_fpr(fp, tn)


# =========================
# main for sauce
# =========================

def main():
    input_filename, stream_size, num_of_asks, output_filename = check_inputs()
    bx = BlackBox()

    #stores for calcs 
    seen_users = set()
    results = []

    #iterate through num_of_asks provided
    for ask in range(num_of_asks):
        #call blackbox for stream based on input params 
        users = bx.ask(input_filename, stream_size)

        #users we get vs seen ones we had 
        #process should add to seen if we see them
        fpr = process_stream(users, seen_users)
        results.append((ask, fpr))

    write_output(output_filename, results)

if __name__ == "__main__":
    main()