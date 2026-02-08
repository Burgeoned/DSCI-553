#homework 5
#task2.py Flajolet-Martin algorithm
#In the first two tasks, you will generate a simulated data stream with the Yelp dataset and implement the Bloom Filtering and Flajolet-Martin algorithm. 
# In the third task, you will do some analysis using Fixed Size Sample (Reservoir Sampling).

#If we cannot call myhashs(s) in task1 and task2 in your script to get the hash value list, there will be a 50% penalty.
# Bloom Filtering algorithm to estimate whether the user_id in the data stream has shown before
#keep a a global filter bit array and the length is 69997
#hash for bloom should be: independent, uniform distribution

#For this task, the size of the stream will be 300 (stream_size). And we will test your code more than 30
# times (num_of_asks). And for your final result, 0.2 <= (sum of all your estimations / sum of all ground
# truths) <= 5.
# The run time should be within 100s for 30 data streams.

#CSV file with the header “Time,Ground Truth,Estimation”
#input: python task2.py <input_filename> stream_size num_of_asks <output_filename>

#Hash every element a to a sufficiently long bit-string (e.g., h(element a) = 1100 – 4 bits)
#Maintain R = length of longest trailing zeros among all bit-strings (e.g., R = 2)
#Estimate count = 2R,   e.g., 22= 4


import sys
import csv
import binascii
from blackbox import BlackBox

#a lot can be reused from task1

# =========================
# global adjustables/variables
# =========================

#large prime for hashing
prime_number = 2147483647


#hash funcs for the flajolet martin algo
num_hash_functions = 100

#hash func must be divisible by group size
group_size = 10
num_groups = num_hash_functions//group_size

#odd numbers    
hash_params = [(2 * i + 1, 2 * i + 3) for i in range(num_hash_functions)]

# =========================
# typical generic easy utilities
# =========================

#old checkinputs, modified for this assignment
def check_inputs():
    if len(sys.argv) != 5:
        print("Usage: task2.py <input_filename> stream_size num_of_asks <output_filename>", file=sys.stderr)
        sys.exit(1)

    #filter threshold, input file name, output file name
    return sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]

#outputs are time, ground truth, estimation
def write_output(output_filename, rows):
     with open(output_filename, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["Time", "Ground Truth", "Estimation"])

        for time, ground_truth, estimation in rows:
            writer.writerow([time, ground_truth, estimation])


# =========================
# math/ flajolet martin algorithm functions
# =========================

#turns uid into int
def string_to_int(uid_string):
    return int(binascii.hexlify(uid_string.encode("utf8")), 16)

#outputs hash values 
def myhashs(uid_string):

    #same as task1,turn to string but hash differently
    x = string_to_int(uid_string)
    hashes = []
    for a, b in hash_params:
        hash_value = (a*x + b) % prime_number
        hashes.append(hash_value)
    return hashes

def count_trailing_zeros(integer):
    count = 0

    #error handling 
    if integer == 0:
        return 32

    #while lsb is 0, keep going     
    while (integer&1) == 0:
        count += 1
        
        #shifts by 1 bit to the right 
        integer >>= 1

    #count of zeros 
    return count

def get_estimate(max_trailing_zeros_list):

    # 2^R from flajolet martin 
    estimates = [2**r for r in max_trailing_zeros_list]

    #store averages of each group 
    group_avgs = []
    for group in range(num_groups):
        start = group * group_size
        end = start + group_size
        #estimate for each number in group
        group = estimates[start:end]

        #avg of each group
        group_avg = sum(group) / float(group_size)

        #append to group avgs 
        group_avgs.append(group_avg)

    #median to reduce variance from ends 
    group_avgs.sort()

    #for median, depends on if even or odd to calc median
    if num_groups % 2 == 1:
        estimation = group_avgs[num_groups // 2]
    else:
        mid = num_groups // 2
        estimation = (group_avgs[mid - 1] + group_avgs[mid]) / 2.0

    # final integer estimate
    return int(round(estimation))

#process strema of users with the funcs built
def process_stream(users):

    #get unique users, then count for ground truth actual user counts
    unique_users = set(users)
    ground_truth = len(unique_users)

    #max is hash functions * array of [0]
    max_trailing_zeros = [0] * num_hash_functions

    #iterate through each uid to hash
    for uid in unique_users:
        hashes = myhashs(uid)

        for i, hash_value in enumerate(hashes):
            #check trailing zeros
            zeros = count_trailing_zeros(hash_value)

            #set new max trailing zeros if exceeds the current amount 
            if zeros > max_trailing_zeros[i]:
                max_trailing_zeros[i] = zeros

    #use max to estimate vs ground truth (FM algo)
    estimation = get_estimate(max_trailing_zeros)

    return ground_truth, estimation
 

# =========================
# main sawce
# =========================

def main():
    #same as the usual
    input_filename, stream_size, num_of_asks, output_filename = check_inputs()
    bx = BlackBox()

    results = []

    for ask in range(num_of_asks):
        batch = bx.ask(input_filename, stream_size)
        ground_truth, estimation = process_stream(batch)
        results.append((ask, ground_truth, estimation))

    write_output(output_filename, results)

if __name__ == "__main__":
    main()