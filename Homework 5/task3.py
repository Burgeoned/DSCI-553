#homework 5
#task3.py 
#The goal of task3 is to implement the fixed-size sampling method (Reservoir Sampling Algorithm).
#we assume that the memory can only save 100 users
#or the first 100 users, you can directly save them in a list. After that, for the nth
#(n startsfrom 101) user in the whole sequence of users, you will keep the nth user with the probability of 100/n,
#otherwise discard it. If you keep the nth user, you need to randomly pick one in the list to be replaced.

#For this task, the size of the stream will be 100 (stream_size). And we will test your code more than 30 times (num_of_asks)
#Be careful here: Please write your random.seed(553) in the main function. Please do not write random.seed(553) in other places.
#The run time should be within 100s for 30 data streams.

import sys
import csv
import random
from blackbox import BlackBox

# =========================
# global adjustables/variables
# =========================

#100 provided by assignment 
reservoir_size = 100

#making these global because it needs to be maintained over iterations of calling reservoir sampling
reservoir = []       
sequence_num = 0   

# =========================
# typical generic easy utilities
# =========================

#old checkinputs, modified for this assignment
def check_inputs():
    if len(sys.argv) != 5:
        print("Usage: task3.py <input_filename> stream_size num_of_asks <output_filename>", file=sys.stderr)
        sys.exit(1)

    #filter threshold, input file name, output file name
    return sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]

def write_output(output_filename, rows):
    with open(output_filename, "w", newline="") as f:
        writer = csv.writer(f)
        # copied column names in the sample
        writer.writerow(["seqnum", "0_id", "20_id", "40_id", "60_id", "80_id"])
        for row in rows:
            writer.writerow(row)

# =========================
# reservoir sampling functions
# =========================

#process uids for reservoir sampling
def process_users(users):

    #calling the res/seq at the top of file
    #sequence is to count total users seen across the streams (or multiple thus far)
    global reservoir, sequence_num

    #iterate through each user in current stream (users from blackbox)
    for uid in users:
        #inc seq each iteration 
        sequence_num += 1


        #fill reservoir initially if smaller than reservoir 
        if sequence_num <= reservoir_size:

            #when not > 100, just use all 100
            reservoir.append(uid)
        else:
            #sampling after first 100
            #random sample, and sequence - 1 (because 0 counting)
            if random.random() < (reservoir_size / float(sequence_num)):
                
                #when kept, choose an index
                index = random.randint(0, reservoir_size - 1)
                reservoir[index] = uid

    # after processing the batch, collect required positions
    # by the time we finish the first batch (stream_size >= 100),
    # reservoir will be full, so these indices are valid.
    indices = [0, 20, 40, 60, 80]

    #should be the users/ids at 1, 21, 41, 61, 81 out of 100 
    selected_ids = [reservoir[i] for i in indices]

    return sequence_num, selected_ids

# =========================
# main sawce
# =========================

def main():
    input_filename, stream_size, num_of_asks, output_filename = check_inputs()
    

    #seed in main 
    random.seed(553)
    bx = BlackBox()

    #store results for output
    results = []

    #iterate through each ask and build to write 
    for ask in range(num_of_asks):
        users = bx.ask(input_filename, stream_size)
        seq_after_batch, selected_ids = process_users(users)
        row = [seq_after_batch] + selected_ids
        results.append(row)

    write_output(output_filename, results)


if __name__ == "__main__":
    main()