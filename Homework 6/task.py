#hw 6
#do bfr on a synthetic dataset
#can use sklearn and numpy, rdd optional but should be faster
#indexed -1 points are outliers
#hw6_cluster.txt is the synthetic clustering dataset
#discardset(ds)/compressionset(cs)/retainedset(rs)
#in ds/cs N = num points, SUM = sum of coordinates of the ponits, SUMQ = sum of square coordinates
#spark-submit task.py <input_file> <n_cluster> <output_file>
#output is text file
#each line start with Round: {i}: for intermediate file

import sys
import math
from collections import defaultdict

import numpy as np
from pyspark import SparkConf, SparkContext
from sklearn.cluster import KMeans

##############################
# adjustables
##############################

#lol copying last assignment
random_seed = 553

#20% 
chunk_ratio = 0.2


##############################
# check inputs, other basic funcs, also timer fo r
##############################

def check_inputs():
    # check inputs and also make sure clusters is int
    if len(sys.argv) != 4:
        print("Usage: spark-submit task.py <input_file> <num_clusters> <output_file>", file=sys.stderr)
        sys.exit(1)

    return sys.argv[1], int(sys.argv[2]), sys.argv[3]


##############################
# preprocessing data, intermediary step 
##############################

#format of line: point_id, true_cluster_label, f1, f2, ..., fd
#ignore the true labels and get (point_id, feature_vector)
def parse_line_to_point(line):

    line = line.strip()
    if not line:
        return None

    pieces = line.split(",")
    if len(pieces) < 3:
        return None

    point_id = int(pieces[0])
    feature_values = [float(value) for value in pieces[2:]]

    return point_id, feature_values


def read_and_preprocess_data(spark_context, input_file_path, random_seed=random_seed):

    #rdd supposed to be faster from the assignment details/instructions
    raw_rdd = spark_context.textFile(input_file_path)

    #check for missing 
    parsed_rdd = raw_rdd.map(parse_line_to_point).filter(lambda x: x is not None)
    parsed_rows = parsed_rdd.collect()
    
    #store ids and features 
    point_ids = []
    feature_rows = []

    #build out mapping 
    for point_id, feature_values in parsed_rows:
        point_ids.append(point_id)
        feature_rows.append(feature_values)

    #i missed numpy so much this semester ksl fd;ls fhsa f fsa
    feature_matrix = np.array(feature_rows, dtype=float)
    num_points = feature_matrix.shape[0]

    #index all of the points so that we can randomize by index 
    global_indices = np.arange(num_points, dtype=int)
    rng = np.random.RandomState(random_seed)
    rng.shuffle(global_indices)

    #point ids, their features, and the randomized index (choose 20% of it )
    return point_ids, feature_matrix, global_indices

#step 1 20% of data randomly 
def build_random_chunks(global_indices, chunk_ratio=0.2):

    #get total points 
    num_points = len(global_indices)

    #build chunks of the data being ~20% each 
    chunk_size = max(1, int(num_points * chunk_ratio))

    chunks = []

    #iterate through index for chunk sizes and build chunks 
    for start_index in range(0, num_points, chunk_size):

        #make chunks actually just the size of the chunks 
        end_index = start_index + chunk_size
        chunk = global_indices[start_index:end_index]
        chunks.append(chunk)

    return chunks

##############################
# summary of clusters for N/SUM/SUMQ 
##############################

#summary is for the discard set of BFR (summarize to reduce need to keep them, hence discard)
def create_cluster_summary(dimension):
    summary = {
        "num_points": 0,
        "sum_vector": np.zeros(dimension, dtype=float),
        "sum_sq_vector": np.zeros(dimension, dtype=float)
    }
    return summary

#creating discard set and the things to summarize
def add_point_to_summary(cluster_summary, feature_vector):
    cluster_summary["num_points"] += 1
    cluster_summary["sum_vector"] += feature_vector
    cluster_summary["sum_sq_vector"] += feature_vector * feature_vector

#when combining clusters, need to also merge their summaries to create final summary
def merge_cluster_summaries(target_summary, source_summary):
    target_summary["num_points"] += source_summary["num_points"]
    target_summary["sum_vector"] += source_summary["sum_vector"]
    target_summary["sum_sq_vector"] += source_summary["sum_sq_vector"]

#get the mean for the cluster (needed for summary)
def cluster_mean(cluster_summary):
    return cluster_summary["sum_vector"] / cluster_summary["num_points"]

#cluster standard deviation
def cluster_std(cluster_summary):
    mean_vector = cluster_mean(cluster_summary)
    variance_vector = (
        cluster_summary["sum_sq_vector"] / cluster_summary["num_points"]
        - mean_vector * mean_vector
    )
    return np.sqrt(variance_vector)

#mahalanobis distance formula 
#sqrt of sum((value-mean)/standard deviation)^2
def mahalanobis_distance(feature_vector, cluster_summary):
    mean_vector = cluster_mean(cluster_summary)
    std_vector = cluster_std(cluster_summary)

    #inside of the fomula
    standardized = (feature_vector - mean_vector) / std_vector

    #sqrt and sum of the square of formula 
    return math.sqrt(np.sum(standardized * standardized))


##############################
# k means as pointed by step 2 in assignment 
##############################

#step 2, run kmeans
#use large k (num_clusters)
def run_kmeans(feature_matrix, num_clusters, random_seed=random_seed):
    
    #get number of points 
    num_points = feature_matrix.shape[0]
    if num_points == 0:
        return None, None

    #cant have more clusters than points set to points if cluster count ends up higher 
    if num_clusters > num_points:
        num_clusters = num_points

    #n_init = restart times, 10 is pretty safe/stable from sources online
    model = KMeans(n_clusters=num_clusters, random_state=random_seed, n_init=10)

    #get results from model for centers and the labels within cluster
    labels = model.fit_predict(feature_matrix)
    centers = model.cluster_centers_
    return labels, centers


##############################
# BFR round 1
##############################

#develops the first ds/rs/cs
def bfr_first_chunk(feature_matrix,
                    first_chunk_indices,
                    num_clusters):

    #dimension of the points 
    dimension = feature_matrix.shape[1]

    #large K for BFR (step 2), 5x clusters 
    big_k = 5 * num_clusters

    #grab just the first 20% of the feature matrix 
    chunk_feature_matrix = feature_matrix[first_chunk_indices]

    #K means and find the left out RS (retained set)
    big_labels, _ = run_kmeans(chunk_feature_matrix, big_k)

    #map cluster id and get indices within chunk
    cluster_to_local_indices = defaultdict(list)
    for local_index, cluster_label in enumerate(big_labels):
        cluster_to_local_indices[cluster_label].append(local_index)

    #RS is retained set, points that dont belong to DS or CS yet
    retained_set_points = {}
    remaining_local_indices = []

    #Step 3 on the pdf clusters with size 1 = RS, others = potential DS points
    for cluster_label, local_list in cluster_to_local_indices.items():
        if len(local_list) == 1:
            #singleton, send to RS
            #basically outlier
            local_index = local_list[0]
            global_index = first_chunk_indices[local_index]
            retained_set_points[global_index] = feature_matrix[global_index]

        else:
            #keep these local indices to build DS in step 4
            for local_index in local_list:

                #build remaining local indices set for ds
                remaining_local_indices.append(local_index)

    remaining_local_indices = np.array(remaining_local_indices, dtype=int)

    #step 4 of pdf KMeans with K = num_clusters on remaining points = DS
    discard_set_clusters = {}     
    discard_set_assignments = {}  

    #if 0, we can build the ds
    if remaining_local_indices.size > 0:
        remaining_feature_matrix = chunk_feature_matrix[remaining_local_indices]
        ds_labels, _ = run_kmeans(remaining_feature_matrix, num_clusters)

        #make empty summaries for each DS cluster (don't actually have summry or need yet)
        for cluster_id in range(num_clusters):
            discard_set_clusters[cluster_id] = create_cluster_summary(dimension)

        #add each remaining point into its DS summary
        for local_position, ds_cluster_id in enumerate(ds_labels):
            global_index = first_chunk_indices[remaining_local_indices[local_position]]
            feature_vector = feature_matrix[global_index]
            add_point_to_summary(discard_set_clusters[ds_cluster_id], feature_vector)
            discard_set_assignments[global_index] = ds_cluster_id

    #step 6 on pdf KMeans on RS to form CS clusters + new RS
    compression_set_clusters = {}    
    compression_set_assignments = {}    

    #get cs clusters and the points for them and the RS 
    compression_set_clusters, compression_set_assignments, retained_set_points = run_kmeans_on_retained_set(
        feature_matrix,
        retained_set_points,
        compression_set_clusters,
        compression_set_assignments,
        num_clusters
    )

    #summary from first part of doing bfr 
    #ds count, cs clusters/counts, rs counts
    num_discard_points = sum(summary["num_points"] for summary in discard_set_clusters.values())
    num_compression_clusters = len(compression_set_clusters)
    num_compression_points = sum(summary["num_points"] for summary in compression_set_clusters.values())
    num_retained_points = len(retained_set_points)

    #put into one for easier managing
    round_statistics = (
        num_discard_points,
        num_compression_clusters,
        num_compression_points,
        num_retained_points
    )

    return (discard_set_clusters,
            discard_set_assignments,
            compression_set_clusters,
            compression_set_assignments,
            retained_set_points,
            round_statistics)

##############################
#rounds 2+ of BFR
##############################

#process rest of chunk for bfr
def process_subsequent_chunk(feature_matrix,
                             chunk_indices,
                             num_clusters,
                             discard_set_clusters,
                             discard_set_assignments,
                             compression_set_clusters,
                             compression_set_assignments,
                             retained_set_points):
    
    #dimension and threshold (2*sqrt(d))
    #threshold was provided in pdf 
    dimension = feature_matrix.shape[1]
    mahalanobis_threshold = 2.0 * math.sqrt(dimension)

    #loop through all points in this chunk 
    for global_index in chunk_indices:
        feature_vector = feature_matrix[global_index]

        #ds
        best_discard_cluster_id = None
        best_discard_distance = float("inf")
        
        #compare to ds to decide discard or not 
        for cluster_id, summary in discard_set_clusters.items():

            #mahalanobis this time
            distance_value = mahalanobis_distance(feature_vector, summary)

            #discard if close enough, build ds
            if distance_value < best_discard_distance:
                best_discard_distance = distance_value
                best_discard_cluster_id = cluster_id

        #building the ds index by setting new cluster ids to it 
        if best_discard_cluster_id is not None and best_discard_distance < mahalanobis_threshold:
            add_point_to_summary(discard_set_clusters[best_discard_cluster_id], feature_vector)
            discard_set_assignments[global_index] = best_discard_cluster_id
            continue

        #cs after ds
        best_compression_cluster_id = None
        best_compression_distance = float("inf")

        #the next closest sets after ds
        for cluster_id, summary in compression_set_clusters.items():
            distance_value = mahalanobis_distance(feature_vector, summary)
            if distance_value < best_compression_distance:
                best_compression_distance = distance_value
                best_compression_cluster_id = cluster_id

        #same as ds, but the other points and clusters left 
        if best_compression_cluster_id is not None and best_compression_distance < mahalanobis_threshold:
            add_point_to_summary(compression_set_clusters[best_compression_cluster_id], feature_vector)
            compression_set_assignments[global_index] = best_compression_cluster_id
            continue

        #rs the leftovers
        retained_set_points[global_index] = feature_vector

    #run kmeans on the RS to run another iteratoin 
    compression_set_clusters, compression_set_assignments, retained_set_points = run_kmeans_on_retained_set(
        feature_matrix,
        retained_set_points,
        compression_set_clusters,
        compression_set_assignments,
        num_clusters
    )

    #step 12 on pdf merge close CS clusters 
    compression_set_clusters, compression_set_assignments = merge_compression_clusters(
        compression_set_clusters,
        compression_set_assignments
    )

    #stats for this round of ds (intermediary step)
    num_discard_points = sum(summary["num_points"] for summary in discard_set_clusters.values())
    num_compression_clusters = len(compression_set_clusters)
    num_compression_points = sum(summary["num_points"] for summary in compression_set_clusters.values())
    num_retained_points = len(retained_set_points)

    round_statistics = (
        num_discard_points,
        num_compression_clusters,
        num_compression_points,
        num_retained_points
    )

    return (discard_set_clusters,
            discard_set_assignments,
            compression_set_clusters,
            compression_set_assignments,
            retained_set_points,
            round_statistics)


##############################
#k means on cs and rs 
##############################

def run_kmeans_on_retained_set(feature_matrix,
                               retained_set_points,
                               compression_set_clusters,
                               compression_set_assignments,

                               num_clusters):
    
    #if no RS points, nothing to do, or if less than 2 points (cant cluster)
    #basically finished
    if (len(retained_set_points) == 0) or (len(retained_set_points) < 2):
        return compression_set_clusters, compression_set_assignments, retained_set_points

    #basically repeating the ds stuff, but now on rs
    dimension = feature_matrix.shape[1]
    big_k = 5 * num_clusters

    #turn RS dict into matrix for kmeans 
    retained_indices = np.array(list(retained_set_points.keys()), dtype=int)
    retained_matrix = feature_matrix[retained_indices]

    #kmeans on the RS
    labels, _ = run_kmeans(retained_matrix, big_k)

    label_to_positions = defaultdict(list)
    for position, label_value in enumerate(labels):
        label_to_positions[label_value].append(position)

    #start CS ids depending on existing cs clusters 
    if len(compression_set_clusters) > 0:
        next_compression_cluster_id = max(compression_set_clusters.keys()) + 1
    else:
        next_compression_cluster_id = 0

    #store new RS points 
    new_retained_set_points = {}

    #clusters with >1 point become CS, others stay RS
    for label_value, position_list in label_to_positions.items():

        #the <=1 case = RS
        if len(position_list) <= 1:
            for pos in position_list:
                global_index = retained_indices[pos]
                new_retained_set_points[global_index] = feature_matrix[global_index]

        #CS the rest
        else:
            cluster_summary = create_cluster_summary(dimension)
            compression_cluster_id = next_compression_cluster_id
            #add another 1 so next cluster id is different
            next_compression_cluster_id += 1

            #run the new summary with change in points of this iteration 
            for pos in position_list:
                global_index = retained_indices[pos]
                feature_vector = feature_matrix[global_index]
                add_point_to_summary(cluster_summary, feature_vector)
                compression_set_assignments[global_index] = compression_cluster_id

            #set the cluster (using the id) to this summary 
            compression_set_clusters[compression_cluster_id] = cluster_summary

    #changed cs/rs 
    return compression_set_clusters, compression_set_assignments, new_retained_set_points

#merging cs to cs 
#step 12 on pdf, merge cs clusters within mahalanobis distance threshold
def merge_compression_clusters(compression_set_clusters,
                               compression_set_assignments):
    
    #if no CS clusters nothing to merge 
    if len(compression_set_clusters) == 0:
        return compression_set_clusters, compression_set_assignments

    #get dimension from first CS cluster
    compression_ids = sorted(compression_set_clusters.keys())
    first_cs_id = compression_ids[0]

    #just copied components earlier, distance for mahalanobis 
    dimension = len(compression_set_clusters[first_cs_id]["sum_vector"])
    distance_threshold = 2.0 * math.sqrt(dimension)

    #store merged clusters mapping
    merged_map = {} 

    #needed to make a function to get the original cluster still 
    def find_original_cluster(cluster_id):
        while cluster_id in merged_map:
            cluster_id = merged_map[cluster_id]
        return cluster_id

    #check distances between the CS centroids 
    for i in range(len(compression_ids)):

        #compare each i and j in both sets 
        for j in range(i + 1, len(compression_ids)):

            #1st and 2nd cluster 
            cluster_i = find_original_cluster(compression_ids[i])
            cluster_j = find_original_cluster(compression_ids[j])

            #skip if already merged to same original
            if cluster_i == cluster_j:
                continue
            
            #summaries of both using func 
            summary_i = compression_set_clusters[cluster_i]
            summary_j = compression_set_clusters[cluster_j]

            mean_i = cluster_mean(summary_i)

            #compare mean of i to cluster j with mahalanobisss
            distance_value = mahalanobis_distance(mean_i, summary_j)

            #if distance is within the threshold, we can merge and remove the CS j 
            if distance_value < distance_threshold:
                merge_cluster_summaries(summary_i, summary_j)
                merged_map[cluster_j] = cluster_i
                del compression_set_clusters[cluster_j]

    #fix assignments to use original cs ids 
    for point_index, cluster_id in list(compression_set_assignments.items()):
        original_id = find_original_cluster(cluster_id)

        #check for id changing 
        if original_id != cluster_id:

            #set to original id to use to retain original still on the new set
            compression_set_assignments[point_index] = original_id

    return compression_set_clusters, compression_set_assignments


def merge_compression_into_discard_at_end(discard_set_clusters,
                                          discard_set_assignments,
                                          compression_set_clusters,
                                          compression_set_assignments,
                                          feature_matrix):
    
    #if no CS clusters, nothing to merge, so just skip and return 
    if len(compression_set_clusters) == 0:
        return discard_set_clusters, discard_set_assignments

    #get dimension from first DS  and other stats (like mahalanobis )
    discard_ids = sorted(discard_set_clusters.keys())
    first_ds_id = discard_ids[0]
    dimension = len(discard_set_clusters[first_ds_id]["sum_vector"])
    distance_threshold = 2.0 * math.sqrt(dimension)

    #loop through each CS and see if it should merge into some DS cluster 
    for compression_id, compression_summary in list(compression_set_clusters.items()):
        compression_mean = cluster_mean(compression_summary)

        best_discard_id = None
        best_distance = float("inf")
        
        #checking stats to compare cs with ds 
        for discard_id, discard_summary in discard_set_clusters.items():
            distance_value = mahalanobis_distance(compression_mean, discard_summary)
            if distance_value < best_distance:
                best_distance = distance_value
                best_discard_id = discard_id

        #merge if distance fits
        if best_discard_id is not None and best_distance < distance_threshold:
            merge_cluster_summaries(discard_set_clusters[best_discard_id], compression_summary)

            #move all points from this CS cluster into that DS cluster
            for point_index, cluster_id in list(compression_set_assignments.items()):
                if cluster_id == compression_id:
                    discard_set_assignments[point_index] = best_discard_id

                    #delete the cs set
                    del compression_set_assignments[point_index]

            #then delete the cs if mere
            del compression_set_clusters[compression_id]

    #new ds outputted 
    return discard_set_clusters, discard_set_assignments


##############################
#write output
##############################

#writing outputs
def write_output_file(output_file_path,
                      intermediate_statistics_list,
                      final_labels_by_point_id):
    
    with open(output_file_path, "w") as f:

        #intermediate results example in pdf
        f.write("The intermediate results:\n")
        for round_index, stats in enumerate(intermediate_statistics_list, start=1):
            num_discard_points, num_compression_clusters, num_compression_points, num_retained_points = stats

            #Round x: n_ds, n_compression_clusters, n_cs, n_rs
            f.write(
                f"Round {round_index}: {num_discard_points},{num_compression_clusters},{num_compression_points},{num_retained_points}\n"
            )

        f.write("\n")
        f.write("The clustering results:\n")

        #write out each point and the label of where its clustered
        #example in pdf: 0,1 or 1,1 or 2,0 or 3,0
        for point_id in sorted(final_labels_by_point_id.keys()):
            f.write(f"{point_id},{final_labels_by_point_id[point_id]}\n")



##############################
#main sawce
##############################

def main():
    input_file_path, num_clusters, output_file_path = check_inputs()

    conf = SparkConf().setAppName("hw6_task")
    spark_context = SparkContext(conf=conf)
    spark_context.setLogLevel("WARN")

    #preprocess data to how we need 
    point_ids, feature_matrix, shuffled_global_indices = read_and_preprocess_data(
        spark_context, input_file_path
    )

    #pid is point id,
    index_to_point_id = {i: pid for i, pid in enumerate(point_ids)}

    #build random chunks to start bfr 
    chunks = build_random_chunks(shuffled_global_indices, chunk_ratio=chunk_ratio)

    intermediate_statistics_list = []

    #initialize the first chunk 
    first_chunk_indices = chunks[0]
    (discard_set_clusters,
     discard_set_assignments,
     compression_set_clusters,
     compression_set_assignments,
     retained_set_points,
     round_stats) = bfr_first_chunk(
        feature_matrix,
        first_chunk_indices,
        num_clusters
    )
    intermediate_statistics_list.append(round_stats)

    #iterate and get stats for the rest of the chunks (skip 1 cuz we did already)
    for chunk_indices in chunks[1:]:

        #i honestly typoed so much cuz this thing is so long lol 
        (discard_set_clusters,
         discard_set_assignments,
         compression_set_clusters,
         compression_set_assignments,
         retained_set_points,
         round_stats) = process_subsequent_chunk(
            feature_matrix,
            chunk_indices,
            num_clusters,
            discard_set_clusters,
            discard_set_assignments,
            compression_set_clusters,
            compression_set_assignments,
            retained_set_points
        )
        intermediate_statistics_list.append(round_stats)

    #merge cs into ds if the fit 
    discard_set_clusters, discard_set_assignments = merge_compression_into_discard_at_end(
        discard_set_clusters,
        discard_set_assignments,
        compression_set_clusters,
        compression_set_assignments,
        feature_matrix
    )

    #store final labels of each point for printing
    final_labels_by_point_id = {}
    #should match point count
    num_points = feature_matrix.shape[0]

    #go through the DS and put point ids with their cluster ids for final output
    for global_index, cluster_id in discard_set_assignments.items():
        point_id = index_to_point_id[global_index]
        final_labels_by_point_id[point_id] = cluster_id

    #go through each point and set the non DS points to -1 (outlier basically )
    for global_index in range(num_points):
        point_id = index_to_point_id[global_index]
        if point_id not in final_labels_by_point_id:
            final_labels_by_point_id[point_id] = -1

    #write to txt file
    write_output_file(output_file_path, intermediate_statistics_list, final_labels_by_point_id)

    spark_context.stop()


if __name__ == "__main__":
    main()