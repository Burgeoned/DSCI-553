#hw4 task 2
#submit format
#spark-submit task2.py <filter threshold> <input_file_path> <betweenness_output_file_path> <community_output_file_path>
#sparkrdd only, no df
#Task 2: Community Detection Based on Girvan-Newman algorithm
#betweenness calculations, save to txt
#community detection, modularity formula

#tips:
#1. For task 2.2, you should take into account the precision. For example: stop the modularity calculation only if there is a 
#significant reduction in the new modularity.
#2. A=1 when BOTH i in j and j in i. Not just i in j or j in i.
#3. For task 2.2 the stopping criteria plays an important role. Again, avoid the temptation to stop your search at the first
#decrease in modularity. Instead, continue exploring all potential partitions to find the global maximum. This comprehensive 
# approach ensures that you don't miss the optimal solution.
#4. If you want to do a thorough checking of the answer, you can always calculate the modularity for all possible 
#communities (stop until no edges remain).
#5. In modularity calculation, For A, using current graph; for kikj, using original graph.

#task 2_1
#rebuild graph from task1, but with rdd, and then calc betweenness
#betweenness format (‘user_id1’, ‘user_id2’), betweenness value
#order: firstly sorted by the betweenness values in descending order and then the first
#user_id in the tuple in lexicographical order (the user_id is type of string). The two user_ids in each tuple
#should also be in lexicographical order.
# round() function to round the betweenness value to five digits after the decimal point

#task 2_2
#divide into global highest modularity
#build: girvan newman
#If the community only has one user node, we still regard it as a valid community

#self reminder: adjacency is essentially matrix of what nodes are connected 

#trying to document better with 
# =========================

import sys, math, csv
from collections import defaultdict, deque
from itertools import combinations
from pyspark import SparkConf, SparkContext

# =========================
#helpers (read files and check inputs)
# =========================

def check_inputs():
    if len(sys.argv) != 5:
        print("Usage: spark-submit task2.py <filter_threshold> <input_csv> <betweenness_output_txt> <community_output_txt>", file=sys.stderr)
        sys.exit(1)
    return int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]

#csv to rdd, taken from old assignments
def csv_to_rdd(lines):
    for row in csv.reader(lines):
        if not row or len(row) < 2:
            continue
        a = row[0].strip()
        b = row[1].strip()

        #skip empties
        if not a or not b:
            continue
        yield (a, b)

#taken from old assignments
def read_csv(sc, input_path, has_header=None):
    rdd = sc.textFile(input_path)

    if has_header is None:
        #column names
        first = rdd.first()
        has_header = ("user_id" in first and "business_id" in first)

    if has_header:
        header = rdd.first()
        rdd = rdd.filter(lambda line: line != header)

    return rdd.mapPartitions(csv_to_rdd)

# =========================
#building task 2_1: betweenness calcs
# =========================

#pairs rdd should be user, business, threshold is input by user
def build_threshold_edges_rdd(pairs_rdd, filter_threshold):

    #flipping this to group users by business
    business_to_users = (pairs_rdd
                         #ub = user business 
                         .map(lambda ub: (ub[1], ub[0]))
                         .groupByKey()
                         #remove duplicates 
                         .mapValues(lambda users: list(set(users))))
    
    #built from above, now pairs with 2 representing
    pairs_by_business = business_to_users.mapValues(lambda users: list(combinations(users, 2)))
    all_pairs = pairs_by_business.flatMap(lambda kv: kv[1])

    #make sure we can make it undirected edges
    def undirected(u, current_node):
        return (u, current_node) if u < current_node else (current_node, u)

    #undirect the pairs, and then sum for overlapts 
    pair_undirected = all_pairs.map(lambda uv: (undirected(uv[0], uv[1]), 1))
    pair_counts = pair_undirected.reduceByKey(lambda x, y: x + y)

    #keep the ones that meet the filter threshold >= 
    edges = pair_counts.filter(lambda kv: kv[1] >= filter_threshold).keys()

    return edges

#take rdd built above
def edge_adjacency(edges_rdd):

    #e = edges, 
    undirected = (edges_rdd
                  #ensure both directions
                  .flatMap(lambda e: [(e[0], e[1]), (e[1], e[0])])
                  #groups by key (node) and then al neighbors
                  .groupByKey()
                  #set and remove dupes
                  .mapValues(lambda n: set(n)))
    
    #materialize for spark 
    return dict(undirected.collect())

#only way i found reasonable to calc is BFS (Brandes) for node betweenness of large sets
#takes from computing ALL down from O(current_node^3) down to O(VE) complexity

def bfs(source_node, neighbors):

    #is the list of nodes before neighbor_node on the shortest path
    predecessors = defaultdict(list)
    #distance from source_node node to current_node
    #-1 for those not counted
    distance = defaultdict(lambda: -1)
    # sigma is num of shortest paths from source_node node to current_node
    sigma = defaultdict(int)

    #start at 0 and 1, distance is 0 and 1 shortest path to itself
    distance[source_node] = 0
    sigma[source_node] = 1

    bfs_order = []
    #double ended queue dequeue
    queue = deque([source_node])

    while queue:
        #go through the queue until empty 
        current_node = queue.popleft()
        bfs_order.append(current_node)

        for neighbor_node in neighbors.get(current_node, []):

            #th first time we see the neighbor (set at -1 above)
            if distance[neighbor_node] < 0:
                #discovered, and +1 to it 
                distance[neighbor_node] = distance[current_node] + 1
                queue.append(neighbor_node)
            
            #if discovered, 
            if distance[neighbor_node] == distance[current_node] + 1:
                #adding the shortest paths to reach current node
                sigma[neighbor_node] += sigma[current_node]
                #add neighbor node to predecessors(all the nodes we pass to reach current node)
                predecessors[neighbor_node].append(current_node)

    #term online from bfs algorithm
    #delta is how much shortest paths flow through the node
    delta = defaultdict(float)

    #how much betweenness contribution from this edge, calc here to store
    edge_credit = defaultdict(float)

    #go from farthest to closest 
    #for eachchild nodeof each parent node
    for child_node in reversed(bfs_order):
        for parent_node in predecessors[child_node]:
            #shortest path of child node exists, then 
            if sigma[child_node] > 0:
                #edge credit formula is just sigmav/sigmaw (1+deltaw)
                credit = (sigma[parent_node] / sigma[child_node]) * (1.0 + delta[child_node])
                
                #undirected, so we set to one direction
                edge_vertex = (parent_node, child_node) if parent_node < child_node else (child_node, parent_node)
                #add how much credit for the whole loop to get final credits
                edge_credit[edge_vertex] += credit
                delta[parent_node] += credit

    #should be the maount of shortest path flow
    return edge_credit


def calc_edge_betweenness(adjacency):
    total_edge_credit = defaultdict(float)
    #iterate through very single node, each as the source 
    for source_node in adjacency.keys():
        #run bfs and get credits for each node 
        per_source_credit = bfs(source_node, adjacency)

        #iterate through all of the edges and their credits 
        for edge_vertex, value in per_source_credit.items():
            #sum into the total 
            total_edge_credit[edge_vertex] += value

    #undirected graphs, need to divide by 2 to ensure proper credit distribution lol 
    #this is from double counting since undirected
    for edge_vertex in list(total_edge_credit.keys()):
        total_edge_credit[edge_vertex] = total_edge_credit[edge_vertex]/2.0

    #should output a betweenness for the whole map of nodes
    return total_edge_credit

# =========================
#building task 2_2:
# =========================
#need to build gn Girvan-Newman algorithm

def build_communities(adjacency):

    #initialize for storing 
    visited_nodes = set()
    communities = []

    #for column of nodes = start_nodes
    for start_node in adjacency.keys():
        if start_node in visited_nodes:
            continue
        
        #hold community of nodes 
        community = []
        queue = deque([start_node])
        visited_nodes.add(start_node)

        #iterate through the queue and add each current node to the community
        while queue:
            current = queue.popleft()
            community.append(current)

            #neighbors = adjacency[current]
            for neighbor in adjacency.get(current, []):

                #add neighbors to visited as needed if we hadn't added it prior 
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append(neighbor)

        #sort each community before adding to communities 
        communities.append(sorted(community))

    #should get lists like [['A','B','C'], ['D','E'], ['F']] sorted 
    return communities

#need to calculate modularity to decide where to split
#modularity = Q value often 0.3 to 0.7
#Q = 1/2m summation[Aij - ki*kj/2m]
#A_ij: from the current(cut) graph
#k_i, k_j, m: from the og graph
def calc_modularity(current_adjacency, communities, original_degree, two_m):
    #m from 2m of the formula 
    m = two_m/2.0
    #Q for modularity 
    Q = 0.0

    #iterat ethrough each community we built in communities 
    for community in communities:
        #make community into a set of members
        members_set = set(community)

        #community version of modularity formula is:
        #summation [Lc/m - (Dc/2m)^2]
        Lc = 0
        for u1 in community:
            neighbors_of_u1 = current_adjacency.get(u1, set())
            # neighbors from the CURRENT graph (A_ij comes from the cut graph)
            for u2 in neighbors_of_u1:
                if u2 in members_set and u1 < u2:
                    #check for both directions before adding +1
                    neighbors_of_u2 = current_adjacency.get(u2, set())
                    if u1 in neighbors_of_u2:
                        Lc += 1

        #sum of community node degrees 
        Dc = sum(original_degree.get(u1, 0) for u1 in community)

        #summation of the community modularity formula
        Q += (Lc / m) - ((Dc / (2.0 * m))**2)

    return Q

#Girvan–Newman says: at each iteration, remove the edge(s) with highest current betweenness.
#If multiple edges tie for max, removing only one is arbitrary and can trap you in a suboptimal path; 
#removing all max edges yields a clean, deterministic split and makes Q search more stable
#use this between iterations to remove current highest betweenness 
def remove_max_betweenness_edges(adjacency, edge_betweenness):

    #error handle nones
    if not edge_betweenness:
        return [], 0.0

    #get highest value edge 
    max_value = max(edge_betweenness.values())
    edges_to_remove = [edge for edge, value in edge_betweenness.items() if value == max_value]

    #remove the edges on both ends because undirected 
    for u1, u2 in edges_to_remove:

        #double check and validate in the set before remove with discard
        if u2 in adjacency.get(u1, set()):
            adjacency[u1].discard(u2)
        if u1 in adjacency.get(u2, set()):
            adjacency[u2].discard(u1)

    #edge to remove and its value 
    return edges_to_remove, max_value


# =========================
#writing outputs
# =========================

#task2_1 output
#value should be descending, and then by user id lexicogrpahically 
def write_betweenness_output(path, edge_betweenness_dict):
    #sorting 
    sorted_items = sorted(edge_betweenness_dict.items(),
                          #desecnding by kv, then make sure [0] before [1]
                          key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    with open(path, "w", encoding="utf-8") as f:
        for (u1, u2), betweeness in sorted_items:
            #round to 5 digits 
            f.write(f"('{u1}', '{u2}'), {betweeness:.5f}\n")

#task2_2 output
def write_communities_output(path, communities):
    #sort the community 
    communities = [sorted(community) for community in communities]
    #sort also by length, then by tuple(m)
    communities.sort(key=lambda c: (len(c), tuple(c)))
    with open(path, "w", encoding="utf-8") as f:
        for members in communities:
            f.write(", ".join(f"'{uid}'" for uid in members) + "\n")


# =========================
#main, put everything together 
# =========================
def main():

    #validate input
    filter_threshold, input_path, betweenness_output_file_path, community_output_file_path = check_inputs()

    #build spark
    conf = SparkConf().setAppName("task2_rdd")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    #materialize spark
    user_business = read_csv(sc, input_path, has_header=None).distinct().cache()
    _ = user_business.count()

    edges = build_threshold_edges_rdd(user_business, filter_threshold).cache()
    _ = edges.count()

    #collect adjacency matrix (kinda matrix) for all users
    adjacency = edge_adjacency(edges)

    
    #degrees and 2m calculations from the original graph (before we start cutting)
    original_adjacency = {u: set(v) for u, v in adjacency.items()}
    original_degree = {u: len(v) for u, v in original_adjacency.items()}
    #2m used in formulas 
    two_m = sum(original_degree.values())

    #run betweenness on the original graph and write output for task 2_1
    base_betweenness = calc_edge_betweenness(original_adjacency)
    write_betweenness_output(betweenness_output_file_path, base_betweenness)

    # below is for task 2_2 gn algorithm 

    #remake of the graph to work off of 
    adjacency_working = {node: set(neighbors) for node, neighbors in original_adjacency.items()}
    #before we remove anything,build communities out 
    current_communities = build_communities(adjacency_working)

    #get Q and partition of the original communities 
    best_modularity_Q = calc_modularity(adjacency_working, current_communities, original_degree, two_m)
    best_partition = current_communities

    while True:

        #recalculate each loop 
        betweenness_now = calc_edge_betweenness(adjacency_working)

        #calculate for which ones should be removed and the max value to detemrine those edges
        edges_removed, max_value = remove_max_betweenness_edges(adjacency_working, betweenness_now)

        #when nothing is removed (nothing left to remove )
        if not edges_removed:
            #build communities to evaluate
            current_communities = build_communities(adjacency_working)
            #evaluate Q modularity of the built communities 
            current_Q = calc_modularity(adjacency_working, current_communities, original_degree, two_m)

            #evaluate and compare, if it is >, then cut and rebuild 
            if current_Q > best_modularity_Q:
                best_modularity_Q = current_Q
                best_partition = current_communities
            break
        
        #rebuild AGAIN to evaluate the newly formed communities 
        current_communities = build_communities(adjacency_working)
        current_Q = calc_modularity(adjacency_working, current_communities, original_degree, two_m)

        #evaluate AGAIN with the newly built communities
        if current_Q > best_modularity_Q:
            best_modularity_Q = current_Q
            best_partition = current_communities

    #write output for task 2_2
    write_communities_output(community_output_file_path, best_partition)

    #stop spark
    sc.stop()


if __name__ == "__main__":
    main()