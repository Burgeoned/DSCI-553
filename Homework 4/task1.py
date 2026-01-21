#hw4 task1
#submit format
#spark-submit --packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 task1.py <filter threshold> <input_file_path> <community_output_file_path>
#txt file, and ‘user_id1’, ‘user_id2’, ‘user_id3’, ‘user_id4’, …

import sys

#other libraries allowed for task1 of this assignment
from pyspark.sql import SparkSession, functions as F, types as T
from graphframes import GraphFrame

#old checkinputs, modified for this assignment
def check_inputs():
    if len(sys.argv) != 4:
        print("Usage: spark-submit --packages graphframes:graphframes:0.8.2-spark3.1-s_2.12" 
              "task1.py <filter_threshold> <input_file_name> <output_file_name>", file=sys.stderr)
        sys.exit(1)

    #filter threshold, input file name, output file name
    return int(sys.argv[1]), sys.argv[2], sys.argv[3]

#load spark data into spark 
def load_data(spark, input_path):
    return (
        spark.read.option("header", True).csv(input_path)
        #user id / business id in spark 
        #thank god lol back to sql
        .select(F.col("user_id").cast(T.StringType()),
                F.col("business_id").cast(T.StringType()))
        .dropna()
        .dropDuplicates()
        .cache()
    )

#final output (should be a txt file)
def write_output(path, communities):
    with open(path, "w", encoding="utf-8") as f:
        #each row = 1 community (instruction provided in pdf)
        #user1, user2, user3, .. etc.
        for members in communities:
            #join uids with '', between each
            f.write(", ".join(f"'{uid}'" for uid in members) + "\n")

#validate >= threhsold number provided 
#>= same number of businesses rated to be counted for building edges between the nodes 
def check_threshold(matching_rates, threshold):
    return matching_rates >= F.lit(threshold)

#excludes isolated users (so dont become vertices/vertex)
def build_vertices_from_edges(edges):
    return (
        edges.select(F.col("e1").alias("id"))
             .union(edges.select(F.col("e2").alias("id")))
             .distinct()
    )

#need to mirror and flip the directions so that they become undirected 
def make_undirected_edges(edges):
    return (
        edges.select(F.col("e1").alias("src"), F.col("e2").alias("dst"))
             .union(edges.select(F.col("e2").alias("src"), F.col("e1").alias("dst")))
             .distinct()
    )

#realize i could have built this directly into main as it's small, but whatever
def run_lpa(graph):
    return graph.labelPropagation(maxIter=5).select("id", "label")


#building edges (between nodes of users), based on >= threshold size given in input
def node_edges(df, threshold):

    #view of the df, view a view b
    a = df.alias("a")
    b = df.alias("b")

    #self join on bid match, but only on b>a 
    edges = (
        a.join(b, on="business_id")
         #ensures we don't do double counting, basically only a,b not counting b,a
         .where(F.col("a.user_id") < F.col("b.user_id"))
         #group by user ids as u1/u2, aggregates by these 2 together 
         .groupBy(F.col("a.user_id").alias("u1"), F.col("b.user_id").alias("u2"))
         .count()
         #visibility
         .withColumnRenamed("count", "overlap_count")
         #check overlaps (for exceeding threshold to build an edge)
         .where(check_threshold(F.col("overlap_count"), threshold))
         #selects these for building edges
         .select(F.col("u1").alias("e1"), F.col("u2").alias("e2"))
         .distinct()
         .cache()
    )
    return edges


#main to put it all together

#added in .count() to materialize (it seems to fix and pass on vocareum with this add)
def main():
    #input validation
    threshold, input_path, output_path = check_inputs()

    spark = SparkSession.builder.appName("task1").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    #load data into df (input provided)
    df = load_data(spark, input_path)

    # "upper triangle" which is one direction (b>a)
    edge_pairs_upper = node_edges(df, threshold)
    #debug
    #print(f"[dbg] edges_upper={edge_pairs_upper.count()}", file=sys.stderr)
    _ = edge_pairs_upper.count()

    vertices = build_vertices_from_edges(edge_pairs_upper)
    edges_undirected = make_undirected_edges(edge_pairs_upper)
    _ = vertices.count(); _ = edges_undirected.count()
    #debug
    #print(f"[dbg] vertices={vertices.count()}, edges_undirected={edges_undirected.count()}", file=sys.stderr)
    
    #build the graph, and run lpa on the graph
    graph = GraphFrame(vertices, edges_undirected)

    #returns 2 column df of id and labels
    lpa = run_lpa(graph)
    _ = lpa.count()
    
    #debug
    #print(f"[dbg] lpa_rows={lpa.count()}", file=sys.stderr)

    #build communities from the lpa 
    communities = (
        #groups by LPA label (groups into communities)
        lpa.groupBy("label")
           .agg(F.collect_list("id").alias("members"))
           .select("members")
           #sorts members within community 
           .rdd.map(lambda r: sorted(r["members"]))
           .collect()
    )
    #sort by size ascending, then user id
    communities.sort(key=lambda m: (len(m), tuple(m)))

    #debug
    #print(f"[dbg] communities={len(communities)}", file=sys.stderr)

    #output (should be txt file)
    write_output(output_path, communities)

    #stop spark
    spark.stop()


if __name__ == "__main__":
    main()