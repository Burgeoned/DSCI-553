# DSCI 553: Foundations and Applications of Data Mining (Fall 2025)
Data Mining & Scalable Product Systems

University of Southern California | MS Applied Data Science


## Overview
This repository serves as a master portfolio of large-scale data mining and product strategy systems. The core focuses on implementing complex algorithms from scratch using Apache Spark (RDD API) to process high-dimensional datasets that exceed single-machine memory constraints.

## Flagship Project
**Objective:** Engineered a hybrid recommendation system to predict user-business ratings for the Yelp dataset.

**Evolution:** This project represents the high-performance evolution of Module 03 (Recommendation Systems), moving from baseline collaborative filtering to an optimized, model-based architecture.

**Outcome:** Achieved a validation RMSE of 0.9769, beating the competitive benchmark of 0.9800.

**Innovation:** Utilized an XGBoost-only framework, choosing to eliminate noisy collaborative filtering signals in favor of high-signal structured features like custom sentiment-scoring for Yelp "tips" and semantic category grouping.

| Module                                                        | Programming                                                  | Tags                                                      | RMSE     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------------------- | -------- |
| [Hybrid Recommendation System](https://github.com/Burgeoned/DSCI-553/blob/main/Competition/DSCI%20553%20Competition%20Project%20-%20F2025%20.docx.pdf) | [Python](https://github.com/Burgeoned/DSCI-553/blob/main/Competition/competition.py) | `XGBoost` `Feature Engineering` `Model-based recommendation system` | 0.976939 |


## Engineering Modules: Large-Scale Distributed Algorithms
Each module represents a fundamental challenge in processing massive, high-dimensional datasets using the Spark RDD API to ensure low-level control over distributed memory and partitioning.

| #     | Module                                                        | Programming                                                  | Focus & Tags                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | [Spark Operations](https://github.com/Burgeoned/DSCI-553/blob/main/Homework%201/Assignment%201%20Fall%202025.pdf) | [Python](https://github.com/Burgeoned/DSCI-553/tree/main/Homework%201) | Distributed Foundations: Transformations, Actions, and Yelp Dataset exploration.                                           |
| 2    | [Frequent Itemset Mining](https://github.com/Burgeoned/DSCI-553/blob/main/Homework%202/Homework%202.pdf) | [Python](https://github.com/Burgeoned/DSCI-553/tree/main/Homework%202) | Pattern Discovery: Implementing the SON Algorithm and A-Priori for distributed passes.                           |
| 3    | [Recommendation Systems](https://github.com/Burgeoned/DSCI-553/blob/main/Homework%203/HW3.pdf) | [Python](https://github.com/Burgeoned/DSCI-553/tree/main/Homework%203) | Precursor to Competition: Locality Sensitive Hashing (LSH) and Collaborative Filtering. |
| 4    | [Community Detection](https://github.com/Burgeoned/DSCI-553/blob/main/Homework%204/Assignment4%20-%20Fall%202025.pdf) | [Python](https://github.com/Burgeoned/DSCI-553/tree/main/Homework%204) | Graph Analytics: Detecting social structures via the Girvan-Newman Algorithm.                      |
| 5    | [Data Stream Processing](https://github.com/Burgeoned/DSCI-553/blob/main/Homework%205/Assignment%205-Fall%202025.pdf) | [Python](https://github.com/Burgeoned/DSCI-553/tree/main/Homework%205) | Real-Time Data: Bloom Filtering, Flajolet-Martin, and Reservoir Sampling. |
| 6    | [Massive Clustering](https://github.com/Burgeoned/DSCI-553/blob/main/Homework%206/Assignment%206-FALL%202025.pdf) | [Python](https://github.com/Burgeoned/DSCI-553/tree/main/Homework%206) | High-Dimensional Analysis: Implementing the BFR Algorithm for massive datasets.             |
