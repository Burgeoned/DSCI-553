# DSCI 553: Foundations and Applications of Data Mining (Fall 2025)
Data Mining & Scalable Product Systems

University of Southern California | MS Applied Data Science


## Overview
This repository serves as a master portfolio of large-scale data mining and product strategy systems. The core of the repository focuses on implementing complex algorithms from scratch using Apache Spark (RDD API) to process high-dimensional datasets that exceed single-machine memory constraints.

## Flagship Project
**Objective:** Engineered a hybrid recommendation system to predict user-business ratings for the Yelp dataset. 

**Outcome:** Achieved a validation RMSE of 0.9769, beating the target competitive benchmark of 0.9800. 

**Architecture:** Utilized an XGBoost-only framework, choosing to eliminate noisy collaborative filtering signals in favor of high-signal structured features. 

**Innovation:** Developed custom sentiment-scoring for Yelp "tips" and implemented semantic category grouping to mitigate the high dimensionality of 1,300+ business categories.

| Topic                                                        | Programming                                                  | Tags                                                      | RMSE     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------------------- | -------- |
| [Hybrid Recommendation System](https://github.com/Burgeoned/DSCI-553/blob/main/Competition/DSCI%20553%20Competition%20Project%20-%20F2025%20.docx.pdf) | [Python]((https://github.com/Burgeoned/DSCI-553/blob/main/Competition/competition.py)) | `XGBoost` `Yelp Data` `Model-based recommendation system` | 0.976939 |

## Technical Implementation
* Distributed Transparency: Strictly utilized Spark RDDs (not DataFrames) to ensure deep control over distributed memory management and partitioning logic. 

* Operational Constraints: All systems were optimized to run on standard cluster configurations with a maximum 4GB memory and strict time limits.

* Environment: Developed for Python 3.6 and Spark 3.1.2.
