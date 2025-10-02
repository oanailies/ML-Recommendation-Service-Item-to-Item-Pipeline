ML Recommendation Service – Item-to-Item Pipeline
Overview

The Item-to-Item Recommendation Pipeline is part of the ML Recommendation Service designed to provide product suggestions based on historical purchases of other customers, without requiring individual user data.

Its primary goal is to suggest complementary or frequently co-purchased products, such as the “Customers also bought” recommendations you often see on e-commerce product pages.

This pipeline uses a hybrid machine learning approach that combines multiple models and techniques to maximize recommendation accuracy, robustness, and diversity.

Features

Hybrid recommendation system: Combines ALS, EASE-R, co-occurrence statistics, and linear blending.

Cold-start support: Provides fallback recommendations for new products or items with limited purchase history.

Diversity-aware recommendations: Uses Maximal Marginal Relevance (MMR) to promote long-tail products and avoid redundant suggestions.

Precomputed product neighbors: Ensures fast retrieval for real-time recommendations.

Rule-based interpretability: Includes a “Bought Together” rules module with metrics such as support, confidence, and lift.

Architecture

The pipeline consists of the following stages:

1. Data Preprocessing

Input: orders.json

Extracted data includes:

Customer ID

Order status

Order date

Total purchase value

Product details: name, brand, category, size, color, price

Temporal split using leave-last-per-user:

Most recent purchase per user → test set

Older purchases → training set

2. Base Model – ALS (Alternating Least Squares)

Trained on implicit feedback from historical purchases.

Weighted using BM25 and recency weighting, prioritizing more recent purchases.

Captures latent item relationships.

3. Hybrid Enhancements

EASE-R: Captures linear item-to-item interactions.

COOC (co-occurrence statistics): Calculates lift, Jaccard, and NPMI between items appearing in the same basket.

BLEND: Linear combination of ALS and EASE to balance local accuracy and generalization.

4. Reranking and Diversity

Scores from ALS, EASE-R, and co-occurrence metrics are combined.

MMR (Maximal Marginal Relevance) is applied to:

Avoid recommending redundant items

Promote long-tail products

Popularity and diversity signals ensure well-rounded suggestions.

5. Cold-Start Handling

For new or sparsely purchased products:

Construct representative vectors based on category, brand, or globally popular items.

Provides fallback recommendations to maintain robustness.

6. “Bought Together” Rules Module

Extracts frequently co-purchased product pairs.

Computes metrics:

Support – How often items are bought together

Confidence – Likelihood that an item is purchased given another

Lift – Strength of association between products

Model Outputs

Saved as a .pkl artifact, containing:

Latent factors (ALS)

Product metadata

Precomputed neighbors for each product

Can be used to generate recommendations in real-time or batch mode.
