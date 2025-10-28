# ML Recommendation Service – Item-to-Item Pipeline

## Overview
The **Item-to-Item Recommendation Pipeline** is a core component of the ML Recommendation Service.  
It provides product suggestions based on historical purchase behavior, without requiring individual user data.  
The goal is to generate complementary or frequently co-purchased product recommendations similar to the "Customers also bought" sections found on e-commerce websites.

This pipeline implements a **hybrid machine learning approach**, combining several algorithms and techniques to ensure accuracy, robustness, and diversity.

---

## Key Features
- **Hybrid Recommendation System**: Combines ALS, EASE-R, co-occurrence statistics, and linear blending.
- **Cold-Start Support**: Provides fallback recommendations for new or low-history products.
- **Diversity-Aware Recommendations**: Applies Maximal Marginal Relevance (MMR) to promote long-tail products and reduce redundancy.
- **Precomputed Product Neighbors**: Enables fast, real-time retrieval.
- **Rule-Based Interpretability**: Includes a “Bought Together” rules module using support, confidence, and lift metrics.

---

## Architecture

### 1. Data Preprocessing
**Input:** `orders.json`

Extracted fields include:
- Customer ID  
- Order status  
- Order date  
- Total purchase value  
- Product details: name, brand, category, size, color, price  

A **temporal split** is applied using *leave-last-per-user*:
- The most recent purchase per user → test set  
- Older purchases → training set

---

### 2. Base Model – ALS (Alternating Least Squares)
- Trained on implicit purchase feedback.  
- Weighted using **BM25** and **recency weighting** to emphasize recent purchases.  
- Captures latent item relationships between products.

---

### 3. Hybrid Enhancements
- **EASE-R**: Captures linear item-to-item interactions.  
- **COOC (Co-occurrence Statistics)**: Computes lift, Jaccard similarity, and NPMI for items appearing together.  
- **BLEND**: Linear combination of ALS and EASE-R scores for balanced accuracy and generalization.

---

### 4. Reranking and Diversity
- Combines scores from ALS, EASE-R, and co-occurrence metrics.  
- Applies **MMR (Maximal Marginal Relevance)** to:
  - Reduce redundant recommendations  
  - Encourage diversity and long-tail exposure  
- Final ranking balances accuracy, popularity, and novelty.

---

### 5. Cold-Start Handling
For new or infrequently purchased items:
- Constructs representative embeddings based on **category**, **brand**, or **global popularity**.  
- Ensures robust fallback recommendations even for unseen products.

---

### 6. “Bought Together” Rules Module
Generates interpretable rule-based associations between products:
- **Support** – Frequency of co-purchase  
- **Confidence** – Probability of buying an item given another  
- **Lift** – Strength of association between items  

These rules supplement the ML models with interpretable, human-readable insights.

---

## Model Outputs
The pipeline produces a `.pkl` artifact containing:
- Latent item factors (ALS)  
- Product metadata  
- Precomputed item neighbors for efficient retrieval  

This artifact can be used for:
- **Real-time recommendations** (e.g., API serving)  
- **Batch recommendations** for offline personalization tasks.

---

## Technologies Used
- Python (NumPy, Pandas, Scipy)
- Implicit library (ALS)
- EASE-R implementation
- Scikit-learn
- TensorFlow / Keras
- Pickle for model serialization

---

## Summary
The ML Recommendation Service provides a scalable, interpretable, and hybrid approach to item-to-item recommendations.  
By combining collaborative filtering, co-occurrence statistics, and diversity-aware reranking, the system effectively supports both high-traffic recommendation APIs and batch personalization workflows.
