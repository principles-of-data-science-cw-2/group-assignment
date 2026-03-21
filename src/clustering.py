import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter

# --- Load the Dataset ---
df = pd.read_csv('../data/cleaned_retail_data.csv')
# Shape of dataset
print(df.shape)
print(df.head())

# --- Data Cleaning - ckeck ---

# Remove missing values
print("Number of missing values")
print(df.isna().sum())

# Remove duplicates
print(f"Number of duplicates: {df.duplicated().sum()}")

# Handle incorrect values (Example: Negative quantities , Invalid prices)
print(f"Negative quantities: {(df[["Quantity"]] < 0).sum()}")
print(f"Negative prices: {(df[["Price"]] < 0).sum()}")

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
print(f"InvoiceDate data type: {df['InvoiceDate'].dtype}")

# -------------------------------------------------
#      Customer Segmentation (RFM Clustering) 
# -------------------------------------------------

# --- Feature Engineering - Calculate RFM metrics ---

# Recency = (reference date) − (last purchase date)

# reference date (usually max(most recent) date in dataset)
reference_date = df["InvoiceDate"].max()
print(reference_date)
# get last purchase per customer
last_purchase = df.groupby("Customer ID")["InvoiceDate"].max()

# Recency (in days)
recency = (reference_date - last_purchase).dt.days

# Frequency
frequency = df.groupby("Customer ID")["Invoice"].nunique()

# Monetary
monetary = df.groupby("Customer ID")["Total_Revenue"].sum()

# Combine them:
rfm = pd.concat([recency, frequency, monetary], axis=1)
rfm.columns = ["Recency","Frequency","Monetary"]
print(rfm.head())

# ---- Feature Scaling - Use Standardization.-----
# Now features have: mean = 0 , standard deviation = 1

scaler = StandardScaler()
scaled_data = scaler.fit_transform(rfm)

rfm_scaled = rfm.copy()
rfm_scaled[["Recency","Frequency","Monetary"]] = scaled_data
print(rfm_scaled)

# --- Determine Optimal Number of Clusters ---
# Method : Elbow Method

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

# Silhouette score (to justify the choice)

X = rfm_scaled  # your scaled RFM data

silhouette_scores = []

k_values = range(2, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

# Plot
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.show()

# ----- Train the Clustering Model - K-Means -----

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(rfm_scaled)

rfm['Cluster'] = clusters
print(rfm.head())

# ------ Cluster Visualization ------
# Using PCA (dimensionality reduction)
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)
print(rfm_pca.shape)
      
plt.scatter(rfm_pca[:,0], rfm_pca[:,1], c=rfm["Cluster"])
plt.show()

# ----- Cluster Profiling (Most Important Step) ------
# Understand what each cluster represents.
cluster_profile = rfm.groupby('Cluster').mean()
print(cluster_profile)


# -------------------------------------------------
#      Demand Pattern Analysis 
# -------------------------------------------------

# ---- Feature Engineering ------

# Recency = (reference date) − (last purchase date)

# reference date (usually max date in dataset)
reference_date = df["InvoiceDate"].max()
print(reference_date)
# get last purchase per customer
last_purchase = df.groupby("Description")["InvoiceDate"].max()

# Recency (in days)
product_recency = (reference_date - last_purchase).dt.days

# Frequency
product_frequency = df.groupby("Description")["Quantity"].nunique()

# Monetary
product_monetary = df.groupby("Description")["Total_Revenue"].sum()

# Combine them:
rfm_product = pd.concat([product_recency, product_frequency, product_monetary], axis=1)
rfm_product.columns = ["Recency","Frequency","Monetary"]
print(rfm_product)

# ----- Feature Scaling - Use Standardization. ---------
# Now features have: mean = 0 , standard deviation = 1
scaler = StandardScaler()

rfm_product_scaled = rfm_product.copy()
rfm_product_scaled[["Recency","Frequency","Monetary"]] = scaler.fit_transform(rfm_product)
print(rfm_product_scaled)

# ------ Determine Optimal Number of Clusters -------------
# Method : Elbow Method
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(rfm_product_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()


# Silhouette score (to justify the choice)
X = rfm_product_scaled  # your scaled RFM data

silhouette_scores = []

k_values = range(2, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

# Plot
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.show()

# The optimal number of clusters = 3

# ------- Train the Clustering Model - K-Means ---------

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(rfm_product_scaled)

rfm_product['Cluster'] = clusters
rfm_product

# --------- Cluster Visualization -----------
# Using PCA (dimensionality reduction)
pca_product = PCA(n_components=2)
rfm_product_pca = pca_product.fit_transform(rfm_product_scaled)
print(rfm_product_pca.shape)
      
plt.scatter(rfm_product_pca[:,0], rfm_product_pca[:,1], c=rfm_product["Cluster"], cmap='viridis')
plt.show()

# --------- Cluster Profiling (Most Important Step) --------
# Understand what each cluster represents.

cluster_profile = rfm_product.groupby('Cluster').mean()
print(cluster_profile)

# -----------------------------------
# The aim of this code is to analyze the most common product-related words associated with customers in Cluster 2.
cluster_2 = rfm_product[rfm_product["Cluster"] == 2]
cluster_2_products = []
for index, row in cluster_2.iterrows():
    cluster_2_products.append(index)

# Combine all words
all_words = []
for desc in cluster_2_products:
    words = desc.upper().split()  # split by space, convert to uppercase
    all_words.extend(words)

# Count frequency
word_counts = Counter(all_words)

# Show top 50 common words
print(word_counts.most_common(50))

# ---------------------------------------
# The aim of this code is to analyze the most common product-related words associated with customers in Cluster 0.
cluster_0 = rfm_product[rfm_product["Cluster"] == 0]
cluster_0_products = []
for index, row in cluster_0.iterrows():
    cluster_0_products.append(index)

# Combine all words
all_words_0 = []
for desc in cluster_0_products:
    words_0 = desc.upper().split()  # split by space, convert to uppercase
    all_words_0.extend(words_0)

# Count frequency
word_counts_0 = Counter(all_words_0)

# Show top 50 common words
print(word_counts_0.most_common(50))

# -------------------------------------------------
# The aim of this code is to analyze the most common product-related words associated with customers in Cluster 1.
cluster_1 = rfm_product[rfm_product["Cluster"] == 1]
cluster_1_products = []
for index, row in cluster_1.iterrows():
    cluster_1_products.append(index)

# Combine all words
all_words_1 = []
for desc in cluster_1_products:
    words_1 = desc.upper().split()  # split by space, convert to uppercase
    all_words_1.extend(words_1)

# Count frequency
word_counts_1 = Counter(all_words_1)

# Show top 50 common words
print(word_counts_1.most_common(50))

