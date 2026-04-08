import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
from collections import Counter

# Load the Dataset
df = pd.read_csv('../data/cleaned_retail_data.csv')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
print(f"Dataset shape: {df.shape}")
df.head()

# ================================================
#
# Part 1 — Customer Segmentation (RFM Clustering)
#
#=================================================

# Step 1 — Data Quality Check
# ---------------------------
print("Missing values:")
print(df.isna().sum())
print(f"\nDuplicates: {df.duplicated().sum()}")
print(f"Negative quantities: {(df['Quantity'] < 0).sum()}")
print(f"Negative prices:     {(df['Price'] < 0).sum()}")
print(f"InvoiceDate dtype:   {df['InvoiceDate'].dtype}")

# Step 2 — Feature Engineering: Compute RFM Metrics
# -------------------------------------------------
# Only use positive-quantity rows for customer RFM (exclude returns)
df_sales = df[df['Quantity'] > 0].copy()

reference_date = df_sales['InvoiceDate'].max()

recency   = (reference_date - df_sales.groupby('Customer ID')['InvoiceDate'].max()).dt.days
frequency = df_sales.groupby('Customer ID')['Invoice'].nunique()
monetary  = df_sales.groupby('Customer ID')['Total_Revenue'].sum()

rfm = pd.concat([recency, frequency, monetary], axis=1)
rfm.columns = ['Recency', 'Frequency', 'Monetary']
print(f"Customers: {len(rfm)}")
print(rfm.describe())

# Step 3 — Outlier Removal
# ------------------------
# Outlier removal using z-score threshold = 3
z_scores = np.abs(stats.zscore(rfm[['Recency','Frequency','Monetary']]))
mask = (z_scores < 3).all(axis=1)
rfm_clean = rfm[mask].copy()

removed = rfm[~mask]
print(f"Original customers : {len(rfm)}")
print(f"After outlier removal: {len(rfm_clean)}")
print(f"Removed {len(removed)} outlier(s):")
print(removed[['Recency','Frequency','Monetary']])

# Step 4 — Feature Scaling (Standardisation)
# ------------------------------------------
scaler = StandardScaler()
rfm_scaled_vals = scaler.fit_transform(rfm_clean[['Recency','Frequency','Monetary']])
rfm_scaled = rfm_clean.copy()
rfm_scaled[['Recency','Frequency','Monetary']] = rfm_scaled_vals
print("Scaled RFM (mean≈0, std≈1):")
print(rfm_scaled.describe().round(2))

# Step 5 — Determine Optimal Number of Clusters
# ---------------------------------------------
# Elbow Method
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_scaled[['Recency','Frequency','Monetary']])
    wcss.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1,11), wcss, marker='o', linewidth=2)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.title('Elbow Method — Customer Segmentation')
plt.xticks(range(1,11))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Silhouette Score
sil_scores = []
k_range = range(2, 10)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(rfm_scaled[['Recency','Frequency','Monetary']])
    sil_scores.append(silhouette_score(rfm_scaled[['Recency','Frequency','Monetary']], labels))

plt.figure(figsize=(8, 4))
plt.plot(k_range, sil_scores, marker='o', color='darkorange', linewidth=2)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method — Customer Segmentation')
plt.xticks(k_range)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

best_k = list(k_range)[sil_scores.index(max(sil_scores))]
print(f"Best k by silhouette: {best_k}  (score: {max(sil_scores):.3f})")

# Step 6 — Apply K-Means Clustering
# ---------------------------------
km_customer = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm_clean['KM_Cluster'] = km_customer.fit_predict(rfm_scaled[['Recency','Frequency','Monetary']])
rfm_clean.head()

# Step 7 — Visualise Clusters with PCA (2D Projection)
# ----------------------------------------------------
pca = PCA(n_components=2, random_state=42)
rfm_pca = pca.fit_transform(rfm_scaled[['Recency','Frequency','Monetary']])
explained = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(9, 6))
colors = ['#2196F3','#FF5722','#4CAF50','#9C27B0']
labels_map = {0:'Cluster 0', 1:'Cluster 1', 2:'Cluster 2', 3:'Cluster 3'}

for c in range(4):
    mask = rfm_clean['KM_Cluster'] == c
    ax.scatter(rfm_pca[mask, 0], rfm_pca[mask, 1],
               c=colors[c], label=labels_map[c], alpha=0.6, s=30)

ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}% variance)')
ax.set_title('K-Means Customer Clusters (PCA projection)')
ax.legend()
plt.tight_layout()
plt.show()

# Step 8 — Cluster Profiling
# --------------------------
km_profile = rfm_clean.groupby('KM_Cluster')[['Recency','Frequency','Monetary']].mean().round(2)
km_profile['Count'] = rfm_clean.groupby('KM_Cluster').size()
km_profile['% Share'] = (km_profile['Count'] / len(rfm_clean) * 100).round(1)
print("K-Means Customer Cluster Profiles:")
print(km_profile)

# Part 1b — Hierarchical Clustering on Customer Segments
# =====================================================

# Dendrogram — Ward linkage
X_cust = rfm_scaled[['Recency','Frequency','Monetary']].values
Z = linkage(X_cust, method='ward')

plt.figure(figsize=(14, 5))
dendrogram(Z, truncate_mode='lastp', p=40,
           leaf_rotation=90, leaf_font_size=8,
           color_threshold=0.6 * max(Z[:, 2]))
plt.title('Hierarchical Clustering Dendrogram — Customer Segmentation (Ward)', fontsize=13)
plt.xlabel('Customer (cluster size in brackets)')
plt.ylabel('Ward Distance')
plt.axhline(y=sorted(Z[:,2])[-3], color='red', linestyle='--',
            linewidth=1.5, label='Cut for 4 clusters')
plt.legend()
plt.tight_layout()
plt.show()

# Apply Agglomerative Clustering with k=4
agg_cust = AgglomerativeClustering(n_clusters=4, linkage='ward')
rfm_clean['HC_Cluster'] = agg_cust.fit_predict(X_cust)

hc_profile = rfm_clean.groupby('HC_Cluster')[['Recency','Frequency','Monetary']].mean().round(2)
hc_profile['Count'] = rfm_clean.groupby('HC_Cluster').size()
print("Hierarchical Clustering — Customer Profiles:")
print(hc_profile)

# Side-by-side comparison: K-Means vs Hierarchical
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
cmaps = ['tab10', 'Set1']
titles = ['K-Means (k=4)', 'Hierarchical / Ward (k=4)']
cluster_cols = ['KM_Cluster', 'HC_Cluster']

for ax, col, title, cmap in zip(axes, cluster_cols, titles, cmaps):
    for c in rfm_clean[col].unique():
        mask = rfm_clean[col] == c
        ax.scatter(rfm_pca[mask.values, 0], rfm_pca[mask.values, 1],
                   label=f'Cluster {c}', alpha=0.6, s=25)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}%)')
    ax.legend(fontsize=8)

plt.suptitle('Customer Segmentation: K-Means vs Hierarchical Clustering', fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

# Measure agreement between both methods
ari = adjusted_rand_score(rfm_clean['KM_Cluster'], rfm_clean['HC_Cluster'])
print(f"Adjusted Rand Index (K-Means vs Hierarchical): {ari:.3f}")
print("Interpretation: 1.0 = perfect agreement  |  0.0 = random  |  <0 = worse than random")


# ============================================================
#
# Part 2 — Demand Pattern Analysis (Product-Level Clustering)
#
# ============================================================

# Use only sales transactions (exclude returns)
reference_date = df_sales['InvoiceDate'].max()

product_recency   = (reference_date - df_sales.groupby('Description')['InvoiceDate'].max()).dt.days
# FIXED: count unique invoices, not unique quantity values
product_frequency = df_sales.groupby('Description')['Invoice'].nunique()
product_monetary  = df_sales.groupby('Description')['Total_Revenue'].sum()

rfm_product = pd.concat([product_recency, product_frequency, product_monetary], axis=1)
rfm_product.columns = ['Recency', 'Frequency', 'Monetary']
print(f"Products: {len(rfm_product)}")
print(rfm_product.describe().round(2))

# Scale Product Features
scaler_prod = StandardScaler()
rfm_product_scaled = rfm_product.copy()
rfm_product_scaled[['Recency','Frequency','Monetary']] = scaler_prod.fit_transform(rfm_product)
print("Scaled product RFM (mean≈0, std≈1):")
print(rfm_product_scaled.describe().round(2))

# Elbow Method
wcss_p = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_product_scaled)
    wcss_p.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1,11), wcss_p, marker='o', linewidth=2)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.title('Elbow Method — Product Demand Patterns')
plt.xticks(range(1,11))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Silhouette Score
sil_p = []
k_range_p = range(2, 10)
for k in k_range_p:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(rfm_product_scaled)
    sil_p.append(silhouette_score(rfm_product_scaled, labels))

plt.figure(figsize=(8, 4))
plt.plot(k_range_p, sil_p, marker='o', color='darkorange', linewidth=2)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method — Product Demand Patterns')
plt.xticks(k_range_p)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

best_kp = list(k_range_p)[sil_p.index(max(sil_p))]
print(f"Optimal k by silhouette: {best_kp}  (score: {max(sil_p):.3f})")

# KMeans Clustering
km_product = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm_product['Cluster'] = km_product.fit_predict(rfm_product_scaled)
rfm_product.head()

# Visualise Product Demand Clusters
pca_p = PCA(n_components=2, random_state=42)
rfm_product_pca = pca_p.fit_transform(rfm_product_scaled)
explained_p = pca_p.explained_variance_ratio_

colors_p = ['#FF6B6B', '#4ECDC4', '#45B7D1']
fig, ax = plt.subplots(figsize=(9, 6))
for c in range(3):
    mask = rfm_product['Cluster'] == c
    ax.scatter(rfm_product_pca[mask, 0], rfm_product_pca[mask, 1],
               c=colors_p[c], label=f'Cluster {c}', alpha=0.5, s=20)
ax.set_xlabel(f'PC1 ({explained_p[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({explained_p[1]*100:.1f}% variance)')
ax.set_title('K-Means Product Demand Clusters (PCA)')
ax.legend()
plt.tight_layout()
plt.show()

# Cluster Profiling
prod_profile = rfm_product.groupby('Cluster')[['Recency','Frequency','Monetary']].mean().round(2)
prod_profile['Count'] = rfm_product.groupby('Cluster').size()
prod_profile['% Share'] = (prod_profile['Count'] / len(rfm_product) * 100).round(1)
print("Product Demand Cluster Profiles:")
print(prod_profile)

# Word Frequency Analysis by Demand Cluster
def word_freq(cluster_df, cluster_id, top_n=20):
    products = cluster_df[cluster_df['Cluster'] == cluster_id].index.tolist()
    words = []
    for desc in products:
        words.extend(str(desc).upper().split())
    counts = Counter(words)
    # Filter generic stop words
    stopwords = {'OF','THE','AND','WITH','A','IN','TO','FOR','&','-','SET','OF'}
    filtered = [(w,c) for w,c in counts.most_common(top_n+len(stopwords))
                if w not in stopwords][:top_n]
    return filtered

for cid, label in [(1,'High-Demand'), (0,'Steady Mid-Tier'), (2,'Low-Demand')]:
    top = word_freq(rfm_product, cid, top_n=15)
    words, freqs = zip(*top)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.barh(words[::-1], freqs[::-1], color=colors_p[cid], alpha=0.8)
    ax.set_title(f'Cluster {cid} — {label}: Top Product Keywords', fontsize=12)
    ax.set_xlabel('Frequency')
    plt.tight_layout()
    plt.show()
    

# ========================================================
#
# Part 3 — Service Performance Clustering (Country-Level)
#
# ========================================================

# Cancellation rate based on invoice prefix
df['Is_Cancelled'] = df['Invoice'].astype(str).str.startswith('C')

country_agg = df.groupby('Country').agg(
    Total_Revenue   = ('Total_Revenue', 'sum'),
    Num_Orders      = ('Invoice', 'nunique'),
    Num_Customers   = ('Customer ID', 'nunique'),
    Num_Cancellations     = ('Is_Cancelled', 'sum'),
    Total_Rows      = ('Invoice', 'count'),
    Product_Diversity = ('StockCode', 'nunique')
).reset_index()

country_agg['Cancellation_Rate']      = (country_agg['Num_Cancellations'] / country_agg['Total_Rows']).round(4)
country_agg['Avg_Basket_Value'] = (country_agg['Total_Revenue'] / country_agg['Num_Orders']).round(2)

print(f"Countries: {len(country_agg)}")
print(country_agg.sort_values('Total_Revenue', ascending=False).head(10).to_string(index=False))

# Scale Country-Level Features
features_sp = ['Total_Revenue','Num_Orders','Num_Customers',
               'Cancellation_Rate','Avg_Basket_Value','Product_Diversity']

scaler_sp = StandardScaler()
X_sp = scaler_sp.fit_transform(country_agg[features_sp])
X_sp_df = pd.DataFrame(X_sp, columns=features_sp)

print("Scaled features (first 5 countries):")
print(X_sp_df.head())

# Determine Optimal Number of Clusters
# Elbow
wcss_sp = []
for k in range(1, min(11, len(country_agg))):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_sp)
    wcss_sp.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(wcss_sp)+1), wcss_sp, marker='o', linewidth=2)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.title('Elbow Method — Service Performance (Country-Level)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Silhouette
sil_sp = []
max_k = min(10, len(country_agg)-1)
k_range_sp = range(2, max_k+1)
for k in k_range_sp:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_sp)
    sil_sp.append(silhouette_score(X_sp, labels))

plt.figure(figsize=(8, 4))
plt.plot(k_range_sp, sil_sp, marker='o', color='darkorange', linewidth=2)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method — Service Performance (Country-Level)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

best_ksp = list(k_range_sp)[sil_sp.index(max(sil_sp))]
print(f"Optimal k: {best_ksp}  (silhouette: {max(sil_sp):.3f})")

# Apply K-Means Clustering
km_sp = KMeans(n_clusters=3, random_state=42, n_init=10)
country_agg['KM_Cluster'] = km_sp.fit_predict(X_sp)

sp_profile = country_agg.groupby('KM_Cluster')[features_sp].mean().round(2)
sp_profile['Country_Count'] = country_agg.groupby('KM_Cluster').size()
print("K-Means Service Performance Cluster Profiles:")
print(sp_profile.to_string())

# Visualise — PCA projection
pca_sp = PCA(n_components=2, random_state=42)
X_sp_pca = pca_sp.fit_transform(X_sp)
explained_sp = pca_sp.explained_variance_ratio_

colors_sp = ['#2ecc71','#e74c3c','#3498db']
fig, ax = plt.subplots(figsize=(10, 7))
for c in range(3):
    mask = country_agg['KM_Cluster'] == c
    ax.scatter(X_sp_pca[mask, 0], X_sp_pca[mask, 1],
               c=colors_sp[c], s=100, alpha=0.8, label=f'Cluster {c}')
    for _, row in country_agg[mask].iterrows():
        idx = country_agg.index.get_loc(row.name)
        ax.annotate(row['Country'], (X_sp_pca[idx,0], X_sp_pca[idx,1]),
                    fontsize=7, ha='center', va='bottom')

ax.set_xlabel(f'PC1 ({explained_sp[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({explained_sp[1]*100:.1f}% variance)')
ax.set_title('K-Means Country Service Performance Clusters (PCA)')
ax.legend()
plt.tight_layout()
plt.show()

# Hierarchical Clustering on Service Performance
Z_sp = linkage(X_sp, method='ward')

plt.figure(figsize=(14, 5))
dendrogram(Z_sp,
           labels=country_agg['Country'].values,
           leaf_rotation=90, leaf_font_size=8,
           color_threshold=0.5 * max(Z_sp[:,2]))
plt.title('Hierarchical Clustering Dendrogram — Country Service Performance (Ward)', fontsize=13)
plt.ylabel('Ward Distance')
plt.axhline(y=sorted(Z_sp[:,2])[-2], color='red', linestyle='--',
            linewidth=1.5, label='Cut for 3 clusters')
plt.legend()
plt.tight_layout()
plt.show()

agg_sp = AgglomerativeClustering(n_clusters=3, linkage='ward')
country_agg['HC_Cluster'] = agg_sp.fit_predict(X_sp)

hc_sp_profile = country_agg.groupby('HC_Cluster')[features_sp].mean().round(2)
hc_sp_profile['Country_Count'] = country_agg.groupby('HC_Cluster').size()
print("Hierarchical Clustering — Service Performance Profiles:")
print(hc_sp_profile.to_string())

# Side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, col, title in zip(axes,
                           ['KM_Cluster','HC_Cluster'],
                           ['K-Means (k=3)','Hierarchical / Ward (k=3)']):
    for c in range(3):
        mask = country_agg[col] == c
        ax.scatter(X_sp_pca[mask, 0], X_sp_pca[mask, 1],
                   c=colors_sp[c], s=80, alpha=0.8, label=f'Cluster {c}')
        for _, row in country_agg[mask].iterrows():
            idx = country_agg.index.get_loc(row.name)
            ax.annotate(row['Country'], (X_sp_pca[idx,0], X_sp_pca[idx,1]),
                        fontsize=6, ha='center', va='bottom')
    ax.set_title(f'Service Performance — {title}')
    ax.set_xlabel(f'PC1 ({explained_sp[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({explained_sp[1]*100:.1f}%)')
    ax.legend()

plt.suptitle('Country Service Performance: K-Means vs Hierarchical', fontsize=13)
plt.tight_layout()
plt.show()

ari_sp = adjusted_rand_score(country_agg['KM_Cluster'], country_agg['HC_Cluster'])
print(f"Adjusted Rand Index: {ari_sp:.3f}")

# Country list per cluster
print("Countries by K-Means Service Performance Cluster:\n")
for c in range(3):
    countries = country_agg[country_agg['KM_Cluster']==c]['Country'].tolist()
    print(f"Cluster {c} ({len(countries)} countries): {', '.join(sorted(countries))}")
    print()
    
