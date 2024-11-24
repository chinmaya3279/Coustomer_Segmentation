import pandas as pd

# Load the normalized dataset
df = pd.read_csv("normalized_customer_segmentation.csv")

# Preview the data
print(df.head())
print(df.columns)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Select normalized features for clustering
features = df[["Income_Norm", "Spending_Score_Norm", "Recency_Norm", "Frequency_Norm"]]

# Determine the optimal number of clusters using the Elbow Method
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.grid()
plt.show()

# Fit K-Means with the chosen number of clusters (e.g., k=5)
optimal_k = 5  # Adjust based on the Elbow Curve
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df["Cluster"] = kmeans.fit_predict(features)

# Save the dataset with clusters
df.to_csv("customer_segmentation_with_clusters.csv", index=False)
print("Clustered dataset saved as 'customer_segmentation_with_clusters.csv'")
