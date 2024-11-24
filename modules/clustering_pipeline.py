# clustering_pipeline.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class ClusteringPipeline:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_normalize(self):
        df = pd.read_csv(self.file_path)
        features = df[["Income", "Spending_Score", "Recency", "Frequency"]]
        scaler = StandardScaler()
        self.scaled_features = scaler.fit_transform(features)
        self.df = df

    def apply_clustering(self, k=5):
        kmeans = KMeans(n_clusters=k, random_state=42)
        self.df["Cluster"] = kmeans.fit_predict(self.scaled_features)
        return self.df

# Usage:
# pipeline = ClusteringPipeline("indian_customer_segmentation.csv")
# pipeline.load_and_normalize()
# clustered_data = pipeline.apply_clustering(k=5)
