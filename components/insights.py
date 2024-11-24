import pandas as pd

def cluster_insights(df):
    # Calculate average metrics for each cluster
    cluster_summary = df.groupby("Cluster").agg({
        "Income": "mean",
        "Spending_Score": "mean",
        "Age": "mean",
        "Recency": "mean",
    }).reset_index()

    cluster_summary.columns = ["Cluster", "Avg Income", "Avg Spending", "Avg Age", "Avg Recency"]
    return cluster_summary
