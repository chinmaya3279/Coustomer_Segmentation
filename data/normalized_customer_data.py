import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("indian_customer_segmentation_detailed.csv")

# Select relevant numerical features for normalization
features = df[["Income", "Spending_Score", "Recency", "Frequency"]]

# Apply StandardScaler
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Add normalized features back to the dataset
normalized_columns = ["Income_Norm", "Spending_Score_Norm", "Recency_Norm", "Frequency_Norm"]
normalized_df = pd.DataFrame(normalized_features, columns=normalized_columns)
df_normalized = pd.concat([df, normalized_df], axis=1)

# Save the normalized dataset
df_normalized.to_csv("normalized_customer_segmentation.csv", index=False)
print("Normalized dataset saved as 'normalized_customer_segmentation.csv'")
