# # app.py
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# # from clustering_pipeline import ClusteringPipeline
# from modules.clustering_pipeline import ClusteringPipeline
# # import sys
# # sys.path.append("/path/to/clustering_pipeline_directory")
# # from modules.clustering_pipeline import ClusteringPipeline



# # Load data
# file_path = (r"C:\\Program Files\\Git\\Coding\\projects\\Customer Segmentation\\customer_segmentation_with_clusters.csv")
# pipeline = ClusteringPipeline(file_path)

# # pipeline = ClusteringPipeline("indian_customer_segmentation.csv")
# pipeline.load_and_normalize()
# df = pipeline.apply_clustering(k=5)

# # Sidebar for filters
# st.sidebar.header("Filter Customers")
# cluster = st.sidebar.selectbox("Select Cluster", sorted(df["Cluster"].unique()))
# city = st.sidebar.selectbox("Select City", sorted(df["City"].unique()))

# filtered_df = df[(df["Cluster"] == cluster) & (df["City"] == city)]

# # Display filtered data
# st.title("Customer Segmentation Dashboard")
# st.write(f"Showing customers in Cluster {cluster} from {city}")
# st.dataframe(filtered_df)

# # Visualize clusters
# fig = px.scatter(
#     df, x="Income", y="Spending_Score", color="Cluster",
#     hover_data=["City", "Age", "Profession"]
# )
# st.plotly_chart(fig)

# # Cluster insights
# st.subheader("Cluster Insights")
# cluster_summary = df.groupby("Cluster").mean()[["Income", "Spending_Score"]]
# st.table(cluster_summary)

# # Export segmented data
# st.download_button(
#     "Download Filtered Data",
#     data=filtered_df.to_csv(index=False).encode("utf-8"),
#     file_name="filtered_customers.csv",
#     mime="text/csv"
# )


# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json


# Load dataset
df = pd.read_csv("data\indian_customer_segmentation_detailed.csv")
# df = pd.read_csv("data\customer_segmentation_with_clusters.csv")

# Load city coordinates
coordinates_path = "./data/city.js"
# with open(coordinates_path, "r") as file:
#     city_coordinates = json.load(file)

try:
    with open("data\city.js", "r") as file:
        city_coordinates = json.load(file)
except FileNotFoundError:
    print("Error: data\city.js file not found.")
    city_coordinates = {}
except json.JSONDecodeError:
    print("Error: Invalid JSON in data\city.js file.")
    city_coordinates = {}



# Sidebar inputs for clustering parameters
st.sidebar.header("Dynamic Clustering Settings")
num_clusters = st.sidebar.slider("Select Number of Clusters (k)", min_value=2, max_value=10, value=5)
selected_city = st.sidebar.selectbox("Filter by City", ["All"] + sorted(df["City"].unique()))

# Normalize relevant features
scaler = StandardScaler()
features = df[["Income", "Spending_Score", "Recency", "Frequency"]]
scaled_features = scaler.fit_transform(features)

# Apply K-Means with dynamic cluster count
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_features)

# Filter data by city if specified
if selected_city != "All":
    filtered_df = df[df["City"] == selected_city]
else:
    filtered_df = df

# Display filtered data
st.title("Dynamic Customer Segmentation Dashboard")
st.write(f"Showing {len(filtered_df)} customers for city: {selected_city}")

st.dataframe(filtered_df)

# Generate dynamic cluster graph
fig = px.scatter(
    filtered_df,
    x="Income",
    y="Spending_Score",
    color="Cluster",
    hover_data=["City", "Age", "Profession"],
    title=f"Customer Clusters with {num_clusters} Clusters",
    labels={"Income": "Annual Income", "Spending_Score": "Spending Score"},
    template="plotly_white"
)
st.plotly_chart(fig)

# Export segmented data
st.download_button(
    "Download Filtered Data",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name=f"filtered_customers_{selected_city}.csv",
    mime="text/csv"
)

# Cluster-specific insights
# st.subheader("Cluster-Specific Insights")
# cluster_summary = filtered_df.groupby("Cluster").mean()[["Income", "Spending_Score", "Recency", "Frequency"]]
# st.table(cluster_summary)

from components.insights import cluster_insights

st.subheader("Cluster Insights")
cluster_summary = cluster_insights(df)
st.write(cluster_summary)


# Age distribution by cluster
st.subheader("Age Distribution by Cluster")
age_fig = px.histogram(
    filtered_df, x="Age", color="Cluster",
    title="Age Distribution by Cluster",
    template="plotly_white"
)
st.plotly_chart(age_fig)

# Spending Score vs. Income Heatmap
st.subheader("Spending Score vs. Income")
heatmap = px.density_heatmap(
    filtered_df, x="Income", y="Spending_Score",
    color_continuous_scale="Viridis",
    title="Spending Score vs. Income Heatmap",
    template="plotly_white"
)
st.plotly_chart(heatmap)


# Add latitude/longitude mapping for cities (preprocess this in your dataset)
city_coordinates = {
    "Delhi": [28.6139, 77.2090],
    "Mumbai": [19.0760, 72.8777],
    # Add more cities...
}
# filtered_df["Latitude"] = filtered_df["City"].map(lambda x: city_coordinates[x][0])
# filtered_df["Longitude"] = filtered_df["City"].map(lambda x: city_coordinates[x][1])

# Safely map latitude and longitude
filtered_df["Latitude"] = filtered_df["City"].map(lambda x: city_coordinates.get(x, [None, None])[0])
filtered_df["Longitude"] = filtered_df["City"].map(lambda x: city_coordinates.get(x, [None, None])[1])

# Fill missing coordinates with defaults
filtered_df["Latitude"].fillna(0, inplace=True)
filtered_df["Longitude"].fillna(0, inplace=True)


# Geo-visualization
st.subheader("Customer Distribution on Map")
map_fig = px.scatter_mapbox(
    filtered_df, lat="Latitude", lon="Longitude", color="Cluster",
    hover_data=["City", "Income", "Spending_Score"],
    mapbox_style="carto-positron",
    title="Customer Clusters on Map"
)
st.plotly_chart(map_fig)


from components.visualization import geographical_distribution

st.subheader("Customer Geographical Distribution")
# geo_fig = geographical_distribution(df)
# st.plotly_chart(geo_fig)

# Map latitude and longitude
df["Latitude"] = df["City"].map(lambda x: city_coordinates.get(x, [None, None])[0])
df["Longitude"] = df["City"].map(lambda x: city_coordinates.get(x, [None, None])[1])

# Generate the plot
geo_fig = px.scatter_geo(
    filtered_df,
    lat="Latitude",
    lon="Longitude",
    hover_name="City",
    size="Spending_Score",
    color="Cluster",
    title="Customer Geographical Distribution"
)

# Display the plot in Streamlit
st.plotly_chart(geo_fig)


from components.search import search_customer

st.sidebar.subheader("Search Customers")
search_term = st.sidebar.text_input("Enter customer name or ID")

if search_term:
    search_results = search_customer(df, search_term)
    st.subheader("Search Results")
    st.dataframe(search_results)


from components.visualization import spending_trend_over_time



st.subheader("Spending Trends Over Time")

import pandas as pd
import plotly.express as px
import streamlit as st

# Sample function to get trend over time for spending (using `Last_Purchase_Date`)
def spending_trend_over_time(df, city=None, customer_id=None):
    # Extract the 'Month' column from 'Last_Purchase_Date' if it doesn't exist
    if 'Month' not in df.columns:
        df['Month'] = pd.to_datetime(df['Last_Purchase_Date']).dt.month

    # Calculate the main trend: average spending score over time (by month)
    trend = df.groupby("Month").agg({"Spending_Score": "mean"}).reset_index()

    # Initialize Plotly figure with the main trend line
    fig = px.line(
        trend,
        x="Month",
        y="Spending_Score",
        title="Spending Trends Over Time",
        labels={"Month": "Month", "Spending_Score": "Average Spending Score"},
    )

    # If a city is selected, filter the data by city and plot the city's spending trend
    if city:
        city_data = df[df['City'] == city]
        city_trend = city_data.groupby("Month").agg({"Spending_Score": "mean"}).reset_index()
        fig.add_scatter(
            x=city_trend["Month"],
            y=city_trend["Spending_Score"],
            mode='lines',
            name=f"City: {city}",
            line=dict(dash='dot', width=2)
        )

    # If a customer is selected, filter the data by customer and plot the customer's spending trend
    if customer_id:
        customer_data = df[df['Customer_ID'] == customer_id]
        customer_trend = customer_data.groupby("Month").agg({"Spending_Score": "mean"}).reset_index()
        fig.add_scatter(
            x=customer_trend["Month"],
            y=customer_trend["Spending_Score"],
            mode='lines',
            name=f"Customer: {customer_id}",
            line=dict(color='green', width=2, dash='dash')
        )

    return fig

# Streamlit UI
st.title("Spending Trends Over Time")

# Dropdown to select City
city = st.selectbox('Select City', df['City'].unique())

# Dropdown to select Customer ID
customer_id = st.selectbox('Select Customer', df['Customer_ID'].unique())

# Plot the combined trend
trend_fig = spending_trend_over_time(df, city=city, customer_id=customer_id)

# Show the trend figure
st.plotly_chart(trend_fig)


import plotly.io as pio

# Save graph as PDF or PNG
if st.button("Export Geographical Distribution"):
    pio.write_image(geo_fig, "assets/geographical_distribution.pdf")
    st.success("Exported as assets/geographical_distribution.pdf")



# Login System
# if "authenticated" not in st.session_state:
#     st.session_state["authenticated"] = False

# if not st.session_state["authenticated"]:
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     if username == "admin" and password == "password123":
#         st.session_state["authenticated"] = True
#         st.success("Login Successful!")
#     else:
#         st.error("Invalid Credentials!")
#         st.stop()


# Dynamic range sliders
st.sidebar.header("Dynamic Filters")
income_range = st.sidebar.slider("Income Range", int(df["Income"].min()), int(df["Income"].max()), (10000, 5000000))
age_range = st.sidebar.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (20, 40))

# Apply filters
filtered_df = filtered_df[
    (filtered_df["Income"].between(*income_range)) &
    (filtered_df["Age"].between(*age_range))
]
st.write(f"Filtered {len(filtered_df)} customers based on dynamic filters.")


# Display filter values (dynamic range sliders values)
st.write(f"Income Range: {income_range[0]} - {income_range[1]}")
st.write(f"Age Range: {age_range[0]} - {age_range[1]}")


# **1. Scatter Plot (Income vs Age)**
scatter_fig = px.scatter(
    filtered_df, 
    x="Income", 
    y="Age", 
    color="Gender",  # Color by gender to see how income and age differ for each gender
    title="Income vs Age of Filtered Customers",
    labels={"Income": "Customer Income", "Age": "Customer Age", "Gender": "Customer Gender"}
)

# Show the scatter plot
st.plotly_chart(scatter_fig)

# **2. Income Group Segmentation (Bar Chart)**
# Segregating the customers into income groups (Low, Medium, High)
# Segregating the customers into income groups (Low, Medium, High)
# Define custom bins for the income range
import pandas as pd
import plotly.express as px

# Define bins and labels for income groups
bins = [10000, 500000, 1000000, 2000000, 3000000, 4000000, 5000000]
labels = ['10000-500000', '500001-1000000', '1000001-2000000', '2000001-3000000', '3000001-4000000', '4000001-5000000']

# Apply the bins to the 'Income' column and create a new column for the income group
filtered_df['Income_Group'] = pd.cut(filtered_df['Income'], bins=bins, labels=labels, right=False)

st.write(filtered_df[['Income', 'Income_Group']].head())
st.write(filtered_df['Income_Group'].unique())


# Ensure all groups are represented, even if empty
income_group_counts = filtered_df["Income_Group"].value_counts().reindex(labels, fill_value=0).reset_index()
income_group_counts.columns = ["Income_Group", "Count"]

# Ensure that all income values are within the defined bins
# st.write(filtered_df['Income_Group'].unique())

# Plot the distribution of customers in each income group

# Generate bar chart
income_group_fig = px.bar(
    income_group_counts,
    x="Income_Group",
    y="Count",
    title="Income Group Distribution of Filtered Customers",
    labels={"Income_Group": "Income Group", "Count": "Number of Customers"},
    category_orders={"Income_Group": labels}
)

# Show the plot
st.plotly_chart(income_group_fig)



# **3. Spending vs Income (Scatter Plot)**
# For additional insights into customer behavior
spending_income_fig = px.scatter(
    filtered_df, 
    x="Income", 
    y="Spending_Score", 
    color="Income_Group",  # Color by income group for insights
    title="Income vs Spending Score of Filtered Customers",
    labels={"Income": "Customer Income", "Spending_Score": "Customer Spending Score"}
)

# Show the scatter plot for income vs spending score
st.plotly_chart(spending_income_fig)

# **4. Customer Segmentation by Spending Behavior (Pie Chart)**
# Segmentation based on Spending Score
spending_score_bins = [0, 30, 60, 100]
spending_labels = ['Low Spending', 'Medium Spending', 'High Spending']
filtered_df['Spending_Score_Group'] = pd.cut(filtered_df['Spending_Score'], bins=spending_score_bins, labels=spending_labels)

spending_score_pie_fig = px.pie(
    filtered_df['Spending_Score_Group'].value_counts().reset_index(),
    names='index',
    values='Spending_Score_Group',
    title="Customer Segmentation Based on Spending Score",
    labels={"index": "Spending Group", "Spending_Score_Group": "Number of Customers"}
)

# Show the pie chart for customer spending score segmentation
st.plotly_chart(spending_score_pie_fig)