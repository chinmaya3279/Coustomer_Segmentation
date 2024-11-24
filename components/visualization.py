import plotly.express as px

def geographical_distribution(df):
    # Generate a map visualization
    fig = px.scatter_geo(
        df,
        locations="State",
        locationmode="country names",
        hover_name="City",
        size="Spending_Score",
        title="Geographical Distribution of Customers",
        template="plotly_dark"
    )


def spending_trend_over_time(df):
    # Example: Assume a "Month" column exists
    trend = df.groupby("Month").agg({"Spending_Score": "mean"}).reset_index()
    fig = px.line(trend, x="Month", y="Spending_Score", title="Spending Trends Over Time")
    return fig
