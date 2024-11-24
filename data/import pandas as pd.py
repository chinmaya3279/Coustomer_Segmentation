import pandas as pd
import numpy as np
import random
from faker import Faker

# Initialize Faker for Indian data
fake = Faker("en_IN")

# Number of customers
num_customers = 5000

# Define all Indian states and their major cities
states_and_cities = {
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Trichy"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Agra"],
    "Karnataka": ["Bengaluru", "Mysuru", "Mangaluru", "Hubli-Dharwad"],
    "West Bengal": ["Kolkata", "Howrah", "Durgapur", "Siliguri"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota"],
    "Punjab": ["Amritsar", "Ludhiana", "Chandigarh", "Patiala"],
    "Kerala": ["Kochi", "Thiruvananthapuram", "Kozhikode", "Thrissur"],
    "Delhi NCR": ["New Delhi", "Noida", "Gurugram", "Ghaziabad"]
}

# Income brackets
income_brackets = {
    "Low Income": (120000, 300000),
    "Middle Income": (300000, 1200000),
    "High Income": (1200000, 5000000)
}

# Spending categories
categories = ["Electronics", "Groceries", "Clothing", "Healthcare", "Entertainment", "Education", "Technology", "Travel", "Food & Beverages", "Luxury Goods"]

# Generate the dataset
data = []
for _ in range(num_customers):
    state = random.choice(list(states_and_cities.keys()))
    city = random.choice(states_and_cities[state])
    income_group = random.choice(list(income_brackets.keys()))
    income = random.randint(*income_brackets[income_group])
    spending_score = random.randint(1, 100)
    recency = random.randint(0, 365)
    total_transactions = random.randint(1, 500)
    total_spent = total_transactions * random.randint(100, 1000)
    frequency = random.randint(1, 50)
    family_size = random.randint(1, 8)
    clv = total_spent * random.uniform(0.1, 0.5)
    
    data.append({
        "Customer_ID": fake.uuid4()[:8],
        "Name": fake.name(),
        "Age": random.randint(18, 70),
        "Gender": random.choice(["Male", "Female", "Other"]),
        "State": state,
        "City": city,
        "Income": income,
        "Income_Group": income_group,
        "Education": random.choice(["Undergraduate", "Postgraduate", "Diploma", "Others"]),
        "Profession": random.choice(["IT Professional", "Business Owner", "Government Employee", "Homemaker", "Student"]),
        "Marital_Status": random.choice(["Single", "Married", "Divorced"]),
        "Family_Size": family_size,
        "Spending_Score": spending_score,
        "Last_Purchase_Date": fake.date_between(start_date="-1y", end_date="today"),
        "Total_Transactions": total_transactions,
        "Total_Spent": total_spent,
        "Recency": recency,
        "Frequency": frequency,
        "Preferred_Category": random.choice(categories),
        "Mobile_App_Usage (mins/day)": random.randint(0, 120),
        "Website_Activity (mins/day)": random.randint(0, 120),
        "Customer_Lifetime_Value": round(clv, 2),
    })

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("indian_customer_segmentation_detailed.csv", index=False)
print("Detailed dataset created and saved as 'indian_customer_segmentation_detailed.csv'")
