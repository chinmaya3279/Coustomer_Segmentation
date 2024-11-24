def search_customer(df, search_term):
    # Search by name or customer ID
    result = df[df["Customer_Name"].str.contains(search_term, case=False, na=False)]
    return result
