# daghan og error

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from tabulate import tabulate
from collections import Counter


def clean_data(df):
    # Removing rows with missing values in 'CustomerID' and 'Description' columns
    df = df.dropna(subset=['CustomerID', 'Description'])
    
    # Removing duplicate rows
    df.drop_duplicates(inplace=True)

    # Filter out the rows with InvoiceNo starting with "C" and create a new column indicating the transaction status
    df['Transaction_Status'] = np.where(df['InvoiceNo'].astype(str).str.startswith('C'), 'Cancelled', 'Completed')

    # Finding the number of unique stock codes
    # unique_stock_codes = df['StockCode'].nunique()

    # Finding the number of numeric characters in each unique stock code
    unique_stock_codes = df['StockCode'].unique()

    # Finding and printing the stock codes with 0 and 1 numeric characters
    anomalous_stock_codes = [code for code in unique_stock_codes if sum(c.isdigit() for c in str(code)) in (0, 1)]

    # Removing rows with anomalous stock codes from the dataset
    df = df[~df['StockCode'].isin(anomalous_stock_codes)]

    service_related_descriptions = ["Next Day Carriage", "High Resolution Image"]
    # Calculate the percentage of records with service-related descriptions
    service_related_percentage = df[df['Description'].isin(service_related_descriptions)].shape[0] / df.shape[0] * 100
    # Remove rows with service-related information in the description
    df = df[~df['Description'].isin(service_related_descriptions)]

    # Standardize the text to uppercase to maintain uniformity across the dataset
    df['Description'] = df['Description'].str.upper()

    # Removing records with a unit price of zero to avoid potential data entry errors
    df = df[df['UnitPrice'] > 0]

    # Resetting the index of the cleaned dataset
    df.reset_index(drop=True, inplace=True)
    
    return df


def feature_engineer(df):
    # Convert InvoiceDate to datetime type
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Convert InvoiceDate to datetime and extract only the date
    df['InvoiceDay'] = df['InvoiceDate'].dt.date

    # Find the most recent purchase date for each customer
    customer_data = df.groupby('CustomerID')['InvoiceDay'].max().reset_index()

    # Find the most recent date in the entire dataset
    most_recent_date = df['InvoiceDay'].max()

    # Convert InvoiceDay to datetime type before subtraction
    customer_data['InvoiceDay'] = pd.to_datetime(customer_data['InvoiceDay'])
    most_recent_date = pd.to_datetime(most_recent_date)

    # Calculate the number of days since the last purchase for each customer
    customer_data['Days_Since_Last_Purchase'] = (most_recent_date - customer_data['InvoiceDay']).dt.days

    # Remove the InvoiceDay column
    customer_data.drop(columns=['InvoiceDay'], inplace=True)

    # Calculate the total number of transactions made by each customer
    total_transactions = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
    total_transactions.rename(columns={'InvoiceNo': 'Total_Transactions'}, inplace=True)

    # Calculate the total number of products purchased by each customer
    total_products_purchased = df.groupby('CustomerID')['Quantity'].sum().reset_index()
    total_products_purchased.rename(columns={'Quantity': 'Total_Products_Purchased'}, inplace=True)

    # Merge the new features into the customer_data dataframe
    customer_data = pd.merge(customer_data, total_transactions, on='CustomerID')
    customer_data = pd.merge(customer_data, total_products_purchased, on='CustomerID')

    # Calculate the total spend by each customer
    df['Total_Spend'] = df['UnitPrice'] * df['Quantity']
    total_spend = df.groupby('CustomerID')['Total_Spend'].sum().reset_index()

    # Calculate the average transaction value for each customer
    average_transaction_value = total_spend.merge(total_transactions, on='CustomerID')
    average_transaction_value['Average_Transaction_Value'] = average_transaction_value['Total_Spend'] / average_transaction_value['Total_Transactions']

    # Merge the new features into the customer_data dataframe
    customer_data = pd.merge(customer_data, total_spend, on='CustomerID')
    customer_data = pd.merge(customer_data, average_transaction_value[['CustomerID', 'Average_Transaction_Value']], on='CustomerID')

    # Calculate the number of unique products purchased by each customer
    unique_products_purchased = df.groupby('CustomerID')['StockCode'].nunique().reset_index()
    unique_products_purchased.rename(columns={'StockCode': 'Unique_Products_Purchased'}, inplace=True)

    # Merge the new feature into the customer_data dataframe
    customer_data = pd.merge(customer_data, unique_products_purchased, on='CustomerID')

    # Extract day of week and hour from InvoiceDate
    df['Day_Of_Week'] = df['InvoiceDate'].dt.dayofweek
    df['Hour'] = df['InvoiceDate'].dt.hour

    # Calculate the average number of days between consecutive purchases
    days_between_purchases = df.groupby('CustomerID')['InvoiceDay'].apply(lambda x: (x.diff().dropna()).apply(lambda y: y.days))
    average_days_between_purchases = days_between_purchases.groupby('CustomerID').mean().reset_index()
    average_days_between_purchases.rename(columns={'InvoiceDay': 'Average_Days_Between_Purchases'}, inplace=True)

    # Find the favorite shopping day of the week
    favorite_shopping_day = df.groupby(['CustomerID', 'Day_Of_Week']).size().reset_index(name='Count')
    favorite_shopping_day = favorite_shopping_day.loc[favorite_shopping_day.groupby('CustomerID')['Count'].idxmax()][['CustomerID', 'Day_Of_Week']]

    # Find the favorite shopping hour of the day
    favorite_shopping_hour = df.groupby(['CustomerID', 'Hour']).size().reset_index(name='Count')
    favorite_shopping_hour = favorite_shopping_hour.loc[favorite_shopping_hour.groupby('CustomerID')['Count'].idxmax()][['CustomerID', 'Hour']]

    # Merge the new features into the customer_data dataframe
    customer_data = pd.merge(customer_data, average_days_between_purchases, on='CustomerID')
    customer_data = pd.merge(customer_data, favorite_shopping_day, on='CustomerID')
    customer_data = pd.merge(customer_data, favorite_shopping_hour, on='CustomerID')

    # Group by CustomerID and Country to get the number of transactions per country for each customer
    customer_country = df.groupby(['CustomerID', 'Country']).size().reset_index(name='Number_of_Transactions')

    # Get the country with the maximum number of transactions for each customer (in case a customer has transactions from multiple countries)
    customer_main_country = customer_country.sort_values('Number_of_Transactions', ascending=False).drop_duplicates('CustomerID')

    # Create a binary column indicating whether the customer is from the UK or not
    customer_main_country['Is_UK'] = customer_main_country['Country'].apply(lambda x: 1 if x == 'United Kingdom' else 0)

    # Merge this data with our customer_data dataframe
    customer_data = pd.merge(customer_data, customer_main_country[['CustomerID', 'Is_UK']], on='CustomerID', how='left')

    # Calculate the total number of transactions made by each customer
    total_transactions = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()

    # Calculate the number of cancelled transactions for each customer
    cancelled_transactions = df[df['Transaction_Status'] == 'Cancelled']
    cancellation_frequency = cancelled_transactions.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
    cancellation_frequency.rename(columns={'InvoiceNo': 'Cancellation_Frequency'}, inplace=True)

    # Merge the Cancellation Frequency data into the customer_data dataframe
    customer_data = pd.merge(customer_data, cancellation_frequency, on='CustomerID', how='left')

    # Replace NaN values with 0 (for customers who have not cancelled any transaction)
    customer_data['Cancellation_Frequency'].fillna(0, inplace=True)

    # Calculate the Cancellation Rate
    customer_data['Cancellation_Rate'] = customer_data['Cancellation_Frequency'] / total_transactions['InvoiceNo']

    # Display the first few rows of the customer_data dataframe
    customer_data.head()

    # Extract month and year from InvoiceDate
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month

    # Calculate monthly spending for each customer
    monthly_spending = df.groupby(['CustomerID', 'Year', 'Month'])['Total_Spend'].sum().reset_index()

    # Calculate Seasonal Buying Patterns: We are using monthly frequency as a proxy for seasonal buying patterns
    seasonal_buying_patterns = monthly_spending.groupby('CustomerID')['Total_Spend'].agg(['mean', 'std']).reset_index()
    seasonal_buying_patterns.rename(columns={'mean': 'Monthly_Spending_Mean', 'std': 'Monthly_Spending_Std'}, inplace=True)

    # Replace NaN values in Monthly_Spending_Std with 0, implying no variability for customers with single transaction month
    seasonal_buying_patterns['Monthly_Spending_Std'].fillna(0, inplace=True)

    # Calculate Trends in Spending 
    # We are using the slope of the linear trend line fitted to the customer's spending over time as an indicator of spending trends
    def calculate_trend(spend_data):
        # If there are more than one data points, we calculate the trend using linear regression
        if len(spend_data) > 1:
            x = np.arange(len(spend_data))
            slope, _, _, _, _ = linregress(x, spend_data)
            return slope
        # If there is only one data point, no trend can be calculated, hence we return 0
        else:
            return 0

    # Apply the calculate_trend function to find the spending trend for each customer
    spending_trends = monthly_spending.groupby('CustomerID')['Total_Spend'].apply(calculate_trend).reset_index()
    spending_trends.rename(columns={'Total_Spend': 'Spending_Trend'}, inplace=True)

    # Merge the new features into the customer_data dataframe
    customer_data = pd.merge(customer_data, seasonal_buying_patterns, on='CustomerID')
    customer_data = pd.merge(customer_data, spending_trends, on='CustomerID')

    return df, customer_data


def fix_outlier(customer_data):
    # Initializing the IsolationForest model with a contamination parameter of 0.05
    model = IsolationForest(contamination=0.05, random_state=0)

    # Fitting the model on our dataset (converting DataFrame to NumPy to avoid warning)
    customer_data['Outlier_Scores'] = model.fit_predict(customer_data.iloc[:, 1:].to_numpy())

    # Creating a new column to identify outliers (1 for inliers and -1 for outliers)
    customer_data['Is_Outlier'] = [1 if x == -1 else 0 for x in customer_data['Outlier_Scores']]

    # Separate the outliers for analysis
    outliers_data = customer_data[customer_data['Is_Outlier'] == 1]

    # Remove the outliers from the main dataset
    customer_data_cleaned = customer_data[customer_data['Is_Outlier'] == 0]

    # Drop the 'Outlier_Scores' and 'Is_Outlier' columns
    customer_data_cleaned = customer_data_cleaned.drop(columns=['Outlier_Scores', 'Is_Outlier'])

    # Reset the index of the cleaned data
    customer_data_cleaned.reset_index(drop=True, inplace=True)

    return customer_data_cleaned, outliers_data


def feature_scale(customer_data_cleaned):
    # Initialize the StandardScaler
    scaler = StandardScaler()

    # List of columns that don't need to be scaled
    columns_to_exclude = ['CustomerID', 'Is_UK', 'Day_Of_Week']

    # List of columns that need to be scaled
    columns_to_scale = customer_data_cleaned.columns.difference(columns_to_exclude)

    # Copy the cleaned dataset
    customer_data_scaled = customer_data_cleaned.copy()

    # Applying the scaler to the necessary columns in the dataset
    customer_data_scaled[columns_to_scale] = scaler.fit_transform(customer_data_scaled[columns_to_scale])

    return customer_data_scaled


def dimensionality_reduction(customer_data_scaled):
    # Setting CustomerID as the index column
    customer_data_scaled.set_index('CustomerID', inplace=True)

    # Apply PCA
    pca = PCA().fit(customer_data_scaled)

    # Creating a PCA object with 6 components
    pca = PCA(n_components=6)

    # Fitting and transforming the original data to the new PCA dataframe
    customer_data_pca = pca.fit_transform(customer_data_scaled)

    # Creating a new dataframe from the PCA dataframe, with columns labeled PC1, PC2, etc.
    customer_data_pca = pd.DataFrame(customer_data_pca, columns=['PC'+str(i+1) for i in range(pca.n_components_)])

    # Adding the CustomerID index back to the new PCA dataframe
    customer_data_pca.index = customer_data_scaled.index

    return customer_data_pca


def kmeans_clustering(customer_data_cleaned, customer_data_pca):
    # Apply KMeans clustering using the optimal k
    kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=100, random_state=0)
    kmeans.fit(customer_data_pca)

    # Get the frequency of each cluster
    cluster_frequencies = Counter(kmeans.labels_)

    # Create a mapping from old labels to new labels based on frequency
    label_mapping = {label: new_label for new_label, (label, _) in 
                    enumerate(cluster_frequencies.most_common())}

    # Reverse the mapping to assign labels as per your criteria
    label_mapping = {v: k for k, v in {2: 1, 1: 0, 0: 2}.items()}

    # Apply the mapping to get the new labels
    new_labels = np.array([label_mapping[label] for label in kmeans.labels_])

    # Append the new cluster labels back to the original dataset
    customer_data_cleaned['cluster'] = new_labels

    # Append the new cluster labels to the PCA version of the dataset
    customer_data_pca['cluster'] = new_labels

    return customer_data_cleaned, customer_data_pca


def recommendation_system(customer_data_cleaned, outliers_data):
    # Step 1: Extract the CustomerIDs of the outliers and remove their transactions from the main dataframe
    outlier_customer_ids = outliers_data['CustomerID'].astype('float').unique()
    df_filtered = df[~df['CustomerID'].isin(outlier_customer_ids)]

    # Step 2: Ensure consistent data type for CustomerID across both dataframes before merging
    customer_data_cleaned['CustomerID'] = customer_data_cleaned['CustomerID'].astype('float')

    # Step 3: Merge the transaction data with the customer data to get the cluster information for each transaction
    merged_data = df_filtered.merge(customer_data_cleaned[['CustomerID', 'cluster']], on='CustomerID', how='inner')

    # Step 4: Identify the top 10 best-selling products in each cluster based on the total quantity sold
    best_selling_products = merged_data.groupby(['cluster', 'StockCode', 'Description'])['Quantity'].sum().reset_index()
    best_selling_products = best_selling_products.sort_values(by=['cluster', 'Quantity'], ascending=[True, False])
    top_products_per_cluster = best_selling_products.groupby('cluster').head(10)

    # Step 5: Create a record of products purchased by each customer in each cluster
    customer_purchases = merged_data.groupby(['CustomerID', 'cluster', 'StockCode'])['Quantity'].sum().reset_index()

    # Step 6: Generate recommendations for each customer in each cluster
    recommendations = []
    for cluster in top_products_per_cluster['cluster'].unique():
        top_products = top_products_per_cluster[top_products_per_cluster['cluster'] == cluster]
        customers_in_cluster = customer_data_cleaned[customer_data_cleaned['cluster'] == cluster]['CustomerID']
        
        for customer in customers_in_cluster:
            # Identify products already purchased by the customer
            customer_purchased_products = customer_purchases[(customer_purchases['CustomerID'] == customer) & 
                                                            (customer_purchases['cluster'] == cluster)]['StockCode'].tolist()
            
            # Find top 3 products in the best-selling list that the customer hasn't purchased yet
            top_products_not_purchased = top_products[~top_products['StockCode'].isin(customer_purchased_products)]
            top_3_products_not_purchased = top_products_not_purchased.head(3)
            
            # Append the recommendations to the list
            recommendations.append([customer, cluster] + top_3_products_not_purchased[['StockCode', 'Description']].values.flatten().tolist())

    # Step 7: Create a dataframe from the recommendations list and merge it with the original customer data
    recommendations_df = pd.DataFrame(recommendations, columns=['CustomerID', 'cluster', 'Rec1_StockCode', 'Rec1_Description', \
                                                    'Rec2_StockCode', 'Rec2_Description', 'Rec3_StockCode', 'Rec3_Description'])
    customer_data_with_recommendations = customer_data_cleaned.merge(recommendations_df, on=['CustomerID', 'cluster'], how='right')

    # Display 10 random rows from the customer_data_with_recommendations dataframe
    customer_data_with_recommendations.set_index('CustomerID').iloc[:, -6:].sample(10, random_state=0)

    customer_data_with_recommendations.to_csv("output.csv", index=False)



df = pd.read_csv('data.csv', encoding="ISO-8859-1")

df = clean_data(df)

df_temp, customer_data = feature_engineer(df)

customer_data_cleaned, outliers_data = fix_outlier(customer_data)

customer_data_scaled = feature_scale(customer_data_cleaned)

customer_data_pca = dimensionality_reduction(customer_data_scaled)

customer_data_cleaned, customer_data_pca = kmeans_clustering(customer_data_cleaned, customer_data_pca)