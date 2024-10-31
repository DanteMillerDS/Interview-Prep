# data_processing.py

import pandas as pd

# 1. Loading Data
# Load data from various sources
data_csv = pd.read_csv('file.csv')
data_excel = pd.read_excel('file.xlsx')
# data_sql = pd.read_sql(query, connection) # Uncomment and define query, connection for SQL
data_json = pd.read_json('file.json')

# 2. Exploring and Understanding Data
# View first few rows and get info
print(data_csv.head())
data_csv.info()
print(data_csv.describe())  # Summary statistics for numerical columns
print(data_csv.isnull().sum())  # Check for missing values

# 3. Filtering and Selecting Data
# Select specific columns
selected_data = data_csv[['column1', 'column2']]

# Filter rows based on conditions
filtered_data = data_csv[data_csv['column'] > 10]

# Filter rows based on multiple conditions
filtered_data_multi = data_csv[(data_csv['column1'] > 10) & (data_csv['column2'] == 'value')]

# 4. Handling Missing Data
# Fill missing values with a constant
data_csv['column'].fillna(0, inplace=True)

# Fill missing values with mean, median, or mode
data_csv['column'].fillna(data_csv['column'].mean(), inplace=True)
data_csv['column'].fillna(data_csv['column'].mode()[0], inplace=True)

# Drop rows or columns with missing values
data_csv.dropna(axis=0, inplace=True)  # Drop rows
data_csv.dropna(axis=1, inplace=True)  # Drop columns

# 5. Transforming Data
# Add new calculated columns
data_csv['new_column'] = data_csv['column1'] + data_csv['column2']

# Apply functions to transform columns
data_csv['transformed_column'] = data_csv['column'].apply(lambda x: x*2 if x > 10 else x)

# Rename columns
data_csv.rename(columns={'old_name': 'new_name'}, inplace=True)

# 6. Grouping and Aggregating Data
# Group by a column and aggregate
grouped_data = data_csv.groupby('column1')['column2'].sum().reset_index()

# Multiple aggregations
agg_data = data_csv.groupby('column1').agg({'column2': 'sum', 'column3': 'mean'}).reset_index()

# Using pivot tables for summarizing data
pivot_table = data_csv.pivot_table(index='column1', columns='column2', values='column3', aggfunc='sum')

# 7. Sorting and Ranking
# Sort by one or multiple columns
sorted_data = data_csv.sort_values(by=['column1', 'column2'], ascending=[True, False])

# Rank values
data_csv['rank'] = data_csv['column'].rank()

# 8. Merging and Concatenating Data
# Concatenate rows or columns
combined_data_rows = pd.concat([data_csv, data_excel], axis=0)  # Rows
combined_data_columns = pd.concat([data_csv, data_excel], axis=1)  # Columns

# Merge two datasets based on a common key
# merged_data = pd.merge(data1, data2, on='common_column', how='inner') # Uncomment and define data1, data2

# 9. Handling Duplicates
# Check for duplicates
print(data_csv.duplicated().sum())

# Remove duplicates
data_csv.drop_duplicates(inplace=True)

# 10. Reshaping Data
# Melt (from wide to long)
melted_data = data_csv.melt(id_vars=['column1'], value_vars=['column2', 'column3'],
                            var_name='variable', value_name='value')

# Pivot (from long to wide)
pivoted_data = data_csv.pivot(index='column1', columns='column2', values='column3')

# 11. Window Functions and Rolling Calculations
# Rolling mean
data_csv['rolling_mean'] = data_csv['column'].rolling(window=3).mean()

# Expanding calculations
data_csv['cumulative_sum'] = data_csv['column'].expanding().sum()

# 12. Exporting Processed Data
# Export to CSV
data_csv.to_csv('processed_data.csv', index=False)

# Export to Excel
data_csv.to_excel('processed_data.xlsx', index=False)
