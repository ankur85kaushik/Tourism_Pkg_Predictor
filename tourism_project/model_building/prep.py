# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("tourism_project/data/tourism.csv")
print("Dataset loaded successfully.")

# Drop the index column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Drop the unique identifier CustomerID
df.drop(columns=['CustomerID'], inplace=True)

# CHECK AND DROP DUPLICATES (Now catches the 117 identical profiles)
duplicate_count = df.duplicated().sum()
print(f"Dropping {duplicate_count} exact duplicate customer profiles...")
df = df.drop_duplicates(keep='first')

# CHECK FOR NULL VALUES (Optional check print, though count is 0)
null_count = df.isnull().sum().sum()
print(f"Total null values remaining in dataset: {null_count}")


# Replace the Fe Male to Female and merge Unmarried and single
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')


# 4. Create 5-year Age Groups( Age min is 18 and Age max is 65)
bins = [18, 24, 29, 34, 39, 44, 49, 54, 59, 65]
labels = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-65']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True).astype(str)
df = df.drop(columns=['Age'])

# Separate columns for Label Encoding vs One-Hot Encoding
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
binary_cols = []
non_binary_cols = []

for col in categorical_cols:
    if df[col].nunique() == 2:
        binary_cols.append(col)
    else:
        non_binary_cols.append(col)

# Apply LabelEncoder for binary fields (0 or 1)
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# Apply One-Hot Encoding for multi-category fields
df = pd.get_dummies(df, columns=non_binary_cols, dtype=int)

target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("tourism_project/data/Xtrain.csv",index=False)
Xtest.to_csv("tourism_project/data/Xtest.csv",index=False)
ytrain.to_csv("tourism_project/data/ytrain.csv",index=False)
ytest.to_csv("tourism_project/data/ytest.csv",index=False)

print("Data prepared: train/test splits written.")
