import joblib
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error


df = pd.read_csv("C:\COLLEGE PROJECT\Electricity_Demand_Prediction")

# Changing timestamp feature to datetime object so that pandas can understand this.
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d-%b-%y")

"""
    Step 1: Imputation of missing values.
"""

if int(df['hour'].isna().sum()) > 0:
    # Function to impute hours
    def impute_hours(group):
        # Find how many rows in this group
        n = len(group)
        # Expected hours from 0 up to 23 (or shorter if partial day data)
        expected_hours = list(range(n))
        # Assign hours where missing
        group["hour"] = group["hour"].fillna(pd.Series(expected_hours, index=group.index))
        return group

    # Apply by date
    df = df.groupby(df["Timestamp"]).apply(impute_hours).reset_index(drop=True)
    df[df['hour'].isna()]


df["dayofweek"] = df["Timestamp"].dt.dayofweek  # Monday=0, Sunday=6
df["month"] = df["Timestamp"].dt.month
df["year"] = df["Timestamp"].dt.year
df["dayofyear"] = df["Timestamp"].dt.dayofyear
df["date"] = df["Timestamp"].dt.day

for col in ["Temperature", "Humidity", "Demand"]:
    df[col] = df[col].fillna(df[col].mean())

"""
    Step 2: Check Duplicates
"""

# check the counts of duplicate rows in dataframe
dup_count = df.duplicated().sum()

if dup_count > 0:
  df = df.drop_duplicates() # method to drop duplicates rows

"""
    Step 3: Feature Transformation
"""

def get_day(day_num):
  # transforming number to day name
  day_dict = {
      0: "Mon",
      1: "Tue",
      2: "Wed",
      3: "Thu",
      4: "Fri",
      5: "Sat",
      6: "Sun"
  }
  return day_dict[day_num]

df['dayofweek'] = df['dayofweek'].apply(get_day)
# Converting float to int for 'hour' feature
df['hour'] = df['hour'].astype(int)

    
"""
    Step 4: Feature Selection

    1. Removing Timestamp is recommended because there are already other time features available which can tell us the same information.
    2. Including timestamp will make this problem into Time-series, which is NOT our goal.
"""

df.drop(columns = ['Timestamp', 'dayofyear'], inplace = True) # inplace = True, will internally update the dataframe

"""
    Step 5: Dealing with Categorical Values using LabelEncoding and OneHotEncoding with .get_dummies() function
"""

categorical_columns = ['hour', 'dayofweek', 'month', 'year', 'date']

# We can remove one feature from each category during OneHotEncoding because we can still identify that category with reduced number of columns using drop_first = True
df_categorical = pd.get_dummies(df, columns = categorical_columns, drop_first=True)
print("Total columns length: ", len(df_categorical.columns))


"""
    Step 6: Splitting data into training and test (90-10 ratio)
"""

X = df_categorical.drop(columns=['Demand'])
y = df_categorical.loc[: , ['Demand']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


"""
    Step 7: Feature Scaling
"""
SCALER_DIR = "scaler"
scaler = MinMaxScaler()

COLUMN_DIR = "columns"
column_path = os.path.join(COLUMN_DIR, f"feature_columns.joblib")
joblib.dump(list(X_train.columns), column_path)
print(f"Saved feature columns to {column_path}")

# X_test is unseen data and we dont know about it at present. so we only use .fit_transform() in X_train.
X_train_scaled = scaler.fit_transform(X_train) # returns numpy array
X_test_scaled = scaler.transform(X_test) # returns numpy array
# y_train and y_test does NOT need to be scalted because the interpretaion of meaning of target column will hold no meaning.

# Save model
scaler_path = os.path.join(SCALER_DIR, f"MinMaxScaler.joblib")
joblib.dump(scaler, scaler_path)
print(f"Saved MinMaxScaler to {scaler_path}")

"""
    Step 8: Model Training and Testing using RMSE (Root Mean Square Error)
"""

models  = {
    'LinearRegression': LinearRegression(),
    'RandomForestRegressor': RandomForestRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'DecisionTreeRegressor': DecisionTreeRegressor()
}

MODEL_DIR = "models"
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train) # Train
    y_pred = model.predict(X_test_scaled) # Test

    rmse_score = root_mean_squared_error(y_test, y_pred)
    print(model_name, rmse_score)
  
    # Save model
    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    print(f"Saved {model_name} to {model_path}")