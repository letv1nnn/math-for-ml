import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor

# datasets: holiday_events, oil, stores and train/test sets.

# train and test datasets
train_dataset_file, test_dataset_file = "store-sales-time-series-forecasting/train.csv", "store-sales-time-series-forecasting/test.csv"
train_dataset, x_test = pd.read_csv(train_dataset_file), pd.read_csv(test_dataset_file)
result_id = x_test["id"]

# splitting data into different datasets, train and cross validation sets
x_train, y_train = train_dataset.drop(columns=["sales"]), train_dataset["sales"]
x_train, x_cv, y_train, y_cv = train_test_split(
    x_train, y_train, random_state=42, test_size=0.2
)

# feature engineering
def dataset_feature_engineering(x: pd.DataFrame):
    x = x.copy()
    # need to handle date and product family cases, because they all are non-numerical objects
    # date: year, month, date
    x["year"], x["month"], x["day"] = (
        x["date"].apply(lambda item: int(item.split('-')[0])),
        x["date"].apply(lambda item: int(item.split('-')[1])),
        x["date"].apply(lambda item: int(item.split('-')[2]))
    )
    # product family:
    product_families = ['AUTOMOTIVE', 'PREPARED FOODS', 'BEAUTY', 'SEAFOOD', 'POULTRY', 'CELEBRATION',
                         'BEVERAGES', 'HARDWARE', 'PLAYERS AND ELECTRONICS', 'BREAD/BAKERY', 'DELI',
                         'CLEANING', 'LIQUOR,WINE,BEER', 'HOME CARE', 'BABY CARE', 'LINGERIE', 'DAIRY',
                         'FROZEN FOODS', 'EGGS', 'BOOKS', 'PERSONAL CARE', 'HOME AND KITCHEN II',
                         'HOME APPLIANCES', 'PRODUCE', 'PET SUPPLIES', 'MAGAZINES',
                         'SCHOOL AND OFFICE SUPPLIES', 'GROCERY I', 'GROCERY II', 'LADIESWEAR',
                         'HOME AND KITCHEN I', 'MEATS', 'LAWN AND GARDEN']
    one_hot = pd.get_dummies(x["family"])
    one_hot = one_hot.reindex(columns=product_families, fill_value=0)
    one_hot = one_hot.astype(np.int8)
    x = pd.concat([x, one_hot], axis=1)

    x = x.drop(columns=["date", "family"])
    return x


def feature_engineer():
    pass

x_train, x_cv, x_test = dataset_feature_engineering(x_train), dataset_feature_engineering(x_cv), dataset_feature_engineering(x_test)
# print(x_train.info())

# scaling the data
scaler = MinMaxScaler()
x_train, x_cv, x_test = scaler.fit_transform(x_train), scaler.transform(x_cv), scaler.transform(x_test)


# creating and training the model

models = {
    "XGBoost": XGBRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# Time-series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

for name, model in models.items():
    print(f"Evaluating {name}")
    scores = cross_val_score(model, x_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
    print(f"Average RMSE: {np.mean(np.sqrt(-scores)):.2f}")

# Hyperparameter tuning (example for XGBoost)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1]
}

grid_search = GridSearchCV(XGBRegressor(random_state=42), param_grid, cv=tscv, scoring='neg_mean_squared_error')
grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_

y_pred_train = best_model.predict(x_train)
y_pred_cv = best_model.predict(x_cv)

def evaluate_regression(y_true, y_pred, label="") -> None:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label} MSE: {mse:.2f}")
    print(f"{label} RMSE: {rmse:.2f}")
    print(f"{label} MAE: {mae:.2f}")
    print(f"{label} R²: {r2:.4f}")
    print("-" * 30)


# evaluate_regression(y_train, y_pred_train, label="Training set")
# evaluate_regression(y_cv, y_pred_cv, label="Validation set")

# Visualization
# difference between actual and predicted values

def visualize_difference(x_train, y_true, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true.values, label='Actual', marker='x', linestyle='None', alpha=0.6)
    plt.plot(y_pred, label='Predicted', marker='o', linestyle='None', alpha=0.6)
    plt.legend()
    plt.title("Actual vs Predicted Sales")
    plt.xlabel("Sample Index")
    plt.ylabel("Sales")
    plt.grid(True)
    plt.show()


visualize_difference(x_train, y_train, y_pred_train)
