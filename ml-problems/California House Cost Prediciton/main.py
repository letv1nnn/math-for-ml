import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from scipy import stats
from scipy.stats import randint

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import tarfile
import urllib.request
import joblib

import sklearn as sl
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (OrdinalEncoder, OneHotEncoder,
                                   MinMaxScaler, StandardScaler,
                                   FunctionTransformer)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer, make_column_selector
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------------------------------------------------------------------------------------|

# LOADING DATA

# ---------------------------------------------------------------------------------------------------------------------|

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
    with tarfile.open(tarball_path) as housing_tarball:
        housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()

# ---------------------------------------------------------------------------------------------------------------------|

# EXPLORE DATA AND GAIN INSIGHTS

# ---------------------------------------------------------------------------------------------------------------------|

# QUICK LOOK AT THE DATA STRUCTURE AND THE DATA IN GENERAL
# print(housing.head())
# print(housing.info())
# print(housing["ocean_proximity"].value_counts())
# print(housing.describe())

# DATA VISUALIZATION
def visualize_initial_dataset():
    housing.hist(bins=50, figsize=(12, 8))
    plt.show()

# visualize_initial_dataset()

# SPLITTING DATA INTO TRAIN AND TEST SETS
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# manual implementation
def shuffle_and_split_data(data, test_ration):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ration)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# splitting the median house district into 6 categories
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
def visualize_income_cat():
    housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
    plt.xlabel("Income category")
    plt.ylabel("Number of districts")
    plt.show()

# visualize_income_cat()

# Ensures all important classes are represented in both sets
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

# print(strat_test_set.info())
# Income category proportion int the test set
# print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# copy of the original training set, so I could do various transformations with it without change the initial one
housing = strat_train_set.copy()

# The area of each circle represents the district's population
# and the color represents the price of the house
def visualize_long_lat():
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2,
                 s=housing["population"] / 100, label="population",
                 c="median_house_value", cmap="jet", colorbar=True,
                 legend=True, sharex=False, figsize=(10, 7)
                 )
    plt.show()

# visualize_long_lat()


# scatter matrix that plots every numerical attribute against every other numerical attribute.
def visualize_correlations():
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.show()

# visualize_correlations()

# ---------------------------------------------------------------------------------------------------------------------|

# EXPERIMENT WITH ATTRIBUTE COMBINATIONS AND PREPARE DATA FOR MACHINE LEARNING ALGORITHMS

# ---------------------------------------------------------------------------------------------------------------------|

housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]


housing = housing.drop(columns=["median_house_value"], axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

imputer = SimpleImputer(strategy="median")
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
# print(imputer.statistics_)
# print(housing_num.median().values)

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
housing_cat = housing[["ocean_proximity"]]

ordinal_encoder = OrdinalEncoder()
housing_cat_encoding = ordinal_encoder.fit_transform(housing_cat)

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# print(housing_cat_1hot) sparse matrix, really efficient way while using OneHotEncoding method
housing_cat_1hot = housing_cat_1hot.toarray()
# print(housing_cat_1hot)

# Data scaling
min_max_scaler = MinMaxScaler()
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

# scaling the data using standardisation
std_scaler = StandardScaler(with_mean=False) # with mean is equal to false for sparse matrices
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

# Using radial bias function to find housing ages that are close to 35
age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)

# Scaling the target
target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

# Creating a Linear Regression model
model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)
some_new_data = housing[["median_income"]].iloc[:5]

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)

# Alternatively
model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)

# Creating a log-transformer and applying it to the population feature
log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])


# Custom transformer that acts much as like the Standard Scaler
class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean: bool=True):
        self.with_mean = with_mean

    def fit(self, X, y=None):
        X = check_array(X)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_
    # add get_features_names_out() and inverse_transform() methods


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]], sample_weight=housing_labels)


# ML PIPELINE
# Many data transformation steps that need to be executed in the right order.
# Pipeline class helps with such sequences of transformation.
# Here is a small pipeline for numerical attributes, which will first impute then scale the input features
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])
# Alternatively, it's possible to use make_pipeline function instead

housing_num_prepared = num_pipeline.fit_transform(housing_num)
df_housing_num_prepared = pd.DataFrame(housing_num_prepared, columns=num_pipeline.get_feature_names_out(), index=housing_num.index)

# A single transformer that handle both numerical and categorical columns
num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())
preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)

housing_prepared = preprocessing.fit_transform(housing)
# print(preprocessing.get_feature_names_out())

# ---------------------------------------------------------------------------------------------------------------------|

# SELECT AND TRAIN A MODEL

# ---------------------------------------------------------------------------------------------------------------------|
'''
forest_reg = make_pipeline(preprocessing, RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1))
forest_reg.fit(housing, housing_labels)
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
                              scoring="neg_root_mean_squared_error", cv=10, n_jobs=-1)

print(pd.Series(forest_rmses).describe())
'''
#---------------------------------------------------------------------------------------------------------------------

# FINE-TUNE THIS MODEL

#---------------------------------------------------------------------------------------------------------------------


full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(n_estimators=10, random_state=42))
])
param_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10],
     'random_forest__max_features': [4, 6, 8]},
    {'preprocessing__geo__n_clusters': [10, 15],
     'random_forest__max_features': [6, 8, 10]},
]

grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
                           scoring='neg_root_mean_squared_error')
grid_search.fit(housing, housing_labels)
# print(grid_search.best_params_)

cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)

# Second method, using RandomizedSearchCV that is more preferable
param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50),
                  'random_forest__max_features': randint(low=2, high=20)}

rnd_search = RandomizedSearchCV(
    full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
    scoring='neg_root_mean_squared_error', random_state=42)

rnd_search.fit(housing, housing_labels)

final_model = rnd_search.best_estimator_
feature_importances = final_model["random_forest"].feature_importances_
# print(feature_importances.round(2))

#print(sorted(zip(feature_importances, final_model["preprocessing"].get_feature_names_out()), reverse=True))

# Testing the model on a test set
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test["rooms_per_house"] = X_test["total_rooms"] / X_test["households"]
X_test["bedrooms_ratio"] = X_test["total_bedrooms"] / X_test["total_rooms"]
X_test["people_per_house"] = X_test["population"] / X_test["households"]

final_predictions = final_model.predict(X_test)

final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
# print(final_rmse)

# For most cases RMSE is enough, but if the model surpass by 0.1 or even less then the previous one.
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
print(np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                               loc=squared_errors.mean(),
                               scale=stats.sem(squared_errors))))

#---------------------------------------------------------------------------------------------------------------------

# LAUNCH, MONITOR, AND MAINTAIN THE SYSTEM

#---------------------------------------------------------------------------------------------------------------------


# saving the model
joblib.dump(final_model, "california_housing_model.pkl")
