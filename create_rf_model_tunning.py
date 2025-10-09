import json
import pathlib
import pickle
from typing import List, Tuple

import pandas
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # path to CSV with demographics

# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
    'floors', 'sqft_above', 'sqft_basement', 'zipcode'
]

OUTPUT_DIR = "model"  # Directory where output artifacts will be saved


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load and merge home sales and demographics data."""
    data = pandas.read_csv(sales_path,
                           usecols=sales_column_selection,
                           dtype={'zipcode': str})
    demographics = pandas.read_csv(demographics_path,
                                   dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")
    y = merged_data.pop('price')
    x = merged_data

    return x, y


def main():
    """Load data, train RandomForest model with CV tuning, and export artifacts."""
    print("Loading and preparing data...")
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, random_state=42, test_size=0.2
    )

    print("Starting RandomForest hyperparameter tuning with 5-fold CV...")

    # Define pipeline
    pipe = pipeline.make_pipeline(
        preprocessing.RobustScaler(),
        RandomForestRegressor(random_state=42)
    )

    # Define parameter grid for tuning
    param_grid = {
        'randomforestregressor__n_estimators': [100, 150, 200],
        'randomforestregressor__max_depth': [10, 20, None],
        'randomforestregressor__min_samples_split': [2, 5, 10],
        'randomforestregressor__min_samples_leaf': [1, 2, 4],
    }

    # Perform cross-validation grid search
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(x_train, y_train)

    print("Best hyperparameters found:")
    print(grid_search.best_params_)

    print("Best cross-validation score (R²):", grid_search.best_score_)

    # Refit best model
    best_model = grid_search.best_estimator_

    # Evaluate on test set
    test_score = best_model.score(x_test, y_test)
    print(f"Test set R² score: {test_score:.4f}")

    # Save model artifacts
    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    pickle.dump(best_model, open(output_dir / "model_rf.pkl", 'wb'))
    json.dump(list(x_train.columns), open(output_dir / "model_features.json", 'w'))

    # Save training summary
    summary = {
        "best_params": grid_search.best_params_,
        "cv_best_score": float(grid_search.best_score_),
        "test_r2": float(test_score)
    }
    json.dump(summary, open(output_dir / "training_summary.json", 'w'), indent=2)

    print("Model and metadata saved successfully in 'model/' directory.")


if __name__ == "__main__":
    main()
