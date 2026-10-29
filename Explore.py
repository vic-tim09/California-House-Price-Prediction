"""Explore the California Housing dataset before training the model."""

import pandas as pd
from sklearn.datasets import fetch_california_housing


def main() -> None:
    # Download/cache the dataset and return its feature names and target.
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()

    # The target is expressed in units of $100,000, not raw dollars.
    df = df.rename(columns={"MedHouseVal": "MedianHouseValue_100k_USD"})

    print("Dataset shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe().round(2))
    print("\nMissing values per column:")
    print(df.isna().sum())


if __name__ == "__main__":
    main()
