import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_as_dataframe():
    base       = os.path.join(os.path.dirname(__file__), "..", "data")
    train_path = os.path.join(base, "adult_train.csv")
    test_path  = os.path.join(base, "adult_test.csv")

    print("  Loading Kaggle Adult dataset...")
    train_df = pd.read_csv(train_path, skipinitialspace=True)
    test_df  = pd.read_csv(test_path,  skipinitialspace=True)

    df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"  Combined: {len(df)} records")

    # Clean
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)

    # Label: Target column, strip trailing dots (UCI artifact)
    df['Target'] = df['Target'].astype(str).str.strip().str.rstrip('.')
    df['hired']  = (df['Target'] == '>50K').astype(int)
    df.drop(columns=['Target', 'fnlwgt'], inplace=True)

    # Protected attributes
    df['sex']  = (df['Sex'].str.strip()  == 'Male').astype(int)
    df['race'] = (df['Race'].str.strip() == 'White').astype(int)
    sensitive_sex = df['sex'].copy()

    df.drop(columns=['Sex', 'Race'], inplace=True)

    # Encode remaining categoricals
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df.drop(columns=['hired', 'sex', 'race'])
    y = df['hired']

    print(f"  Shortlist rate — Overall: {y.mean():.2%}")
    print(f"  Shortlist rate — Male:    {y[sensitive_sex==1].mean():.2%}")
    print(f"  Shortlist rate — Female:  {y[sensitive_sex==0].mean():.2%}")

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive_sex,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    print(f"  Train: {len(X_train)} | Test: {len(X_test)} | Features: {X_train.shape[1]}")
    print(f"  Group — Male: {(s_test==1).sum()} | Female: {(s_test==0).sum()}")

    return X_train, X_test, y_train, y_test, s_train, s_test