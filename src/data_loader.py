import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_as_dataframe():
    """
    Loads UCI Adult Income dataset and frames it as a hiring/resume screening problem.
    Label: 1 = shortlisted (income >50K proxy), 0 = rejected
    Protected attributes: sex, race
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    col_names = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week",
        "native_country", "income"
    ]

    print("  Loading UCI Adult dataset...")
    try:
        df = pd.read_csv(url, header=None, names=col_names,
                         na_values=' ?', skipinitialspace=True)
        print(f"  Downloaded {len(df)} records.")
    except Exception:
        print("  Network unavailable — using local fallback...")
        df = _generate_synthetic_hiring_data()

    # ── Clean ──────────────────────────────────────────────────────────────────
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)

    # ── Label: shortlisted = income >50K ──────────────────────────────────────
    df['hired'] = (df['income'].str.strip() == '>50K').astype(int)
    df.drop(columns=['income', 'fnlwgt'], inplace=True)

    # ── Protected attributes ───────────────────────────────────────────────────
    df['sex']  = (df['sex'].str.strip() == 'Male').astype(int)   # 1=male, 0=female
    df['race'] = (df['race'].str.strip() == 'White').astype(int) # 1=white, 0=non-white

    sensitive_sex  = df['sex'].copy()
    sensitive_race = df['race'].copy()

    # ── Features ───────────────────────────────────────────────────────────────
    # Drop direct identifiers but keep skill-relevant features
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df.drop(columns=['hired', 'sex', 'race'])
    y = df['hired']

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive_sex,       # using sex as primary protected attr
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    print(f"  Train: {len(X_train)} | Test: {len(X_test)} | Features: {X_train.shape[1]}")
    print(f"  Shortlist rate — Overall: {y.mean():.2%}")
    print(f"  Group — Male: {(s_test==1).sum()} | Female: {(s_test==0).sum()}")

    return X_train, X_test, y_train, y_test, s_train, s_test


def _generate_synthetic_hiring_data():
    """Offline fallback — mimics Adult dataset distribution with documented bias."""
    np.random.seed(42)
    n = 5000
    sex  = np.random.choice([' Male', ' Female'], n, p=[0.67, 0.33])
    race = np.random.choice([' White', ' Black', ' Asian-Pac-Islander',
                              ' Amer-Indian-Eskimo', ' Other'], n,
                             p=[0.85, 0.09, 0.03, 0.01, 0.02])
    edu_num = np.random.randint(1, 16, n)
    age     = np.random.randint(20, 65, n)
    hours   = np.random.randint(20, 60, n)

    # Base hire probability driven by qualifications
    p_hire = 0.1 + 0.04 * edu_num + 0.003 * (hours - 40)
    # Inject gender bias
    p_hire[sex == ' Female'] *= 0.65
    # Inject race bias
    p_hire[race != ' White'] *= 0.75
    p_hire = np.clip(p_hire, 0.05, 0.95)
    income = np.array([
        np.random.choice([' >50K', ' <=50K'], p=[p, 1-p]) for p in p_hire
    ])

    df = pd.DataFrame({
        'age': age,
        'workclass': np.random.choice(
            [' Private',' Self-emp-not-inc',' Government'], n, p=[0.7,0.15,0.15]),
        'fnlwgt': np.random.randint(10000, 500000, n),
        'education': np.random.choice(
            [' Bachelors',' HS-grad',' Masters',' Some-college'], n),
        'education_num': edu_num,
        'marital_status': np.random.choice(
            [' Married-civ-spouse',' Never-married',' Divorced'], n),
        'occupation': np.random.choice(
            [' Tech-support',' Craft-repair',' Exec-managerial',' Prof-specialty'], n),
        'relationship': np.random.choice(
            [' Husband',' Not-in-family',' Wife',' Own-child'], n),
        'race': race, 'sex': sex,
        'capital_gain': np.random.randint(0, 5000, n),
        'capital_loss': np.random.randint(0, 2000, n),
        'hours_per_week': hours,
        'native_country': ' United-States',
        'income': income
    })
    return df