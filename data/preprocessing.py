""" Phantom Positions — Shared Preprocessing
===================================================
This module intentionally centralizes preprocessing logic to eliminate
inconsistencies between model experiments.

It handles all data loading, feature construction, splitting, scaling,
and class balancing for the EMSCAD fraud detection pipeline. Both the SVM and
Random Forest notebooks import from here to ensure identical preprocessing
across all conditions and models.

HOW TO USE
----------
At the top of your model notebook, add:

    import sys
    sys.path.append('..')
    from data.preprocess import load_and_split

Then call load_and_split() with one of three feature conditions:

    X_train, Y_train, X_test, Y_test = load_and_split(feature_set='text_only')
    X_train, Y_train, X_test, Y_test = load_and_split(feature_set='metadata_only')
    X_train, Y_train, X_test, Y_test = load_and_split(feature_set='combined')

WHAT THIS MODULE HANDLES
---------------------------------
- Stratified train/test split (70/30, random_state=5)
- StandardScaler fit on training data only, applied to both splits
- SMOTE applied to training data only to address class imbalance
- Test set is never resampled or modified, preserving real-world class distribution

WHAT GETS RETURNED
------------------
    X_train: resampled, scaled training features (post-SMOTE)
    Y_train: resampled training labels (post-SMOTE)
    X_test: scaled test features (original distribution, untouched)
    Y_test: test labels (original distribution, untouched) """

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE        # reference: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
import scipy.sparse as sp                       # reference: https://docs.scipy.org/doc/scipy/reference/sparse.html


# ----------------------------
# Column definitions
# ----------------------------

# Text-based fields used for NLP feature extraction
TEXT_COLS = ['title', 'location', 'company_profile', 
             'description', 'requirements', 'benefits']

# Categorical metadata fields for one-hot encoding
CAT_COLS  = ['department', 'salary_range', 'employment_type', 'required_experience',
             'required_education', 'industry', 'function']

def load_data():
    """
    Load the dataset from the local CSV file.

    Returns
    -------
    pandas.DataFrame
        Raw dataset containing job postings and labels.
    """
    data_path = Path(__file__).parent / "fake_job_postings.csv"
    df = pd.read_csv(data_path)
    print(f"=== data loaded successfully ===")
    print(f"Shape: {df.shape}")
    return df


def build_features(df, feature_set='combined'):
    """
    Construct feature matrix and target vector based on selected feature set.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing job postings.
    feature_set : str, optional
        Feature configuration to use:
            - 'metadata_only'
            - 'text_only'
            - 'combined' (default)

    Returns
    -------
    X : array-like or sparse matrix
        Feature matrix.
    Y : pandas.Series
        Target labels (fraudulent indicator).

    Raises
    ------
    ValueError
        If an invalid feature_set is provided.
    """
    Y = df['fraudulent']

    # ----------------------------
    # Metadata only
    # ----------------------------
    if feature_set == 'metadata_only':
        X = df.drop(columns=['fraudulent', 'job_id'] + TEXT_COLS)
        X = pd.get_dummies(X, columns=CAT_COLS)
        return X, Y

    # ----------------------------
    # Text only (TF-IDF)
    # ----------------------------
    elif feature_set == 'text_only':
        # Combine all text columns into a single string per posting
        text_data = df[TEXT_COLS].fillna('').agg(' '.join, axis=1)
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(text_data)
        return X, Y

    # ----------------------------
    # Combined (metadata + text)
    # ----------------------------
    elif feature_set == 'combined':
        # Metadata side
        meta = df.drop(columns=['fraudulent', 'job_id'] + TEXT_COLS)
        meta = pd.get_dummies(meta, columns=CAT_COLS)

        # Text side
        text_data = df[TEXT_COLS].fillna('').agg(' '.join, axis=1)
        vectorizer = TfidfVectorizer(max_features=5000)
        text_matrix = vectorizer.fit_transform(text_data)

        # Convert metadata to sparse and combine
        meta_sparse = sp.csr_matrix(meta.values.astype(float))
        X = sp.hstack([meta_sparse, text_matrix])
        return X, Y

    else:
        raise ValueError(f"Invalid feature_set '{feature_set}'. Choose 'metadata_only', 'text_only', or 'combined'.")


def load_and_split(feature_set='combined'):
    """
    Full preprocessing pipeline:
        - Load data
        - Build features
        - Train/test split (stratified)
        - Scale features
        - Apply SMOTE to training data

    Parameters
    ----------
    feature_set : str, optional
        Feature configuration to use (default is 'combined').

    Returns
    -------
    X_train_res : Training features after SMOTE.
    X_test : Test features (unmodified by SMOTE).
    Y_train_res : Resampled training labels.
    Y_test : Test labels.
    """
    df = load_data()
    X, Y = build_features(df, feature_set)

    # ----------------------------
    # Train/test split
    # ----------------------------
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=5, stratify=Y
    )

    # Uncomment next line to Check fraud counts before SMOTE
    # print(f"Before SMOTE: {Y_train.value_counts().to_dict()}") 

    # ----------------------------
    # Feature scaling
    # ----------------------------
    scaler = StandardScaler(with_mean=False)  # with_mean=False required for sparse matrices
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ----------------------------
    # SMOTE (training data only)
    # ----------------------------
    sm = SMOTE(random_state=5, k_neighbors=3)
    X_train_res, Y_train_res = sm.fit_resample(X_train, Y_train)

    """ Remove Block Comment to: 
    # Check fraud counts after SMOTE
    print(f"After SMOTE: {pd.Series(Y_train_res).value_counts().to_dict()}")

    # Indices of minority class in resampled set
    real_fraud = X_train_res[:606]
    synthetic_fraud = X_train_res[606:]

    # Compare variance across features
    real_var = np.var(real_fraud, axis=0).mean()
    synthetic_var = np.var(synthetic_fraud, axis=0).mean()

    print(f"Mean feature variance - Real fraud: {real_var:.4f}")
    print(f"Mean feature variance - Synthetic fraud: {synthetic_var:.4f}") """

    print(f"Feature set: {feature_set}")
    print(f"Training samples after SMOTE: {X_train_res.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    return X_train_res, X_test, Y_train_res, Y_test
