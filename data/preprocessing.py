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

# 'telecommuting', 'has_company_logo', and 'has_questions' are binary, so they get passed through automatically to the dataframe, hence not being added to the lists above.

DROP_COLS = []

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
        X = df.drop(columns=['fraudulent', 'job_id'] + TEXT_COLS + DROP_COLS, errors='ignore')
        available_cat_cols = [col for col in CAT_COLS if col in X.columns]
        X = pd.get_dummies(X, columns=available_cat_cols)
        return X, Y, None

    # ----------------------------
    # Text only (TF-IDF)
    # ----------------------------
    elif feature_set == 'text_only':
        # Combine all text columns into a single string per posting
        text_data = df[TEXT_COLS].fillna('').agg(' '.join, axis=1)
        return None, Y, text_data

    # ----------------------------
    # Combined (metadata + text)
    # ----------------------------
    elif feature_set == 'combined':
        # Metadata side
        meta = df.drop(columns=['fraudulent', 'job_id'] + TEXT_COLS + DROP_COLS, errors='ignore')
        available_cat_cols = [col for col in CAT_COLS if col in meta.columns]
        meta = pd.get_dummies(meta, columns=available_cat_cols)

        # Text side
        text_data = df[TEXT_COLS].fillna('').agg(' '.join, axis=1)
        return meta, Y, text_data

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
    meta, Y, text_data = build_features(df, feature_set)

    # ----------------------------
    # Train/test split
    # For the Logistic Regression model, the max_features parameter in TfidfVectorizer() should be set to 3000 for optimal performance. 
    # ----------------------------
    if feature_set == 'metadata_only':
        X_train, X_test, Y_train, Y_test = train_test_split(
            meta, Y, test_size=0.3, random_state=5, stratify=Y
        )

    elif feature_set == 'text_only':
        text_train, text_test, Y_train, Y_test = train_test_split(
            text_data, Y, test_size=0.3, random_state=5, stratify=Y
        )
        # Fit vectorizer on training text only
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2), min_df=5, max_df=0.9)
        X_train = vectorizer.fit_transform(text_train)
        X_test  = vectorizer.transform(text_test)

    elif feature_set == 'combined':
        # Split metadata and text in parallel using the same indices
        meta_train, meta_test, text_train, text_test, Y_train, Y_test = train_test_split(
            meta, text_data, Y, test_size=0.3, random_state=5, stratify=Y
        )
        # Fit vectorizer on training text only
        vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1,2), min_df=5, max_df=0.9)
        text_train_vec = vectorizer.fit_transform(text_train)
        text_test_vec  = vectorizer.transform(text_test)

        # Combine metadata and text
        meta_train_sparse = sp.csr_matrix(meta_train.values.astype(float))
        meta_test_sparse  = sp.csr_matrix(meta_test.values.astype(float))
        X_train = sp.hstack([meta_train_sparse, text_train_vec])
        X_test  = sp.hstack([meta_test_sparse,  text_test_vec])

    # ----------------------------
    # Feature scaling
    # ----------------------------
    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ----------------------------
    # SMOTE (training data only)
    # ----------------------------
    sm = SMOTE(random_state=5, k_neighbors=3)
    X_train_res, Y_train_res = sm.fit_resample(X_train, Y_train)

    print(f"Feature set: {feature_set}")
    print(f"Training samples after SMOTE: {X_train_res.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    return X_train_res, X_test, Y_train_res, Y_test
