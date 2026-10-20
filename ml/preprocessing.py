import pandas as pd
import numpy as np
import re
from html import unescape
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to clean text columns.
    """
    def __init__(self, text_columns):
        self.text_columns = text_columns
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        for col in self.text_columns:
            # Handle NaNs
            X_copy[col] = X_copy[col].fillna("")
            
            # Unescape HTML entities (e.g., &amp; -> &)
            X_copy[col] = X_copy[col].apply(unescape)
            
            # Remove actual HTML tags
            X_copy[col] = X_copy[col].str.replace(r'<[^>]+>', ' ', regex=True)
            
            # Remove weird encoding artifacts like  without destroying punctuation
            X_copy[col] = X_copy[col].str.replace(r'\uFFFD', ' ', regex=True)
            
            # Normalize excessive whitespace
            X_copy[col] = X_copy[col].str.replace(r'\s+', ' ', regex=True).str.strip()
            
        return X_copy


def parse_salary(salary_str):
    """
    Parses a salary string like '95000-115000' into min, max, midpoint, and presence flag.
    Returns (min, max, midpoint, has_salary_range)
    """
    if pd.isna(salary_str):
        return np.nan, np.nan, np.nan, 0
    
    match = re.search(r'(\d+)\s*-\s*(\d+)', str(salary_str))
    if match:
        try:
            s_min = float(match.group(1))
            s_max = float(match.group(2))
            s_mid = (s_min + s_max) / 2.0
            return s_min, s_max, s_mid, 1
        except ValueError:
            return np.nan, np.nan, np.nan, 0
    return np.nan, np.nan, np.nan, 0


def preprocess_data(df):
    """
    Initial preprocessing steps that transform the raw pandas DataFrame.
    Returns the processed features (X) and the target labels (y).
    """
    df_clean = df.copy()
    
    # 1. Extract target if present
    y = None
    if 'fraudulent' in df_clean.columns:
        y = df_clean['fraudulent']
        df_clean = df_clean.drop(columns=['fraudulent'])
        
    # 2. Drop pure metadata (job_id)
    if 'job_id' in df_clean.columns:
        df_clean = df_clean.drop(columns=['job_id'])
        
    # 3. Handle salary
    if 'salary_range' in df_clean.columns:
        parsed_salary = df_clean['salary_range'].apply(parse_salary)
        df_clean['salary_min'] = parsed_salary.apply(lambda x: x[0])
        df_clean['salary_max'] = parsed_salary.apply(lambda x: x[1])
        df_clean['salary_midpoint'] = parsed_salary.apply(lambda x: x[2])
        df_clean['has_salary_range'] = parsed_salary.apply(lambda x: x[3])
        df_clean = df_clean.drop(columns=['salary_range'])
        
    # 4. Handle department
    if 'department' in df_clean.columns:
        df_clean['has_department'] = df_clean['department'].notna().astype(int)
        
    # 5. Fill missing values for categorical features
    cat_cols = ['department', 'employment_type', 'required_experience', 'required_education', 'industry', 'function']
    for col in cat_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("Unknown")
            
    # 6. Fill missing text columns
    text_cols = ['title', 'location', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("")
            
    # 7. Combine text fields into a single rich text feature
    text_components = [
        'title', 'location', 'company_profile', 'description', 'requirements', 'benefits',
        'department', 'employment_type', 'required_experience', 'required_education', 'industry', 'function'
    ]
    
    df_clean['combined_text'] = ""
    for col in text_components:
        if col in df_clean.columns:
            df_clean['combined_text'] += df_clean[col].astype(str) + " "
            
    # Apply custom text cleaning to the combined text
    cleaner = TextCleaner(text_columns=['combined_text'])
    df_clean = cleaner.transform(df_clean)
    
    return df_clean, y


def build_feature_pipeline():
    """
    Constructs a scikit-learn ColumnTransformer to vectorize text, one-hot encode
    categorical features, and scale/impute numerical features.
    """
    numeric_features = [
        'telecommuting', 
        'has_company_logo', 
        'has_questions', 
        'has_salary_range', 
        'has_department',
        'salary_min',
        'salary_max',
        'salary_midpoint'
    ]
    
    categorical_features = [
        'department',
        'employment_type', 
        'required_experience', 
        'required_education', 
        'industry', 
        'function'
    ]
    
    text_feature = 'combined_text'
    
    # We impute NaNs in numeric features with a constant value (-1) for linear models/SVMs,
    # and then scale them.
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('text', TfidfVectorizer(max_features=3000, stop_words='english'), text_feature)
        ]
    )
    
    return preprocessor
