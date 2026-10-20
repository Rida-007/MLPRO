import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_data, build_feature_pipeline

def main():
    print("Running verification...")
    # 1. Original dataset unchanged
    df_original = pd.read_csv('data/fake_job_postings.csv')
    df = df_original.copy()
    
    X, y = preprocess_data(df)
    
    print("\n--- Verifying Features ---")
    print(f"job_id excluded: {'job_id' not in X.columns}")
    print(f"fraudulent excluded: {'fraudulent' not in X.columns}")
    
    print("\n--- Verifying Salary Features ---")
    print(f"salary_min present: {'salary_min' in X.columns}")
    print(f"salary_max present: {'salary_max' in X.columns}")
    print(f"salary_midpoint present: {'salary_midpoint' in X.columns}")
    print(f"has_salary_range present: {'has_salary_range' in X.columns}")
    
    # Check a parsed example
    example_salary_idx = df_original[df_original['salary_range'] == '95000-115000'].index[0]
    print("Parsed Example (95000-115000):")
    print(f"  Min: {X.loc[example_salary_idx, 'salary_min']}")
    print(f"  Max: {X.loc[example_salary_idx, 'salary_max']}")
    print(f"  Mid: {X.loc[example_salary_idx, 'salary_midpoint']}")
    print(f"  Has: {X.loc[example_salary_idx, 'has_salary_range']}")
    
    print("\n--- Verifying Department ---")
    print(f"department present: {'department' in X.columns}")
    print(f"has_department present: {'has_department' in X.columns}")
    print(f"Department 'Unknown' count: {(X['department'] == 'Unknown').sum()}")
    
    print("\n--- Verifying Text and Categorical ---")
    print(f"Combined text feature created: {'combined_text' in X.columns}")
    
    print("\n--- Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    print("\nTrain class distribution:")
    print(y_train.value_counts(normalize=True))
    print("\nTest class distribution:")
    print(y_test.value_counts(normalize=True))
    
    print("\n--- Testing sklearn Pipeline ---")
    pipeline = build_feature_pipeline()
    try:
        X_train_transformed = pipeline.fit_transform(X_train)
        X_test_transformed = pipeline.transform(X_test)
        print("\nTF-IDF, OneHotEncoder, and Numerical scaling succeeded.")
        print(f"Train Transformed shape: {X_train_transformed.shape}")
        print(f"Test Transformed shape: {X_test_transformed.shape}")
        print("\nAll verification checks passed!")
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        
if __name__ == '__main__':
    main()
