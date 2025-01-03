import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from pytorch_tabnet.tab_model import TabNetClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import uproot
import torch
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
import time
import warnings
import scipy.stats as stats
warnings.filterwarnings('ignore')  

# Data path update
data_path = r"C:\Users\berke\OneDrive\Masaüstü\simulation\myz\data"

def load_data_from_directories(base_path=data_path):
    print(f"Loading data from {base_path}...")
    
    if not os.path.exists(base_path):
        raise ValueError(f"Directory not found: {base_path}")
    
    satellite_data = {}  # We'll keep data from each satellite separately
    root_files = [f for f in os.listdir(base_path) if f.endswith('.root')]
    
    if not root_files:
        raise ValueError(f"No root files found in directory: {base_path}")
    
    print(f"Found root files: {root_files}")
    
    # Dictionary for tree names and reading methods
    file_configs = {
        'filtered_grb_table_with_epeak.root': {
            'tree_name': 'GRB_Tree',
            'library': 'np',
            'satellite':'Swift BAT'
        },
        'GRB_Data_with_Hardness_Ratio.root': {
            'tree_name': 'GRBTree',
            'library': 'np',
            'satellite': 'Fermi GBM'
        },
        'output_data.root': {
            'tree_name': 'TriggerData',
            'library': 'pd',
            'satellite': 'CGRO BATSE'
        }
    }
    
    for file in root_files:
        file_path = os.path.join(base_path, file)
        try:
            config = file_configs.get(file)
            if config is None:
                print(f"Warning: Configuration not found for {file}")
                continue
            
            with uproot.open(file_path) as root_file:
                try:
                    tree = root_file[config['tree_name']]
                    print(f"File being read: {file}")
                    print(f"Tree name: {config['tree_name']}")
                    
                    # First read as numpy array
                    arrays = tree.arrays(library=config['library'])
                    
                    if config['library'] == 'np':
                        # Convert numpy array to DataFrame
                        df = pd.DataFrame()
                        for branch in tree.keys():
                            branch_name = str(branch).lower().strip()
                            df[branch_name] = arrays[branch]
                    else:
                        df = arrays
                    
                    # Clean column names
                    df.columns = [str(col).lower().strip() for col in df.columns]
                    print(f"Read columns: {df.columns.tolist()}")
                    
                    # Store satellite data separately
                    satellite_data[config['satellite']] = df
                    
                except Exception as e:
                    print(f"Tree reading error ({config['tree_name']}): {str(e)}")
                    continue
            
        except Exception as e:
            print(f"File reading error ({file_path}): {str(e)}")
            continue
    
    # Analyze T90 distribution for each data set
    for satellite, df in satellite_data.items():
        try:
            t90_values = None
            
            if satellite == 'Fermi GBM':
                t90_values = df['t90'].astype(float)
            else:
                t90_col = [col for col in df.columns if 't90' in col.lower()]
                if t90_col:
                    t90_values = pd.to_numeric(df[t90_col[0]], errors='coerce')
            
            if t90_values is not None:
                plt.figure(figsize=(12, 7))
                
                # Prepare T90 data
                t90_values = t90_values.replace([np.inf, -np.inf], np.nan)
                t90_values = t90_values[t90_values > 0]
                t90_values = t90_values.dropna()
                
                if len(t90_values) > 0:
                    # Calculate log10(T90) values
                    log_t90 = np.log10(t90_values)
                    
                    # Calculate binomial distribution parameters
                    n_trials = len(log_t90)
                    p_success = 0.05
                    
                    # Calculate thresholds
                    binom_threshold_upper = stats.binom.ppf(1 - p_success, n_trials, p_success) / n_trials
                    binom_threshold_lower = stats.binom.ppf(p_success, n_trials, p_success) / n_trials
                    
                    upper_base = np.percentile(log_t90, (1 - p_success) * 100)
                    lower_base = np.percentile(log_t90, p_success * 100)
                    
                    # Calculate error margin
                    t90_error = 0.1 * t90_values
                    log_t90_error = np.abs(t90_error / (t90_values * np.log(10)))
                    error_margin = np.mean(log_t90_error) * 1.96
                    
                    threshold_upper = upper_base * (1 + binom_threshold_upper) + error_margin
                    threshold_lower = lower_base * (1 + binom_threshold_lower) - error_margin
                    
                    # Plot histogram
                    counts, bins, _ = plt.hist(log_t90, bins=50, density=True, 
                                             alpha=0.6, color='skyblue',
                                             label='T90 Distribution')
                    
                    # Draw threshold lines
                    plt.axvline(x=threshold_lower, color='red', linestyle='--', 
                              label='Short-Long Boundary')
                    plt.axvline(x=threshold_upper, color='green', linestyle='--', 
                              label='Long-Ultra Long Boundary')
                    
                    # Graph formatting
                    plt.title(f'{satellite} GRB T90 Distribution')
                    plt.xlabel('log10(T90) [s]')
                    plt.ylabel('Density')
                    plt.grid(True, alpha=0.3)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.xlim(-2, 4)
                    
                    # Print classification results
                    conditions = [
                        (log_t90 <= threshold_lower),
                        (log_t90 > threshold_lower) & (log_t90 <= threshold_upper),
                        (log_t90 > threshold_upper)
                    ]
                    classifications = np.select(conditions, [0, 1, 2])
                    
                    unique, counts = np.unique(classifications, return_counts=True)
                    total = len(classifications)
                    
                    print("\nT90 Statistics for {satellite}:")
                    print(f"Total number of GRBs: {total}")
                    class_names = ['Short', 'Long', 'Ultra-Long']
                    for val, count in zip(unique, counts):
                        print(f"Number of {class_names[val]} GRBs: {count} "
                              f"({count/total*100:.1f}%)")
                    
                    plt.tight_layout()
                    plt.show()
                else:
                    print(f"\nNo T90 data available for {satellite} or it could not be processed")
                    plt.close()
                    
        except Exception as e:
            print(f"\nData visualization error for {satellite}: {str(e)}")
            plt.close()
            continue
    
    # Normalize and proceed with subsequent steps
    normalized_dfs = []
    for satellite, df in satellite_data.items():
        print(f"\nNormalizing {satellite} data...")
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        # Normalize each numeric column
        for col in numeric_cols:
            if 't90' not in col.lower():  # Normalize t90 values
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                df[col] = df[col].clip(lower_bound, upper_bound)
                
                # Min-max normalization
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
        
        # Add satellite information
        df['satellite_source'] = satellite
        normalized_dfs.append(df)
    
    try:
        if not normalized_dfs:
            raise ValueError("No data was read!")
        
        # Combine normalized data
        final_df = pd.concat(normalized_dfs, ignore_index=True)
        
        # Clean duplicate columns
        duplicate_columns = final_df.columns[final_df.columns.duplicated()]
        for col in duplicate_columns:
            col_indices = [i for i, name in enumerate(final_df.columns) if name == col]
            for i, idx in enumerate(col_indices[1:], 1):
                new_name = f"{col}_{i}"
                final_df.columns.values[idx] = new_name
        
        # Clean column names
        final_df.columns = [str(col).lower().strip() for col in final_df.columns]
        
        # Fill missing values
        for col in final_df.columns:
            if col != 'satellite_source':  # Exclude satellite information
                if final_df[col].dtype in ['float64', 'int64']:
                    # Calculate median for each satellite
                    medians = final_df.groupby('satellite_source')[col].transform('median')
                    final_df[col].fillna(medians, inplace=True)
        
        if 't90' in final_df.columns:
            final_df['target'] = (final_df['t90'] > 2.0).astype(int)
        
        print(f"\nLoaded total {len(final_df)} rows of data.")
        print(f"Columns: {final_df.columns.tolist()}")
        
        # Show data distribution
        print("\nData distribution by satellite:")
        print(final_df['satellite_source'].value_counts())
        
        return final_df, satellite_data
        
    except Exception as e:
        print(f"Data loading error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

def prepare_features(X):
    # Check for duplicate columns
    duplicate_cols = X.columns[X.columns.duplicated()].unique()
    if len(duplicate_cols) > 0:
        print(f"Duplicate columns found: {duplicate_cols}")
        # Remove duplicate columns
        X = X.loc[:, ~X.columns.duplicated()]
    return X

# Update target variable transformation
def transform_to_three_classes(data):
    # Combine and clean T90 values
    if 'bat t90 _sec_' in data.columns:
        data['t90'] = data['bat t90 _sec_']
    elif 't90' not in data.columns:
        print("Warning: T90 column not found!")
        return data
    
    # Convert T90 values to numeric and clean
    t90_values = pd.to_numeric(data['t90'], errors='coerce')
    t90_values = t90_values.replace([np.inf, -np.inf], np.nan)
    t90_values = t90_values.fillna(t90_values.median())
    t90_values = np.maximum(t90_values, 0.001)  # Fix negative values
    
    # Logarithmic transformation
    log_t90 = np.log10(t90_values)
    
    # Assume typical error rate for T90 (e.g., 10%)
    t90_error = 0.1 * t90_values
    log_t90_error = np.abs(t90_error / (t90_values * np.log(10)))
    
    # z-score for confidence interval (1.96 for 95% confidence)
    z_score = 1.96
    
    # Calculate binomial distribution parameters
    n_trials = len(log_t90)
    p_success = 0.05  # Approximate ratio for Ultra-long GRBs
    
    # Calculate lower and upper threshold values using binomial distribution
    binom_threshold_upper = stats.binom.ppf(1 - p_success, n_trials, p_success) / n_trials
    binom_threshold_lower = stats.binom.ppf(p_success, n_trials, p_success) / n_trials
    
    # Calculate error margin
    upper_base = np.percentile(log_t90, (1 - p_success) * 100)
    lower_base = np.percentile(log_t90, p_success * 100)
    
    error_margin = np.mean(log_t90_error) * z_score
    
    # Lower and upper threshold values
    threshold_upper = upper_base * (1 + binom_threshold_upper) + error_margin
    threshold_lower = lower_base * (1 + binom_threshold_lower) - error_margin
    
    # Classify based on T90 values
    conditions = [
        (log_t90 > threshold_upper),  # Ultra-long
        (log_t90 > threshold_lower) & (log_t90 <= threshold_upper),  # Long
        (log_t90 <= threshold_lower)  # Short
    ]
    values = [2, 1, 0]  # 2: Ultra-long, 1: Long, 0: Short
    
    # Calculate classification results
    data['grb_type'] = np.select(conditions, values)
    
    # Show class distribution
    print("\nGRB Class Distribution:")
    unique, counts = np.unique(data['grb_type'], return_counts=True)
    total = len(data['grb_type'])
    for class_val, count in zip(unique, counts):
        class_name = {0: 'Short', 1: 'Long', 2: 'Ultra-long'}[class_val]
        percentage = (count / total) * 100
        print(f"{class_name} GRB: {count} ({percentage:.2f}%)")
    
    # Print threshold values
    print(f"\nThreshold Values:")
    print(f"Lower Threshold: 10^{threshold_lower:.2f} seconds")
    print(f"Upper Threshold: 10^{threshold_upper:.2f} seconds")
    
    return data

# Update data preprocessing function
def preprocess_data(data):
    # Select numeric columns
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    
    # For each numeric column
    for column in numeric_columns:
        # Convert infinite values to NaN
        data[column] = data[column].replace([np.inf, -np.inf], np.nan)
        
        # Clean extremely large values
        percentile_99 = data[column].quantile(0.99)
        data.loc[data[column] > percentile_99, column] = percentile_99
        
        # If column has missing values
        if data[column].isnull().any():
            # Use median for time series data
            if 't90' in column.lower() or 'time' in column.lower():
                median_value = data[column].median()
                data[column].fillna(median_value, inplace=True)
            # Use mean for other features
            else:
                mean_value = data[column].mean()
                data[column].fillna(mean_value, inplace=True)
    
    # Convert data types to float64
    for column in numeric_columns:
        data[column] = data[column].astype(np.float64)
    
    return data

# Scale features before SMOTE
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    
    # Convert to DataFrame
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    
    # Select only numeric columns
    numeric_columns = X_train.select_dtypes(include=['float64', 'int64']).columns
    
    # Store non-numeric columns
    non_numeric_train = X_train.select_dtypes(exclude=['float64', 'int64'])
    non_numeric_test = X_test.select_dtypes(exclude=['float64', 'int64'])
    
    # Process numeric columns
    X_train_numeric = X_train[numeric_columns].copy()
    X_test_numeric = X_test[numeric_columns].copy()
    
    # Check for NaN and infinite values and clean
    X_train_numeric = X_train_numeric.replace([np.inf, -np.inf], np.nan)
    X_test_numeric = X_test_numeric.replace([np.inf, -np.inf], np.nan)
    
    # Fill missing values with column median
    for col in X_train_numeric.columns:
        col_median = X_train_numeric[col].median()
        X_train_numeric[col] = X_train_numeric[col].fillna(col_median)
        X_test_numeric[col] = X_test_numeric[col].fillna(col_median)
    
    # Scale features
    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_test_scaled = scaler.transform(X_test_numeric)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=numeric_columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=numeric_columns)
    
    # Add back non-numeric columns
    for col in non_numeric_train.columns:
        X_train_scaled[col] = non_numeric_train[col]
        X_test_scaled[col] = non_numeric_test[col]
    
    return X_train_scaled, X_test_scaled

# Helper function for TabNet predictions
def get_tabnet_predictions(model, X):
    try:
        # Convert data to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Check and fix data type
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        
        # Make predictions on GPU
        with torch.cuda.device('cuda:0'):
            with torch.no_grad():
                preds = model.predict(X)
                proba = model.predict_proba(X)
        
        return preds, proba
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print("\nError details:")
        import traceback
        print(traceback.format_exc())
        raise

# Function to prepare data for TabNet
def prepare_data_for_tabnet(X_train, X_test, y_train, y_test):
    # Convert to DataFrame
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    
    # Clean missing values
    for col in X_train.columns:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
    
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Convert to NumPy array
    X_train = X_train.values.astype(np.float32)
    X_test = X_test.values.astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    
    # Check data dimensions and print
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

# Load data and preprocess
print("Starting data loading...")
data, satellite_data = load_data_from_directories()

# Preprocess data
data = preprocess_data(data)  # First remove missing and infinite values
data = transform_to_three_classes(data)  # Then perform classification

# Check class balance and update SMOTE parameters if necessary
def check_class_balance(y):
    class_counts = np.bincount(y)
    total = len(y)
    print("\nClass Ratios:")
    for i, count in enumerate(class_counts):
        print(f"Class {i}: {count} ({count/total*100:.2f}%)")
    return class_counts

# Update SMOTE parameters
def apply_balanced_smote(X, y):
    class_counts = check_class_balance(y)
    
    # Check for NaN values and clean
    if isinstance(X, pd.DataFrame):
        # Fill missing values with column median
        for col in X.columns:
            if X[col].isnull().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
    else:
        # For NumPy array
        if np.isnan(X).any():
            # Calculate column median
            col_median = np.nanmedian(X, axis=0)
            # Fill NaN values
            nan_mask = np.isnan(X)
            for col_idx in range(X.shape[1]):
                col_nan_mask = nan_mask[:, col_idx]
                X[col_nan_mask, col_idx] = col_median[col_idx]
    
    # Check for infinite values and clean
    if isinstance(X, pd.DataFrame):
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            if X[col].isnull().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
    else:
        X[~np.isfinite(X)] = np.nan
        col_median = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        for col_idx in range(X.shape[1]):
            col_nan_mask = nan_mask[:, col_idx]
            X[col_nan_mask, col_idx] = col_median[col_idx]
    
    # Target class with the most samples
    max_class_count = np.max(class_counts)
    
    # SMOTE for balancing - balance all classes to the level of the most populated class
    sampling_strategy = {
        0: max_class_count,  # Short GRBs
        1: max_class_count,  # Long GRBs
        2: max_class_count   # Ultra-Long GRBs
    }
    
    try:
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print("\nClass distribution after SMOTE:")
        check_class_balance(y_resampled)
        
        return X_resampled, y_resampled
        
    except Exception as e:
        print(f"Error during SMOTE: {str(e)}")
        print("Returning original data...")
        return X, y

# Separate target variables
X = data.drop(['grb_type', 'target', 'grb_type_detailed', 'satellite_source', 'grb', 'name'], axis=1, errors='ignore')
X = prepare_features(X)  # Check for duplicate columns and remove
y_type = data['grb_type']

# Remove categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns
X = X.drop(columns=categorical_columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_type, test_size=0.2, random_state=42, stratify=y_type
)

# Scale features
X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

# Check for NaN values
print("\nChecking for NaN values before SMOTE:")
if isinstance(X_train_scaled, pd.DataFrame):
    nan_cols = X_train_scaled.columns[X_train_scaled.isnull().any()].tolist()
    if nan_cols:
        print(f"Columns containing NaN values: {nan_cols}")

# Apply SMOTE
X_train_balanced, y_train_balanced = apply_balanced_smote(X_train_scaled, y_train)

# First model: XGBoost
print("Training XGBoost Model (Multi-Class Classification)...")
print("\nStarting XGBoost Model Training...")

# Get feature names
feature_names = X_train_balanced.columns.tolist()
X_test_scaled.columns = feature_names  # Update test data columns

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.01,
    max_depth=8,
    random_state=42,
    objective='multi:softmax',
    num_class=3,
    eval_metric=['mlogloss', 'merror'],
    verbosity=1,
    early_stopping=30,
    feature_names=feature_names
)

# XGBoost training
print("\nStarting XGBoost Model Training...")
start_time = time.time()

# Update eval_set for balanced data
eval_set = [(X_train_balanced, y_train_balanced), (X_test_scaled, y_test)]

# Model training
xgb_model.fit(
    X_train_balanced, y_train_balanced,
    eval_set=eval_set,
    verbose=True
)

# Use same feature order for predictions
xgb_time = time.time() - start_time
xgb_f1 = f1_score(y_test, xgb_model.predict(X_test_scaled[feature_names]), average='weighted')
print(f"XGBoost - Time: {xgb_time:.2f} seconds, F1 Score: {xgb_f1:.4f}")

print(f"\nXGBoost Results:")
print(f"Training Accuracy: {accuracy_score(y_train_balanced, xgb_model.predict(X_train_balanced[feature_names])):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, xgb_model.predict(X_test_scaled[feature_names])):.4f}")

# Get XGBoost predictions - maintain feature order
xgb_train_pred = xgb_model.predict(X_train_balanced[feature_names])
xgb_test_pred = xgb_model.predict(X_test_scaled[feature_names])
xgb_train_pred_proba = xgb_model.predict_proba(X_train_balanced[feature_names])
xgb_test_pred_proba = xgb_model.predict_proba(X_test_scaled[feature_names])

# Update LightGBM model parameters
params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': ['multi_logloss', 'multi_error'],
    'learning_rate': 0.05,
    'max_depth': 6,
    'num_leaves': 50,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'min_data_in_leaf': 20,  
    'min_gain_to_split': 1e-3,  
    'verbose': 1,
    'min_child_samples': 20,  
    'min_child_weight': 1e-3  
}

# Update LightGBM training function
def prepare_lightgbm_data(X_train, X_test):
    # Convert to DataFrame
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    
    # Make column names unique
    column_names = []
    for col in X_train_df.columns:
        base_name = str(col)
        if base_name in column_names:
            # If name already exists, add number
            counter = 1
            while f"{base_name}_{counter}" in column_names:
                counter += 1
            column_names.append(f"{base_name}_{counter}")
        else:
            column_names.append(base_name)
    
    # Assign new column names
    X_train_df.columns = column_names
    X_test_df.columns = column_names
    
    return X_train_df, X_test_df

# Prepare data for LightGBM
X_train_lgb, X_test_lgb = prepare_lightgbm_data(X_train_balanced, X_test_scaled)

# LightGBM callback function
def lgb_training_callback(env):
    print(f"LightGBM Epoch {env.iteration + 1:3d} | "
          f"Train Loss: {env.evaluation_result_list[0][2]:.4f} | "
          f"Val Loss: {env.evaluation_result_list[1][2]:.4f}")

# LightGBM training
print("\nStarting LightGBM Model Training...")
lgb_train = lgb.Dataset(X_train_lgb, y_train_balanced)
lgb_eval = lgb.Dataset(X_test_lgb, y_test, reference=lgb_train)

# LightGBM training
lgb_model = lgb.train(
    params,
    lgb_train,
    num_boost_round=100,
    valid_sets=[lgb_train, lgb_eval],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.log_evaluation(period=1),
        lgb.early_stopping(30),
        lgb_training_callback
    ]
)

# Get LightGBM predictions
lgb_train_pred = np.argmax(lgb_model.predict(X_train_lgb), axis=1)
lgb_test_pred = np.argmax(lgb_model.predict(X_test_lgb), axis=1)
lgb_train_pred_proba = lgb_model.predict(X_train_lgb)
lgb_test_pred_proba = lgb_model.predict(X_test_lgb)

# Prepare data for TabNet
X_train_tabnet, X_test_tabnet, y_train_tabnet, y_test_tabnet = prepare_data_for_tabnet(X_train, X_test, y_train, y_test)

# Define TabNet model
tabnet_model = TabNetClassifier(
    n_d=64,
    n_a=64,
    n_steps=3,
    gamma=1.5,
    n_independent=2,
    n_shared=2,
    lambda_sparse=1e-4,
    momentum=0.3,
    clip_value=2,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size": 10, "gamma": 0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    epsilon=1e-15,
    device_name='cuda:0',
    verbose=1  
)

# TabNet training
print("\nStarting TabNet Model Training...")
tabnet_model.fit(
    X_train_tabnet, y_train_tabnet,
    eval_set=[(X_train_tabnet, y_train_tabnet), (X_test_tabnet, y_test_tabnet)],
    eval_name=['train', 'valid'],
    eval_metric=['accuracy'],
    max_epochs=100,
    patience=20,
    batch_size=512,
    virtual_batch_size=64,
    num_workers=0,
    drop_last=False
)

# Update helper function for TabNet predictions
def get_tabnet_predictions(model, X):
    try:
        # Convert data to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Check and fix data type
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        
        # Make predictions on GPU
        with torch.cuda.device('cuda:0'):
            with torch.no_grad():
                preds = model.predict(X)
                proba = model.predict_proba(X)
        
        return preds, proba
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print("\nError details:")
        import traceback
        print(traceback.format_exc())
        raise

# Get TabNet predictions
tabnet_train_pred, tabnet_train_pred_proba = get_tabnet_predictions(tabnet_model, X_train_tabnet)
tabnet_test_pred, tabnet_test_pred_proba = get_tabnet_predictions(tabnet_model, X_test_tabnet)

# Update CatBoost training function
print("\nStarting CatBoost Model Training...")
catboost_model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=True,
    min_data_in_leaf=20,
    loss_function='MultiClass',
    eval_metric='MultiClass',
    classes_count=3,
    auto_class_weights='Balanced'
)

# CatBoost training
catboost_model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    verbose=True, 
    plot=False
)

# Get CatBoost predictions
catboost_train_pred = catboost_model.predict(X_train)
catboost_test_pred = catboost_model.predict(X_test)
catboost_train_pred_proba = catboost_model.predict_proba(X_train)
catboost_test_pred_proba = catboost_model.predict_proba(X_test)

# Update final predictions function
def get_ensemble_predictions(predictions_list, probas_list):
    # Majority voting for class predictions
    ensemble_pred = np.zeros(len(predictions_list[0]))
    for i in range(len(ensemble_pred)):
        votes = [pred[i].item() if isinstance(pred[i], np.ndarray) else pred[i] for pred in predictions_list]
        ensemble_pred[i] = max(set(votes), key=votes.count)
    
    # Average probabilities
    normalized_probas = []
    for proba in probas_list:
        if isinstance(proba, np.ndarray):
            # If predictions are binary, convert to three classes
            if proba.shape[1] == 2:
                # Add zero probability for 3rd class
                zeros = np.zeros((proba.shape[0], 1))
                proba = np.hstack((proba, zeros))
            normalized_probas.append(proba)
        else:
            # Special handling for TabNet output
            proba_array = np.array(proba)
            if proba_array.shape[1] == 2:
                zeros = np.zeros((proba_array.shape[0], 1))
                proba_array = np.hstack((proba_array, zeros))
            normalized_probas.append(proba_array)
    
    # Check shapes of probability arrays
    shapes = [p.shape for p in normalized_probas]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError(f"Probability arrays are still incompatible: {shapes}")
    
    # Average probabilities
    ensemble_proba = np.mean(normalized_probas, axis=0)
    
    return ensemble_pred, ensemble_proba

# List model predictions
predictions_list = [
    xgb_test_pred,
    lgb_test_pred,
    tabnet_test_pred,
    catboost_test_pred
]

# Format probabilities
probas_list = [
    xgb_test_pred_proba,
    lgb_test_pred_proba,
    tabnet_test_pred_proba,
    catboost_test_pred_proba
]

# Get ensemble predictions
ensemble_pred, ensemble_proba = get_ensemble_predictions(predictions_list, probas_list)

# Update visualization section
plt.figure(figsize=(20, 6))

# Class distribution
plt.subplot(1, 3, 1)
for i, model in enumerate(['XGBoost', 'LightGBM', 'TabNet', 'CatBoost', 'Ensemble']):
    if i < 4:
        preds = predictions_list[i]
    else:
        preds = ensemble_pred
    plt.hist(preds, alpha=0.5, label=model, bins=3)
plt.title('Class Distribution of Model Predictions')
plt.xlabel('Class')
plt.ylabel('Count')
plt.legend()

# Model performance comparison
plt.subplot(1, 3, 2)
accuracies = []
for pred in predictions_list + [ensemble_pred]:
    if len(pred) != len(y_test):
        pred = pred[:len(y_test)]
    accuracies.append(accuracy_score(y_test, pred))

colors = ['blue', 'green', 'red', 'purple', 'orange']
plt.bar(range(len(accuracies)), accuracies, color=colors)
plt.xticks(range(len(accuracies)), 
          ['XGBoost', 'LightGBM', 'TabNet', 'CatBoost', 'Ensemble'], 
          rotation=45)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')

# Add values to graph
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center')

# Confusion Matrix visualization
plt.subplot(1, 3, 3)
cm = confusion_matrix(y_test, ensemble_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Ensemble Model Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')

plt.tight_layout(pad=3.0)
plt.show()

# Individual confusion matrix visualization for each model
plt.figure(figsize=(20, 5))
model_names = ['XGBoost', 'LightGBM', 'TabNet', 'CatBoost']

for i, (preds, name) in enumerate(zip(predictions_list, model_names)):
    plt.subplot(1, 4, i+1)
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')

plt.tight_layout(pad=3.0)
plt.show()

# Model performance reports
for i, (preds, name) in enumerate(zip(predictions_list + [ensemble_pred], 
                                    ['XGBoost', 'LightGBM', 'TabNet', 'CatBoost', 'Ensemble'])):
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, preds, 
                              target_names=['Short', 'Long', 'Ultra-Long']))

# XGBoost results
print(f"\nXGBoost Results:")
print(f"Training Accuracy: {accuracy_score(y_train_balanced, xgb_model.predict(X_train_balanced[feature_names])):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, xgb_model.predict(X_test_scaled[feature_names])):.4f}")
print(f"XGBoost - Time: {xgb_time:.2f} seconds, F1 Score: {xgb_f1:.4f}")

def add_classification_results(data, predictions_dict):
    # Add predictions for each model
    data['xgboost_prediction'] = predictions_dict['XGBoost']
    data['lightgbm_prediction'] = predictions_dict['LightGBM']
    data['tabnet_prediction'] = predictions_dict['TabNet']
    data['catboost_prediction'] = predictions_dict['CatBoost']
    data['ensemble_prediction'] = predictions_dict['Ensemble']
    
    # Convert prediction values to labels
    prediction_labels = {
        0: 'Short GRB',
        1: 'Long GRB',
        2: 'Ultra-Long GRB'
    }
    
    # Create labeled columns for each model
    for model in ['xgboost', 'lightgbm', 'tabnet', 'catboost', 'ensemble']:
        col_name = f'{model}_class'
        data[col_name] = data[f'{model}_prediction'].map(prediction_labels)
    
    return data

# Model predictions in a dictionary - feature_names used
test_predictions = {
    'XGBoost': xgb_model.predict(X_test_scaled[feature_names]),
    'LightGBM': lgb_test_pred,
    'TabNet': tabnet_test_pred,
    'CatBoost': catboost_test_pred,
    'Ensemble': ensemble_pred
}

# Add predictions to test data
X_test_with_predictions = X_test_scaled.copy() 
X_test_with_predictions = add_classification_results(X_test_with_predictions, test_predictions)

# Show results
print("\nFirst 5 rows of prediction results:")
prediction_columns = [col for col in X_test_with_predictions.columns if 'prediction' in col or 'class' in col]
print(X_test_with_predictions[prediction_columns].head())

# Show prediction distributions for each model
print("\nPrediction distributions for each model:")
for model in ['xgboost', 'lightgbm', 'tabnet', 'catboost', 'ensemble']:
    print(f"\n{model.capitalize()} class distribution:")
    print(X_test_with_predictions[f'{model}_class'].value_counts())

def save_classification_to_csv(data, satellite_data):
    for satellite, df in satellite_data.items():
        try:
            print(f"\nProcessing: {satellite}")
            
            # Find T90 column and clean values
            if satellite == 'Fermi GBM':
                t90_col = 't90'
                t90_values = df[t90_col].astype(float)
            else:
                t90_col = [col for col in df.columns if 't90' in col.lower()][0]
                t90_values = pd.to_numeric(df[t90_col], errors='coerce')
            
            # Logarithmic transformation and error calculation
            log_t90 = np.log10(t90_values.values)
            t90_error = 0.1 * t90_values.values
            log_t90_error = np.abs(t90_error / (t90_values.values * np.log(10)))
            
            # Binomial distribution parameters
            n_trials = len(log_t90)
            p_success = 0.05
            
            # Threshold calculation
            binom_threshold_upper = stats.binom.ppf(1 - p_success, n_trials, p_success) / n_trials
            binom_threshold_lower = stats.binom.ppf(p_success, n_trials, p_success) / n_trials
            
            upper_base = np.percentile(log_t90, (1 - p_success) * 100)
            lower_base = np.percentile(log_t90, p_success * 100)
            
            error_margin = np.mean(log_t90_error) * 1.96
            
            threshold_upper = upper_base * (1 + binom_threshold_upper) + error_margin
            threshold_lower = lower_base * (1 + binom_threshold_lower) - error_margin
            
            # Classification
            conditions = [
                (log_t90 > threshold_upper),
                (log_t90 > threshold_lower) & (log_t90 <= threshold_upper),
                (log_t90 <= threshold_lower)
            ]
            values = [2, 1, 0]
            
            df = df.copy()
            df['grb_class'] = np.select(conditions, values, default=np.nan)
            
            # Save as CSV
            output_path = os.path.join(data_path, f'{satellite.replace(" ", "_")}_classified.csv')
            df.to_csv(output_path, index=False)
            print(f"{satellite} data saved: {output_path}")
            
            # Show class distribution
            print(f"{satellite} GRB Class Distribution:")
            class_dist = df['grb_class'].value_counts().sort_index()
            total = len(df[df['grb_class'].notna()])
            for class_val, count in class_dist.items():
                class_name = {0: 'Short', 1: 'Long', 2: 'Ultra-long'}[int(class_val)]
                print(f"{class_name} GRB: {count} ({count/total*100:.2f}%)")
            
            print(f"\nThreshold Values:")
            print(f"Lower Threshold: 10^{threshold_lower:.2f} seconds")
            print(f"Upper Threshold: 10^{threshold_upper:.2f} seconds")
            
        except Exception as e:
            print(f"\nError for {satellite}: {str(e)}")
            print("Error details:")
            import traceback
            print(traceback.format_exc())
            continue

# Function call
save_classification_to_csv(data, satellite_data)