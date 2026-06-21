import pandas as pd
import numpy as np

def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Analyzes the uploaded DataFrame and generates a structured data profile.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    dict: A dictionary containing profiling details and a data quality score.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    
    if total_rows == 0:
        return {
            "error": "Dataset is empty.",
            "total_rows": 0,
            "total_cols": total_cols,
            "quality_score": 0
        }

    # Memory usage
    try:
        memory_bytes = df.memory_usage(deep=True).sum()
        if memory_bytes < 1024:
            memory_str = f"{memory_bytes} B"
        elif memory_bytes < 1024 * 1024:
            memory_str = f"{memory_bytes / 1024:.2f} KB"
        else:
            memory_str = f"{memory_bytes / (1024 * 1024):.2f} MB"
    except Exception:
        memory_str = "Unknown"

    # Duplicates
    num_duplicates = int(df.duplicated().sum())
    duplicate_ratio = num_duplicates / total_rows

    # Parse columns and identify types
    col_profiles = {}
    numerical_cols = []
    categorical_cols = []
    datetime_cols = []
    text_cols = []
    
    total_cells = total_rows * total_cols
    total_missing_cells = 0

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        total_missing_cells += missing_count
        missing_pct = (missing_count / total_rows) * 100
        num_unique = series.nunique()
        
        # Determine column type
        inferred_type = "categorical"
        
        # Check if already datetime or can be converted easily
        if pd.api.types.is_datetime64_any_dtype(series):
            inferred_type = "datetime"
        elif pd.api.types.is_numeric_dtype(series):
            # Check if boolean-like or float/int
            inferred_type = "numerical"
        else:
            # Let's try converting a sample to datetime to see if it fits
            sample_non_null = series.dropna().head(100)
            try:
                # If it's a string, see if it is mostly datetime
                if len(sample_non_null) > 0 and isinstance(sample_non_null.iloc[0], str):
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        parsed_sample = pd.to_datetime(sample_non_null, errors='coerce')
                    if parsed_sample.notna().sum() / len(sample_non_null) > 0.8:
                        inferred_type = "datetime"
            except Exception:
                pass
                
            if inferred_type != "datetime":
                # Check if it is a text column (high cardinality and long strings)
                avg_length = 0
                sample_str = series.dropna().astype(str).head(100)
                if len(sample_str) > 0:
                    avg_length = sample_str.str.len().mean()
                
                if avg_length > 50 and num_unique / total_rows > 0.5:
                    inferred_type = "text"
                else:
                    inferred_type = "categorical"

        # Categorize
        if inferred_type == "numerical":
            numerical_cols.append(col)
        elif inferred_type == "categorical":
            categorical_cols.append(col)
        elif inferred_type == "datetime":
            datetime_cols.append(col)
        else:
            text_cols.append(col)

        # Base profile info
        col_info = {
            "name": col,
            "type": inferred_type,
            "dtype": str(series.dtype),
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": num_unique,
            "samples": [str(val) for val in series.dropna().unique()[:5]]
        }

        # Dtype specific stats
        if inferred_type == "numerical":
            col_info.update({
                "min": float(series.min()) if not pd.isna(series.min()) else None,
                "max": float(series.max()) if not pd.isna(series.max()) else None,
                "mean": float(series.mean()) if not pd.isna(series.mean()) else None,
                "median": float(series.median()) if not pd.isna(series.median()) else None,
                "std": float(series.std()) if not pd.isna(series.std()) else None,
                "skew": float(series.skew()) if not pd.isna(series.skew()) else 0.0
            })
        elif inferred_type == "categorical":
            mode_val = series.mode()
            top_val = str(mode_val.iloc[0]) if len(mode_val) > 0 else None
            top_freq = int(series.value_counts().iloc[0]) if len(series.dropna()) > 0 else 0
            col_info.update({
                "top_value": top_val,
                "top_frequency": top_freq,
                "top_percentage": (top_freq / total_rows) * 100 if total_rows > 0 else 0
            })
            
        col_profiles[col] = col_info

    # Quality Score Calculation
    quality_score = 100.0
    
    # 1. Missing data penalty (max -30)
    missing_pct_overall = (total_missing_cells / total_cells) if total_cells > 0 else 0
    quality_score -= (missing_pct_overall * 30.0)
    
    # 2. Duplicate rows penalty (max -15)
    quality_score -= (duplicate_ratio * 15.0)
    
    # 3. Columns with 100% missing values penalty (max -15)
    cols_all_missing = sum(1 for col in col_profiles.values() if col["missing_pct"] == 100.0)
    quality_score -= min(cols_all_missing * 5.0, 15.0)
    
    # 4. Extreme skewness penalty for numerical features (max -10)
    skewed_cols = 0
    for col in numerical_cols:
        skew = col_profiles[col].get("skew", 0.0)
        if abs(skew) > 3.0:
            skewed_cols += 1
    quality_score -= min(skewed_cols * 2.0, 10.0)

    # 5. Extremely high cardinality in categorical (that are not text) (max -10)
    high_card_cols = 0
    for col in categorical_cols:
        unique_cnt = col_profiles[col]["unique_count"]
        # If cardinality is high relative to row count and column is not identifier-like
        if unique_cnt > 1 and unique_cnt / total_rows > 0.4:
            high_card_cols += 1
    quality_score -= min(high_card_cols * 2.0, 10.0)

    quality_score = max(0.0, round(quality_score, 2))

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "memory_usage": memory_str,
        "duplicates": num_duplicates,
        "duplicate_ratio": duplicate_ratio,
        "quality_score": quality_score,
        "column_types": {
            "numerical": numerical_cols,
            "categorical": categorical_cols,
            "datetime": datetime_cols,
            "text": text_cols
        },
        "columns": col_profiles
    }
