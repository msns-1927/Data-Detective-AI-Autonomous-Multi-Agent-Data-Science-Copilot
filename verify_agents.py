import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our custom agents
from profiling_agent import profile_dataset
from analytics_agent import detect_outliers, analyze_correlations, analyze_feature_importance
from report_agent import get_ml_recommendations, generate_pdf_report

def run_verification():
    print("🚀 Starting Data Detective AI Agent Verification...")
    
    # 1. Create a synthetic dataset
    print("\n--- Step 1: Creating Synthetic Dataset ---")
    np.random.seed(42)
    n_samples = 150
    
    # Linear relationship: price is roughly 300 * size + noise
    size = np.random.normal(1500, 300, n_samples)
    price = 300 * size + np.random.normal(0, 20000, n_samples)
    
    # Inject outliers in price
    price[0] = 2500000.0  # extreme high outlier
    price[1] = 5000.0     # extreme low outlier
    
    rooms = np.random.randint(1, 6, n_samples)
    
    # Categoricals
    neighborhoods = np.random.choice(["Downtown", "Suburbs", "Uptown", "Rural"], size=n_samples)
    has_pool = np.random.choice(["Yes", "No"], size=n_samples, p=[0.2, 0.8])
    
    df = pd.DataFrame({
        "house_size": size,
        "price": price,
        "rooms": rooms,
        "neighborhood": neighborhoods,
        "has_pool": has_pool
    })
    
    # Inject some missing values
    df.loc[10:20, "house_size"] = np.nan
    df.loc[40:42, "neighborhood"] = np.nan
    
    print(f"Created synthetic DataFrame with shape: {df.shape}")
    print("Missing counts:")
    print(df.isna().sum())
    
    # 2. Run Profiling Agent
    print("\n--- Step 2: Running Profiling Agent ---")
    profile = profile_dataset(df)
    
    assert profile["total_rows"] == n_samples
    assert profile["total_cols"] == 5
    assert "house_size" in profile["column_types"]["numerical"]
    assert "price" in profile["column_types"]["numerical"]
    assert "neighborhood" in profile["column_types"]["categorical"]
    
    print("Profiling Successful!")
    print(f"Data Quality Score: {profile['quality_score']}/100")
    print(f"Memory Usage: {profile['memory_usage']}")
    print(f"Detected columns: Numerical: {profile['column_types']['numerical']}, Categorical: {profile['column_types']['categorical']}")

    # 3. Run Analytics Agent
    print("\n--- Step 3: Running Analytics Agent ---")
    num_cols = profile["column_types"]["numerical"]
    outliers = detect_outliers(df, num_cols)
    correlations = analyze_correlations(df, num_cols)
    
    # Verify outliers detected in price
    assert "price" in outliers
    print(f"Outliers detected in 'price': {outliers['price']['count']} outliers ({outliers['price']['percentage']}%).")
    
    # Verify correlation
    top_corr = correlations["top_pairs"][0]
    print(f"Top correlation: {top_corr['feature_a']} and {top_corr['feature_b']} = {top_corr['coefficient']:.4f}")
    assert top_corr["abs_coefficient"] > 0.3
    
    # Run Feature Importance
    importance = analyze_feature_importance(df, "price", profile)
    assert "importances" in importance
    assert "house_size" in importance["importances"]
    print("Feature importances computed successfully:")
    for feat, val in importance["importances"].items():
        print(f"  - {feat}: {val:.4f}")
        
    # 4. Model Recommendation check
    print("\n--- Step 4: Model Recommendation Engine ---")
    recs = get_ml_recommendations(profile, "price", importance)
    assert len(recs) > 0
    print(f"Generated {len(recs)} models suggestions. Top suggestion: {recs[0]['name']}")
    
    # 5. Compile PDF Report Agent
    print("\n--- Step 5: Generating PDF Report ---")
    output_pdf = "verification_report.pdf"
    
    # Mock some AI insights for test PDF
    mock_ai_insights = {
        "success": True,
        "insights": {
            "executive_summary": "This is a synthetic housing price dataset generated for validation. It features size and room variables which exhibit expected linear correlations with price.",
            "data_quality": "The dataset contains standard missing values in 'house_size' and 'neighborhood' columns. Outliers are present in the price column, which may require robust methods or scaling.",
            "relationships": "There is a strong positive correlation between 'house_size' and 'price'. Outliers in price represent ultra-premium properties.",
            "action_plan": "We recommend using Random Forest or XGBoost models to predict housing prices, since they are robust to the outliers injected in this dataset."
        }
    }
    
    generate_pdf_report(
        profile_data=profile,
        analytics_data={"outliers": outliers, "correlations": correlations},
        ai_insights=mock_ai_insights,
        target_col="price",
        importance_data=importance,
        output_path=output_pdf
    )
    
    assert os.path.exists(output_pdf)
    print(f"PDF Report generated successfully at: {os.path.abspath(output_pdf)}")
    
    print("\n🎉 ALL AGENTS VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    run_verification()
