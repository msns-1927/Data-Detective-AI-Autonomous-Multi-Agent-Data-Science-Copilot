import os
# pyrefly: ignore [missing-import]
import google.generativeai as genai

def generate_ai_insights(
    profile_data: dict, 
    analytics_data: dict, 
    target_col: str = None, 
    importance_data: dict = None, 
    api_key: str = None
) -> dict:
    """
    Constructs a comprehensive prompt for Gemini and fetches structured narrative insights.
    
    Returns a dictionary of insights.
    """
    # 1. Check API Key
    effective_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not effective_key:
        return {
            "success": False,
            "error": "Gemini API key not found. Please provide an API key in the sidebar."
        }
        
    try:
        genai.configure(api_key=effective_key)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to configure Gemini Client: {str(e)}"
        }

    # 2. Extract details to construct the prompt
    rows = profile_data.get("total_rows", 0)
    cols = profile_data.get("total_cols", 0)
    memory = profile_data.get("memory_usage", "Unknown")
    duplicates = profile_data.get("duplicates", 0)
    quality_score = profile_data.get("quality_score", 100.0)
    
    types = profile_data.get("column_types", {})
    num_cols = list(types.get("numerical", []))
    cat_cols = list(types.get("categorical", []))
    dt_cols = list(types.get("datetime", []))
    txt_cols = list(types.get("text", []))
    
    missing_summary = []
    for col_name, info in profile_data.get("columns", {}).items():
        if info["missing_count"] > 0:
            missing_summary.append(f"- {col_name}: {info['missing_count']} missing ({info['missing_pct']:.1f}%)")
            
    outliers = analytics_data.get("outliers", {})
    outlier_summary = []
    for col_name, out_info in outliers.items():
        outlier_summary.append(
            f"- {col_name}: {out_info['count']} outliers ({out_info['percentage']}%), "
            f"bounds: [{out_info['lower_bound']:.2f}, {out_info['upper_bound']:.2f}]"
        )
        
    correlations = analytics_data.get("correlations", {}).get("top_pairs", [])
    corr_summary = []
    for c in correlations:
        corr_summary.append(f"- {c['feature_a']} and {c['feature_b']}: Pearson = {c['coefficient']:.4f}")
        
    importance_summary = []
    task_type = "None"
    if target_col and importance_data:
        task_type = importance_data.get("task_type", "Unknown")
        importances = importance_data.get("importances", {})
        if "error" not in importance_data:
            for feat, val in importances.items():
                importance_summary.append(f"- {feat}: {val:.4f} importance score")
        else:
            importance_summary.append(f"Error computing importance: {importance_data.get('error')}")

    # Build Prompt
    prompt = f"""
You are the "Data Detective AI" - an autonomous expert data scientist. 
Analyze the provided metadata and metrics from a dataset and write a professional, detailed analytical report.

---
DATASET STRUCTURE:
- Shape: {rows} rows and {cols} columns
- Memory usage: {memory}
- Duplicates: {duplicates}
- Data Quality Score: {quality_score}/100

COLUMN BREAKDOWN:
- Numerical Columns ({len(num_cols)}): {", ".join(num_cols) if num_cols else "None"}
- Categorical Columns ({len(cat_cols)}): {", ".join(cat_cols) if cat_cols else "None"}
- Datetime Columns ({len(dt_cols)}): {", ".join(dt_cols) if dt_cols else "None"}
- Text/Description Columns ({len(txt_cols)}): {", ".join(txt_cols) if txt_cols else "None"}

MISSING VALUES:
{chr(10).join(missing_summary) if missing_summary else "No missing values found."}

OUTLIERS DETECTED:
{chr(10).join(outlier_summary) if outlier_summary else "No outliers detected."}

TOP PEARSON CORRELATIONS:
{chr(10).join(corr_summary) if corr_summary else "No significant correlations found."}
"""

    if target_col:
        prompt += f"""
TARGET ANALYSIS FOR FEATURE IMPORTANCE:
- Selected Target Column: {target_col}
- Inferred Task: {task_type}
- Feature Importances (Random Forest):
{chr(10).join(importance_summary) if importance_summary else "No feature importance data."}
"""

    prompt += """
---
INSTRUCTION FOR RESPONSE:
Write a comprehensive, four-part analysis. Use clean, professional formatting with markdown. Avoid using placeholders or generalities. Address the specific columns and metrics provided above.

PART 1: EXECUTIVE SUMMARY
Provide a brief but engaging summary of the dataset. What kind of dataset does this appear to be? What are the key properties, its general size, and main takeaways?

PART 2: DATA QUALITY & CLEANING RECOMMENDATIONS
Assess the data quality score. What are the main issues (missing values, duplicates, skewness, outliers)? Detail concrete preprocessing steps (e.g., specific imputation strategies for specific columns, scaling, dropping high cardinality fields) that are necessary before running any machine learning.

PART 3: ANOMALIES, RELATIONSHIPS & PATTERNS
Interpret the outliers and correlations. Explain what the correlation coefficients tell us about the relationship between specific features in a business context. What could explain the outliers or skewness observed in specific variables?

PART 4: MACHINE LEARNING & ACTION PLAN
If a target column was selected, explain how the top features drive predictions for the target. Suggest an action plan for the business or user based on these top drivers. If no target was selected, suggest 2 potential predictive modeling objectives that would yield high business value from this dataset.

Make each part detailed and well-written.
Return your response structured clearly, separating the sections with '### PART 1: EXECUTIVE SUMMARY', '### PART 2: DATA QUALITY & CLEANING RECOMMENDATIONS', '### PART 3: ANOMALIES, RELATIONSHIPS & PATTERNS', and '### PART 4: MACHINE LEARNING & ACTION PLAN'.
"""

    try:
        # Try a list of model names to ensure compatibility with different library and API version setups
        models_to_try = [
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-2.5-flash",
            "gemini-1.5-pro",
            "gemini-pro"
        ]
        
        text = ""
        last_error = None
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                text = response.text
                if text:
                    break
            except Exception as e:
                last_error = e
                continue
        
        if not text and last_error:
            raise last_error
            
        # Parse text into components
        parts = {
            "full_text": text,
            "executive_summary": "",
            "data_quality": "",
            "relationships": "",
            "action_plan": ""
        }
        
        # Helper to extract parts
        import re
        p1 = re.search(r"PART 1: EXECUTIVE SUMMARY(.*?)### PART 2", text, re.DOTALL | re.IGNORECASE)
        p2 = re.search(r"PART 2: DATA QUALITY & CLEANING RECOMMENDATIONS(.*?)### PART 3", text, re.DOTALL | re.IGNORECASE)
        p3 = re.search(r"PART 3: ANOMALIES, RELATIONSHIPS & PATTERNS(.*?)### PART 4", text, re.DOTALL | re.IGNORECASE)
        p4 = re.search(r"PART 4: MACHINE LEARNING & ACTION PLAN(.*)", text, re.DOTALL | re.IGNORECASE)
        
        parts["executive_summary"] = p1.group(1).strip() if p1 else "Executive Summary generation failed. Please see full text below."
        parts["data_quality"] = p2.group(1).strip() if p2 else "Data Quality details failed to parse."
        parts["relationships"] = p3.group(1).strip() if p3 else "Relationships analysis failed to parse."
        parts["action_plan"] = p4.group(1).strip() if p4 else "Action Plan details failed to parse."
        
        # If extraction failed because Gemini didn't use the exact header pattern, fall back to displaying the full text
        if not p1 or not p2 or not p3 or not p4:
            parts["executive_summary"] = text
            parts["data_quality"] = "See Executive Summary tab (Full Text parsed)"
            parts["relationships"] = "See Executive Summary tab (Full Text parsed)"
            parts["action_plan"] = "See Executive Summary tab (Full Text parsed)"

        return {
            "success": True,
            "insights": parts
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error running Gemini API: {str(e)}"
        }
