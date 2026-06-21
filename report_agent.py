import os
import pandas as pd
import numpy as np
from fpdf import FPDF

def clean_pdf_text(text: str) -> str:
    """
    Cleans Unicode characters to make them compatible with FPDF standard fonts (Latin-1).
    """
    if not text:
        return ""
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2022": "*",
        "\u2026": "...",
        "\u2212": "-",
        "\u00ae": "(R)",
        "\u2122": "(TM)",
        "\u00a9": "(C)",
        "\u20ac": "EUR",
        "\u00a3": "GBP"
    }
    for search, replace in replacements.items():
        text = text.replace(search, replace)
    return text.encode("latin-1", "replace").decode("latin-1")

def get_ml_recommendations(profile_data: dict, target_col: str, importance_data: dict) -> list:
    """
    Recommends ML models based on dataset shapes, data types, and targets.
    """
    if not target_col:
        # Unsupervised Learning Recommendations
        return [
            {
                "name": "K-Means Clustering",
                "type": "Unsupervised Learning",
                "rationale": "Useful for grouping data points into distinct, homogeneous segments (clusters) to discover hidden behaviors.",
                "preprocessing": "StandardScaler for numerical features, and One-Hot Encoding for categorical features.",
                "evaluation": "Silhouette Score, Elbow Method (Inertia)",
                "code": """import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load data (drop target columns or text)
df = pd.read_csv('dataset.csv')
X = df.select_dtypes(include=['number']).dropna() # simple numeric subset

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=3, random_state=42))
])

df['cluster'] = pipeline.fit_predict(X)
print(df['cluster'].value_counts())
"""
            },
            {
                "name": "Principal Component Analysis (PCA)",
                "type": "Dimensionality Reduction",
                "rationale": "Reduces dataset dimensions while preserving variance, enabling 2D/3D visualizations and mitigating the curse of dimensionality.",
                "preprocessing": "Standardization is strictly required so that features are on the same scale.",
                "evaluation": "Cumulative Explained Variance Ratio",
                "code": """import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv('dataset.csv')
X = df.select_dtypes(include=['number']).dropna()

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2))
])

pca_features = pipeline.fit_transform(X)
print("Explained Variance Ratio:", pipeline.named_steps['pca'].explained_variance_ratio_)
"""
            }
        ]

    # Supervised Learning Recommendations
    task_type = importance_data.get("task_type", "classification")
    num_cols = profile_data.get("column_types", {}).get("numerical", [])
    cat_cols = profile_data.get("column_types", {}).get("categorical", [])
    
    # Filter features used
    features_used = importance_data.get("features_used", [col for col in profile_data.get("columns", {}).keys() if col != target_col])
    num_features = [c for c in num_cols if c in features_used]
    cat_features = [c for c in cat_cols if c in features_used]

    if task_type == "classification":
        return [
            {
                "name": "Random Forest Classifier",
                "type": "Supervised - Ensemble Classification",
                "rationale": "Excellent general-purpose classifier. Handles non-linear decision boundaries, handles outliers well, and requires minimal tuning.",
                "preprocessing": "Imputation of missing values. Target needs label encoding.",
                "evaluation": "Classification Report (Precision, Recall, F1), ROC-AUC Score",
                "code": f"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('dataset.csv')
X = df[{features_used}]
y = df['{target_col}']

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), {num_features}),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), {cat_features})
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
"""
            },
            {
                "name": "XGBoost Classifier",
                "type": "Supervised - Gradient Boosted Classification",
                "rationale": "State-of-the-art classifier for tabular datasets. Performs extremely well, handles missing values natively, and allows regularization to prevent overfitting.",
                "preprocessing": "Missing value handling is native. Categorical variables must be One-Hot/Target encoded.",
                "evaluation": "F1-Score, Log Loss, ROC-AUC",
                "code": f"""import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report

df = pd.read_csv('dataset.csv')
X = df[{features_used}]
# Map target classes to 0, 1, 2...
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df['{target_col}'])

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), {cat_features})
], remainder='passthrough')

model = Pipeline([
    ('preprocessor', preprocessor),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
"""
            },
            {
                "name": "Logistic Regression (Regularized)",
                "type": "Supervised - Linear Classification",
                "rationale": "Simple, highly interpretable baseline. With L1/L2 regularization, it performs well on linearly separable problems and is extremely fast.",
                "preprocessing": "Scaling is mandatory. Missing value imputation is required.",
                "evaluation": "Accuracy, F1-Score, Confusion Matrix",
                "code": f"""import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

df = pd.read_csv('dataset.csv')
X = df[{features_used}]
y = df['{target_col}']

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), {num_features}),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), {cat_features})
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
"""
            }
        ]
    else:
        # Regression models
        return [
            {
                "name": "Random Forest Regressor",
                "type": "Supervised - Ensemble Regression",
                "rationale": "Very strong and flexible non-linear model. Captures complex relationships, does not require normal scaling of features, and is robust to outliers.",
                "preprocessing": "Numerical imputation, One-Hot Encoding for categorical features.",
                "evaluation": "R-squared (R2), Root Mean Squared Error (RMSE)",
                "code": f"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('dataset.csv')
X = df[{features_used}]
y = df['{target_col}']

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), {num_features}),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), {cat_features})
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("R2 Score:", r2_score(y_test, preds))
print("RMSE:", mean_squared_error(y_test, preds, squared=False))
"""
            },
            {
                "name": "XGBoost Regressor",
                "type": "Supervised - Gradient Boosted Regression",
                "rationale": "Optimized gradient boosting. Offers high accuracy, speed, and standard hyperparameter options to build high-performance regressors.",
                "preprocessing": "Categorical feature encoding required.",
                "evaluation": "R-squared (R2), Mean Absolute Error (MAE)",
                "code": f"""import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('dataset.csv')
X = df[{features_used}]
y = df['{target_col}']

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), {cat_features})
], remainder='passthrough')

model = Pipeline([
    ('preprocessor', preprocessor),
    ('xgb', XGBRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("R2 Score:", r2_score(y_test, preds))
print("MAE:", mean_absolute_error(y_test, preds))
"""
            },
            {
                "name": "Ridge Regression",
                "type": "Supervised - Regularized Linear Regression",
                "rationale": "Standard linear regression baseline that uses L2 weight decay regularization to avoid overfitting on collinear features.",
                "preprocessing": "Standard Scaling is required. Imputation of numerical columns.",
                "evaluation": "R-squared (R2), MAE, coefficients evaluation",
                "code": f"""import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

df = pd.read_csv('dataset.csv')
X = df[{features_used}]
y = df['{target_col}']

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), {num_features}),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), {cat_features})
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("R2 Score:", r2_score(y_test, preds))
"""
            }
        ]

class PDFReport(FPDF):
    def header(self):
        # Draw background color banner for header
        self.set_fill_color(24, 28, 56) # Deep Navy Indigo
        self.rect(0, 0, 210, 20, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('helvetica', 'B', 11)
        self.cell(0, -6, 'DATA DETECTIVE AI - AUTO-GENERATED ANALYTICS REPORT', align='L')
        self.set_x(170)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, -6, 'CONFIDENTIAL', align='R')
        self.ln(12)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')
        
    def section_title(self, label):
        self.set_font('helvetica', 'B', 14)
        self.set_text_color(0, 150, 136) # Teal Accent
        self.cell(0, 8, label, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 150, 136)
        self.set_line_width(0.5)
        self.line(self.get_x(), self.get_y(), 200, self.get_y())
        self.ln(4)
        
    def section_body(self, text):
        if not text:
            return
            
        lines = text.split("\n")
        
        for line in lines:
            if not line.strip():
                self.ln(4)
                continue
                
            stripped = line.strip()
            
            # Reset standard body styling
            self.set_font('helvetica', '', 10)
            self.set_text_color(50, 50, 50)
            
            # Identify leading space indentation
            leading_spaces = len(line) - len(line.lstrip())
            indent = 10
            content = line
            
            # Render Markdown headers inside blocks
            if stripped.startswith("# "):
                self.set_font('helvetica', 'B', 15)
                self.set_text_color(24, 28, 56)
                self.write(7, clean_pdf_text(stripped[2:]))
                self.ln(8)
                continue
            elif stripped.startswith("## "):
                self.set_font('helvetica', 'B', 13)
                self.set_text_color(24, 28, 56)
                self.write(6, clean_pdf_text(stripped[3:]))
                self.ln(7)
                continue
            elif stripped.startswith("### "):
                self.set_font('helvetica', 'B', 11)
                self.set_text_color(0, 150, 136)
                self.write(6, clean_pdf_text(stripped[4:]))
                self.ln(6)
                continue
                
            # Render Bullet Points or Lists
            if stripped.startswith("* ") or stripped.startswith("- "):
                indent = 15 + leading_spaces * 1.5
                self.set_x(indent)
                self.write(5, "- ")
                content = stripped[2:]
            elif stripped.startswith("• "):
                indent = 15 + leading_spaces * 1.5
                self.set_x(indent)
                self.write(5, "- ")
                content = stripped[2:]
            elif ". " in stripped and stripped.split(". ")[0].isdigit() and stripped.split(". ")[0] != "":
                num_part = stripped.split(". ")[0] + ". "
                indent = 15 + leading_spaces * 1.5
                self.set_x(indent)
                self.write(5, num_part)
                content = stripped[len(num_part):]
            else:
                if leading_spaces > 0:
                    self.set_x(10 + leading_spaces * 1.5)
                    
            # Parse bold elements inside lines (alternating odd indexes as Bold)
            parts = content.split("**")
            for idx, part in enumerate(parts):
                cleaned_part = clean_pdf_text(part)
                if idx % 2 == 1:
                    self.set_font('helvetica', 'B', 10)
                else:
                    self.set_font('helvetica', '', 10)
                self.write(5, cleaned_part)
                
            self.ln(5)

def generate_pdf_report(
    profile_data: dict, 
    analytics_data: dict, 
    ai_insights: dict, 
    target_col: str, 
    importance_data: dict, 
    output_path: str
):
    """
    Compiles data profiling metrics, statistical analyses, Gemini insights,
    and model recommendations into a highly polished business PDF report.
    """
    pdf = PDFReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Document Title
    pdf.ln(5)
    pdf.set_font('helvetica', 'B', 22)
    pdf.set_text_color(24, 28, 56)
    pdf.cell(0, 10, clean_pdf_text('Dataset Analysis & Insights Report'), new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.set_font('helvetica', 'I', 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, clean_pdf_text('Powered by Data Detective AI & Gemini'), new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(5)
    
    
    # Executive Metadata Table
    pdf.set_fill_color(245, 245, 250)
    pdf.set_draw_color(220, 220, 225)
    
    pdf.set_font('helvetica', 'B', 10)
    pdf.set_text_color(24, 28, 56)
    
    # Grid metrics
    pdf.cell(45, 8, 'Total Rows', 1, 0, 'C', True)
    pdf.cell(45, 8, 'Total Columns', 1, 0, 'C', True)
    pdf.cell(50, 8, 'Memory Footprint', 1, 0, 'C', True)
    pdf.cell(50, 8, 'Data Quality Score', 1, 1, 'C', True)
    
    pdf.set_font('helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(45, 8, str(profile_data['total_rows']), 1, 0, 'C')
    pdf.cell(45, 8, str(profile_data['total_cols']), 1, 0, 'C')
    pdf.cell(50, 8, str(profile_data['memory_usage']), 1, 0, 'C')
    
    score = profile_data['quality_score']
    pdf.set_font('helvetica', 'B', 10)
    if score >= 80:
        pdf.set_text_color(46, 125, 50) # Green
    elif score >= 50:
        pdf.set_text_color(239, 108, 0) # Orange
    else:
        pdf.set_text_color(198, 40, 40) # Red
    pdf.cell(50, 8, f"{score}/100", 1, 1, 'C')
    
    pdf.ln(8)
    
    
    # Section 1: Executive Summary
    pdf.section_title('1. Executive Summary')
    if ai_insights.get("success", False):
        summary_text = ai_insights["insights"]["executive_summary"]
    else:
        summary_text = "This report summarizes the structure and core characteristics of the uploaded dataset. AI insights were not generated."
    pdf.section_body(summary_text)
    

    # Section 2: Data Quality & Cleaning Details
    pdf.add_page()
    pdf.section_title('2. Data Quality & Preprocessing Plan')
    
    # Subheader
    pdf.set_font('helvetica', 'B', 11)
    pdf.set_text_color(24, 28, 56)
    pdf.cell(0, 6, 'Data Cleaning Requirements:', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    
    if ai_insights.get("success", False):
        quality_text = ai_insights["insights"]["data_quality"]
    else:
        quality_text = "No custom recommendations generated. Inspect missing columns and outlier features in the dashboard."
    pdf.section_body(quality_text)
    
    # Outliers Table if present
    outliers = analytics_data.get("outliers", {})
    if outliers:
        pdf.set_font('helvetica', 'B', 10)
        pdf.set_text_color(24, 28, 56)
        pdf.cell(0, 6, 'Detected Numerical Outliers (IQR):', new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        
        pdf.set_fill_color(245, 245, 250)
        pdf.cell(50, 7, 'Column', 1, 0, 'L', True)
        pdf.cell(30, 7, 'Outliers Count', 1, 0, 'C', True)
        pdf.cell(30, 7, 'Percentage', 1, 0, 'C', True)
        pdf.cell(80, 7, 'Normal Boundary', 1, 1, 'C', True)
        
        pdf.set_font('helvetica', '', 9)
        pdf.set_text_color(50, 50, 50)
        for col_name, o_info in outliers.items():
            pdf.cell(50, 7, clean_pdf_text(col_name), 1, 0, 'L')
            pdf.cell(30, 7, str(o_info['count']), 1, 0, 'C')
            pdf.cell(30, 7, f"{o_info['percentage']}%", 1, 0, 'C')
            pdf.cell(80, 7, f"[{o_info['lower_bound']:.2f}, {o_info['upper_bound']:.2f}]", 1, 1, 'C')
            
        pdf.ln(6)
        

    # Section 3: Relationship & Anomaly Analysis
    pdf.section_title('3. Relationship & Trend Discoveries')
    if ai_insights.get("success", False):
        relations_text = ai_insights["insights"]["relationships"]
    else:
        relations_text = "No relationships text generated."
    pdf.section_body(relations_text)
    
    # Top Correlations Table
    correlations = analytics_data.get("correlations", {}).get("top_pairs", [])
    if correlations:
        pdf.set_font('helvetica', 'B', 10)
        pdf.set_text_color(24, 28, 56)
        pdf.cell(0, 6, 'Key Linear Relationships (Pearson Correlation):', new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        
        pdf.set_fill_color(245, 245, 250)
        pdf.cell(70, 7, 'Feature A', 1, 0, 'L', True)
        pdf.cell(70, 7, 'Feature B', 1, 0, 'L', True)
        pdf.cell(50, 7, 'Correlation Coefficient', 1, 1, 'C', True)
        
        pdf.set_font('helvetica', '', 9)
        pdf.set_text_color(50, 50, 50)
        for c in correlations[:5]:  # show top 5 in PDF to keep clean
            pdf.cell(70, 7, clean_pdf_text(c['feature_a']), 1, 0, 'L')
            pdf.cell(70, 7, clean_pdf_text(c['feature_b']), 1, 0, 'L')
            coeff = c['coefficient']
            pdf.set_font('helvetica', 'B', 9)
            if coeff > 0.5:
                pdf.set_text_color(46, 125, 50) # strong positive - green
            elif coeff < -0.5:
                pdf.set_text_color(198, 40, 40) # strong negative - red
            else:
                pdf.set_text_color(50, 50, 50)
            pdf.cell(50, 7, f"{coeff:+.4f}", 1, 1, 'C')
            pdf.set_font('helvetica', '', 9)
            pdf.set_text_color(50, 50, 50)
            
        pdf.ln(6)


    # Section 4: Machine Learning & Action Plan
    pdf.add_page()
    pdf.section_title('4. Business Action Plan & Predictive modeling')
    
    if ai_insights.get("success", False):
        action_text = ai_insights["insights"]["action_plan"]
    else:
        action_text = "No action plan text generated."
    pdf.section_body(action_text)
    
    # Model Recommendations
    pdf.ln(2)
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_text_color(24, 28, 56)
    pdf.cell(0, 6, 'AI-Recommended Machine Learning Models:', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    
    recs = get_ml_recommendations(profile_data, target_col, importance_data or {})
    for idx, r in enumerate(recs, 1):
        pdf.set_fill_color(245, 245, 250)
        pdf.set_font('helvetica', 'B', 10)
        pdf.set_text_color(24, 28, 56)
        pdf.cell(0, 6, f"Model {idx}: {r['name']} ({r['type']})", new_x="LMARGIN", new_y="NEXT", fill=True)
        
        pdf.set_font('helvetica', '', 9)
        pdf.set_text_color(50, 50, 50)
        # Parse bold tags inside recommended model lists
        details_list = [
            f"**Suitability:** {r['rationale']}",
            f"**Preprocessing:** {r['preprocessing']}",
            f"**Metrics:** {r['evaluation']}"
        ]
        for item in details_list:
            parts = item.split("**")
            for idx, part in enumerate(parts):
                cleaned_part = clean_pdf_text(part)
                if idx % 2 == 1:
                    pdf.set_font('helvetica', 'B', 9)
                else:
                    pdf.set_font('helvetica', '', 9)
                pdf.write(4.5, cleaned_part)
            pdf.ln(5)
        pdf.ln(2)

    # Output report
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pdf.output(output_path)
