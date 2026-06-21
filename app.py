import streamlit as st
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import agents
from profiling_agent import profile_dataset
from analytics_agent import detect_outliers, analyze_correlations, analyze_feature_importance
from insight_agent import generate_ai_insights
from report_agent import get_ml_recommendations, generate_pdf_report

# Page Configuration
st.set_page_config(
    page_title="Data Detective AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Styling
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background-color: #0f111a;
        color: #e5e7eb;
    }
    
    /* Header styling */
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #151824;
        border-right: 1px solid #23273a;
    }
    
    /* Card design */
    .dashboard-card {
        background-color: #1a1e2f;
        border: 1px solid #2d334d;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    .dashboard-card:hover {
        border-color: #00f2fe;
        transform: translateY(-2px);
    }
    .card-num {
        font-size: 2.2rem;
        font-weight: bold;
        color: #00f2fe;
        margin-bottom: 0.2rem;
    }
    .card-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Custom buttons */
    .stButton>button {
        background: linear-gradient(135deg, #009688 0%, #00796b 100%) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #00bfa5 0%, #009688 100%) !important;
        box-shadow: 0px 4px 15px rgba(0, 150, 136, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Tab formatting */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #151824;
        padding: 8px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        border-radius: 6px;
        background-color: transparent;
        color: #94a3b8;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #00f2fe;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a1e2f !important;
        color: #00f2fe !important;
        border: 1px solid #2d334d !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to display custom KPI card
def kpi_card(label, value, color="#00f2fe"):
    st.markdown(f"""
    <div class="dashboard-card">
        <div class="card-num" style="color: {color}">{value}</div>
        <div class="card-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

# Main Application Title
st.title("🔍 Data Detective AI")
st.caption("Autonomous Agent-Based Data Analysis & Predictive Recommendations")

# Sidebar - Configuration and Uploader
with st.sidebar:
    st.markdown("<h1 style='text-align: left; font-size: 3.5rem; margin-top: -20px; margin-bottom: -10px;'>🕵️‍♂️</h1>", unsafe_allow_html=True)
    st.header("Upload Center")
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    
    st.markdown("---")
    st.subheader("Gemini Credentials")
    api_key_input = st.text_input("Enter Gemini API Key", type="password", help="Providing an API key here overrides the environment variable GEMINI_API_KEY.")
    
    # API status indicator
    gemini_key = api_key_input or os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        st.success("Gemini API key configured!")
    else:
        st.warning("Provide an API key to enable AI insights.")

    st.markdown("---")
    st.subheader("Target Configuration")
    target_placeholder = st.empty()

# Analysis Logic & State Management
if uploaded_file is not None:
    # 1. Read Data
    @st.cache_data
    def load_csv(file):
        return pd.read_csv(file)
        
    try:
        df = load_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        st.stop()
        
    # Check if target can be selected
    with target_placeholder:
        all_cols = ["None"] + list(df.columns)
        target_col = st.selectbox("Select Target Variable (Optional)", options=all_cols, index=0)
    
    target_val = None if target_col == "None" else target_col

    # 2. Cache Agent computations in Session State to prevent rerunning on every interactive change
    state_key = f"analysis_{uploaded_file.name}_{target_val}"
    
    if state_key not in st.session_state:
        with st.spinner("🕵️‍♂️ Detective is inspecting the dataset..."):
            # A. Data Profiling Agent
            profile = profile_dataset(df)
            
            # B. Analytics Agent
            num_cols = profile["column_types"]["numerical"]
            outliers = detect_outliers(df, num_cols)
            correlations = analyze_correlations(df, num_cols)
            
            # Feature Importance
            importance = None
            if target_val:
                importance = analyze_feature_importance(df, target_val, profile)
                
            analytics = {
                "outliers": outliers,
                "correlations": correlations,
                "importance": importance
            }
            
            st.session_state[state_key] = {
                "profile": profile,
                "analytics": analytics
            }
            
            # Reset AI insights state when dataset changes
            st.session_state[f"ai_insights_{state_key}"] = None
            
    # Load analysis outputs
    analysis_data = st.session_state[state_key]
    profile = analysis_data["profile"]
    analytics = analysis_data["analytics"]
    
    
    # TABBED MAIN LAYOUT
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard Overview", 
        "📋 Data Profiling", 
        "📈 EDA & Plots", 
        "🔬 Advanced Analytics", 
        "🧠 Detective AI Insights", 
        "📑 ML Recs & Report"
    ])
    
    
    # TAB 1: DASHBOARD OVERVIEW
    with tab1:
        st.subheader("Dataset Summary & Health Score")
        
        # Row of Metric Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            kpi_card("Total Rows", f"{profile['total_rows']:,}")
        with col2:
            kpi_card("Total Columns", f"{profile['total_cols']:,}")
        with col3:
            kpi_card("Memory Footprint", profile['memory_usage'])
        with col4:
            # Color health score
            qs = profile['quality_score']
            h_color = "#2e7d32" if qs >= 80 else ("#ef6c00" if qs >= 50 else "#c62828")
            kpi_card("Data Quality Score", f"{qs}%", h_color)
        with col5:
            dup_p = (profile['duplicate_ratio'] * 100)
            kpi_card("Duplicate Rows", f"{profile['duplicates']} ({dup_p:.1f}%)", "#e0a96d")
            
        st.markdown("### Raw Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)


    # TAB 2: DATA PROFILING
    with tab2:
        st.subheader("Column Types & Quality Audit")
        
        col_list = []
        for col_name, info in profile["columns"].items():
            col_list.append({
                "Column Name": col_name,
                "Inferred Type": info["type"],
                "Native Dtype": info["dtype"],
                "Unique Values": info["unique_count"],
                "Missing Values": info["missing_count"],
                "Missing %": f"{info['missing_pct']:.2f}%",
                "Sample Values": ", ".join(info["samples"])
            })
        
        profile_df = pd.DataFrame(col_list)
        st.dataframe(profile_df, use_container_width=True, hide_index=True)
        
        # Missing values visualization
        missing_df = pd.DataFrame([
            {"Column": c, "Missing %": info["missing_pct"]} 
            for c, info in profile["columns"].items()
        ]).sort_values("Missing %", ascending=False)
        
        if missing_df["Missing %"].sum() > 0:
            st.markdown("### Missing Values Distribution")
            fig = px.bar(
                missing_df[missing_df["Missing %"] > 0], 
                x="Column", 
                y="Missing %", 
                title="Percentage of Missing Values per Feature",
                color="Missing %",
                color_continuous_scale="reds"
            )
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("Fantastic! No missing values detected in the dataset.")


    # TAB 3: EDA & PLOTS
    with tab3:
        st.subheader("Exploratory Data Analysis")
        
        eda_mode = st.radio("Choose Plot Type", ["Numerical Correlation Heatmap", "Feature Distributions", "Bivariate Relationships"], horizontal=True)
        
        if eda_mode == "Numerical Correlation Heatmap":
            num_cols = profile["column_types"]["numerical"]
            if len(num_cols) >= 2:
                corr_dict = analytics["correlations"]["matrix"]
                corr_matrix = pd.DataFrame(corr_dict)
                
                fig = px.imshow(
                    corr_matrix, 
                    text_auto=".2f", 
                    aspect="auto", 
                    color_continuous_scale="RdBu", 
                    range_color=[-1, 1],
                    title="Pearson Correlation Coefficient Heatmap"
                )
                fig.update_layout(template="plotly_dark", height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("At least two numerical columns are required to draw a correlation heatmap.")
                
        elif eda_mode == "Feature Distributions":
            sel_col = st.selectbox("Select Column to Analyze", options=df.columns)
            is_num = sel_col in profile["column_types"]["numerical"]
            
            if is_num:
                fig = px.histogram(
                    df, 
                    x=sel_col, 
                    marginal="box", 
                    title=f"Distribution & Spread of {sel_col}",
                    color_discrete_sequence=["#00f2fe"]
                )
                fig.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                top_n = df[sel_col].value_counts().head(20).reset_index()
                top_n.columns = [sel_col, "Count"]
                fig = px.bar(
                    top_n, 
                    x=sel_col, 
                    y="Count", 
                    title=f"Top Categorical Distribution for {sel_col} (Max 20 categories)",
                    color="Count",
                    color_continuous_scale="blues"
                )
                fig.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)
                
        elif eda_mode == "Bivariate Relationships":
            col_x = st.selectbox("Select X Axis Variable", options=df.columns, index=0)
            col_y = st.selectbox("Select Y Axis Variable", options=df.columns, index=min(1, len(df.columns)-1))
            
            fig = px.scatter(
                df, 
                x=col_x, 
                y=col_y, 
                title=f"{col_y} vs {col_x}", 
                opacity=0.7,
                color_discrete_sequence=["#4facfe"]
            )
            fig.update_layout(template="plotly_dark", height=550)
            st.plotly_chart(fig, use_container_width=True)


    # TAB 4: ADVANCED ANALYTICS
    with tab4:
        st.subheader("Anomaly and Predictive Importance Insights")
        
        # Split into Outliers and Feature Importances
        col_out, col_imp = st.columns([1, 1])
        
        with col_out:
            st.markdown("### 🚨 Detected Outliers (IQR)")
            outliers = analytics["outliers"]
            if outliers:
                out_summary = []
                for k, v in outliers.items():
                    out_summary.append({
                        "Column": k,
                        "Count": v["count"],
                        "Percentage": f"{v['percentage']}%",
                        "Lower Bound": f"{v['lower_bound']:.2f}",
                        "Upper Bound": f"{v['upper_bound']:.2f}"
                    })
                st.dataframe(pd.DataFrame(out_summary), use_container_width=True, hide_index=True)
                
                # Outlier Box Plot
                sel_out_col = st.selectbox("Boxplot outlier check for:", options=list(outliers.keys()))
                fig = px.box(df, y=sel_out_col, points="outliers", color_discrete_sequence=["#e0a96d"])
                fig.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No numerical outliers detected using IQR boundaries (1.5x IQR).")
                
        with col_imp:
            st.markdown("### 🎯 Feature Importance to Target")
            if target_val:
                importance_data = analytics["importance"]
                if "error" in importance_data:
                    st.error(importance_data["error"])
                else:
                    st.info(f"Target variable: **{target_val}** | Model Inferred Task: **{importance_data['task_type']}**")
                    
                    # Convert to df
                    imp_df = pd.DataFrame([
                        {"Feature": k, "Importance": v} 
                        for k, v in importance_data["importances"].items()
                    ]).sort_values("Importance", ascending=True)
                    
                    fig = px.bar(
                        imp_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title=f"Relative Feature Importance (Random Forest)",
                        color="Importance",
                        color_continuous_scale="viridis"
                    )
                    fig.update_layout(template="plotly_dark", height=450)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select a Target Variable in the sidebar to run feature importance analysis.")


    # TAB 5: DETECTIVE AI INSIGHTS
    with tab5:
        st.subheader("🧠 Gemini Detective AI Commentary")
        
        # Check if AI insights are already generated
        insights_key = f"ai_insights_{state_key}"
        
        if not gemini_key:
            st.warning("Please provide your Gemini API key in the sidebar to activate the AI Detective agent.")
        else:
            if st.session_state.get(insights_key) is None:
                # Trigger button to avoid running API billing on load
                if st.button("Activate AI Detective (Generate Insights)"):
                    with st.spinner("Analyzing profile datasets and prompting Gemini..."):
                        res = generate_ai_insights(
                            profile_data=profile,
                            analytics_data=analytics,
                            target_col=target_val,
                            importance_data=analytics.get("importance"),
                            api_key=gemini_key
                        )
                        st.session_state[insights_key] = res
            
            ai_data = st.session_state.get(insights_key)
            if ai_data:
                if not ai_data["success"]:
                    st.error(ai_data["error"])
                else:
                    # Show insights in clean inner tabs
                    in_tab1, in_tab2, in_tab3, in_tab4 = st.tabs([
                        "1️⃣ Executive Summary",
                        "2️⃣ Data Quality Plan",
                        "3️⃣ Trends & Relationships",
                        "4️⃣ Business Action Plan"
                    ])
                    with in_tab1:
                        st.markdown(ai_data["insights"]["executive_summary"])
                    with in_tab2:
                        st.markdown(ai_data["insights"]["data_quality"])
                    with in_tab3:
                        st.markdown(ai_data["insights"]["relationships"])
                    with in_tab4:
                        st.markdown(ai_data["insights"]["action_plan"])
            else:
                st.info("Click 'Activate AI Detective' above to fetch analytical narratives.")


    # TAB 6: ML RECS & REPORT
    with tab6:
        st.subheader("📑 Machine Learning Model Suggestions")
        
        # Recommendations Expanders
        recs = get_ml_recommendations(profile, target_val, analytics.get("importance") or {})
        
        for idx, r in enumerate(recs, 1):
            with st.expander(f"Model Option {idx}: {r['name']} ({r['type']})"):
                st.markdown(f"**Suitability / Rationale:**\n{r['rationale']}")
                st.markdown(f"**Required Preprocessing:**\n{r['preprocessing']}")
                st.markdown(f"**Recommended Validation Metrics:**\n{r['evaluation']}")
                st.markdown("**Python Pipeline Code:**")
                st.code(r["code"], language="python")
                
        st.markdown("---")
        st.subheader("📥 Export Professional PDF Business Report")
        
        # Verify if AI Insights are present to put in the PDF
        ai_data = st.session_state.get(f"ai_insights_{state_key}")
        if not ai_data or not ai_data.get("success"):
            st.info("💡 Pro-Tip: Activate the AI Detective in the previous tab before generating the PDF to include full Gemini narrative analysis in your report.")
            
        if st.button("Generate Report PDF"):
            with st.spinner("Compiling data tables and AI insights into PDF layout..."):
                # Setup temp output path
                temp_dir = tempfile.gettempdir()
                pdf_filename = f"Data_Detective_Report_{uploaded_file.name.split('.')[0]}.pdf"
                pdf_path = os.path.join(temp_dir, pdf_filename)
                
                # Run PDF agent
                generate_pdf_report(
                    profile_data=profile,
                    analytics_data=analytics,
                    ai_insights=ai_data or {"success": False},
                    target_col=target_val,
                    importance_data=analytics.get("importance"),
                    output_path=pdf_path
                )
                
                # Load PDF content for download
                try:
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    
                    st.success("PDF generated successfully!")
                    st.download_button(
                        label="Download PDF Business Report",
                        data=pdf_bytes,
                        file_name=pdf_filename,
                        mime="application/pdf"
                    )
                except Exception as ex:
                    st.error(f"Error compiling PDF: {str(ex)}")

else:
    # Landing page style when no dataset is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem;">
        <div style="font-size: 5rem; margin-bottom: 1.5rem;">🕵️‍♂️</div>
        <h2>Welcome to Data Detective AI!</h2>
        <p style="color: #94a3b8; font-size: 1.1rem; max-width: 600px; margin: 0 auto 2rem auto;">
            The autonomous AI data analyst that automatically profiles datasets, detects outliers, maps linear correlations, estimates predictive feature importance, suggests machine learning models, and generates custom business insights.
        </p>
        <div style="border: 2px dashed #2d334d; border-radius: 12px; padding: 3rem; background-color: #151824; max-width: 500px; margin: 0 auto;">
            <p style="color: #64748b; font-weight: 600; margin-bottom: 0;">Upload a CSV file in the sidebar to begin investigation.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
