# 🔍 Data-Detective-AI-Autonomous-Multi-Agent-Data-Science-Copilot
An AI-powered multi-agent platform that automatically profiles datasets, performs advanced analytics, identifies hidden patterns, recommends machine learning models, generates business insights, and produces professional reports.

Transform any dataset into actionable intelligence using AI agents and Gemini-powered reasoning.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Gemini](https://img.shields.io/badge/Google-Gemini-green)
![License](https://img.shields.io/badge/License-Apache_2.0-blue)

## 📖 Overview:

**Data Detective AI** is an autonomous multi-agent data science copilot designed to simplify and accelerate the data analysis workflow. The platform enables users to upload datasets and automatically performs data profiling, quality assessment, exploratory analytics, anomaly detection, correlation analysis, feature importance evaluation, AI-powered business insights, machine learning model recommendations, and professional PDF report generation.

Powered by **Google Gemini API**, the system leverages specialized AI agents to transform raw data into actionable business intelligence with minimal manual intervention. The application is deployed on **Google Cloud Platform (GCP)** using **Google Compute Engine** and provides an interactive web interface built with **Streamlit**, allowing users to access advanced analytics directly from their browser.

Whether analyzing business, healthcare, finance, retail, or research datasets, Data Detective AI helps analysts, students, and organizations make faster, data-driven decisions through intelligent automation.


## ✨ Features:

### 📂 Intelligent Dataset Upload:

* Upload CSV datasets through an intuitive Streamlit web interface.
* Automatically validates and loads datasets for analysis.

### 📊 Data Profiling Agent:

* Detects data types and schema.
* Calculates data quality score.
* Identifies missing values and duplicate records.
* Generates statistical summaries for numerical features.

### 📈 Analytics Agent:

* Performs outlier detection using the IQR method.
* Conducts correlation analysis between numerical features.
* Identifies the most influential features using feature importance analysis.
* Automatically detects whether the dataset is suitable for classification or regression tasks.

### 🧠 AI Insight Agent:

* Generates executive summaries using Google Gemini.
* Provides AI-powered business insights and interpretations.
* Identifies hidden patterns and trends within the dataset.
* Recommends actionable business decisions based on analytical results.

### 🤖 Machine Learning Recommendation Agent:

* Suggests the most suitable machine learning models for the uploaded dataset.
* Recommends preprocessing techniques and evaluation metrics.
* Assists users in selecting appropriate predictive modeling approaches.

### 📄 Automated Report Generation:

* Creates comprehensive PDF reports containing:

  * Dataset overview
  * Data quality assessment
  * Statistical analysis
  * Correlation insights
  * Feature importance
  * AI-generated business recommendations
  * Machine learning model suggestions

### 📊 Interactive Dashboard:

* Displays analytical results through an interactive Streamlit interface.
* Presents visualizations for profiling, analytics, and AI-generated insights.
* Enables users to explore datasets without writing code.

### ☁️ Cloud Deployment:

* Successfully deployed on Google Cloud Platform (GCP).
* Hosted using Google Compute Engine for scalable and remote access.
* Accessible through any modern web browser without local installation.

### 🔒 AI-Powered Decision Support:

* Reduces manual exploratory data analysis.
* Accelerates data-driven decision making.
* Converts raw datasets into actionable business intelligence within minutes.



## 🏗️ System Architecture:

Data Detective AI follows a modular multi-agent architecture in which each agent is responsible for a specific stage of the data science workflow. The application is deployed on Google Cloud Platform (GCP) using Google Compute Engine and provides a cloud-hosted Streamlit interface for users to interact with the system.

The workflow begins when a user uploads a CSV dataset through the Streamlit web application. The uploaded dataset is first processed by the Data Profiling Agent, which performs schema analysis, data quality assessment, missing value detection, duplicate identification, type inference, and statistical summarization.

The profiling results are then passed to the Analytics Agent, where advanced exploratory analysis is performed. This includes outlier detection, correlation analysis, feature importance evaluation, and automatic identification of the machine learning problem type (classification or regression).

Next, the Insight Agent communicates with the Google Gemini API to generate AI-powered executive summaries, business insights, trend analysis, and actionable recommendations based on the analytical findings.

The generated insights are forwarded to the Report Agent, which compiles machine learning model recommendations, visualizations, and AI-generated interpretations into a professional PDF report.

Finally, the processed results are displayed through the interactive Streamlit dashboard, allowing users to explore analytics, review AI-generated insights, download comprehensive reports, and make informed data-driven decisions.

<div align="center">
  <img src="https://github.com/user-attachments/assets/65577bc0-b803-4da5-8464-39a4088172f0" alt="Data Detective AI System Architecture" width="70%">
</div>

## 🔄 Workflow:

The workflow of **Data Detective AI** is designed to automate the end-to-end data analysis process through a sequence of specialized AI agents.

### Step 1: Dataset Upload:

The user uploads a CSV dataset through the Streamlit web interface hosted on **Google Cloud Platform (GCP)**.

### Step 2: Data Profiling:

The **Data Profiling Agent** analyzes the dataset by identifying data types, calculating data quality scores, detecting missing values and duplicates, generating statistical summaries, and assessing the overall health of the dataset.

### Step 3: Advanced Analytics:

The **Analytics Agent** performs exploratory data analysis by detecting outliers, analyzing feature correlations, computing feature importance, and automatically identifying whether the problem is a classification or regression task.

### Step 4: AI-Powered Insights:

The processed analytical results are sent to the **Google Gemini API** through the **Insight Agent**, which generates executive summaries, business insights, trend analysis, and actionable recommendations.

### Step 5: Machine Learning Recommendations:

Based on the dataset characteristics and analytical findings, the **Recommendation Engine** suggests suitable machine learning algorithms, preprocessing techniques, and evaluation metrics for predictive modeling.

### Step 6: Report Generation:

The **Report Agent** compiles all analytical results, AI-generated insights, and machine learning recommendations into a professional PDF report for easy sharing and decision-making.

### Step 7: Results & Visualization:

The Streamlit dashboard presents interactive visualizations, profiling statistics, analytical findings, AI-generated insights, machine learning recommendations, and allows users to download the generated PDF report.

### Workflow Summary:

```text
User
   │
   ▼
Upload CSV Dataset
   │
   ▼
Data Profiling Agent
   │
   ▼
Analytics Agent
   │
   ▼
Insight Agent (Google Gemini)
   │
   ▼
Machine Learning Recommendation Engine
   │
   ▼
Report Generation Agent
   │
   ▼
Interactive Dashboard + PDF Report
```



## 🛠️ Tech Stack:

### 💻 Programming Language:
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### 📊 Data Processing & Analysis:
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)

### 🤖 Machine Learning:
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-006400?style=for-the-badge)

### 📈 Data Visualization:
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

### 🧠 Generative AI:
![Google Gemini](https://img.shields.io/badge/Google_Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)

### 🌐 Frontend:
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

### 📄 Report Generation:
![FPDF2](https://img.shields.io/badge/FPDF2-009688?style=for-the-badge)

### ☁️ Cloud Deployment:
![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)
![Compute Engine](https://img.shields.io/badge/Compute_Engine-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)

### 🛠️ Development Tools:
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![VS Code](https://img.shields.io/badge/VS_Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white)

### ⚙️ Environment:
![Python-dotenv](https://img.shields.io/badge/python--dotenv-4EAA25?style=for-the-badge)
![Python venv](https://img.shields.io/badge/Python_Venv-3776AB?style=for-the-badge&logo=python&logoColor=white)



## ⚙️ Installation:

Follow the steps below to set up and run **Data Detective AI** on your local machine.

### 1️⃣ Clone the Repository:

```bash
git clone https://github.com/msns-1927/Data-Detective-AI.git
cd Data-Detective-AI
```

### 2️⃣ Create a Virtual Environment:

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies:

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables:

Create a `.env` file in the project root directory and add your Google Gemini API key:

```env
GEMINI_API_KEY=your_google_gemini_api_key
```

### 5️⃣ Run the Application:

```bash
streamlit run app.py
```

### 6️⃣ Access the Application:

Open your browser and navigate to:

```text
http://localhost:0000
```

You can now upload a CSV dataset and start exploring automated profiling, analytics, AI-generated insights, machine learning recommendations, and downloadable PDF reports.

## ☁️ Deployment:

Data Detective AI is successfully deployed on **Google Cloud Platform (GCP)** using **Google Compute Engine**, making the application accessible through a web browser without requiring local installation.

### Deployment Infrastructure:

* **Cloud Provider:** Google Cloud Platform (GCP)
* **Compute Service:** Google Compute Engine (Ubuntu VM)
* **Application Framework:** Streamlit
* **Programming Language:** Python 3.10+
* **Virtual Environment:** Python venv
* **AI Service:** Google Gemini API
* **Access Method:** Public IP Address

### Deployment Workflow:

```text
Developer
     │
     ▼
Push Source Code to GitHub
     │
     ▼
Launch Google Compute Engine VM
     │
     ▼
Clone Repository
     │
     ▼
Create Virtual Environment
     │
     ▼
Install Dependencies
     │
     ▼
Configure Gemini API Key (.env)
     │
     ▼
Run Streamlit Application
     │
     ▼
Expose Port 8501 via Firewall
     │
     ▼
Access Application through Public IP
```
### Deployment Highlights:

* ☁️ Hosted on Google Cloud Platform (GCP)
* 🚀 Deployed using Google Compute Engine
* 📊 Interactive Streamlit web application
* 🤖 Integrated with Google Gemini API
* 🔒 Environment variables managed securely using `.env`
* 🌍 Accessible from any modern web browser
* 📈 Scalable cloud-based deployment for remote access


## 🚀 Usage:

Using **Data Detective AI** is simple and requires no prior coding experience. Follow the steps below to analyze your dataset and generate AI-powered insights.

### Step 1: Launch the Application:

Run the Streamlit application or access the deployed application hosted on **Google Cloud Platform (GCP)**.

### Step 2: Upload a Dataset:

* Upload a CSV dataset using the **Upload Dataset** option.
* The application automatically validates and loads the dataset for analysis.

### Step 3: Profile the Dataset:

The **Data Profiling Agent** automatically analyzes the uploaded dataset and provides:

* Dataset overview
* Data quality score
* Missing value analysis
* Duplicate detection
* Data type identification
* Statistical summary

### Step 4: Perform Advanced Analytics:

The **Analytics Agent** performs:

* Outlier detection
* Correlation analysis
* Feature importance analysis
* Automatic identification of classification or regression tasks

### Step 5: Generate AI Insights:

The **Insight Agent**, powered by **Google Gemini**, generates:

* Executive summary
* Business insights
* Pattern analysis
* Actionable recommendations

### Step 6: Review Machine Learning Recommendations:

The platform automatically recommends:

* Suitable machine learning algorithms
* Data preprocessing techniques
* Model evaluation metrics

### Step 7: Download the Report:

Generate and download a professional PDF report containing:

* Dataset profile
* Data quality assessment
* Analytics summary
* AI-generated insights
* Machine learning recommendations
* Business action plan

### Example Workflow:

```text
Upload CSV Dataset
        │
        ▼
Automatic Data Profiling
        │
        ▼
Advanced Analytics
        │
        ▼
AI-Powered Insights
        │
        ▼
ML Model Recommendations
        │
        ▼
Generate PDF Report
```

> **Supported Input:** CSV datasets

> **Generated Outputs:** Interactive Dashboard, AI Insights, Machine Learning Recommendations, and Downloadable PDF Analytics Report.

## 📈 Results:

Successfully analyzed:

- 6,497 records
- 13 features
- 95.29 Data Quality Score

Generated:

- Automated EDA
- Feature Importance Analysis
- Business Recommendations
- ML Model Recommendations
- PDF Executive Report


## 🌍 Use Cases:

- Customer Churn Analysis
- Sales Analytics
- Healthcare Analytics
- Financial Risk Analysis
- Fraud Detection
- Employee Attrition Prediction
- Manufacturing Quality Control


## 🔮 Future Roadmap:

- Conversational Data Chat
- Forecasting Agent
- Root Cause Analysis Agent
- Auto Dashboard Generation
- LangGraph Multi-Agent Orchestration
- Cloud Deployment
- Explainable AI Module


## 👨‍💻 Author:

Siva Narayana Muppidi

B.Tech - Artificial Intelligence & Data Science

### LinkedIn:
https://www.linkedin.com/in/siva-narayana-muppidi-413259230/

### GitHub:
https://github.com/msns-1927


## 📜 License:

Apache License 2.0
