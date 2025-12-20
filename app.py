import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Walmart Sales Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DARK THEME PROFESSIONAL CSS
# ============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

/* App Background */
.stApp {
    background: #0f172a;
}

/* Main Container */
.main .block-container {
    padding: 2rem 3rem;
    max-width: 1400px;
}

/* Hero */
.hero-header {
    background: linear-gradient(135deg, #020617, #1e293b);
    border-radius: 16px;
    padding: 3rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
}

.hero-header h1 {
    color: #f8fafc !important;
    font-size: 2.8rem;
}

.hero-header p {
    color: #94a3b8;
}

/* Cards */
.modern-card, .metric-card {
    background: #020617;
    border-radius: 14px;
    padding: 2rem;
    border: 1px solid #1e293b;
    box-shadow: 0 8px 25px rgba(0,0,0,0.5);
}

.modern-card h3,
.metric-card h2 {
    color: #38bdf8;
}

.modern-card p,
.metric-card p {
    color: #cbd5f5;
}

/* Info Box */
.info-box {
    background: #020617;
    border-left: 4px solid #38bdf8;
    padding: 1.2rem;
    border-radius: 10px;
}

/* Section Headers */
.section-header {
    background: linear-gradient(135deg, #020617, #1e293b);
    color: #f8fafc;
    border-left: 5px solid #38bdf8;
    border-radius: 12px;
    padding: 1.2rem;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #38bdf8, #0ea5e9);
    color: #020617;
    font-weight: 800;
    border-radius: 10px;
    padding: 1rem;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #0ea5e9, #0284c7);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #020617);
    border-right: 1px solid #1e293b;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #020617;
}

.stTabs [aria-selected="true"] {
    background: #38bdf8;
    color: #020617;
}

/* DataFrame */
[data-testid="stDataFrame"] {
    background: #020617;
    border: 1px solid #1e293b;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #38bdf8;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8;
}

/* Progress Bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #38bdf8, #0ea5e9);
}

/* Alerts */
.stAlert {
    background: #020617;
    border-left: 4px solid #38bdf8;
    color: #e2e8f0;
}

/* Footer */
.footer {
    background: #020617;
    border-radius: 12px;
    border: 1px solid #1e293b;
}

.footer p {
    color: #94a3b8;
}

/* General Text */
h1,h2,h3,h4,h5,h6 { color: #f8fafc !important; }
p, label { color: #cbd5f5 !important; }
            /* Info Card */
.info-card {
    background: #020617;
    border-radius: 14px;
    padding: 2rem;
    border: 1px solid #1e293b;
    box-shadow: 0 8px 25px rgba(0,0,0,0.5);
}

/* Feature List Fix */
.feature-list {
    list-style: none;          /* remove default bullets */
    padding-left: 0;
    margin-top: 1rem;
}

.feature-list li {
    color: #e2e8f0;
    background: #020617;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
    border-radius: 10px;
    border-left: 4px solid #38bdf8;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    line-height: 1.6;
}

/* Strong text inside list */
.feature-list strong {
    color: #38bdf8;
}
/* Boxed list with bullets */
.bullet-box {
    list-style-type: disc;      /* enable bullets */
    padding-left: 1.5rem;
    margin-top: 1rem;
}

.bullet-box li {
    background: #020617;
    color: #e2e8f0;
    margin-bottom: 0.7rem;
    padding: 0.8rem 1rem;
    border-radius: 10px;
    border: 1px solid #1e293b;
    box-shadow: 0 4px 10px rgba(0,0,0,0.4);
    line-height: 1.6;
}

/* Bullet color */
.bullet-box li::marker {
    color: #38bdf8;
    font-size: 1.1rem;
}

/* Highlight text */
.bullet-box strong {
    color: #38bdf8;
}


</style>
""", unsafe_allow_html=True)
# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown("# 🛒 Walmart Sales Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate to:",
    ["📖 Introduction", "📈 EDA Dashboard", "🔮 Sales Prediction", "📊 Model Performance", "📘 Conclusion"],
    label_visibility="collapsed"
)

# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/cleaned_engineered_data_cleaned.csv")
        return df
    except FileNotFoundError:
        st.error("❌ Data file not found: 'data/cleaned_engineered_data_cleaned.csv'")
        return None

@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load("rf_model.pkl")
        xgb_model = joblib.load("xgb_model.pkl")  # XGBoost added
        features = joblib.load("input_features.pkl")
        return rf_model, xgb_model, features
    except FileNotFoundError:
        st.error("❌ Model files not found. Please run the modeling notebook first.")
        return None, None, None

@st.cache_data
def load_store_stats():
    try:
        return pd.read_csv("store_reference_stats.csv")
    except FileNotFoundError:
        return None

# ============================================================
# LOAD MODEL METRICS
# ============================================================
@st.cache_data
def load_metrics():
    import os
    rf_metrics = xgb_metrics = None
    st.write("Files in app folder:", os.listdir())  # debug
    if os.path.exists("model_metrics.csv"):
        rf_metrics = pd.read_csv("model_metrics.csv")
    else:
        st.warning("Random Forest metrics not found")

    if os.path.exists("model_metrics_xgb.csv"):
        xgb_metrics = pd.read_csv("model_metrics_xgb.csv")
    else:
        st.warning("XGBoost metrics not found")

    return rf_metrics, xgb_metrics


# ============================================================
# IMPROVED PREDICTION FUNCTION
# ============================================================
def create_prediction_input(store, dept, temperature, fuel_price, cpi, unemployment, 
                           is_holiday, date, markdowns, feature_names, df, store_stats):
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Basic features
    if "Store" in feature_names: input_df.at[0, "Store"] = int(store)
    if "Dept" in feature_names: input_df.at[0, "Dept"] = int(dept)
    if "Temperature" in feature_names: input_df.at[0, "Temperature"] = float(temperature)
    if "Fuel_Price" in feature_names: input_df.at[0, "Fuel_Price"] = float(fuel_price)
    if "CPI" in feature_names: input_df.at[0, "CPI"] = float(cpi)
    if "Unemployment" in feature_names: input_df.at[0, "Unemployment"] = float(unemployment)
    if "IsHoliday" in feature_names: input_df.at[0, "IsHoliday"] = int(is_holiday)
    
    # Date features
    if "Month" in feature_names: input_df.at[0, "Month"] = date.month
    if "Day" in feature_names: input_df.at[0, "Day"] = date.day
    if "DayOfWeek" in feature_names: input_df.at[0, "DayOfWeek"] = date.weekday()
    if "Week" in feature_names: input_df.at[0, "Week"] = date.isocalendar().week
    if "Year" in feature_names: input_df.at[0, "Year"] = int(df['Year'].median()) if 'Year' in df.columns else 2011
    
    # Markdowns
    for i, md_val in enumerate(markdowns, 1):
        md_col = f"MarkDown{i}"
        if md_col in feature_names: input_df.at[0, md_col] = float(md_val)
    
    # Store/Dept features
    if 'Store' in df.columns:
        store_data = df[df['Store'] == store]
        if len(store_data) > 0:
            if "Size" in feature_names and "Size" in df.columns: input_df.at[0, "Size"] = store_data['Size'].iloc[0]
            if "Type_B" in feature_names and "Type_B" in df.columns: input_df.at[0, "Type_B"] = store_data['Type_B'].iloc[0]
            if "Type_C" in feature_names and "Type_C" in df.columns: input_df.at[0, "Type_C"] = store_data['Type_C'].iloc[0]

            dept_data = store_data[store_data['Dept'] == dept] if 'Dept' in store_data.columns else store_data
            if len(dept_data) > 0 and 'Weekly_Sales' in dept_data.columns:
                dept_mean_sales = dept_data['Weekly_Sales'].mean()
                for roll_col in ['Weekly_Sales_Rolling4', 'Weekly_Sales_Rolling8', 'Weekly_Sales_Rolling12']:
                    if roll_col in feature_names: input_df.at[0, roll_col] = dept_data[roll_col].median() if roll_col in dept_data.columns else dept_mean_sales
                if "Sales_vs_StoreMean" in feature_names:
                    input_df.at[0, "Sales_vs_StoreMean"] = dept_data['Sales_vs_StoreMean'].median() if "Sales_vs_StoreMean" in dept_data.columns else dept_mean_sales - store_data['Weekly_Sales'].mean()
                for lag in [1,2,4]:
                    lag_col = f"Weekly_Sales_Lag{lag}"
                    if lag_col in feature_names: input_df.at[0, lag_col] = dept_data[lag_col].median() if lag_col in dept_data.columns else dept_mean_sales
        elif store_stats is not None and store in store_stats['Store'].values:
            store_ref = store_stats[store_stats['Store'] == store].iloc[0]
            if "Size" in feature_names and 'size' in store_ref: input_df.at[0, "Size"] = store_ref['size']
            if "Type_B" in feature_names and 'type_b' in store_ref: input_df.at[0, "Type_B"] = store_ref['type_b']
            if "Type_C" in feature_names and 'type_c' in store_ref: input_df.at[0, "Type_C"] = store_ref['type_c']
    
    # Fill remaining with dataset medians
    for col in input_df.columns:
        if pd.isna(input_df.at[0, col]) or input_df.at[0, col] == 0:
            if col in df.columns:
                median_val = df[col].median()
                if not pd.isna(median_val): input_df.at[0, col] = median_val
    
    input_df = input_df.astype(float)
    return input_df
# ============================================================
# PAGE 1: INTRODUCTION
# ============================================================
if page == "📖 Introduction":
    st.title("🛒 Walmart Sales Forecasting System")
    st.markdown("### Advanced Machine Learning for Retail Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='info-card'>
            <h3 style='color: #a5b4fc;'>📊 Dataset</h3>
            <p style='font-size: 14px;'>Historical sales data from 45 Walmart stores</p>
            <ul style='font-size: 13px;'>
                <li>2 Stores</li>
                <li>99 Departments</li>
                <li>10000 Records</li>
                <li>2010-2012 Data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
            <h3 style='color: #c4b5fd;'>🤖 Machine Learning</h3>
            <p style='font-size: 14px;'>Advanced Random Forest algorithm</p>
            <ul style='font-size: 13px;'>
                <li>300 Decision Trees</li>
                <li>R² Score > 0.95</li>
                <li>MAE < $5,000</li>
                <li>Fast Inference</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='info-card'>
            <h3 style='color: #86efac;'>🎯 Business Value</h3>
            <p style='font-size: 14px;'>Data-driven decision making</p>
            <ul style='font-size: 13px;'>
                <li>Inventory Planning</li>
                <li>Resource Allocation</li>
                <li>Demand Forecasting</li>
                <li>Profit Optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("""
<h3>Key Features:</h3>
<ul class="feature-list bullet-box">
    <li>📍 <strong>Store & Department Level</strong>: Individual predictions for each store-department combination</li>
    <li>🌡️ <strong>Environmental Factors</strong>: Temperature and fuel prices</li>
    <li>💼 <strong>Economic Indicators</strong>: CPI and unemployment rate</li>
    <li>🎉 <strong>Holiday Effects</strong>: Special events and holiday weeks</li>
    <li>💰 <strong>Promotional Markdowns</strong>: Five categories of price reductions</li>
    <li>📈 <strong>Time Series Features</strong>: Seasonal patterns and trends</li>
    <li>🏪 <strong>Store Characteristics</strong>: Size and type classification</li>
</ul>
""", unsafe_allow_html=True)


    
    st.markdown("---")
    st.info("👉 Navigate using the sidebar to explore data analysis, make predictions, or view model performance!")

# ============================================================
# PAGE 2: EDA DASHBOARD
# ============================================================
elif page == "📈 EDA Dashboard":
    st.title("📈 Exploratory Data Analysis")
    
    df = load_data()
    if df is None:
        st.stop()
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Stores", df['Store'].nunique() if 'Store' in df.columns else 0)
    with col3:
        st.metric("Total Departments", df['Dept'].nunique() if 'Dept' in df.columns else 0)
    with col4:
        st.metric("Avg Weekly Sales", f"${df['Weekly_Sales'].mean():,.0f}" if 'Weekly_Sales' in df.columns else "N/A")
    
    st.markdown("---")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Overview", "🏬 Store Analysis", "📊 Time Trends", "🔗 Correlations"])
    
    with tab1:
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
        st.subheader("📈 Summary Statistics")
        st.dataframe(df.describe().T.style.background_gradient(cmap='Blues'), use_container_width=True)
        
        if 'Weekly_Sales' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x='Weekly_Sales', nbins=50, 
                                 title='Weekly Sales Distribution',
                                 color_discrete_sequence=['#6366f1'])
                fig.update_layout(
                    paper_bgcolor='#1f2937',
                    plot_bgcolor='#1f2937',
                    font=dict(color='#e5e7eb'),
                    xaxis=dict(gridcolor='#374151'),
                    yaxis=dict(gridcolor='#374151')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(df, y='Weekly_Sales', title='Weekly Sales Box Plot',
                           color_discrete_sequence=['#8b5cf6'])
                fig.update_layout(
                    paper_bgcolor='#1f2937',
                    plot_bgcolor='#1f2937',
                    font=dict(color='#e5e7eb'),
                    yaxis=dict(gridcolor='#374151')
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if 'Store' in df.columns and 'Weekly_Sales' in df.columns:
            st.subheader("🏬 Store Performance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                store_sales = df.groupby("Store")["Weekly_Sales"].mean().reset_index()
                store_sales = store_sales.sort_values('Weekly_Sales', ascending=False)
                
                fig = px.bar(store_sales, x="Store", y="Weekly_Sales",
                            title="Average Weekly Sales by Store",
                            color='Weekly_Sales', 
                            color_continuous_scale='Viridis')
                fig.update_layout(
                    paper_bgcolor='#1f2937',
                    plot_bgcolor='#1f2937',
                    font=dict(color='#e5e7eb'),
                    xaxis=dict(gridcolor='#374151'),
                    yaxis=dict(gridcolor='#374151')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Dept' in df.columns:
                    dept_sales = df.groupby("Dept")["Weekly_Sales"].mean().nlargest(10).reset_index()
                    
                    fig = px.bar(dept_sales, x="Dept", y="Weekly_Sales",
                                title="Top 10 Departments by Avg Sales",
                                color='Weekly_Sales', 
                                color_continuous_scale='Blues')
                    fig.update_layout(
                        paper_bgcolor='#1f2937',
                        plot_bgcolor='#1f2937',
                        font=dict(color='#e5e7eb'),
                        xaxis=dict(gridcolor='#374151'),
                        yaxis=dict(gridcolor='#374151')
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Holiday comparison
            if 'IsHoliday' in df.columns:
                st.subheader("🎉 Holiday Impact on Sales")
                
                holiday_sales = df.groupby("IsHoliday")["Weekly_Sales"].agg(['mean', 'count']).reset_index()
                holiday_sales['IsHoliday'] = holiday_sales['IsHoliday'].map({0: 'Non-Holiday', 1: 'Holiday'})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(holiday_sales, x="IsHoliday", y="mean",
                                title="Average Sales: Holiday vs Non-Holiday",
                                color='IsHoliday',
                                color_discrete_map={'Holiday': '#ef4444', 'Non-Holiday': '#3b82f6'})
                    fig.update_layout(
                        paper_bgcolor='#1f2937',
                        plot_bgcolor='#1f2937',
                        font=dict(color='#e5e7eb'),
                        yaxis_title="Average Sales ($)",
                        xaxis=dict(gridcolor='#374151'),
                        yaxis=dict(gridcolor='#374151')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.pie(holiday_sales, values='count', names='IsHoliday',
                                title='Data Distribution: Holiday vs Non-Holiday',
                                color='IsHoliday',
                                color_discrete_map={'Holiday': '#ef4444', 'Non-Holiday': '#3b82f6'})
                    fig.update_layout(
                        paper_bgcolor='#1f2937',
                        font=dict(color='#e5e7eb')
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Sales Trends Over Time")
        
        # Check if we have date columns
        if 'Year' in df.columns and 'Month' in df.columns:
            # Create date column for time series
            df_temp = df.copy()
            df_temp['Date'] = pd.to_datetime(df_temp['Year'].astype(str) + '-' + df_temp['Month'].astype(str) + '-01')
            
            # Monthly trend
            monthly_sales = df_temp.groupby('Date')['Weekly_Sales'].sum().reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_sales['Date'], 
                y=monthly_sales['Weekly_Sales'],
                mode='lines+markers',
                line=dict(color='#6366f1', width=3),
                marker=dict(size=8, color='#8b5cf6'),
                name='Total Sales'
            ))
            fig.update_layout(
                title='Total Sales Over Time',
                xaxis_title='Date',
                yaxis_title='Total Sales ($)',
                paper_bgcolor='#1f2937',
                plot_bgcolor='#1f2937',
                font=dict(color='#e5e7eb'),
                xaxis=dict(gridcolor='#374151'),
                yaxis=dict(gridcolor='#374151')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)

            with col1:
                # Monthly patterns
                month_sales = df.groupby("Month")["Weekly_Sales"].mean().reset_index()
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                month_sales['MonthName'] = month_sales['Month'].apply(
                    lambda x: month_names[int(x)-1] if 1 <= x <= 12 else str(x))

                fig = px.bar(
                    month_sales,
                    x="MonthName",
                    y="Weekly_Sales",
                    title="Average Sales by Month",
                    color='Weekly_Sales',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(
                    paper_bgcolor='#1f2937',
                    plot_bgcolor='#1f2937',
                    font=dict(color='#e5e7eb'),
                    xaxis=dict(gridcolor='#374151'),
                    yaxis=dict(gridcolor='#374151')
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                if 'Temperature' in df.columns and 'Weekly_Sales' in df.columns:
                    sample_df = df.sample(min(1000, len(df)))
                    fig = px.scatter(
                        sample_df,
                        x='Temperature',
                        y='Weekly_Sales',
                        title='Temperature vs Sales',
                        opacity=0.6,
                        color='Weekly_Sales',
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(
                        paper_bgcolor='#1f2937',
                        plot_bgcolor='#1f2937',
                        font=dict(color='#e5e7eb'),
                        xaxis=dict(gridcolor='#374151'),
                        yaxis=dict(gridcolor='#374151')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Temperature or Weekly_Sales column not found")
    with tab4:
        st.subheader("🔗 Feature Correlations")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        priority_cols = ['Weekly_Sales', 'Store', 'Dept', 'Temperature',
                         'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday', 'Size']

        selected_cols = [col for col in priority_cols if col in numeric_cols]
        selected_cols += [col for col in numeric_cols if col not in selected_cols][:20]

        corr_df = df[selected_cols].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            hovertemplate='X: %{x}<br>Y: %{y}<br>Correlation: %{z:.2f}<extra></extra>'
        ))

        fig.update_layout(
            title='Feature Correlation Heatmap',
            paper_bgcolor='#1f2937',
            plot_bgcolor='#1f2937',
            font=dict(color='#e5e7eb'),
            xaxis=dict(tickangle=45, gridcolor='#374151'),
            yaxis=dict(gridcolor='#374151'),
            margin=dict(l=100, r=50, t=80, b=100)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class='warning-box'>
            ⚠️ Correlations show linear relationships only. High correlation does not imply causation.
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# 🔮 SALES PREDICTION
# ============================================================
elif page == "🔮 Sales Prediction":
    st.title("🔮 Weekly Sales Prediction")

    df = load_data()
    rf_model, xgb_model, feature_names = load_models()
    store_stats = load_store_stats()

    if df is None or rf_model is None or xgb_model is None:
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        store = st.selectbox("🏬 Store", sorted(df["Store"].unique()))
        dept = st.selectbox("📦 Department", sorted(df["Dept"].unique()))
        date = st.date_input("📅 Date", value=datetime(2015,1,1), min_value=datetime(2010,1,1), max_value=datetime(2026,12,31))
        is_holiday = st.selectbox("🎉 Holiday", [0,1])
    with col2:
        temperature = st.number_input("🌡️ Temperature", min_value=1.0, value=70.0, step=1.0)
        fuel_price = st.number_input("⛽ Fuel Price", min_value=1.0, value=3.5, step=0.1)
        cpi = st.number_input("📈 CPI", min_value=1.0, value=220.0, step=1.0)
        unemployment = st.number_input("📉 Unemployment", min_value=0.0, value=7.0, step=0.1)

    st.markdown("### 💰 Promotional Markdowns")
    markdowns = [st.number_input(f"MarkDown{i+1}", min_value=0.0, value=0.0, step=100.0) for i in range(5)]

    model_option = st.radio("Select Model", ["Random Forest", "XGBoost"])  # MODEL SELECTION

    if st.button("🚀 Predict Weekly Sales"):
        input_df = create_prediction_input(
            store, dept, temperature, fuel_price, cpi, unemployment,
            is_holiday, pd.to_datetime(date), markdowns, feature_names, df, store_stats
        )

        if model_option == "Random Forest":
            prediction = rf_model.predict(input_df)[0]
        else:
            prediction = xgb_model.predict(input_df)[0]

        st.markdown(
            f"<div class='prediction-box'><h2>📊 Predicted Weekly Sales ({model_option})</h2><h1>${prediction:,.2f}</h1></div>",
            unsafe_allow_html=True
        )

        # Historical context
        store_avg = df[df["Store"] == store]["Weekly_Sales"].mean()
        dept_avg = df[(df["Store"]==store)&(df["Dept"]==dept)]["Weekly_Sales"].mean()
        overall_avg = df["Weekly_Sales"].mean()

        st.markdown("""<div class="modern-card"><h3>🔍 Prediction Context</h3><p>The predicted value is generated for a <strong>specific department</strong> inside the selected store and should not be directly compared with store-wide or global averages.</p></div>""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("🏬 Store Avg", f"${store_avg:,.0f}")
        c2.metric("📦 Dept Avg", f"${dept_avg:,.0f}")
        c3.metric("🌍 Overall Avg", f"${overall_avg:,.0f}")
        comparison_df = pd.DataFrame({
            "Category": ["Prediction", "Dept Avg", "Store Avg", "Overall Avg"],
            "Sales": [prediction, dept_avg, store_avg, overall_avg]
        })

        fig = px.bar(
            comparison_df,
            x="Category",
            y="Sales",
            title="📊 Prediction vs Historical Averages",
            color="Category"
        )
        fig.update_layout(
            paper_bgcolor="#1f2937",
            plot_bgcolor="#1f2937",
            font=dict(color="#e5e7eb"),
            yaxis_title="Weekly Sales ($)"
        )

        st.plotly_chart(fig, use_container_width=True)
        if dept_avg > 0:
            percent_diff = ((prediction - dept_avg) / dept_avg) * 100
            direction = "higher 📈" if percent_diff > 0 else "lower 📉"

            st.info(
                f"📦 **This prediction is {abs(percent_diff):.2f}% {direction} than the department's historical average sales.**"
            )



elif page == "📊 Model Performance":
    st.title("📊 Model Performance Comparison")

    rf_metrics, xgb_metrics = load_metrics()


    tab1, tab2 = st.tabs(["🌲 Random Forest", "⚡ XGBoost"])

    def render_metrics(metrics_df, model_name):
        if metrics_df is None:
            st.warning(f"{model_name} metrics not found")
            return

        st.subheader("📌 Test Set Metrics")

        # Use TEST metrics only
        metrics_df = metrics_df.set_index("Metric")

        c1, c2, c3 = st.columns(3)

        if "R2" in metrics_df.index:
            c1.metric("R² Score", f"{metrics_df.loc['R2', 'Test']:.3f}")
        if "MAE" in metrics_df.index:
            c2.metric("MAE", f"${metrics_df.loc['MAE', 'Test']:,.0f}")
        if "RMSE" in metrics_df.index:
            c3.metric("RMSE", f"${metrics_df.loc['RMSE', 'Test']:,.0f}")

        # Bar chart (Test metrics)
        plot_df = metrics_df.loc[["MAE", "RMSE", "R2"]].reset_index()
        plot_df["Value"] = plot_df["Test"]

        fig = px.bar(
            plot_df,
            x="Metric",
            y="Value",
            title=f"{model_name} – Test Metrics",
            color="Metric"
        )

        fig.update_layout(
            paper_bgcolor="#1f2937",
            plot_bgcolor="#1f2937",
            font=dict(color="#e5e7eb"),
            yaxis_title="Metric Value"
        )

        st.plotly_chart(fig, use_container_width=True)

    # --------- Tabs ----------
    with tab1:
        render_metrics(rf_metrics, "Random Forest")

    with tab2:
        render_metrics(xgb_metrics, "XGBoost")
        st.write("Files in repo:", os.listdir())
st.write("Current working directory:", os.getcwd())




elif page == "📘 Conclusion":
    st.title("📘 Project Conclusion")

    st.markdown("""
    <div class='info-card'>
        <h3>✅ Key Achievements</h3>
        <ul>
            <li>Accurate weekly sales forecasting using Random Forest</li>
            <li>Comprehensive feature engineering including time-series patterns</li>
            <li>Interactive EDA and visualization dashboard</li>
            <li>Real-time prediction system for business use</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-card'>
        <h3>📈 Business Impact</h3>
        <ul>
            <li>Improved inventory planning</li>
            <li>Optimized staffing decisions</li>
            <li>Better promotional strategy evaluation</li>
            <li>Reduced operational risk</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.success("🎉 Walmart Sales Forecasting System successfully demonstrates the power of machine learning in retail analytics.")
