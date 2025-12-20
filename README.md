# walmart-sales-prediction-app

AI-powered Walmart sales prediction (by DEPARTMENT) with high accuracy using machine learning.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red?style=for-the-badge&logo=streamlit)](https://walmart-sales-forecasting-app-nmx4u44ddrmfjgruoyambb.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/mohid525/walmart-sales-forecasting-app)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data)

## Live Demo

**Try it now:** [walmart-sales-forecasting-app.streamlit.app](https://walmart-sales-forecasting-app-nmx4u44ddrmfjgruoyambb.streamlit.app/)

## Key Features

- **Accurate Sales Predictions** by department
- **Real-time Forecasting** for Walmart stores
- **Interactive Dashboard** with visualizations
- **EDA Tools** for exploring trends and patterns
- **Supports Multiple Departments** for granular analysis

## 🛠 Tech Stack

**ML**: Scikit-learn | **Web**: Streamlit | **Data**: Pandas, NumPy | **Viz**: Plotly, Matplotlib

## 📁 Project Structure

├── app.py # Main Streamlit application
├── walmart_model.py # Model training script
├── walmart_dataset.csv # Dataset
├── walmart_model.pkl # Trained model
├── scaler.pkl # Feature scaler
├── feature_names.pkl # Feature list
├── preprocessing_info.pkl # Preprocessing config
├── EDA_Walmart.ipynb # Data analysis notebook
└── requirements.txt # Dependencies


## 🚀 Quick Start


# Clone repository
git clone https://github.com/mohid525/walmart-sales-forecasting-app.git
cd walmart-sales-forecasting-app

# Install dependencies
pip install -r requirements.txt

# Train model (optional)
python walmart_model.py

# Run app
streamlit run app.py
## 📊 Dataset

**Source:** [Kaggle - Extrovert vs Introvert Behavior Data](https://www.kaggle.com/datasets/micgonzalez/walmart-store-sales-forecasting/data)

- **Samples:** 10k records
- **Features:** Store, Dept, Weekly Sales, CPI, Unemployment, etc.
- **Target:** Weekly Sales
- **Distribution:** Multiple stores & departments
- 
## 💻 Application Pages

1. **Into** - Overview
2. **Data Analysis** - EDA with interactive visualizations
3. **Model performance** - Architecture and performance details
4. **Predict** - Real-time predictions
5. **Conclusion** - Technical documentation

## 👨‍💻 Author

**Muhammad mohid**
<div align="center">
Made with mohid
</div>



