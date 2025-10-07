Project Title: 🧠 Dynamic Pricing Strategy Based on Demand Forecasting

📘 Overview

This project focuses on building a Dynamic Pricing Strategy powered by Demand Forecasting using machine learning.
It predicts future sales using historical data and dynamically adjusts prices to maximize profit while maintaining competitiveness.
A hybrid model combining SARIMA (for time-series trends) and LightGBM (for feature-based learning) ensures robust and accurate forecasting.

📊 Dataset

Files used:

📂 train.csv → date, store_id, item_id, sales

📂 test.csv → id, date, store_id, item_id

📂 sample_submission.csv → id, sales

🧩 Synthetic columns price and cost were generated to simulate real-world pricing

Date format: dd-mm-yyyy

⚙️ Methodology
🧹 A) Data Preprocessing

Converted date to datetime and extracted temporal features (day, month, week, is_weekend)

Created lag features (1, 7, 30 days) and rolling averages (7-day, 30-day)

Added synthetic price and cost per item_id to simulate pricing scenarios

🤖 B) Model & Techniques

SARIMA: Captures seasonality and long-term temporal patterns in sales data

LightGBM: Models complex nonlinear relationships among features

Hybrid Model: Combines SARIMA’s time-series strengths with LightGBM’s predictive power

Dynamic Pricing Module: Uses price elasticity of demand to adjust prices dynamically for optimal profit

📏 C) Evaluation Metrics

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

MAPE (Mean Absolute Percentage Error)

Profit Gain & Hit Rate – to measure pricing efficiency and profitability impact

📈 Results
Metric	Value
MAE	7.50
RMSE	9.75
MAPE	16.17%

The hybrid model accurately captured demand fluctuations and seasonality.

Dynamic pricing simulation showed potential profit gains and improved price efficiency.

Visualization of Actual vs Predicted Sales showed strong alignment, confirming model reliability.

💡 Key Insights

Combining time-series forecasting (SARIMA) with machine learning (LightGBM) enhances accuracy.

Dynamic pricing based on demand elasticity can significantly improve revenue.

The model framework is adaptable across industries like retail, finance, and e-commerce.

🚀 Future Enhancements

Integrate competitor pricing, inventory levels, and real-time demand signals

Implement reinforcement learning for autonomous dynamic pricing

Deploy as a web-based dashboard (Power BI / Streamlit) for real-time visualization

🛠️ Technologies Used

🐍 Python

📦 Libraries: Pandas, NumPy, Matplotlib, Seaborn, LightGBM, statsmodels, scikit-learn

💻 Jupyter Notebook for model experimentation and visualization
