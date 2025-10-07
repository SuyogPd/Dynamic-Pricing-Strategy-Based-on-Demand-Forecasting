import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------------
# Helper function for evaluation
# -------------------------------
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    print("Model Evaluation Metrics:")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.2f}%")
    return mae, rmse, mape

# -------------------------------
# Load data
# -------------------------------
train = pd.read_csv("train.csv", parse_dates=['date'], dayfirst=True)
test = pd.read_csv("test.csv", parse_dates=['date'], dayfirst=True)

# -------------------------------
# Feature Engineering
# -------------------------------
def create_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.isocalendar().week.fillna(0).astype(int)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    return df

train = create_features(train)
test = create_features(test)

# -------------------------------
# Lag Features
# -------------------------------
train = train.sort_values(by=['store_id','item_id','date'])
for lag in [1,7,30]:
    train[f'lag{lag}'] = train.groupby(['store_id','item_id'])['sales'].shift(lag)

train = train.fillna(0)

# -------------------------------
# SARIMA Forecast Function
# -------------------------------
def sarima_forecast(ts_train, steps, seasonal_period=7):
    try:
        model = SARIMAX(ts_train,
                        order=(1,1,1),
                        seasonal_order=(1,1,1,seasonal_period),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=steps)
    except:
        forecast = np.repeat(ts_train.mean(), steps)
    return np.round(forecast).astype(int)

# -------------------------------
# Add SARIMA Forecasts as Feature
# -------------------------------
train['sarima'] = 0
test['sarima'] = 0

store_item_groups_train = train.groupby(['store_id','item_id'])
store_item_groups_test = test.groupby(['store_id','item_id'])

# Generate SARIMA forecasts for training (shifted for feature)
for (store, item), group in tqdm(store_item_groups_train, desc="SARIMA feature for train"):
    ts = group.sort_values('date')['sales']
    sarima_pred = sarima_forecast(ts, len(ts), seasonal_period=7)
    train.loc[group.index, 'sarima'] = sarima_pred

# -------------------------------
# Features for ML
# -------------------------------
features = ['store_id','item_id','year','month','day','dayofweek',
            'weekofyear','is_weekend','lag1','lag7','lag30','sarima']

X = train[features]
y = train['sales']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=False)

# -------------------------------
# Train LightGBM
# -------------------------------
model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric='mae',
    callbacks=[]  # remove verbose errors
)

# -------------------------------
# Evaluate
# -------------------------------
y_pred_valid = np.round(model.predict(X_valid)).astype(int)
evaluate_model(y_valid, y_pred_valid)

# Plot Actual vs Predicted
plt.figure(figsize=(12,6))
plt.plot(y_valid.values, label='Actual')
plt.plot(y_pred_valid, label='Predicted')
plt.title("Hybrid SARIMA + LightGBM Forecast (Validation)")
plt.xlabel("Samples")
plt.ylabel("Sales")
plt.legend()
plt.show()

# -------------------------------
# Prepare Test Set
# -------------------------------
# Lag features for test (using last known train sales)
for lag in [1,7,30]:
    test[f'lag{lag}'] = 0  # placeholder, more advanced approach can use rolling window

# SARIMA feature for test
for (store, item), group in tqdm(store_item_groups_test, desc="SARIMA feature for test"):
    ts_train = train[(train['store_id']==store) & (train['item_id']==item)].sort_values('date')['sales']
    forecast = sarima_forecast(ts_train, len(group), seasonal_period=7)
    test.loc[group.index, 'sarima'] = forecast

X_test = test[features]
test['sales'] = np.round(model.predict(X_test)).astype(int)

# -------------------------------
# Save Submission
# -------------------------------
submission = test[['id','sales']]
submission.to_csv("submission_hybrid.csv", index=False)
print("✅ Hybrid SARIMA + LightGBM submission saved as submission_hybrid.csv")


from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

# y_valid = actual sales
# y_pred_valid = predicted sales from model

mae = mean_absolute_error(y_valid, y_pred_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
mape = mean_absolute_percentage_error(y_valid, y_pred_valid) * 100

print(f"MAE  : {mae:.2f}")     # Average absolute error
print(f"RMSE : {rmse:.2f}")    # Penalizes large errors
print(f"MAPE : {mape:.2f}%")   # Average % error



import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(y_valid.values, label='Actual', alpha=0.7)
plt.plot(y_pred_valid, label='Predicted', alpha=0.7)
plt.title("Actual vs Predicted Sales (Validation)")
plt.xlabel("Samples")
plt.ylabel("Sales")
plt.legend()
plt.show()



plt.figure(figsize=(8,8))
plt.scatter(y_valid, y_pred_valid, alpha=0.5)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Scatter Plot")
plt.show()
