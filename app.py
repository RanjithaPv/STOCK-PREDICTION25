import os
import pandas as pd
from flask import Flask, jsonify, request, render_template
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ===== CONFIG =====
DATA_PATH = "C:\\Users\\ranji\\OneDrive\\Desktop\\12STOCK PREDICT\\nse_cleaned.csv"

# ===== INITIALIZE FLASK =====
app = Flask(__name__)
data_loaded = False
available_symbols = []
stock_data = {}

# ===== LOAD DATA =====
def load_data():
    global available_symbols, data_loaded, stock_data
    if not os.path.exists(DATA_PATH):
        print(f"❌ CSV not found: {DATA_PATH}")
        return False
    try:
        df = pd.read_csv(DATA_PATH)
        if "symbol" not in df.columns:
            print("❌ 'symbol' column missing in CSV")
            return False
        # Normalize symbols to uppercase
        df["symbol"] = df["symbol"].str.upper()
        available_symbols = sorted(df["symbol"].unique().tolist())
        stock_data = {sym: df[df["symbol"] == sym].reset_index(drop=True) for sym in available_symbols}
        data_loaded = True
        print(f"✅ Data loaded for {len(available_symbols)} symbols")
        return True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

data_loaded = load_data()

# ===== ROUTES =====
@app.route("/")
def index():
    return render_template("welcome.html")

@app.route("/dashboard")
def dashboard():
    return render_template("prediction.html")

@app.route("/api/status")
def api_status():
    return jsonify({
        "status": "running" if data_loaded else "error",
        "data_loaded": data_loaded,
        "available_symbols": len(available_symbols)
    })

@app.route("/api/symbols")
def api_symbols():
    return jsonify({"symbols": available_symbols})

@app.route("/api/predict")
def api_predict():
    symbol = request.args.get("symbol", "").upper()
    if not symbol:
        return jsonify({"error": "Symbol parameter is missing"}), 400
    if symbol not in stock_data:
        return jsonify({"error": f"Symbol '{symbol}' not found"}), 400

    df = stock_data[symbol].copy()
    if len(df) < 30:
        return jsonify({"error": f"Not enough data for {symbol} (need >=30 rows)"}), 400

    # Features: OHLC, volume, moving averages
    df["MA5"] = df["close"].rolling(5).mean()
    df["MA10"] = df["close"].rolling(10).mean()
    df.dropna(inplace=True)

    required_cols = ["open", "high", "low", "volume", "MA5", "MA10"]
    if not all(col in df.columns for col in required_cols):
        return jsonify({"error": f"Missing required columns in CSV for {symbol}"}), 400

    X = df[required_cols]
    y = df["close"]

    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Train model
    model = XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)

    # RMSE calculation compatible with all scikit-learn versions
    try:
        rmse = mean_squared_error(y_test, preds, squared=False)
    except TypeError:
        rmse = mean_squared_error(y_test, preds) ** 0.5

    mae = mean_absolute_error(y_test, preds)

    # Next day prediction
    last_row = X.iloc[[-1]]
    next_day_pred = float(model.predict(last_row)[0])

    # Convert dates to string
    dates = df.iloc[train_size:].index.astype(str).tolist()

    return jsonify({
        "symbol": symbol,
        "dates": dates,
        "actual": y_test.tolist(),
        "predicted": preds.tolist(),
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "next_day_prediction": next_day_pred
    })

# ===== MAIN =====
if __name__ == "__main__":
    print("🚀 Starting Stock Prediction API...")
    app.run(debug=True, host="0.0.0.0", port=5000)