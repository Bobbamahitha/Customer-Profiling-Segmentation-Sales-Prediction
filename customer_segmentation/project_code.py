from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statsmodels.api as sm
import os
from sklearn.model_selection import train_test_split
import seaborn as sns
import io
import base64

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

app = Flask(__name__)
CORS(app)

data = None
latest_results = {"segmentation": None, "sales_prediction": None, "regression_comparison": None}

# --- Utility Functions ---
def kmeans_clustering(data, clustering_features, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = model.fit_predict(data[clustering_features])
    data['KMeans_Cluster'] = clusters
    return data, model

def dbscan_clustering(data, clustering_features, eps=1.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = model.fit_predict(data[clustering_features])
    data['DBSCAN_Cluster'] = clusters
    return data, model

# Update in project_mod.py

import numpy as np  # Add this import if not present

def sales_prediction(data):
    # Generate random dates for the Date column if not present
    if 'Date' not in data.columns:
        num_rows = len(data)
        start_date = pd.to_datetime('2020-01-01')
        date_range = pd.date_range(start_date, periods=num_rows, freq='D')
        data['Date'] = np.random.choice(date_range, size=num_rows, replace=False)

    # Ensure the date column is in datetime format and set as index
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Sort data by date
    data.sort_index(inplace=True)

    # Select the sales column for forecasting
    sales_data = data['Order_Frequency']  # Replace with your actual target column

    # Split into training and testing
    train_size = int(len(sales_data) * 0.8)
    train, test = sales_data[:train_size], sales_data[train_size:]

    # Fit ARIMA model
    order = (5, 1, 0)  # Example order, tune this based on data
    model = sm.tsa.ARIMA(train, order=order)
    model_fit = model.fit()

    # Forecast
    predictions = model_fit.predict(start=len(train), end=len(sales_data)-1, dynamic=False)

    return train, test, predictions, model_fit


from flask import Flask, request, jsonify

def sentiment_analysis(order_frequency):
    """
    Perform sentiment analysis based on Order_Frequency.
    """
    if order_frequency < 4:
        sentiment_label = "Negative"
    elif 4 <= order_frequency <= 7:
        sentiment_label = "Neutral"
    else:  # order_frequency > 7
        sentiment_label = "Positive"

    sentiment_score = {
        "Order_Frequency": order_frequency,
        "sentiment_label": sentiment_label
    }
    return sentiment_score
    
# --- Routes ---
@app.route('/')
def home():
    return "Customer Insights and Sales Prediction API is running."

# --- File Upload ---
@app.route('/upload', methods=['GET', 'POST'])
def handle_upload():
    global data
    if request.method == 'GET':
        if data is None:
            return jsonify({"error": "No data uploaded."}), 400
        return jsonify({"message": "Data summary.", "columns": list(data.columns), "rowCount": len(data)}), 200

    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        try:
            file_size = len(file.read())
            file.seek(0)  # Reset file pointer to the beginning

            if file.filename.endswith('.csv'):
                data = pd.read_csv(file)
            elif file.filename.endswith('.xlsx'):
                data = pd.read_excel(file)
            else:
                return jsonify({"error": "Unsupported file format"}), 400

            return jsonify({"message": "File uploaded successfully!", "columns": list(data.columns), "rowCount": len(data), "fileSize": file_size}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

# --- Customer Segmentation ---
@app.route('/segmentation', methods=['GET', 'POST'])
def handle_segmentation():
    global data, latest_results
    if request.method == 'GET':
        if latest_results["segmentation"] is None:
            return jsonify({"error": "No segmentation performed yet."}), 400
        return jsonify({"message": "Latest segmentation results.", "results": latest_results["segmentation"]}), 200

    if request.method == 'POST':
        if data is None:
            return jsonify({"error": "No data uploaded."}), 400

        clustering_features = request.json.get('features', [])
        if not all(feature in data.columns for feature in clustering_features):
            return jsonify({"error": "Required features not found in data."}), 400

        try:
            data, kmeans_model = kmeans_clustering(data, clustering_features, n_clusters=3)
            data, dbscan_model = dbscan_clustering(data, clustering_features)

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            sns.scatterplot(x=data[clustering_features[0]], y=data[clustering_features[1]],
                            hue=data['KMeans_Cluster'], palette='viridis', ax=ax[0])
            ax[0].set_title("K-Means Clustering")

            sns.scatterplot(x=data[clustering_features[0]], y=data[clustering_features[1]],
                            hue=data['DBSCAN_Cluster'], palette='viridis', ax=ax[1])
            ax[1].set_title("DBSCAN Clustering")

            img_io = io.BytesIO()
            plt.savefig(img_io, format='png')
            img_io.seek(0)
            img_base64 = base64.b64encode(img_io.read()).decode()

            latest_results["segmentation"] = {"plot": img_base64}
            return jsonify({"message": "Segmentation completed.", "plot": img_base64}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

# --- Sales Prediction ---
@app.route('/sales-prediction', methods=['GET', 'POST'])
def handle_sales_prediction():
    global data, latest_results
    if request.method == 'GET':
        if latest_results["sales_prediction"] is None:
            return jsonify({"error": "No sales prediction performed yet."}), 400
        return jsonify({"message": "Latest sales prediction results.", "results": latest_results["sales_prediction"]}), 200

    if request.method == 'POST':
        if data is None:
            return jsonify({"error": "No data uploaded."}), 400

        if 'product' not in data.columns or 'Order_Frequency' not in data.columns:
            return jsonify({"error": "Required columns not found in data."}), 400

        try:
            train, test, predictions, model_fit = sales_prediction(data)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(test.index, test.values, label="Actual Sales")
            ax.plot(test.index, predictions.values, label="Predicted Sales", linestyle="--")
            ax.legend()
            ax.set_title("ARIMA Sales Forecasting")
            ax.set_xlabel("Date")
            ax.set_ylabel("Sales")

            img_io = io.BytesIO()
            plt.savefig(img_io, format='png')
            img_io.seek(0)
            img_base64 = base64.b64encode(img_io.read()).decode()

            latest_results["sales_prediction"] = {"plot": img_base64}
            return jsonify({"message": "Sales prediction completed.", "plot": img_base64}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

# --- Sentiment Analysis ---
# --- Sentiment Analysis ---
@app.route('/sentiment-analysis', methods=['GET', 'POST'])
def handle_sentiment_analysis():
    global data  # Access the uploaded dataset

    if request.method == 'GET':
        return jsonify({"message": "Send a POST request with 'Customer_ID' to analyze sentiment."}), 200

    if request.method == 'POST':
        try:
            # Ensure data is loaded
            if data is None:
                return jsonify({"error": "No data uploaded. Please upload the dataset first."}), 400

            # Check if request has JSON data
            if not request.is_json:
                return jsonify({"error": "Request must include JSON data."}), 400

            # Extract and validate Customer_ID
            request_data = request.get_json()
            if 'Customer_ID' not in request_data:
                return jsonify({"error": "Customer_ID is required in the request."}), 400

            try:
                customer_id = int(request_data['Customer_ID'])
            except ValueError:
                return jsonify({"error": "Customer_ID must be a valid integer."}), 400

            # Verify Order_Frequency column exists
            if 'Order_Frequency' not in data.columns:
                return jsonify({"error": "Order_Frequency column not found in dataset."}), 400

            # Find the customer in the dataset
            customer_data = data[data['Customer_ID'] == customer_id]
            if customer_data.empty:
                return jsonify({"error": f"No record found for Customer_ID: {customer_id}"}), 404

            # Safely get Order_Frequency and ensure it's a number
            try:
                order_frequency = float(customer_data.iloc[0]['Order_Frequency'])
            except (ValueError, TypeError):
                return jsonify({"error": "Invalid Order_Frequency value in data."}), 400

            # Determine sentiment based on Order_Frequency
            if order_frequency < 4:
                sentiment_label = "Negative"
            elif 4 <= order_frequency <= 7:
                sentiment_label = "Neutral"
            else:
                sentiment_label = "Positive"

            # Return the sentiment analysis result
            return jsonify({
                "sentiment_score": {
                    "Customer_ID": customer_id,
                    "Order_Frequency": float(order_frequency),
                    "sentiment_label": sentiment_label
                }
            }), 200

        except Exception as e:
            print("Debug: Exception occurred:", str(e))
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500
        
        
@app.route('/regression-comparison', methods=['GET', 'POST'])
def handle_regression_comparison():
    global data, latest_results

    if request.method == 'GET':
        if latest_results["regression_comparison"] is None:
            return jsonify({"error": "No regression comparison performed yet."}), 400
        return jsonify({"message": "Latest regression comparison results.", "results": latest_results["regression_comparison"]}), 200

    if request.method == 'POST':
        if data is None:
            return jsonify({"error": "No data uploaded."}), 400

        # Get features and target from the request
        features = request.json.get('features', [
            'Age', 'Purchase_History', 'Product_Views', 'Abandoned_Carts',
            'Order_Frequency', 'Average_Order_Value', 'Email_Interactions',
            'Ad_Engagement', 'Social_Media_Activity'
        ])
        target = request.json.get('Product_Views', 'Order_Frequency')

        # Check if required features exist in the data
        if not all(feature in data.columns for feature in features):
            return jsonify({"error": "Required features not found in data."}), 400

        if target not in data.columns:
            return jsonify({"error": "Target column not found in data."}), 400

        try:
            # Prepare the data for training and testing
            X = data[features]
            y = data[target]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Linear Regression
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred_lr = lr.predict(X_test)
            lr_mse = mean_squared_error(y_test, y_pred_lr)
            lr_r2 = r2_score(y_test, y_pred_lr)

            # Decision Tree
            dt = DecisionTreeRegressor(random_state=42)
            dt.fit(X_train, y_train)
            y_pred_dt = dt.predict(X_test)
            dt_mse = mean_squared_error(y_test, y_pred_dt)
            dt_r2 = r2_score(y_test, y_pred_dt)

            # Random Forest
            rf = RandomForestRegressor(random_state=42)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            rf_mse = mean_squared_error(y_test, y_pred_rf)
            rf_r2 = r2_score(y_test, y_pred_rf)

            # Store results
            results = {
                "Linear Regression": {"MSE": lr_mse, "R2": lr_r2},
                "Decision Tree": {"MSE": dt_mse, "R2": dt_r2},
                "Random Forest": {"MSE": rf_mse, "R2": rf_r2}
            }
            latest_results["regression_comparison"] = results
            return jsonify({"message": "Regression comparison completed.", "results": results}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
