import dash
import time
import pandas as pd
import yfinance as yf
from prophet import Prophet
from dotenv import load_dotenv
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the .env file
load_dotenv('/Users/abhi/Desktop/projects/python_stock_project/.env')

# Step 1: Gather Data

# Download historical data as pandas DataFrame
data = yf.download('AAPL','2020-01-01','2023-12-31')

# Step 2: Feature Engineering

data['Rolling'] = data['Close'].rolling(window=5).mean()

# Step 3: Simple ML model

# Drop any rows with missing values
data.dropna(inplace=True)

# The feature is the rolling window
X = data[['Rolling']]

# The target variable is the next day's closing price
y = data['Close'].shift(-1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, random_state=0)

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Now we can make predictions on the test set
predictions = model.predict(X_test)

# Step 4: Build the dashboard
app = dash.Dash(__name__)

# Prepare data for Prophet
prophet_data = data[['Close']].reset_index()
prophet_data.columns = ['ds', 'y']

# Instantiate a new Prophet object
m = Prophet(daily_seasonality=True)

# Fit the Prophet model on our data
m.fit(prophet_data)

# Create a dataframe for future predictions
future = m.make_future_dataframe(periods=60)

# Use the model to make predictions
forecast = m.predict(future)

# Plotly figure setup
fig = go.Figure()

# Add actual data to the plot
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Actual'))

# Add forecast data to the plot
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))

# Add confidence interval
fig.add_trace(go.Scatter(
    name='Upper Bound',
    x=forecast['ds'],
    y=forecast['yhat_upper'],
    mode='lines',
    marker=dict(color="#444"),
    line=dict(width=0),
    showlegend=False))
fig.add_trace(go.Scatter(
    name='Lower Bound',
    x=forecast['ds'],
    y=forecast['yhat_lower'],
    marker=dict(color="#444"),
    line=dict(width=0),
    mode='lines',
    fillcolor='rgba(68, 68, 68, 0.3)',
    fill='tonexty',
    showlegend=False))

# Set plot title and labels
fig.update_layout(
    title_text='AAPL Stock Price Forecast',
    yaxis_title='Stock Price',
    xaxis_title='Date',
    hovermode='x'
)

# Step 5: Generate Data Summary

def generate_summary(data, prediction):
    message = f"""
    The stock had a closing price of ${round(data['Close'].iloc[-1], 2)} today. 
    The rolling average over the past 5 days was ${round(data['Rolling'].iloc[-1], 2)}. 
    The predicted closing price for tomorrow is ${round(prediction, 2)}.
    """
    return message

# Generate the summary after making the prediction
summary = generate_summary(data, model.predict([[data['Rolling'].iloc[-1]]])[0])

# Define Dash application layout
app.layout = html.Div([
    dcc.Graph(figure=fig),
    html.Div(children=[
        html.P(id='summary', children=[generate_summary(data, model.predict([[data['Rolling'].iloc[-1]]])[0])])
    ], style={'textAlign': 'center'})
])

if __name__ == '__main__':
    app.run_server(debug=True)
