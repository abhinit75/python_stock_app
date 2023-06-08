import os
import dash
from dash.dependencies import Input, Output, State
from datetime import datetime
import openai
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

# Access the API key
api_key = os.getenv("OPENAPI_KEY")
openai.api_key = api_key

# Step 1: Gather Data


def get_stock_data(ticker):
    data = yf.download(ticker, '2020-01-01',
                       datetime.today().strftime('%Y-%m-%d'))
    return data

# Step 2: Feature Engineering


def calculate_rolling_average(data):
    data['Rolling'] = data['Close'].rolling(window=5).mean()
    return data

# Step 3: Simple ML model


def train_linear_regression_model(data):
    # Drop any rows with missing values
    data.dropna(inplace=True)

    # The feature is the rolling window
    X = data[['Rolling']]

    # The target variable is the next day's closing price
    y = data['Close'].shift(-1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X[:-1], y[:-1], test_size=0.2, random_state=0)

    # Create and fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


# Step 4: Build the dashboard
app = dash.Dash(__name__)

# Prepare data for Prophet


def prepare_prophet_data(data):
    prophet_data = data[['Close']].reset_index()
    prophet_data.columns = ['ds', 'y']
    return prophet_data

# Instantiate a new Prophet object


def fit_prophet_model(data):
    m = Prophet(daily_seasonality=True)
    m.fit(data)
    return m

# Create a dataframe for future predictions


def make_prophet_forecast(model):
    future = model.make_future_dataframe(periods=120)
    forecast = model.predict(future)
    return forecast

# Set plot title and labels


def set_plot_layout(fig):
    fig.update_layout(
        title_text='Stock Price Forecast',
        yaxis_title='Stock Price',
        xaxis_title='Date',
        hovermode='x'
    )

# Generate the summary after making the prediction


def generate_summary(data, prediction):
    message = f"""
    The stock had a closing price of ${round(data['Close'].iloc[-1], 2)} today. 
    The rolling average over the past 5 days was ${round(data['Rolling'].iloc[-1], 2)}. 
    The predicted closing price for tomorrow is ${round(prediction, 2)}.
    """
    return message


# Define Dash application layout
app.layout = html.Div([
    html.Div(style={'height': '50px'}),  # Add some vertical space
    html.Div(
        html.H2("Enter the ticker symbol below"),
        style={'textAlign': 'center'}  # Center the text header
    ),
    html.Div(
        dcc.Input(id='ticker-input', type='text', value='AAPL',
                  placeholder='Enter stock ticker'),
        style={'textAlign': 'center'}  # Center the input box
    ),
    html.Div(
        html.Button(id='submit-button', children='Submit', n_clicks=0),
        style={'textAlign': 'center'}  # Center the button
    ),
    dcc.Graph(id='graph'),
    html.Div(id='summary', style={'textAlign': 'center'})
])


@app.callback(
    [Output('graph', 'figure'), Output('summary', 'children')],
    [Input('submit-button', 'n_clicks')],
    [State('ticker-input', 'value')]
)
def update_output(n_clicks, ticker):
    # Gather Data
    data = get_stock_data(ticker)

    # Feature Engineering
    data = calculate_rolling_average(data)

    # Simple ML model
    model = train_linear_regression_model(data)

    # Build the Prophet model
    prophet_data = prepare_prophet_data(data)
    m = fit_prophet_model(prophet_data)
    forecast = make_prophet_forecast(m)

    # Plotly figure setup
    fig = go.Figure()

    # Add actual data to the plot
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Close'], mode='lines', name='Actual'))

    # Add forecast data to the plot
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))

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
    set_plot_layout(fig)

    # Generate the summary after making the prediction
    summary = generate_summary(
        data, model.predict([[data['Rolling'].iloc[-1]]])[0])

    return fig, summary


if __name__ == '__main__':
    app.run_server(debug=True)
