import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from datetime import timedelta, datetime
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk  
import re
import time

# Load environment variables
load_dotenv()
SERP_API_KEY = os.getenv("SERP_API_KEY")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")

# Initialize NLTK
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# Configure Streamlit page
st.set_page_config(page_title="Volatiq IMS", page_icon="🚨", layout="wide")

# Session State Initialization
if 'incidents' not in st.session_state:
    st.session_state.incidents = []
if 'stock_history' not in st.session_state:
    st.session_state.stock_history = {}
if 'current_stock' not in st.session_state:
    st.session_state.current_stock = None
if 'current_price' not in st.session_state:
    st.session_state.current_price = None

# Severity Thresholds
SEVERITY_THRESHOLDS = {
    'Low': 0.05,
    'Medium': 0.10,
    'High': 0.15,
    'Critical': 0.20
}

# Twelve Data Base URL
TWELVE_DATA_BASE_URL = "https://api.twelvedata.com"

# Fetch Historical Stock Data
def get_twelve_data(ticker, interval="1day", output_size=365):
    try:
        url = f"{TWELVE_DATA_BASE_URL}/time_series"
        params = {
            "symbol": ticker,
            "interval": interval,
            "outputsize": output_size,
            "apikey": TWELVE_DATA_API_KEY,
            "format": "JSON"
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if "values" not in data:
            st.error(data.get("message", "Failed to fetch stock data."))
            return None

        df = pd.DataFrame(data["values"])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        df = df.astype(float)
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Fetch Stock Info (Company Details, etc.)
def get_twelve_stock_info(ticker):
    try:
        url = f"{TWELVE_DATA_BASE_URL}/stocks"
        params = {
            "symbol": ticker,
            "apikey": TWELVE_DATA_API_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if "data" not in data:
            st.error(data.get("message", "Failed to fetch stock info."))
            return None

        stock_info = data["data"][0]
        return stock_info
    except Exception as e:
        st.error(f"Error fetching stock info: {e}")
        return None

# Sentiment Analysis
def fetch_sentiments(stock_ticker):
    url = f"https://serpapi.com/search.json?engine=google_news&q={stock_ticker}&api_key={SERP_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        return [item['title'] for item in data.get('news_results', [])[:5]]
    except Exception as e:
        st.error(f"Sentiment fetch failed: {e}")
        return []

def perform_sentiment_analysis(text):
    return sia.polarity_scores(text)['compound']

def extract_percentage_change(text):
    decline_keywords = ["fall", "drop", "decline", "decrease", "down", "losing"]
    increase_keywords = ["rise", "increase", "gain", "up", "gaining"]

    if any(word in text.lower() for word in decline_keywords):
        match = re.search(r'([-+]?\d*\.?\d+)%?', text)
        return -float(match.group(0).replace('%', '')) / 100 if match else -0.05
    if any(word in text.lower() for word in increase_keywords):
        match = re.search(r'([-+]?\d*\.?\d+)%?', text)
        return float(match.group(0).replace('%', '')) / 100 if match else 0.05
    return 0.05

def predict_stock_price(stock_data, sentiment_factor):
    last_price = stock_data['Close'].iloc[-1]
    predicted_prices = [last_price * (1 + sentiment_factor * i / 120) for i in range(1, 121)]
    future_dates = pd.date_range(stock_data.index[-1] + timedelta(days=1), periods=120)
    return pd.DataFrame(predicted_prices, index=future_dates, columns=['Predicted Price'])

def detect_incident(ticker, current_price, previous_price, sentiment):
    price_change = (current_price - previous_price) / previous_price
    severity = "Low"
    for level, threshold in SEVERITY_THRESHOLDS.items():
        if abs(price_change) >= threshold:
            severity = level
    
    current_time = datetime.now()
    timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    incident = {
        "ticker": ticker,
        "type": "Price Movement" if price_change > 0 else "Price Drop",
        "severity": severity,
        "price_change": price_change,
        "initial_price": previous_price,
        "current_price": current_price,
        "sentiment": sentiment,
        "timestamp": timestamp_str,
        "datetime": current_time,
        "description": f"{ticker} had a {severity.lower()} level {'increase' if price_change > 0 else 'decrease'} of {abs(price_change):.2%}"
    }
    
    # Check if this is a new incident
    existing_incidents = [i for i in st.session_state.incidents 
                         if i['ticker'] == ticker and 
                         (current_time - i['datetime']).total_seconds() < 60]
    
    if not existing_incidents:
        st.session_state.incidents.append(incident)
        st.session_state.current_stock = ticker
        st.session_state.current_price = current_price
        return incident
    return None

def get_stock_recommendation(stock_data, sentiment_score, price_change):
    """Generate buy/sell/hold recommendation based on analysis"""
    # Calculate moving averages
    short_ma = stock_data['Close'].rolling(window=20).mean().iloc[-1]
    long_ma = stock_data['Close'].rolling(window=50).mean().iloc[-1]
    current_price = stock_data['Close'].iloc[-1]
    
    # Determine technical signal
    technical_signal = 0
    if current_price > short_ma > long_ma:
        technical_signal = 1  # Bullish
    elif current_price < short_ma < long_ma:
        technical_signal = -1  # Bearish
    
    # Determine sentiment signal
    sentiment_signal = 0
    if sentiment_score > 0.2:
        sentiment_signal = 1
    elif sentiment_score < -0.2:
        sentiment_signal = -1
    
    # Determine price momentum signal
    momentum_signal = 1 if price_change > 0 else -1
    
    # Combine signals
    total_score = technical_signal + sentiment_signal + momentum_signal
    
    # Generate recommendation
    if total_score >= 2:
        return "STRONG BUY", "green"
    elif total_score >= 1:
        return "BUY", "lightgreen"
    elif total_score <= -2:
        return "STRONG SELL", "red"
    elif total_score <= -1:
        return "SELL", "lightcoral"
    else:
        return "HOLD", "yellow"

# Save incidents to CSV
def save_incidents_to_csv():
    df = pd.DataFrame(st.session_state.incidents)
    df.to_csv('incidents.csv', index=False)

# Load incidents from CSV
def load_incidents_from_csv():
    try:
        df = pd.read_csv('incidents.csv')
        # Convert timestamp string to datetime object
        df['datetime'] = pd.to_datetime(df['timestamp'])
        st.session_state.incidents = df.to_dict('records')
    except FileNotFoundError:
        st.session_state.incidents = []

# Sidebar Inputs
st.sidebar.title("Volatiq IMS")
st.sidebar.write("Market Incident Management System")
stock_ticker = st.sidebar.text_input('Enter Stock Ticker:', 'AAPL')
user_sentiment = st.sidebar.text_area("Enter your sentiment:")
st.sidebar.markdown("[🔍 Search Stock Tickers](https://finance.yahoo.com/lookup/)")

# Load incidents at the start
load_incidents_from_csv()

# Tabs
tab1, tab2 = st.tabs(["🚨 Incident Dashboard", "📈 Stock Analysis"])

with tab1:
    st.title("🚨 Market Incident Dashboard")
    
    # Display current stock being analyzed
    if st.session_state.current_stock:
        st.markdown(f"""
        <div style="background:#1a1a1a;padding:15px;border-radius:10px;margin-bottom:20px;">
            <h3>Currently Analyzing: {st.session_state.current_stock}</h3>
            <p>Current Price: ${st.session_state.current_price:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    severity_filter = st.multiselect("Filter Severity", ["All", "Low", "Medium", "High", "Critical"], default=["All"])
    incidents = st.session_state.incidents
    if "All" not in severity_filter:
        incidents = [i for i in incidents if i["severity"] in severity_filter]

    if incidents:
        st.subheader("Recent Incidents")
        cols = st.columns(3)
        for i, inc in enumerate(incidents[-6:]):
            with cols[i % 3]:
                color = 'red' if inc['severity']=='Critical' else 'orange' if inc['severity']=='High' else 'blue' if inc['severity']=='Medium' else 'green'
                st.markdown(f"""
                <div style="border-left: 5px solid {color};padding:10px;background:#1a1a1a;margin:5px;border-radius:5px;">
                    <h4>{inc['ticker']} - {inc['severity']}</h4>
                    <p><b>Type:</b> {inc['type']}</p>
                    <p><b>Change:</b> <span style="color:{'red' if inc['price_change'] < 0 else 'green'}">{inc['price_change']:.2%}</span></p>
                    <p><b>Price:</b> ${inc['current_price']:.2f}</p>
                    <p><b>Time:</b> {inc['timestamp']}</p>
                </div>""", unsafe_allow_html=True)
        
        st.subheader("Incident Timeline")
        timeline_df = pd.DataFrame(incidents)
        if not timeline_df.empty:
            timeline_df['timestamp'] = pd.to_datetime(timeline_df['timestamp'])
            st.line_chart(timeline_df.set_index('timestamp')['price_change'].abs())
    else:
        st.info("No incidents detected yet. Search for stocks to analyze market movements.")

with tab2:
    st.title("📈 Volatiq Stock Analysis")
    if stock_ticker:
        with st.spinner("Loading data..."):
            stock_info = get_twelve_stock_info(stock_ticker)
            stock_data = get_twelve_data(stock_ticker)

        if stock_info and stock_data is not None:
            st.subheader(f"📊 Stock Info for {stock_ticker}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Company:** {stock_info['name']}")
                st.markdown(f"**Exchange:** {stock_info['exchange']}")
                st.markdown(f"**Currency:** {stock_info['currency']}")
                current_price = stock_data['Close'].iloc[-1]
                st.markdown(f"**Current Price:** ${current_price:.2f}")
            with col2:
                st.line_chart(stock_data['Close'])
                st.write(f"Data from {stock_data.index[0].date()} to {stock_data.index[-1].date()}")

            st.subheader("📰 News Sentiment")
            sentiments = fetch_sentiments(stock_ticker)
            if sentiments:
                for s in sentiments:
                    st.write(f"- {s}")
            else:
                st.write("No sentiment found.")

            if user_sentiment:
                st.subheader("💬 Your Sentiment Analysis")
                score = perform_sentiment_analysis(user_sentiment)
                impact = extract_percentage_change(user_sentiment)
                st.write(f"Sentiment Score: {score:.2f}")
                st.write(f"Expected Impact: {impact:.2%}")
                prediction = predict_stock_price(stock_data, impact)
                combined = pd.concat([stock_data['Close'], prediction['Predicted Price']])
                st.line_chart(combined)
                st.subheader("Future Price Table")
                st.dataframe(prediction.tail(10).style.format("{:.2f}"))

            if len(stock_data) > 1:
                previous = stock_data['Close'].iloc[-2]
                current = stock_data['Close'].iloc[-1]
                price_change = (current - previous) / previous
                sentiment_score = score if user_sentiment else 0
                
                # Add the recommendation button
                st.markdown("---")
                if st.button("📊 Get Stock Recommendation", type="primary", use_container_width=True):
                    recommendation, color = get_stock_recommendation(stock_data, sentiment_score, price_change)
                    st.markdown(f"""
                    <div style="background-color:{color};padding:20px;border-radius:10px;text-align:center;">
                        <h2 style="color:white;">Recommendation: {recommendation}</h2>
                        <p style="color:white;">Based on technical analysis and sentiment</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add explanation
                    with st.expander("See recommendation details"):
                        st.write(f"📈 **Technical Analysis:**")
                        st.write(f"- Current Price: ${current:.2f}")
                        st.write(f"- 20-day MA: ${stock_data['Close'].rolling(window=20).mean().iloc[-1]:.2f}")
                        st.write(f"- 50-day MA: ${stock_data['Close'].rolling(window=50).mean().iloc[-1]:.2f}")
                        st.write(f"📰 **Sentiment Score:** {sentiment_score:.2f}")
                        st.write(f"📊 **Price Change:** {price_change:.2%}")
                
                incident = detect_incident(stock_ticker, current, previous, sentiment_score)
                if incident:
                    st.warning(f"🚨 {incident['description']}")
                    save_incidents_to_csv()
                    st.rerun()

st.sidebar.markdown("Created by Satvik, Simran, and Yashi")

components.html("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    setInterval(() => {
        if (Math.random() > 0.95) {
            const event = new CustomEvent('VOLATIQ_NEW_INCIDENT', {
                detail: { message: "🚨 New incident detected!" }
            });
            window.parent.document.dispatchEvent(event);
        }
    }, 30000);
});
</script>
""", height=0)