import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

nltk.download('vader_lexicon')

def load_data():
    tweets_path = r'C:\Users\Satvik Pandey\OneDrive\Desktop\Stock Market Predictor\Datasets\stock_tweets.csv'
    stock_data_path = r'C:\Users\Satvik Pandey\OneDrive\Desktop\Stock Market Predictor\Datasets\stock_yfinance_data.csv'
    
    tweets_df = pd.read_csv(tweets_path)
    stock_data_df = pd.read_csv(stock_data_path)

    return tweets_df, stock_data_df

def perform_sentiment_analysis(tweets_df):
    sia = SentimentIntensityAnalyzer()
    tweets_df['Sentiment Score'] = tweets_df['Tweet'].apply(lambda x: sia.polarity_scores(x)['compound'])
    return tweets_df

def prepare_data(tweets_df, stock_data_df):
    
    merged_df = pd.merge(stock_data_df, tweets_df, on=['Date', 'Stock Name'], how='left')
    
   
    merged_df['Sentiment Score'].fillna(0, inplace=True)

  
    X = merged_df[['Open', 'High', 'Low', 'Volume', 'Sentiment Score']]
    y = merged_df['Close']
    
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    
    joblib.dump(model, 'stock_price_predictor_model.pkl')

    return model

def main():
    tweets_df, stock_data_df = load_data()
    tweets_df = perform_sentiment_analysis(tweets_df)
    X, y = prepare_data(tweets_df, stock_data_df)
    model = train_model(X, y)
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    main()
