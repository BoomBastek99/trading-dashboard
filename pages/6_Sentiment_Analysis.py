import streamlit as st
import pandas as pd
import numpy as np
import requests
import feedparser
from datetime import datetime, timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
from design_system import *

# Download NLTK data if needed
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# Apply theme
apply_theme()

# Sidebar
st.sidebar.header("Sentiment Analysis")

default_tickers = ["SPY", "AAPL", "NVDA", "TSLA", "BTC-USD"]
tickers = st.sidebar.multiselect("Tickers", options=default_tickers + ["Add Custom"], default=default_tickers[:3])

custom_ticker = st.sidebar.text_input("Custom Ticker")
if custom_ticker and custom_ticker not in tickers:
    tickers.append(custom_ticker)

newsapi_key = st.sidebar.text_input("NewsAPI Key (optional)", type="password")
lookback_days = st.sidebar.slider("Article Lookback (days)", min_value=1, max_value=7, value=3)

refresh = st.sidebar.button("Refresh Sentiment")

@st.cache_data
def fetch_news_newsapi(ticker, api_key, days):
    if not api_key:
        return []
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = f"https://newsapi.org/v2/everything?q={ticker}&from={start_date.date()}&to={end_date.date()}&sortBy=publishedAt&apiKey={api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        articles = []
        for article in data.get('articles', []):
            articles.append({
                'title': article['title'],
                'source': article['source']['name'],
                'published': article['publishedAt'],
                'snippet': article['description'] or article['title'],
                'url': article['url']
            })
        return articles
    except:
        return []

@st.cache_data
def fetch_news_google_rss(ticker, days):
    # Google News RSS feed
    url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    
    try:
        feed = feedparser.parse(url)
        articles = []
        cutoff = datetime.now() - timedelta(days=days)
        
        for entry in feed.entries:
            pub_date = datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now()
            if pub_date >= cutoff:
                articles.append({
                    'title': entry.title,
                    'source': entry.source.title if hasattr(entry, 'source') else 'Google News',
                    'published': pub_date.isoformat(),
                    'snippet': entry.summary if hasattr(entry, 'summary') else entry.title,
                    'url': entry.link
                })
        return articles
    except:
        return []

@st.cache_data
def analyze_sentiment(articles):
    sentiments = []
    for article in articles:
        sentiment = sia.polarity_scores(article['snippet'])
        sentiments.append({
            'title': article['title'],
            'source': article['source'],
            'published': article['published'],
            'sentiment': sentiment['compound'],
            'snippet': article['snippet']
        })
    return sentiments

@st.cache_data
def aggregate_sentiment(sentiments, days):
    if not sentiments:
        return 0, 0, []
    
    df = pd.DataFrame(sentiments)
    df['published'] = pd.to_datetime(df['published'])
    df['age_days'] = (datetime.now() - df['published']).dt.days
    
    # Weight recent articles more
    weights = np.exp(-df['age_days'] / days)
    weighted_sentiment = np.average(df['sentiment'], weights=weights)
    
    # Momentum: compare to older articles
    recent_mask = df['age_days'] <= days / 2
    old_mask = df['age_days'] > days / 2
    recent_avg = df[recent_mask]['sentiment'].mean() if recent_mask.any() else 0
    old_avg = df[old_mask]['sentiment'].mean() if old_mask.any() else 0
    momentum = recent_avg - old_avg
    
    # Top 5 by absolute sentiment
    top_articles = df.reindex(df['sentiment'].abs().sort_values(ascending=False).index).head(5).to_dict('records')
    
    return weighted_sentiment, momentum, top_articles

if refresh or 'sentiment_data' not in st.session_state:
    with st.spinner("Fetching and analyzing news sentiment..."):
        sentiment_data = {}
        total_articles = 0
        
        for ticker in tickers:
            if ticker == "Add Custom":
                continue
            
            # Try NewsAPI first, fallback to Google RSS
            articles = fetch_news_newsapi(ticker, newsapi_key, lookback_days)
            if not articles:
                articles = fetch_news_google_rss(ticker, lookback_days)
            
            sentiments = analyze_sentiment(articles)
            sentiment, momentum, top_articles = aggregate_sentiment(sentiments, lookback_days)
            
            sentiment_data[ticker] = {
                'sentiment': sentiment,
                'momentum': momentum,
                'articles': sentiments,
                'top_articles': top_articles
            }
            total_articles += len(sentiments)
        
        st.session_state['sentiment_data'] = sentiment_data
        st.session_state['total_articles'] = total_articles

if 'sentiment_data' in st.session_state:
    sentiment_data = st.session_state['sentiment_data']
    total_articles = st.session_state['total_articles']
    
    # Top bar
    st.markdown(f"""
    <div style="
        background-color: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    ">
        <h1 style="color: {TEXT_PRIMARY}; margin: 0;">MARKET SENTIMENT BRIEFING</h1>
        <p style="color: {TEXT_SECONDARY}; margin: 5px 0 0 0;">{datetime.now().strftime('%B %d, %Y')} | {datetime.now().strftime('%I:%M %p')}</p>
        <p style="color: {TEXT_MUTED}; margin: 5px 0 0 0;">{total_articles} articles analyzed</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Ticker cards
    cols = st.columns(min(len(sentiment_data), 5))
    selected_ticker = st.selectbox("Select ticker for details", list(sentiment_data.keys()))
    
    for i, (ticker, data) in enumerate(sentiment_data.items()):
        if i >= len(cols):
            break
        with cols[i]:
            sentiment = data['sentiment']
            momentum = data['momentum']
            
            # Gauge color
            if sentiment > 0.1:
                gauge_color = ACCENT_GREEN
            elif sentiment < -0.1:
                gauge_color = ACCENT_RED
            else:
                gauge_color = TEXT_SECONDARY
            
            # Momentum arrow
            if momentum > 0.05:
                arrow = "↑ Improving"
                arrow_color = ACCENT_GREEN
            elif momentum < -0.05:
                arrow = "↓ Declining"
                arrow_color = ACCENT_RED
            else:
                arrow = "→ Stable"
                arrow_color = TEXT_SECONDARY
            
            # Semicircular gauge
            angle = (sentiment + 1) / 2 * 180  # 0 to 180 degrees
            
            st.markdown(f"""
            <div style="text-align: center; margin: 10px 0;">
                <h3 style="color: {TEXT_PRIMARY}; margin: 0 0 10px 0;">{ticker}</h3>
                <div style="
                    width: 120px;
                    height: 60px;
                    border-radius: 60px 60px 0 0;
                    background: conic-gradient({gauge_color} 0deg {angle}deg, {BG_CARD} {angle}deg 180deg);
                    margin: 0 auto;
                    position: relative;
                ">
                    <div style="
                        position: absolute;
                        bottom: 0;
                        left: 50%;
                        transform: translateX(-50%);
                        width: 4px;
                        height: 30px;
                        background-color: {gauge_color};
                        transform-origin: bottom;
                        transform: translateX(-50%) rotate({angle - 90}deg);
                        box-shadow: 0 0 10px {gauge_color}80;
                    "></div>
                    <div style="
                        position: absolute;
                        bottom: -20px;
                        left: 50%;
                        transform: translateX(-50%);
                        color: {gauge_color};
                        font-size: 18px;
                        font-weight: bold;
                        text-shadow: 0 0 5px {gauge_color}40;
                    ">{sentiment:.2f}</div>
                </div>
                <p style="color: {arrow_color}; margin: 10px 0 0 0; font-size: 14px;">{arrow}</p>
                <p style="color: {TEXT_MUTED}; margin: 5px 0 0 0; font-size: 12px;">{len(data['articles'])} articles</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Detail panel
    if selected_ticker in sentiment_data:
        data = sentiment_data[selected_ticker]
        st.markdown(section_header(f"Key Drivers — {selected_ticker}"), unsafe_allow_html=True)
        
        for article in data['top_articles']:
            border_color = ACCENT_GREEN if article['sentiment'] > 0 else ACCENT_RED if article['sentiment'] < 0 else TEXT_SECONDARY
            dot_color = border_color
            
            st.markdown(f"""
            <div style="
                background-color: {BG_CARD};
                border-left: 4px solid {border_color};
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
            ">
                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <span style="color: {TEXT_MUTED}; font-size: 12px;">{article['source']} | {pd.to_datetime(article['published']).strftime('%m/%d')}</span>
                    <span style="
                        display: inline-block;
                        width: 8px;
                        height: 8px;
                        border-radius: 50%;
                        background-color: {dot_color};
                        margin-left: 10px;
                    "></span>
                    <span style="color: {dot_color}; font-size: 12px; margin-left: 5px;">{article['sentiment']:.2f}</span>
                </div>
                <h4 style="color: {TEXT_PRIMARY}; margin: 0 0 5px 0;">{article['title']}</h4>
            </div>
            """, unsafe_allow_html=True)
    
    # Bottom aggregate bar
    st.markdown(section_header("Overall Market Sentiment"), unsafe_allow_html=True)
    
    sentiments = [data['sentiment'] for data in sentiment_data.values()]
    tickers_list = list(sentiment_data.keys())
    
    fig = go.Figure()
    
    # Positive sentiments
    pos_sentiments = [max(0, s) for s in sentiments]
    fig.add_trace(go.Bar(
        x=tickers_list,
        y=pos_sentiments,
        name='Bullish',
        marker_color=ACCENT_GREEN,
        showlegend=False
    ))
    
    # Negative sentiments
    neg_sentiments = [min(0, s) for s in sentiments]
    fig.add_trace(go.Bar(
        x=tickers_list,
        y=neg_sentiments,
        name='Bearish',
        marker_color=ACCENT_RED,
        showlegend=False
    ))
    
    fig.update_layout(
        **get_plotly_layout(),
        barmode='relative',
        height=200,
        xaxis_title='',
        yaxis_title='Sentiment',
        yaxis_range=[-1, 1]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Disclaimer
    st.markdown(f"""
    <div style="
        background-color: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 15px;
        margin: 20px 0;
        color: {TEXT_SECONDARY};
        font-style: italic;
    ">
    Sentiment scores are based on automated text analysis and may misinterpret context. Use as one input among many.
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Configure tickers and click 'Refresh Sentiment' to analyze news sentiment.")