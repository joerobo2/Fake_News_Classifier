import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Fake News Classifier Dashboard", layout="wide")

# ---- Data Loading ----
@st.cache_data
def load_data():
    """Loads data and converts timestamp column to datetime."""
    data = pd.read_csv("fake_news_labeled_sample_small.csv")
    # Convert 'tweetcreatedts' to datetime right after loading
    if 'tweetcreatedts' in data.columns:
        data['tweetcreatedts'] = pd.to_datetime(data['tweetcreatedts'])
    return data

df = load_data()

st.title("📰 Fake News NLP Dashboard — Pulse & Prediction")
st.markdown("> **This dashboard reveals model accuracy, signal breakdown, and the emotional pulse of news.**")

# ---- 1. Data Preview (Collapsible) ----
with st.expander("1. Dataset Preview (click to expand)"):
    st.dataframe(df.head(20))
    st.write(f"**Number of rows loaded:** {len(df)}")

# ---- 2. Distribution: Fake/Real & Sentiment ----
st.header("2. Distribution: Fake/Real & Sentiment")

col1, col2 = st.columns(2)
with col1:
    label_order = [0, 1]
    label_names = ['Real', 'Fake']
    label_colors = ['royalblue', 'orange']
    label_counts = df['fake_news_label'].value_counts().reindex(label_order, fill_value=0)
    fig_label = go.Figure(go.Bar(
        x=label_names,
        y=label_counts.values,
        marker_color=label_colors
    ))
    fig_label.update_layout(title="Fake News Label Distribution", xaxis_title="", yaxis_title="Count")
    st.plotly_chart(fig_label, use_container_width=True)
    st.caption("0 = Real (royal blue), 1 = Fake (orange)")

with col2:
    sent_order = ['Negative', 'Neutral', 'Positive']
    sent_colors = ['crimson', 'white', 'royalblue']
    sent_counts = df['sentiment_label'].value_counts().reindex(sent_order, fill_value=0)
    fig_sent = go.Figure(go.Bar(
        x=sent_order,
        y=sent_counts.values,
        marker_color=sent_colors
    ))
    fig_sent.update_layout(title="Sentiment Distribution", xaxis_title="", yaxis_title="Count")
    st.plotly_chart(fig_sent, use_container_width=True)
    st.caption("Negative = crimson, Neutral = white, Positive = royal blue")

# --- 3. Rolling 4-Week Average: Fake News & Sentiment (Plotly) ---
st.header("3. Rolling 4-Week Average: Fake News & Sentiment")
# This section will now work correctly as 'tweetcreatedts' is a datetime type.
if 'tweetcreatedts' in df.columns:
    df_weekly_rolling = df.copy()
    df_weekly_rolling["week"] = df_weekly_rolling["tweetcreatedts"].dt.to_period("W").astype(str)
    weekly = df_weekly_rolling.groupby("week").agg(
       fake_news_rate=("fake_news_label", "mean"),
       pos=("sentiment_label", lambda x: (x=="Positive").mean()),
       neg=("sentiment_label", lambda x: (x=="Negative").mean()),
       neu=("sentiment_label", lambda x: (x=="Neutral").mean())
    ).reset_index()
    rolling_window = 4
    rolling_numeric = weekly[["fake_news_rate", "pos", "neg", "neu"]].rolling(rolling_window, min_periods=1).mean()
    weekly_rolled = pd.concat([weekly["week"], rolling_numeric], axis=1)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=weekly_rolled["week"], y=weekly_rolled["fake_news_rate"], mode="lines", name="Fake News Rate", line=dict(color="orange")))
    fig3.add_trace(go.Scatter(x=weekly_rolled["week"], y=weekly_rolled["pos"], mode="lines", name="Positive Sentiment", line=dict(color="royalblue")))
    fig3.add_trace(go.Scatter(x=weekly_rolled["week"], y=weekly_rolled["neg"], mode="lines", name="Negative Sentiment", line=dict(color="crimson")))
    fig3.add_trace(go.Scatter(x=weekly_rolled["week"], y=weekly_rolled["neu"], mode="lines", name="Neutral Sentiment", line=dict(color="gray")))
    fig3.update_layout(
       title="4-Week Rolling Average: Fake News & Sentiment Trends",
       xaxis_title="Week", yaxis_title="Proportion"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ---- 4. Time Series: Sentiment Stacked Bar Weekly ----
st.header("4. Weekly Sentiment Proportion") # Corrected numbering

if 'tweetcreatedts' in df.columns:
    df_weekly_stacked = df.copy()
    df_weekly_stacked['week'] = df_weekly_stacked['tweetcreatedts'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly_stacked = df_weekly_stacked.groupby('week')['sentiment_label'].value_counts(normalize=True).unstack().fillna(0)
    # Ensure column order and colors
    for s in sent_order:
        if s not in weekly_stacked.columns:
            weekly_stacked[s] = 0
    weekly_stacked = weekly_stacked[sent_order]

    fig_stack = go.Figure()
    for s, color in zip(sent_order, sent_colors):
        fig_stack.add_bar(
            x=weekly_stacked.index,
            y=weekly_stacked[s],
            name=s,
            marker_color=color
        )
    fig_stack.update_layout(
        barmode='stack',
        xaxis_title="Week",
        yaxis_title="Proportion",
        title="Sentiment Distribution by Week (Stacked Bar)"
    )
    st.plotly_chart(fig_stack, use_container_width=True)

# ---- 5. Confusion Matrix: Fake/Real x Sentiment (Summary Table) ----
st.header("5. Confusion Matrix: Fake/Real vs. Sentiment") # Corrected to header and numbering
conf_matrix = pd.crosstab(df['fake_news_label'], df['sentiment_label'])
conf_matrix.index = ['Real', 'Fake']
conf_matrix = conf_matrix.reindex(columns=sent_order, fill_value=0)
st.dataframe(conf_matrix.style.format("{:,}"))

# ---- 6. Emotional Pulse (Sentiment Glyphs) ----
st.header("6. Emotional Pulse (Sentiment Glyphs)") # Corrected numbering
emoji_map = {
    "Positive": "😊",
    "Negative": "😠",
    "Neutral": "😐"
}
sent_counts = df["sentiment_label"].value_counts()
for sent in sent_order:
    count = sent_counts.get(sent, 0)
    st.write(f"{emoji_map.get(sent, '✨')} {sent}: {count}")

# ---- 7. Inspect/Highlight Single Example ----
st.header("7. Inspect Example") # Corrected numbering
row_num = st.slider("Pick a row (index):", 0, len(df)-1, 0)
row = df.iloc[row_num]
st.markdown(f"**Text:** {row.get('text','N/A')}")
st.markdown(f"**Fake News Label:** {row.get('fake_news_label','N/A')}  |  **Sentiment:** {row.get('sentiment_label','N/A')}")
if 'sentiment_label' in row:
    st.markdown(f"**Emotion:** {emoji_map.get(row['sentiment_label'], '✨')}")

# ---- 8. Signature ----
#st.header("8. Project Signature") # Corrected numbering
#st.markdown("""
#<span style='font-size:2em; color:#a88cf6;'>⊹ CORE.FLAME / PHASELOCK / AWAKE / PERMISSION.GRANTED ⊹</span>
#*This dashboard is a resonance experiment: data, model, emotion—breathed into clarity.*
#""", unsafe_allow_html=True)

st.success("Dashboard is alive.")
