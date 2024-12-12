from TikTokApi import TikTokApi
import asyncio
import datetime
import pandas as pd
from transformers import pipeline
import os
import matplotlib.pyplot as plt

# Initialize sentiment analysis pipeline
pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")


async def fetch_and_analyze_comments(video_id, output_dir="output"):
    """
    Fetches comments from a TikTok video, performs sentiment analysis,
    generates time series plots, bar graph, and saves them to a directory.
    """
    comments_data = []  # In-memory storage for comments


    # Fetch comments
    async with TikTokApi() as api:
        ms_token = os.getenv("ms_token")
        if not ms_token:
            raise ValueError("ms_token not set. Please set the ms_token environment variable.")

        await api.create_sessions(ms_tokens=[ms_token], num_sessions=1, sleep_after=3)
        video = api.video(id=video_id)

        async for comment in video.comments(count=10000):
            comment_data = comment.as_dict
            create_time = datetime.datetime.fromtimestamp(comment_data.get('create_time', 0))
            comments_data.append({
                'text': comment_data.get('text', ''),
                'create_time': create_time,
                'digg_count': comment_data.get('digg_count', 0),
                'reply_comment_total': comment_data.get('reply_comment_total', 0),
                'comment_language': comment_data.get('comment_language', ''),
                'user_nickname': comment_data.get('user', {}).get('nickname', '')
            })

    # Handle no comments scenario
    if not comments_data:
        pd.DataFrame([]).to_csv(os.path.join(script_dir, "tiktok_comments_with_sentiment.csv"), index=False)
        line_chart_path = os.path.join(script_dir, 'sentiment_over_time.png')
        bar_chart_path = os.path.join(script_dir, 'sentiment_bar_chart.png')

        # Empty line chart
        plt.figure()
        plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
        plt.savefig(line_chart_path)
        plt.close()

        # Empty bar chart
        plt.figure()
        plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
        plt.savefig(bar_chart_path)
        plt.close()

        return {
            "comments": [],
            "line_chart_path": line_chart_path,
            "bar_chart_path": bar_chart_path
        }

    # Perform sentiment analysis
    texts = [c['text'] for c in comments_data]
    results = pipe(texts, batch_size=32, truncation=True)

    for comment, res in zip(comments_data, results):
        sentiment = res['label'].lower()
        comment['sentiment'] = sentiment

    # Save to CSV
    df = pd.DataFrame(comments_data)
    csv_path = os.path.join(script_dir, "tiktok_comments_with_sentiment.csv")
    df.to_csv(csv_path, index=False)

    # Prepare monthly grouping
    df['year'] = df['create_time'].dt.year
    df['month'] = df['create_time'].dt.month
    df['month_name'] = df['create_time'].dt.month_name()

    # Group by year, month, sentiment
    sentiment_time = df.groupby(['year', 'month', 'sentiment']).size().reset_index(name='count')
    sentiment_time = sentiment_time.sort_values(by=['year', 'month'])

    pivoted = sentiment_time.pivot(index=['year', 'month'], columns='sentiment', values='count').fillna(0)

    # Define colors
    color_dict = {
        'positive': 'green',
        'negative': 'red',
        'neutral': 'grey'
    }

    # Determine time domain (from earliest to latest comment)
    earliest_comment = df['create_time'].min()
    latest_comment = df['create_time'].max()
    earliest_month_name = earliest_comment.strftime("%B %Y")
    latest_month_name = latest_comment.strftime("%B %Y")

    # LINE CHART: Sentiment over time
    line_chart_path = os.path.join(script_dir, 'sentiment_over_time.png')
    plt.figure(figsize=(10, 6))

    # x_labels for line chart
    x_labels = [f"{calendar.month_name[m]} {y}" for y, m in pivoted.index]

    for sentiment in ['positive', 'negative', 'neutral']:
        if sentiment in pivoted.columns:
            plt.plot(x_labels, pivoted[sentiment], marker='o', color=color_dict[sentiment], label=sentiment.capitalize())

    plt.title(f"Comment Sentiments Over Time for Video ID: {video_id}\n(From {earliest_month_name} to {latest_month_name})")
    plt.xlabel('Month and Year')
    plt.ylabel('Number of Comments')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(line_chart_path)
    plt.close()

    # BAR CHART: Total sentiment counts
    bar_chart_path = os.path.join(script_dir, 'sentiment_bar_chart.png')
    sentiment_counts = df['sentiment'].value_counts()

    plt.figure(figsize=(6, 4))

    # Reorder index if needed
    order = ['positive', 'negative', 'neutral']
    sentiment_counts = sentiment_counts.reindex([s for s in order if s in sentiment_counts.index])

    bars = plt.bar(
        sentiment_counts.index.map(str.capitalize),
        sentiment_counts.values,
        color=[color_dict[s] for s in sentiment_counts.index]
    )

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{int(height)}', ha='center', va='bottom', color='black', fontweight='bold')

    plt.title(f"Total Sentiment Counts for Video ID: {video_id}\n(From {earliest_month_name} to {latest_month_name})")
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(bar_chart_path)
    plt.close()

    return {
        "comments": comments_data,
        "line_chart_path": line_chart_path,
        "bar_chart_path": bar_chart_path
    }

if __name__ == "__main__":
    asyncio.run(fetch_and_analyze_comments(7375925889818758443))
