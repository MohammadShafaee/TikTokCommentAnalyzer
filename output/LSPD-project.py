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

    async with TikTokApi() as api:
        await api.create_sessions(ms_tokens=[os.getenv("ms_token")], num_sessions=1, sleep_after=3)
        video = api.video(id=video_id)
        async for comment in video.comments(count=100000):
            comment_data = comment.as_dict
            create_time = datetime.datetime.fromtimestamp(comment_data.get('create_time', 0))

            # Store comments in memory
            comments_data.append({
                'text': comment_data.get('text', ''),
                'create_time': create_time,
                'digg_count': comment_data.get('digg_count', 0),
                'reply_comment_total': comment_data.get('reply_comment_total', 0),
                'comment_language': comment_data.get('comment_language', ''),
                'user_nickname': comment_data.get('user', {}).get('nickname', ''),
            })

    # Perform sentiment analysis in memory
    for comment in comments_data:
        text = comment['text']
        try:
            result = pipe(text)[0]  # Perform sentiment analysis
            comment['sentiment'] = result['label']
        except Exception:
            comment['sentiment'] = "Error"

    # Convert comments data to a DataFrame
    df = pd.DataFrame(comments_data)

    # Prepare for time series analysis
    df['create_time'] = pd.to_datetime(df['create_time'])
    df.set_index('create_time', inplace=True)

    # Count sentiments
    sentiment_counts = df['sentiment'].value_counts()

    # Generate the video URL using the video ID
    video_url = f"https://www.tiktok.com/@user/video/{video_id}"

    # Bar graph for sentiment counts
    plt.figure(figsize=(10, 6))
    ax = sentiment_counts.plot(kind='bar', color=['grey', 'green', 'red'], alpha=0.8)

    # Add total comments and video URL to the title
    total_comments = len(comments_data)
    plt.title(
        f"Sentiment Counts for TikTok Video\n"
        f"Total Comments: {total_comments} | Video URL: {video_url}",
        fontsize=12,
    )
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=0)

    # Display the value of each bar on the graph
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}", 
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha='center', va='bottom', fontsize=10
        )

    plt.tight_layout()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the bar graph
    bar_graph_path = os.path.join(output_dir, "sentiment_counts_bar_graph.png")
    plt.savefig(bar_graph_path)

    # Resample data monthly
    monthly_time_series = df.groupby('sentiment').resample('ME').size().unstack(0, fill_value=0)

    # Define custom colors
    sentiment_colors = {'positive': 'green', 'negative': 'red', 'neutral': 'grey'}

    # Plot the monthly trends
    plt.figure(figsize=(14, 7))
    for sentiment in monthly_time_series.columns:
        plt.plot(
            monthly_time_series.index,
            monthly_time_series[sentiment],
            label=f"{sentiment.capitalize()}",
            color=sentiment_colors[sentiment.lower()],
            linewidth=2,
        )

    # Add titles and labels
    plt.title('Monthly Trends of TikTok Comments Sentiments', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Comments', fontsize=12)
    plt.xticks(monthly_time_series.index, monthly_time_series.index.strftime('%Y-%m'), rotation=45)
    plt.legend(title='Sentiments', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the time series plot
    graph_output_path = os.path.join(output_dir, "monthly_trends.png")
    plt.savefig(graph_output_path)

    # Save the analyzed comments to a CSV
    csv_output_path = os.path.join(output_dir, "tiktok_comments_with_sentiment.csv")
    df.to_csv(csv_output_path)

    return {
        "comments": comments_data,
        "graph_output_path": graph_output_path,
        "bar_graph_path": bar_graph_path,
        "csv_output_path": csv_output_path,
    }


if __name__ == "__main__":
    result = asyncio.run(fetch_and_analyze_comments(7416899434308537643))
    print(f"Monthly Trends Graph saved at: {result['graph_output_path']}")
    print(f"Bar Graph saved at: {result['bar_graph_path']}")
    print(f"CSV saved at: {result['csv_output_path']}")
