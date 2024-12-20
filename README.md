# Documentation for TikTok Comment Sentiment Analysis

## Overview

The provided Python code is designed to fetch comments from a specified TikTok video, analyze their sentiment using a pre-trained sentiment analysis model, and generate visualizations of sentiment trends over time. The results are saved in a CSV file and visualized as bar and line charts. 

---

# TikTokCommentAnalyzer
Using this code you can analyze the sentiments of the comments under a TikTok post. This code uses the unofficial TikTok API by David Teather to scrape TikTok data, then using the "cardiffnlp/twitter-roberta-base-sentiment-latest" model of the Transformers library, it performs a sentiment analysis on the comments. With this code, you are able to have a CSV dataset, a bar graph of counts, and a time series of a post's comments with their sentiments. In cases where no comments are available, empty visualizations are generated.

---

## Getting Started

**Note:** Installation requires Python 3.9+
1. Install the requirements by running:
   ```
   pip install -r requirements.txt
   ```

2. Set your `ms_token` in your system variables. You can find `ms_token` by visiting TikTok.com and opening the inspection mode of your browser. Navigate to the Application tab, where you will find the variable "msToken" and its value. Copy the value and run the following command in your terminal:
   ```
   set ms_token=the_value_you_copied
   ```
Now you have all the dependencies installed and can run the script. 

---

## API Calls

1. **TikTokApi:**

   - **Purpose:** Fetch comments from a specified TikTok video.
   - **Functionality:**
     - Uses the `ms_token` (TikTok authentication token) to establish a session.
     - Retrieves up to 10,000 comments for a specified video ID. You can change this according to your need.
   - **API Calls Performed:**
     - `api.create_sessions()` to initiate TikTok sessions.
     - `video.comments()` to fetch comments asynchronously.

2. **Hugging Face Transformers:**

   - **Purpose:** Perform sentiment analysis on the fetched comments.
   - **Model Used:** `cardiffnlp/twitter-roberta-base-sentiment-latest` trained on ~124 million twitter comments.
   - **API Calls Performed:**
     - `pipeline()` for sentiment classification.

---

## Keywords Used

- **TikTokApi:** `video.comments`, `ms_token`
- **Sentiment Analysis Model:** `text-classification`, `cardiffnlp/twitter-roberta-base-sentiment-latest`

---

## Dataset Description

### Source

The dataset is generated by fetching comments from TikTok videos via the `TikTokApi` and augmenting them with sentiment labels using a pre-trained Hugging Face model.

### Structure

1. **Number of Entries:**
   - Depends on the number of comments fetched (up to 10,000 per video). If no comments are available, an empty dataset is created.
2. **Types of Entries:**
   - `text` (string): The content of the comment.
   - `create_time` (datetime): Timestamp of when the comment was created.
   - `digg_count` (integer): Number of likes the comment received.
   - `reply_comment_total` (integer): Number of replies to the comment.
   - `comment_language` (string): The language code of the comment (e.g., `en` for English).
   - `user_nickname` (string): The nickname of the user who posted the comment.
   - `sentiment` (string): Sentiment classification of the comment (positive, negative, or neutral).

---

## Requirements

### Installed Libraries

The following libraries from the `requirements` file are used:

- **aiohttp**: To handle asynchronous HTTP requests.
- **async-timeout**: For setting timeouts in asynchronous operations.
- **pandas**: To manage and process tabular data.
- **numpy**: For numerical operations and data transformations.
- **matplotlib**: To create visualizations of sentiment trends.
- **transformers**: For the sentiment analysis pipeline.
- **TikTokApi**: To interact with the TikTok API and fetch comments.
- **torch**: Backend dependency for `transformers`.

---

## Code Functionality

### 1. **Fetching Comments**

- The function `fetch_and_analyze_comments` initializes a TikTok API session using the `ms_token` environment variable.
- Comments are fetched asynchronously, with a maximum count of 10,000 per video.
- If no comments are retrieved, the function creates an empty CSV file and generates placeholder visualizations.

### 2. **Sentiment Analysis**

- The text of each comment is passed to the sentiment analysis pipeline.
- Results are batched (batch size of 32) for efficiency, and each comment is labeled as positive, negative, or neutral.

### 3. **Data Augmentation and Export**

- Comments are augmented with their respective sentiment labels.
- A pandas DataFrame is created and saved as a CSV file (`tiktok_comments_with_sentiment.csv`).

### 4. **Data Visualization**

#### Line Chart:

- Sentiment counts are grouped by year and month.
- A line chart (`sentiment_over_time.png`) visualizes the sentiment trends over time.
- If no data is available, a placeholder chart with a "No data available" message is generated.

#### Bar Chart:

- Total counts of each sentiment are visualized as a bar chart (`sentiment_bar_chart.png`).
- If no data is available, a placeholder chart with a "No data available" message is generated.

---

## Outputs

### CSV File

- **Filename:** `tiktok_comments_with_sentiment.csv`
- **Columns:**
  - `text`, `create_time`, `digg_count`, `reply_comment_total`, `comment_language`, `user_nickname`, `sentiment`

### Visualizations

1. **Line Chart:** Sentiment trends over time (`sentiment_over_time.png`).
2. **Bar Chart:** Total sentiment counts (`sentiment_bar_chart.png`).
---

## Example Dataset

**Sample Entries:**

| text                                             | create\_time        | digg\_count | reply\_comment\_total | comment\_language | user\_nickname | sentiment |
| ------------------------------------------------ | ------------------- | ----------- | --------------------- | ----------------- | -------------- | --------- |
| "mini audio"                                     | 2022-06-17 00:03:59 | 4           | 0                     | en                | YT PGH-life    | neutral   |
| "So I can’t skip songs or turn the volume..."    | 2022-08-14 14:55:09 | 0           | 1                     | en                | Paolommerda    | negative  |
| "young mister bean got me thinking of buying..." | 2022-06-17 21:19:18 | 2           | 0                     | en                | （*´∀\`*)       | neutral   |

---

## Conclusion

This code provides a robust solution for extracting TikTok comments, analyzing their sentiment, and generating insightful visualizations. It is suitable for marketing analysis, trend identification, and understanding audience sentiment.
