# TikTokCommentAnalyzer
 Using this code you can analyze the sentiments of the comments under a tiktok post. 
This code uses the unofficial Tiktok api by David Teather to scrape tiktok data, then using the "cardiffnlp/twitter-roberta-base-sentiment-latest" model of transformers library, it performs a sentiment analysis on the comments. 
With this code you are able to have a csv dataset, a bar graph of counts, and a time series of a post's comments with their sentiments.

# Getting Started 
Note: Installation requires python 3.9+
In order to run this script you have to first install the TikTokApi by running the following code in your terminal:
pip install TikTokApi
python -m playwright install
Make sure you have installed the version 1.36.0 of playwright. 
Second step is to set your ms_token in your system variables. You can find ms_token by going to the tiktok.com and open the inspection mode of your browser, then navigate to Application tab and you will find the variable "msToken" and its value. Copy the value and open cmd. Set the environment variable by running this code: 'set ms_token=the value you copied'.
The next step is to install the transformers library. You can do so by simply running: "pip install transformers".
Now you have all the dependencies and can run the script.

