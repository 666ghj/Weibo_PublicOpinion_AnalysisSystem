# Weibo Public Opinion Analysis System
This project is a **Social Network Public Opinion Analysis System** designed for monitoring, analyzing, and predicting public opinion trends using data from social media platforms such as Weibo.

**Keywords**: Deep Learning, Web Scraping, Full-Stack Development, Natural Language Processing (NLP), Transformers, Flask, Sentiment Analysis, Topic Classification, Data Visualization, Real-time Monitoring, Machine Learning

## Features

- **Real-time Data Collection**: Scrapes and processes data from social platforms.
- **Data Cleaning & Processing**: Cleans and processes collected data for analysis.
- **Topic Classification**: Categorizes posts and comments into relevant topics using machine learning.
- **Sentiment Analysis**: Detects emotional tone (positive, neutral, or negative) in text.
- **Trend Prediction**: Predicts future trends in public opinion based on historical data.

## Installation & Setup

1. Install the necessary environment dependencies (optional):

   ```bash
   conda install --file requirements.txt
   ```

2. Configure your MySQL database:

   - Run `createTables.sql` to set up the required tables.
   - Modify the MySQL configuration in the program accordingly.

3. Start the project with Flask:

   ```bash
   python app.py
   ```
