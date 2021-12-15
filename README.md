# Problem Statement
As part of the data science team for Facebook Watch , we wish to conduct a training to the e-commerce food businesses to help them to better understand their business and understand viewer sentiments and discussion themes. As such, they will be able to provide more appropriate products and improved services to their viewers over their live streaming videos, which will in turn increase their revenue from their live streaming videos.

# Background
Live streaming videos have become increasingly popular, especially with the COVID-19 pandemic that has made individuals feel increasingly isolated in due to the decrease in physical social interactions. This too has forced many businesses to rethink how sales can be done. Traditionally, sales were done primarily through retail stores, online e-commerce platforms and/or websites, or possibly even over the phone. With the rise of live streaming platforms, many businesses have hence turned to them to promote the sale of their products and/or services over live streaming videos. 

The barriers to entry for businesses to utilize live streaming videos are low as one merely needs a phone and a voice to speak out. With such low barriers to entry, many businesses have taken the plunge to engage with their new & existing customers over live streaming videos which often come with attractive perks for the customers. From the special time-limited discount codes to the limited edition quality products that are sold over the live videos on a first-come-first-serve basis, this attractive qualities that live streaming videos provide is refreshing & new to the scene of sales. Furthermore, customers can engage in having their queries and doubts answered promptly and immediately over the live streaming videos. All this varying benefits from shopping via live streaming videos at the expense of the comfort of one's home, is indeed attractive.

However, are business owners able to fully understand their customers? The lack of physical social interaction and inability for them to take social cues, can be a gap between the business owners and customers. 

# Process
The notebook consists of 3 sections: codes, data and presentation. The codes are divided into 3 folders: data collection, data cleaning and data analysis & modelling.

# Data Collection
The data collection process allows us to use Selenium to scrap the comments from Facebook Live Sales videos. In addition to the comments being scrapped, the comments' author, comments' time, number of emoji reactions to the video and total number of views are scrapped as well. This scrapped data will form our dataset from 15 Facebook Live Sales videos. Among this 15 Facebook Live Sales Videos, we have equally scrapped 5 videos each from 3 different sellers. The one thing the sellers have in common is that they are food sellers that specialize in selling seafood.

# Data Cleaning
In the data cleaning process, we use regular expression to clean our raw & uncleaned data for each video individually. This is because the language used in each video is unique, and only the videos that are by the same seller bear a greater amount of similarity. Some notably cleaning were to remove the HTML parsers to the emojis, tagged names and whitespaces. Additionally, standard cleaning like the removal of links and HTML special entities, and dropping of empty comments were done as well. After the data has been cleaned, feature engineering was done to provide us with features that will might possibly affect the total revenue for the videos. 

Before the EDA process, all the individually cleaned data for the 15 videos are concatenated to 1 dataframe. There are 2 main dataframes that have been concatenated: 1 for the video attributes, and 1 for the comments attributes. Below is the data dictionary that clarifies and further explains the features for each dataframe. 

## Data Dictionary
|Feature|Type|Dataset|Description|
|---|---|---|---| 
|video_for|*object*|all_video_attributes.csv|This identifies in the following order: the name of the seller's Facebook account, followed by the word video to determine the data is collect from a video (instead of other sources like a Facebook post or photo for example), and the string of numbers is the unique video number belonging to the particular video where the data is scrapped from.| 
|totalEmojiReaction|*int*|all_video_attributes.csv|The total number of emoji reactions the video has received from Facebook users.| 
|views|*int*|all_video_attributes.csv|The total number of views the video has received from Facebook users. | 
|videoLength|*int*|all_video_attributes.csv|The length of the video has received from Facebook users in seconds.|
|numSellerComments|*int*|all_video_attributes.csv|The total number of times the seller has posted a comment in the video.|
|numComments|*int*|all_video_attributes.csv|The total number of comments by all the Facebook users (including the seller & customers) for the video.|
|lnsQuantity|*int*|all_video_attributes.csv|The total number of comments where the acronym 'LNS' has been mentioned in the video by all the Facebook users for the video. The acronym 'LNS' stands for 'liked & shared' which can be an indication of user engagement for the video.|
|salesQuantity|*int*|all_video_attributes.csv & all_comments_with_sa.csv|The total quantity of sales made by the customers for the video.|
|numProducts|*int*|all_video_attributes.csv|The total quantity of unique products offered by the seller in the video.|
|totalRevenue|*float*|all_video_attributes.csv|The total revenue of the sales from the video in Singapore Dollars.|
|frequencySeller|*float*|all_video_attributes.csv|The frequency of the seller posting a comment in the video.|
|averageCompound|*float*|all_video_attributes.csv|The average compound score from the VADER Sentiment Analysis for the comments for the video|
|postCommentAuthor|*object*|all_comments_with_sa.csv|The Facebook name of the user who posted the comment in the video.|
|postCommentTime_final|*object*|all_comments_with_sa.csv|The timestamp of the comment in the video in the format of HH:MM:SS.|
|isSeller|*float*|all_comments_with_sa.csv|This identifies if the comment has been posted by the seller or not. If it is posted by the seller, it returns '1', otherwise '0'.|
|postCommentLength|*float*|all_comments_with_sa.csv|The total number of words in the comment posted for the video.|
|lns|*float*|all_comments_with_sa.csv|The total number of comments where the acronymn 'LNS' has been mentioned in the video.|
|revenue|*float*|all_comments_with_sa.csv|The total revenue achieved at that particular comment and timestamp for the video.|
|seller|*object*|all_comments_with_sa.csv|The name of the seller for the video.|
|postCommentProcessed|*object*|all_comments_with_sa.csv|The processed comment posted in the video.|
|compound|*float*|all_comments_with_sa.csv|The compound score using VADER for the comment.|
|sentiment_category|*object*|all_comments_with_sa.csv|The sentiment category of the comment. There are 3 categories: neutral, positive and negative. Neutral comments have a compound score of 0, while positive comments have a compound score greater than 0, and negative comments have a compound score of less than 0.|
|normalized_sentiment_score|*float*|all_comments_with_sa.csv|The comments' normalized sentiment score which has a range of scores are between -1 and +1.|
|positive_negative_ratio|*float*|all_comments_with_sa.csv|The comments' ratio of positive to negative sentiments, which has a range of score between 0 to infinity and a score of around 1 is considered to be neutral.|
    
# EDA
The EDA process can be split into 2 parts: for the video attributes, and for the comments attributes. For the video attributes, we place our target variable as the total revenue of the video and look into the correlation of the features with the target variable. Features that had a notable correlation with total revenue of the video are: the sales quantity, number of times the acronym for 'like and share' was being mentioned in the comments section of the video, number of unique products provided by the seller, number of comments made by the seller, and the frequency of comments made by the seller. For the comments attributes, we studied the sentiment analysis of the comments and obtained the sentiment category for each comment. Ngram visualizations were generated as well to see the top words among the comments, regardless of their sentiments. 

# Modelling
For modelling, we engaged in topic modelling to understand the sentiments themes for the viewers for each seller. We used Latent Dirichlet Allocation for Topic Modeling. As a whole, all 3 sellers bear a significant amount of similarity in their sentiments discussion themes. Neutral sentiments often revolve around product and general enquires, and greetings. Positive sentiments often revolve around specific products that the viewers are interested in, satisfactory products & services provided by the sellers, and promotional events such as giveaways & sales. Negative sentiments mostly revolve around the viewers' general worries & concerns and negative remarks to the sellers.

# Conclusion
There are certain attributes, with regards to their Facebook Live Sales videos, that the businesses can focus on or place less emphasis as they better understand themselves. 

Businesses should be more confident in generating genuine customer engagement while be consider limiting the following as well:
- the number of unique products they offer to prevent confusion among the viewers, 
- the number of comments made by the seller to prevent viewers from being turned-off,
- the frequency of the comments made by the seller to appear approachable yet respectful.

Additionally, the sellers should focus on increasing their positive & neutral sentiments, while taking feedback from their negative sentiments' discussion themes. 

# Recommendations
The businesses can consider the following recommendations:
- Be bold in verbally engaging with the customers
- Provide recipes on how to cook the products offered by the seller
- Suggest products & recipes in accordance to the viewers' mood
- Continually provide satisfactory products & services as feedback by the customers
- Mixing the order of the sale of their products
- Use promotional words both verbally and in the comments section
- Provide logical product codes