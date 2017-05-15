#FRAUD DETECTION CASE STUDY  

INTRODUCTION

This case study was focused around predicting whether or not a given event is fraud as new events are created by users. The events were music festivals and the fraud occurred when a user sold tickets to a fake event. Here is an example of the information provided in the data:


{ "org_name": "DREAM Project Foundation", "name_length": 51, "event_end": 1363928400, "venue_latitude": 42.9630578, "event_published": 1361978554.0, "user_type": 1, "channels": 11, "currency": "USD", "org_desc": "", "event_created": 1361291193, "event_start": 1363914000, "has_logo": 1, "email_domain": "dreamprojectfoundation.org", "user_created": 1361290985, "payee_name": "", "payout_type": "ACH", "venue_name": "Grand Rapids Brewing Co", "sale_duration2": 30, "venue_address": "1 Ionia Avenue Southwest", "approx_payout_date": 1364360400, "org_twitter": 13.0, "gts": 537.4, "listed": "y", "ticket_types": [{"event_id": 5558108, "cost": 50.0, "availability": 1, "quantity_total": 125, "quantity_sold": 10}], "org_facebook": 13.0, "num_order": 7, "user_age": 0, "body_length": 1474, "description": Come enjoy a night of music and beer tasting at the new Grand Rapids Brewery while we make an effort to create awareness and raise funds for Dream Project Foundation. The night will include music, Grand Rapids Brewery's finest beer to sample, heavy hors d'oeuvre's and silent auction of artwork directly from the young artists of Dream House.

\r\n


\r\n
Who We Are:

\r\n
DREAM Project Foundation is a small American 501c3 registered non-profit organization, working to break the cycle of human trafficking through community development. As a small, grass roots organization, we focus primarily on prevention and protection which begins with shelter and continues with education, so those vulnerable are aware of the dangers and able to protect themselves.

\r\n
DREAM Project Foundation was officially founded in 2011 to support the DREAM House children's home based in Thailand on the border of Myanar (Burma). While helping children stay safe from the trafficing is the heart of our mission, we know that in order to end trafficking it must be a collaborative effort for all people and communities.

\r\n
We at DREAM Project Foundation are determined to fight against this atrocity, focusing on the factors that cause people to be vulnerable targets to traffickers, with most of our work based in SE Asia as it is a major international hub of human trafficking.

", "object_id": 5558108, "venue_longitude": -85.6706147, "venue_country": "US", "previous_payouts": [{"name": "", "created": "2013-04-19 03:25:42", "country": "US", "state": "", "amount": 500.0, "address": "", "zip_code": "", "event": 5558108, "uid": 52778636}], "sale_duration": 22.0, "num_payouts": 0, "name": "DREAM Project Foundation - Taste of a Better Future", "country": "US", "delivery_method": 0.0, "has_analytics": 0, "fb_published": 0, "venue_state": "MI", "has_header": null, "show_map": 1 }}


With four collaborators we created a pipeline to predict the likelihood of fraud for a new event.

CLEANING THE DATA

Our first objective was to determine what classified fraud as fraud. The account type column provided this information. We decided to classify fraud as any event that had an account type of either ['fraudster_event', 'fraudster', 'fraudster_att']. Once classifying fraud and popping the binary column into an array for the y-portion of the model we had to created the training data. One of the first objectives we had was to remove columns with no signal and columns that could potentially contain leakage. Although it is difficult to infer how/if a column contains signal without actually testing it against a base model, in the interest of time we decided to be cautious and only drop seemingly obvious columns. Before dropping columns we converting categorical columns to numeric (['listed', 'ticket_types']), summing relevant columns (['total_quantity','num_previous_payouts'] and replacing missing values with NAN. In addition, we made 'has_header' into dummy variables. Now that the relevant columns were ready to be put into a model, we decided on the columns to drop. For Logistic Regression these columns ended up being  ['approx_payout_date', 'country', 'currency', 'description', 'email_domain', 'event_created', 'event_end', 'event_published', \
                  'event_start', 'fb_published', 'gts', 'has_analytics', 'has_logo', 'name', 'object_id', 'org_desc', 'org_name', \
                  'payee_name', 'payout_type', 'previous_payouts', 'sale_duration', 'sale_duration2', 'show_map', 'ticket_types', \
                  'user_created', 'user_type', 'venue_address', 'venue_country', 'venue_latitude', 'venue_longitude', 'venue_name', 'venue_state', 'has_header']
For our other models (SVM, Random Forests, AGABoost) we didnt need to remove one of each of the dummy variables. These dropped columns ended up being ['acct_type', 'approx_payout_date', 'country', 'currency', 'description', 'email_domain', 'event_created', 'event_end', 'event_published', \
                  'has_header', 'event_start', 'fb_published', 'gts', 'has_analytics', 'has_logo', 'name', 'object_id', 'org_desc', 'org_name', \
                  'payee_name', 'payout_type', 'previous_payouts', 'sale_duration', 'sale_duration2', 'show_map', 'ticket_types', \
                  'user_created', 'user_type', 'venue_address', 'venue_country', 'venue_latitude', 'venue_longitude', 'venue_name', 'venue_state']
However, given more time I would have liked to work with the location column, as there was probably hidden signal.

MODELS

Our baseline Logistic Regression Model had an accuracy and F1 score in the low seventies and high sixties, respectively. This was with very little feature engineering, other than converting categorical columns to numeric representations. Our baseline GradientBoosted model and SVM models had  F1 scores significantly higher than the Logistic Regression, in the high eighties for each. Once transforming our dataframe into the form mentioned above, we improved our F1 scores for SVM:RBF and Gradient Boosted models to the mid nineties. Eventually we decided to use the Gradient Boosting Classifier as our final model.

CONCLUSION

Our final Gradient Boosted model had an accuracy of ~97% and an F1 score of ~95%. To put our model into practice we pickled it and incorporated it into an app hosted online with Flask. The app was designed to receive new event data points in real time and make a prediction on whether they are fraud or not. In conclusion, our model had strong predictive power. In addition, the use of a gradient boosted model provided better interpretability in relation to the features that contained the most signal. 
