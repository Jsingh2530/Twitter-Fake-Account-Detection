
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, r2_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier


import tweepy
import pprint
from datetime import datetime, timezone


# Replace with your own API keys
consumer_key = "l1cd0VmH2fxLHgjgrVQGt39bu"
consumer_secret = "Bmtty1FXChzlNEtbBfdMj6BjZwt4ApLoDdMdIZCPxJDusxn0q0"
access_token = "1269668956555657217-UAszZGK3S1Gvw1kYUFQ59S9VYRxPFS"
access_token_secret = "yYbKgDsm1dzWAn1AbDcyuDTGZsDwS8PgCGmr1ngKMr4Mi"

# # Replace with your own API keys
# consumer_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# consumer_secret = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# access_token = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# access_token_secret = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

# Authenticate with Twitter's API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

print("API is Working ...")

api = tweepy.API(auth)
userID_list = [""]

for user in userID_list:
    
    #Real Account
    # user = api.get_user(screen_name='SSadagopan')
    
    #Fake Account
    # user = api.get_user(screen_name='Shyaimsek')
    
    #My Account
    user = api.get_user(screen_name='JITENDR83104348')
    
    user_info = {
            "Username": [user.name],
            "Followers": [user.followers_count],
            "Friends": [user.friends_count],
            "Tweets": [user.statuses_count],
            "Verified": [1 if user.description else 0],
            "Profile Image": [1 if user.description else 0],
            "Likes": [user.favourites_count],
            "Comments": [user.statuses_count],
            "Location": [user.location],
            "Description": [1 if user.description else 0],
            "URL": [1 if user.description else 0],
            "Status_Count": [user.statuses_count],
            "ID": [user.id],
            "Favourite_Count": [user.favourites_count],
            "Listed_Count": [user.listed_count],
            "Created_Date": [user.created_at.strftime("%d %m %Y")]          
    }
    
account_creation_date = user.created_at
current_date = datetime.now(timezone.utc)
account_age_seconds = (current_date - account_creation_date).total_seconds()
account_age_years = account_age_seconds / 60 / 60 / 24 / 365

user_info['Age'] = round(account_age_years)
user_info['followers_friends_Ratio'] = (user.followers_count/user.friends_count)


# # Get list of tweets from user's timeline
# tweets = tweepy.Cursor(api.user_timeline, screen_name="JITENDR83104348").items()

# # Loop through tweets and get total number of retweets
# total_retweets = 0
# for tweet in tweets:
#     total_retweets += tweet.retweet_count

# print("Total retweets for user:", total_retweets)

pprint.pprint(user_info,)
print("\n")

# Load the dataset
data = pd.read_csv("Dataset/3000datamore.csv",encoding='latin1')

# Split the data into features (X) and target (y)
X = data.drop(["is_fake","name","screen_name","Account_Create_date"], axis=1)
y = data["is_fake"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# # preprocess the data using TfidfVectorizer
# vectorizer = TfidfVectorizer(stop_words='english')
# X_train = vectorizer.fit_transform(X_train['statuses_count'].values.astype('U'))
# X_test = vectorizer.transform(X_test['statuses_count'].values.astype('U'))
# # define the base classifiers
# rf = RandomForestClassifier(n_estimators=200, random_state=42)
# gb = GradientBoostingClassifier(n_estimators=200, random_state=42)
# svm = SVC(kernel='linear', probability=True, random_state=42)
# # define the stacking classifier
# estimators = [('rf', rf), ('gb', gb)]
# stacking = StackingClassifier(estimators=estimators, final_estimator=svm)
# # train the model
# stacking.fit(X_train, y_train)
# # make predictions on the test set
# y_pred = stacking.predict(X_test)
# # evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)
# print("Stacking Classifier\n")
# print("Accuracy: {:.2f}%".format(accuracy * 100),'\n')







# Train the Random Forest Classifier
classifierRF = RandomForestClassifier(n_estimators=100, random_state=42)
classifierRF.fit(X_train, y_train)
# Make predictions on the test set
y_pred = classifierRF.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
sensitivity = recall_score(y_test, y_pred, average='macro')
specificity = cm[0,0] / (cm[0,0] + cm[0,1])
# report = classification_report(y_test, y_pred)
print("Random Forest Classifier\n\nConfusion Matrix:\n", cm,"\n")
print("Accuracy: {:.2f}%".format(accuracy * 100))
# print(f'Report: {report}')
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}","\n")


# Train and evaluate the Logistic Regression Classifier
classifierLR = LogisticRegression(random_state=0)
classifierLR.fit(X_train, y_train)
y_pred = classifierLR.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
sensitivity = recall_score(y_test, y_pred, average='macro')
specificity = cm[0,0] / (cm[0,0] + cm[0,1])
print("Logistic Regression Classifier\n\nConfusion Matrix:\n", cm)
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred)*100))
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}","\n")


# # Train and evaluate the Linear Regression Classifier

# classifierLinear = LinearRegression()
# classifierLinear.fit(X_train,y_train)
# y_pred = classifierLinear.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)
# print("Linear Regression Classifier\nConfusion Matrix:\n", cm)
# print("Accuracy: {:.2f}%".format(accuracy_score(r2_score(y_test, y_pred))*100), '\n')


# Train and evaluate the K-Nearest Neighbors Classifier

classifierKNN = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifierKNN.fit(X_train, y_train)
y_pred = classifierKNN.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
sensitivity = recall_score(y_test, y_pred, average='macro')
specificity = cm[0,0] / (cm[0,0] + cm[0,1])
print("K-Nearest Neighbors Classifier\n\nConfusion Matrix:\n", cm)
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred)*100))
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}","\n")

# # Train and evaluate the Support Vector Machines Classifier

# classifierSVM = SVC(kernel='linear', random_state=0)
# classifierSVM.fit(X_train, y_train)
# y_pred = classifierSVM.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)
# print("Support Vector Machines Classifier\nConfusion Matrix:\n", cm)
# print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred)*100), '\n')
                      
                      
# Train and evaluate the Gaussian Naive Bayes Classifier
classifierG = GaussianNB()
classifierG.fit(X_train, y_train)
y_pred = classifierG.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
sensitivity = recall_score(y_test, y_pred, average='macro')
specificity = cm[0,0] / (cm[0,0] + cm[0,1])
print("Gaussian Naive Bayes Classifier\n\nConfusion Matrix:\n", cm)
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred)*100))
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}","\n")


# Train the Decision Tree Classifier
classifierDT = DecisionTreeClassifier(criterion="entropy", random_state=42)
classifierDT.fit(X_train, y_train)
y_pred = classifierDT.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
sensitivity = recall_score(y_test, y_pred, average='macro')
specificity = cm[0,0] / (cm[0,0] + cm[0,1])
print("Decision Tree Classifier\n\nConfusion Matrix:\n", cm)
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred)*100))
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}","\n")


# Train the AdaBoost classifier
adaboost = AdaBoostClassifier(n_estimators=100)
adaboost.fit(X_train, y_train)
y_pred = adaboost.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
sensitivity = recall_score(y_test, y_pred, average='macro')
specificity = cm[0,0] / (cm[0,0] + cm[0,1])
print("AdaBoost Classifier\n\nConfusion Matrix:\n", cm)
print("Accuracy: {:.2f}%".format(acc*100))
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}","\n")



# Train an XGBoost model
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)
precision = precision_score(y_test, y_pred, average='macro')
sensitivity = recall_score(y_test, y_pred, average='macro')
specificity = cm[0,0] / (cm[0,0] + cm[0,1])
print("XGBoost classifier\n\nConfusion Matrix:\n", cm)
print("Accuracy: {:.2f}%".format(xgb_acc*100))
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}","\n")


# Function to predict from Random Forest Classifier whether an input account is fake or real
def predict_fake_account_RF(classifier, account_features):
    prediction = classifierRF.predict([account_features])
    return "Fake" if prediction[0] else "Real"

# Function to predict from Logistic Regression Classifier whether an input account is fake or real
def predict_fake_account_LR(classifier, account_features):
    prediction = classifierLR.predict([account_features])
    return "Fake" if prediction[0] else "Real"

# Function to predict from K-Nearest Neighbors Classifier whether an input account is fake or real
def predict_fake_account_KNN(classifier, account_features):
    prediction = classifierKNN.predict([account_features])
    return "Fake" if prediction[0] else "Real"

# # Function to predict from Support Vector Machines Classifier whether an input account is fake or real
# def predict_fake_account_SVM(classifier, account_features):
#     prediction = classifierSVM.predict([account_features])
#     return "Fake" if prediction[0] else "Real"

# Function to predict from Gaussian Naive Bayes Classifier whether an input account is fake or real
def predict_fake_account_G(classifier, account_features):
    prediction = classifierG.predict([account_features])
    return "Fake" if prediction[0] else "Real"

# Function to predict from Decision Tree Classifier whether an input account is fake or real
def predict_fake_account_DT(classifier, account_features):
    prediction = classifierDT.predict([account_features])
    return "Fake" if prediction[0] else "Real"

# Function to predict from Ada Boost Classifier whether an input account is fake or real
def predict_fake_account_AB(classifier, account_features):
    prediction = adaboost.predict([account_features])
    return "Fake" if prediction[0] else "Real"

# Function to predict from XG Boost Classifier whether an input account is fake or real
def predict_fake_account_XGB(classifier, account_features):
    prediction = xgb.predict([account_features])
    return "Fake" if prediction[0] else "Real"


# Example usage

id = user.id
statuses_count = user.statuses_count
followers_count = user.followers_count
friends_count = user.friends_count
favourites_count = user.favourites_count
listed_count = user.listed_count
profile_image = 1 if user.description else 0
description = 1 if user.description else 0
Age = round(account_age_years)
Followers_Freinds = (user.followers_count/user.friends_count)
Verified = 1 if user.description else 0
Protected = 1 if user.protected else 0
Retweets = user.statuses_count

input_account = [id,statuses_count,followers_count,friends_count,favourites_count,listed_count,profile_image,description,Age,Followers_Freinds,Verified,Protected,Retweets]
# input_account = [45255362,1351,35,501,0,1,1,1,8,0.1521452,0,1,78]

print("Input Account is from Random Forest Classifier is :", predict_fake_account_RF(classifierRF, input_account),'\n')

print("Input Account is Logistic Regression Classifier is :", predict_fake_account_LR(classifierLR, input_account),'\n')

print("Input Account is K-Nearest Neighbors Classifier is :", predict_fake_account_KNN(classifierKNN, input_account),'\n')

# print("Input Account is Support Vector Machines Classifier is :", predict_fake_account_SVM(classifierSVM, input_account),'\n')

print("Input Account is Gaussian Naive Bayes Classifier is :", predict_fake_account_G(classifierG, input_account),'\n')

print("Input Account is Decision Tree Classifier is :", predict_fake_account_DT(classifierDT, input_account),'\n')

print("Input Account is Ada Boost Classifier is :", predict_fake_account_AB(classifierDT, input_account),'\n')

# print("Input Account is XGB Classifier is :", predict_fake_account_XGB(classifierDT, input_account),'\n')


