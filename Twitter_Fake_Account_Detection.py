
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, r2_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression  
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier


import tweepy
import pprint
from datetime import datetime, timezone



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
    
    # Real Account
    # user = api.get_user(screen_name='imVkohli')
    
    # # Fake Account
    # user = api.get_user(screen_name='greentexts_bot')
    
    # #My Account
    user = api.get_user(screen_name='SonuSood')
    
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
user_info['followers_friends_Ratio'] = (user.friends_count/user.followers_count)


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
data = pd.read_csv("Dataset/Final-Final.csv",encoding='latin1')



# Split the data into features (X) and target (y)
X = data.drop(["is_fake","name","screen_name","Account_Create_date","id"], axis=1)
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







# # Train the Random Forest Classifier
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
f1 = f1_score(y_test, y_pred, average='macro')
# report = classification_report(y_test, y_pred)
print("Random Forest Classifier\n\nConfusion Matrix:\n", cm,"\n")
print("Accuracy: {:.2f}%".format(accuracy * 100))
# print(f'Report: {report}')
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"F1 Score: {f1:.2f}","\n")


# Train and evaluate the Logistic Regression Classifier
classifierLR = LogisticRegression(random_state=0)
classifierLR.fit(X_train, y_train)
y_pred = classifierLR.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
sensitivity = recall_score(y_test, y_pred, average='macro')
specificity = cm[0,0] / (cm[0,0] + cm[0,1])
f1 = f1_score(y_test, y_pred, average='macro')
print("Logistic Regression Classifier\n\nConfusion Matrix:\n", cm)
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred)*100))
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"F1 Score: {f1:.2f}","\n")



# Make predictions on the testing data with a threshold of 0.3
y_pred_custom = classifierLR.predict(X_test, threshold=0.3)
cm = confusion_matrix(y_test, y_pred_custom)
precision = precision_score(y_test, y_pred_custom, average='macro')
sensitivity = recall_score(y_test, y_pred_custom, average='macro')
specificity = cm[0,0] / (cm[0,0] + cm[0,1])
f1 = f1_score(y_test, y_pred_custom, average='macro')
print("Logistic Regression Classifier\n\nConfusion Matrix:\n", cm)
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred_custom)*100))
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"F1 Score: {f1:.2f}","\n")

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
f1 = f1_score(y_test, y_pred, average='macro')
print("K-Nearest Neighbors Classifier\n\nConfusion Matrix:\n", cm)
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred)*100))
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"F1 Score: {f1:.2f}","\n")

# # Train and evaluate the Support Vector Machines Classifier

# classifierSVM = SVC(kernel='linear')
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
f1 = f1_score(y_test, y_pred, average='macro')
print("Gaussian Naive Bayes Classifier\n\nConfusion Matrix:\n", cm)
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred)*100))
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"F1 Score: {f1:.2f}","\n")


# # Train the Decision Tree Classifier
classifierDT = DecisionTreeClassifier(criterion="entropy", random_state=42)
classifierDT.fit(X_train, y_train)
y_pred = classifierDT.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
sensitivity = recall_score(y_test, y_pred, average='macro')
specificity = cm[0,0] / (cm[0,0] + cm[0,1])
f1 = f1_score(y_test, y_pred, average='macro')
print("Decision Tree Classifier\n\nConfusion Matrix:\n", cm)
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred)*100))
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"F1 Score: {f1:.2f}","\n")


# Train the AdaBoost classifier
adaboost = AdaBoostClassifier(n_estimators=100)
adaboost.fit(X_train, y_train)
y_pred = adaboost.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
sensitivity = recall_score(y_test, y_pred, average='macro')
specificity = cm[0,0] / (cm[0,0] + cm[0,1])
f1 = f1_score(y_test, y_pred, average='macro')
print("AdaBoost Classifier\n\nConfusion Matrix:\n", cm)
print("Accuracy: {:.2f}%".format(acc*100))
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"F1 Score: {f1:.2f}","\n")



# Train an XGBoost model
xgb = XGBClassifier(n_estimators=100)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)
precision = precision_score(y_test, y_pred, average='macro')
sensitivity = recall_score(y_test, y_pred, average='macro')
specificity = cm[0,0] / (cm[0,0] + cm[0,1])
f1 = f1_score(y_test, y_pred, average='macro')
print("XGBoost classifier\n\nConfusion Matrix:\n", cm)
print("Accuracy: {:.2f}%".format(xgb_acc*100))
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"F1 Score: {f1:.2f}","\n")


# Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_clf.fit(X_train, y_train)
y_pred = gb_clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
GB_acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
sensitivity = recall_score(y_test, y_pred, average='macro')
specificity = cm[0,0] / (cm[0,0] + cm[0,1])
f1 = f1_score(y_test, y_pred, average='macro')
print("Gradient Boosting classifier\n\nConfusion Matrix:\n", cm)
print("Accuracy: {:.2f}%".format(xgb_acc*100))
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"F1 Score: {f1:.2f}","\n")


# CatBoost Classifier
cat_clf = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=3, loss_function='Logloss', random_state=42)
cat_clf.fit(X_train, y_train)
y_pred = cat_clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
Cat_acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
sensitivity = recall_score(y_test, y_pred, average='macro')
specificity = cm[0,0] / (cm[0,0] + cm[0,1])
f1 = f1_score(y_test, y_pred, average='macro')
print("CatBoost classifier\n\nConfusion Matrix:\n", cm)
print("Accuracy: {:.2f}%".format(Cat_acc*100))
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"F1 Score: {f1:.2f}","\n")




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

# Function to predict from Gradient Boosting Classifier whether an input account is fake or real
def predict_fake_account_GB(classifier, account_features):
    prediction = gb_clf.predict([account_features])
    return "Fake" if prediction[0] else "Real"

# Function to predict from CatBoost Classifier whether an input account is fake or real
def predict_fake_account_CAT(classifier, account_features):
    prediction = cat_clf.predict([account_features])
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
# Account Details fetching from Twitter API
# input_account = [statuses_count,followers_count,friends_count,favourites_count,listed_count,profile_image,description,Age,Followers_Freinds,Verified,Protected,Retweets]

# # Fake Accounts Details
input_account = [61,13,582,0,0,1,0,5,0.046099291,1,0,26]

# input_account = [36,16,451,0,0,0,0,3,0.036099291,1,0,47]

# # Real Accounts Details
# input_account = [56,26,611,0,0,1,0,4,0.042553191,0,0,36]

print("Input Account is from Random Forest Classifier is :", predict_fake_account_RF(classifierRF, input_account),'\n')

print("Input Account is Logistic Regression Classifier is :", predict_fake_account_LR(classifierLR, input_account),'\n')

print("Input Account is K-Nearest Neighbors Classifier is :", predict_fake_account_KNN(classifierKNN, input_account),'\n')

# # print("Input Account is Support Vector Machines Classifier is :", predict_fake_account_SVM(classifierSVM, input_account),'\n')

print("Input Account is Gaussian Naive Bayes Classifier is :", predict_fake_account_G(classifierG, input_account),'\n')

print("Input Account is Decision Tree Classifier is :", predict_fake_account_DT(classifierDT, input_account),'\n')

print("Input Account is Ada Boost Classifier is :", predict_fake_account_AB(classifierDT, input_account),'\n')

# print("Input Account is XGB Classifier is :", predict_fake_account_XGB(classifierDT, input_account),'\n')

print("Input Account is Gradient Boosting Classifier is :", predict_fake_account_GB(classifierDT, input_account),'\n')

print("Input Account is CatBoost Classifier is :", predict_fake_account_CAT(classifierDT, input_account),'\n')


# Calculate the majority vote
vote = [predict_fake_account_RF(classifierDT, input_account), predict_fake_account_LR(classifierDT, input_account), predict_fake_account_KNN(classifierDT, input_account), predict_fake_account_G(classifierDT, input_account), predict_fake_account_DT(classifierDT, input_account), predict_fake_account_AB(classifierDT, input_account), predict_fake_account_GB(classifierDT, input_account), predict_fake_account_CAT(classifierDT, input_account)]

print(vote,'\n')

num_real = vote.count('Real')
num_fake = vote.count('Fake')
print('\n')

if num_real > num_fake:
    print('The final prediction for this account is : Real Account')
else:
    print('The final prediction for this account is : Fake Account')

print('\n')

# majority_vote = max(set(vote), key = vote.count)

# # Print the majority vote
# print("Majority Vote Result: ", majority_vote)


# # Define weights for each classifier
# weights = [0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1]

# # Calculate weighted vote
# weighted_vote = sum([weights[i]*vote[i] for i in range(len(vote))])

# # Print the weighted vote
# print("Weighted Vote Result: ", weighted_vote)

# # Define weights for each classifier
# weights = [0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1]

# # Get predicted class labels for each classifier
# labels = [predict_fake_account_RF, predict_fake_account_LR, predict_fake_account_KNN, predict_fake_account_G, predict_fake_account_DT, predict_fake_account_AB, predict_fake_account_GB, predict_fake_account_CAT]
# # Calculate the frequency of each label


# # Calculate the frequency of each label
# label_counts = {}
# for label in labels:
#     if label not in label_counts:
#         label_counts[label] = 0
#     label_counts[label] += 1

# # Get the label with the highest frequency or highest weighted vote in case of a tie
# max_count = max(label_counts.values())
# predicted_labels = [label for label, count in label_counts.items() if count == max_count]
# if len(predicted_labels) == 1:
#     predicted_label = predicted_labels[0]
# else:
#     predicted_label = max(predicted_labels, key=lambda label: labels.index(label))
#     predicted_label = predicted_label()  # Call the function to get the numerical value

# # Print the predicted label
# print("Ensemble Prediction: ", predicted_label)
