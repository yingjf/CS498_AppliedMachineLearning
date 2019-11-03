import numpy as np
from numpy import genfromtxt
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def updateThreshold(logisticRegr, x_train, y_train, x_test, y_test, threshold):
    # Get the train data score
    prob_positive_train = logisticRegr.predict_proba(x_train)
    prediction_train = np.full((1, prob_positive_train.shape[0]), 1)[0]
    prediction_train[list(np.where(prob_positive_train[:, 1] >= threshold)[0])] = 5
    score_updated_train = accuracy_score(prediction_train, y_train)

    # Get the test data score
    prob_test = logisticRegr.predict_proba(x_test)
    prob_positive_test = prob_test[:, 1]
    prediction_test = np.full((1, prob_positive_test.shape[0]), 1)[0]
    prediction_test[list(np.where(prob_positive_test >= threshold)[0])] = 5
    score_updated_test = accuracy_score(prediction_test, y_test)

    print("Get the test score {} with threshold {}.".format(score_updated_test, threshold))
    return prediction_train

# ======================== Part 1 ==============================================================
data_path = 'yelp_2k.csv'
with open(data_path, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    # get header from first row
    headers = next(reader)
    # get all the rows as a list
    data = list(reader)
    # transform data into numpy array
    data = np.array(data).astype('str')

x_data = data[:, 5]
y_data = data[:, 3].astype(np.int)
x_data = np.array([x.lower() if isinstance(x, str) else x for x in x_data]) # to Lower case
# Represent with bag of words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(x_data).toarray()

# Get the word count vs word rank
wordCount = np.sum(X, axis=0)
wordCountSort = np.sort(wordCount)[::-1]
plt.plot(wordCountSort)
plt.ylabel('Word Count')
plt.xlabel('Word Rank')
plt.title('Word Frequency')
plt.grid(True)
plt.show()

# Get the stop words
sortIndex = np.argsort(wordCount)[::-1].tolist()
features = vectorizer.get_feature_names()
featuresSort = [features[i] for i in sortIndex]
print(featuresSort)

print(X)
print(vectorizer.get_feature_names())

# Get document frequency
df = np.sum(X, axis=0)
df_index = np.argsort(df)[::-1].tolist()
df_sort = [df[i] for i in df_index]
features_df_Sort = [features[i] for i in df_index]
np.savetxt("document_frequency.csv", np.array([features_df_Sort, df_sort]).T, delimiter=",", fmt="%s")

# By looking at the list of the document frequency,
# determine that the max occurrence is 787 (which contains "service")
# The min occurrence is 2
maxdf = float(787) / X.shape[0]
mindf = 1.0 / X.shape[0]
# Represent with bag of words using max_df and min_df
vectorizer = CountVectorizer(max_df=maxdf, min_df=mindf)
X = vectorizer.fit_transform(x_data).toarray()
# Get the word count vs word rank
wordCount = np.sum(X, axis=0)
wordCountSort = np.sort(wordCount)[::-1]
plt.plot(wordCountSort)
plt.ylabel('Word Count')
plt.xlabel('Word Rank')
plt.title('Word Frequency')
plt.grid(True)
plt.show()

# ======================== Part 2 ===========================================================
# # nbrs = NearestNeighbors(n_neighbors=5, metric='cosine').fit(X)
# # X_test = vectorizer.transform(['Horrible customer service'])
# # res = nbrs.kneighbors(X_test, return_distance=False)
# # print(x_data[res[0]])
# # np.savetxt("horrible_customer_service_reviews.txt", x_data[res[0]], fmt="%s", delimiter='\t')
#
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X)
# neigh = NearestNeighbors(n_neighbors=5, n_jobs=-1)
# neigh.fit(X_train_tfidf)
# X_test_counts = vectorizer.transform(['Horrible customer service'])
# X_test_tfidf = tfidf_transformer.transform(X_test_counts)
# res = neigh.kneighbors(X_test_tfidf)
# # print(x_data[res[0]])
# # np.savetxt("horrible_customer_service_reviews_Tfidf.txt", x_data[res[0]], fmt="%s", delimiter='\t')
#
# # Find all the distance scores
# neigh = NearestNeighbors(n_neighbors=X.shape[0], n_jobs=-1)
# neigh.fit(X_train_tfidf)
# res_all = neigh.kneighbors(X_test_tfidf)
# reviews = np.append(res_all[0][0], x_data[res_all[1][0]], axis=1)
# # np.savetxt("distance_score.csv", res_all[0][0], fmt="%s")
# # By looking at the distance, 34 documents are matching


# ======================== Part 3 ===========================================================
x_train, x_test, y_train, y_test = train_test_split(X, y_data, test_size=0.1, random_state=0)
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
score_train = logisticRegr.score(x_train, y_train)
score_test = logisticRegr.score(x_test, y_test)

prob_train = logisticRegr.predict_proba(x_train)
prob_positive_train = prob_train[:, 1]
index_pos = np.where(y_train==5)
index_neg = np.where(y_train==1)

plt.hist(prob_positive_train[index_pos], np.arange(0.0, 1.1, 0.01), color='green', label='pos')
plt.hist(prob_positive_train[index_neg], np.arange(0.0, 1.1, 0.01), color='blue', label='neg')
plt.ylabel('Count of predictions in bucket')
plt.xlabel('Predicted Score')
plt.title('Histogram of predicted scores')
plt.xticks(np.arange(0, 1.1, 0.1))
plt.legend()


# By looking at the histogram, set threshold as 0.7
threshold_list = [0.6, 0.7, 0.8]
for threshold in threshold_list:
    updateThreshold(logisticRegr, x_train, y_train, x_test, y_test, threshold)

# Plot ROC curve
prob_test = logisticRegr.predict_proba(x_test)
prob_positive_test = prob_test[:, 1]
y_predict = np.full((1, prob_positive_test.shape[0]), 1)[0]
y_predict[list(np.where(prob_positive_test >= 0.5)[0])] = 5
fpr, tpr, _ = roc_curve(y_predict, y_test, pos_label=5)
roc_auc = auc(fpr, tpr)
lw = 2
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()

# FPR is 0.07142


my_data = genfromtxt('yelp_2k.csv', delimiter=",")
x = 1
