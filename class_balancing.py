# example of oversampling a multi-class classification dataset
from pandas import read_csv
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, classification_report, f1_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# define the dataset location
url = 'C:/Users/ASIM/Desktop/RESEARCH/POS/earthquake_POS.csv'
# load the csv file as a data frame
df = read_csv(url, header=None)
print(df)
data = df.values
print(data)
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# label encode the target variable
print(y)
y = LabelEncoder().fit_transform(y)

# transform the dataset
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
# summarize distribution
counter = Counter(y)
for k, v in counter.items():
    per = v / len(y) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# plot the distribution
pyplot.bar(counter.keys(), counter.values())
pyplot.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

start = time.time()
classifier = RandomForestClassifier(n_estimators=42, criterion='entropy')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cv = ShuffleSplit(n_splits=5, test_size=0.3)
scores = cross_val_score(classifier, X, y, cv=10)
target_names = ['direct-eyewitness', 'dont know', 'non-eyewitness']
print(classification_report(y_test, y_pred, target_names=target_names))
print("Random Forest accuracy after 10 fold CV: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2) + ", " + str(
    round(time.time() - start, 3)) + "s")
print("******************************")
print("******************************")
print("******************************")

# print ('   Accuracy:', accuracy_score(y_test, y_pred))
print('scores.mean:', scores.mean())
accuracy = scores.mean()
print("______________________________")
print('Precision:', precision_score(y_test,
                                    y_pred, average='weighted'))
precision = precision_score(y_test, y_pred, average='weighted')
# print ('Precision:', precision_score(y_test, y_pred))
print("______________________________")
print('Recall:', recall_score(y_test, y_pred, average='weighted'))
recall = recall_score(y_test, y_pred, average='weighted')
# print ('Recall:', recall_score(y_test, y_pred))
print("______________________________")
print('F1 score:', f1_score(y_test, y_pred, average='weighted'))
f1score = f1_score(y_test, y_pred, average='weighted')
# print ('F1 score:', f1_score(y_test, y_pred))
print("______________________________")

print("******************************************************************************************")

# url = 'C:/Users/ASIM/Desktop/RESEARCH/POS/earthquake_POS.csv'
