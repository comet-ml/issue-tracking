from __future__ import print_function

from comet_ml import Experiment

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score


experiment  = Experiment(api_key="Jon-Snow", project_name='my project')

# Get dataset and put into train,test lists
categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)

experiment.log_dataset_hash(twenty_train)
# Build training pipeline

text_clf = Pipeline([('vect', CountVectorizer()), # Counts occurrences of each word
                     ('tfidf', TfidfTransformer()), # Normalize the counts based on document length
                     ('clf', SGDClassifier(loss='hinge', penalty='l2', # Call classifier with vector
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None)),
                     ])

text_clf.fit(twenty_train.data,twenty_train.target)
#
# Predict unseen test data based on fitted classifer
predicted = text_clf.predict(twenty_test.data)

# Compute accuracy
print(accuracy_score(twenty_test.target, predicted))

