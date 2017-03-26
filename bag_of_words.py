import pandas as pd
import numpy as np

df = pd.read_csv('./movie_data.csv', encoding = "ISO-8859-1") #http://stackoverflow.com/questions/18171739/unicodedecodeerror-when-reading-csv-file-in-pandas-with-python
df.head(3)

# convert words into vectors
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
docs = np.array([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining and the weather is sweet'])

bag = count.fit_transform(docs)
print(bag)
print(count.vocabulary_)
print(bag.toarray())

# Term Frequency Inverse Document Frequency
from sklearn.feature_extraction.text import TfidfTransformer
tfidF = TfidfTransformer()
np.set_printoptions(precision=2)
print(tfidF.fit_transform(count.fit_transform(docs)).toarray())

import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+',' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

print(preprocessor(df.loc[0, 'review'][-50:]))
print(preprocessor('</a>This :) is :(a test :-)!'))

# 토큰화
def tokenizer(text):
    return text.split()

print(tokenizer('runners like running and thus they run'))

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

print(tokenizer_porter('runners like running and thus they run'))

# 불용어 제거
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop = stopwords.words('english')
print([w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]) # [-10:]?

X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

if __name__ == '__main__':
    from sklearn.grid_search import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
    param_grid = [{'vect__ngram_range': [(1, 1)],
    'vect__stop_words': [stop, None],
    'vect__tokenizer': [tokenizer, tokenizer_porter],
    'clf__penalty': ['l1', 'l2'],
    'clf__C': [1.0, 10.0, 100.0]},
    {'vect__ngram_range': [(1, 1)],
    'vect__stop_words': [stop, None],
    'vect__tokenizer': [tokenizer, tokenizer_porter],
    'vect__use_idf':[False],
    'vect__norm':[None],
    'clf__penalty': ['l1', 'l2'],
    'clf__C': [1.0, 10.0, 100.0]},
    ]

    lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=0))])
    gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv =5, verbose=1, n_jobs=-1)
    gs_lr_tfidf.fit(X_train, y_train)

    print('Best parameter set: %s'% gs_lr_tfidf.best_params_)
    print('CV Accuracy: %.3f'%gs_lr_tfidf.best_score_)
    clf = gs_lr_tfidf.best_estimator_
    print('Test Accuracy: %.3f'%clf.score(X_test, y_test))

