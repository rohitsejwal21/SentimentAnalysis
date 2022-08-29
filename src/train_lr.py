import config 
import re 
import string 

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer


def remove_stopwords(s):

    stop_words = list(stopwords.words('english'))
    word_list = s.split()

    for word in word_list:
        if word in stop_words:
            word_list.remove(word)
    
    return ' '.join(word_list)

def clean_text(df):
    
    df = df.str.lower()
    df = df.apply(lambda x: re.sub(r'[.|,|\/|_|:|;|~]', ' ', str(x)))
    df = df.apply(lambda x: re.sub(r'https:', ' ', str(x)))
    df = df.apply(lambda x: re.sub(r'http:', ' ', str(x)))
    df = df.apply(lambda x: re.sub(r'[#|@|$|%|^|&|(|)|!|"]', '', str(x)))
    df = df.apply(lambda x: re.sub(r'\`', '\'', str(x)))
    df = df.apply(lambda x: re.sub(r'[0-9]', '', str(x)))
    df = df.apply(lambda x: re.sub(r'\'re', ' are', str(x)))
    df = df.apply(lambda x: re.sub(r'\'s', ' is', str(x)))
    df = df.apply(lambda x: re.sub(r' com ', ' ', str(x)))
    df = df.apply(lambda x: re.sub(r'\'ve', ' have', str(x)))
    df = df.apply(lambda x: re.sub(r'don\'t', 'do not', str(x)))
    df = df.apply(lambda x: re.sub(r'doesn\'t', 'does not', str(x)))
    df = df.apply(lambda x: re.sub(r'won\'t', 'will not', str(x)))
    df = df.apply(lambda x: re.sub(r'wouldn\'t', 'would not', str(x)))
    df = df.apply(lambda x: re.sub(r'shouldn\'t', 'should not', str(x)))
    df = df.apply(lambda x: re.sub(r'\'m', ' am', str(x)))
    df = df.apply(lambda x: re.sub(' +', ' ', str(x)))
    #df = df.apply(lambda x: re.sub('', '', str(x)))

    return df 


def test_predictions(tfidf, model):

    test_df = pd.read_csv(config.TEST_FILE)
    
    # Clean
    test_df['tweet'] = clean_text(test_df['tweet'])
    test_df.loc[:, 'tweet'] = test_df['tweet'].apply(remove_stopwords)

    # Transform
    X_test = tfidf.transform(test_df['tweet'])

    # Predict
    preds_test = model.predict(X_test)

    # Submission
    test_submission = pd.DataFrame(
        list(zip(test_df['id'], preds_test)),
        columns=['id', 'label']
    )

    test_submission.to_csv('../input/test_submission.csv', index=False)

def run(fold, model):

    df = pd.read_csv(config.TRAINING_FILE_FOLDS)
    
    #print(df['tweet'].tail(20))
    df['tweet'] = clean_text(df['tweet'])
    #print(df['tweet'].tail(20))

    df.loc[:, 'tweet'] = df['tweet'].apply(remove_stopwords)

    train_df = df[df['kfold'] != fold].reset_index(drop=True)
    cv_df = df[df['kfold'] == fold].reset_index(drop=True)

    tfidf = TfidfVectorizer()
    tfidf.fit(train_df['tweet'])

    X_train = tfidf.transform(train_df['tweet'])
    X_cv = tfidf.transform(cv_df['tweet'])

    best_c = 0
    best_score = 0

    for c in [1, 10, 50, 100, 200, 500, 1000, 10000]:
        model = LogisticRegression(C=c)
        model.fit(X_train, train_df['label'])

        preds_cv = model.predict(X_cv)
        f1 = f1_score(cv_df['label'], preds_cv)
        if f1 > best_score:
            best_score = f1
            best_c = c
        print(f1)

    model = LogisticRegression(C=best_c)
    model.fit(X_train, train_df['label'])

    test_predictions(tfidf, model)

if __name__ == '__main__':

    for f in range(0, 1):
        run(fold=f, model='lr')