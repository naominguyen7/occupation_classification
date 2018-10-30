import pandas as pd, numpy as np
from csv import DictWriter

from scipy.sparse import hstack
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

mydata = pd.read_csv('log_data.csv')
mydata = mydata.fillna('')
num_vars = ['favourites_count','followers_count','friends_count','listed_count','statuses_count',
                'avg_fav_cnt', 'avg_rt_cnt','max_fav_cnt','max_rt_cnt']



def tfidf_cat(X_train, X_test, y_train, var_name):
    categories = ['Politician', 'Trader', 'Journalist']
    text_cat = [' '.join(X_train[var_name][y_train == cat]) for cat in categories]
    tfidf = TfidfVectorizer(min_df=1, sublinear_tf=True, stop_words='english')
    score_cat = tfidf.fit_transform(text_cat).toarray()
    tfidf_train = tfidf.transform(X_train[var_name])
    tfidf_test = tfidf.transform(X_test[var_name])
    return {'tfidf': tfidf,
            'score_cat': score_cat,
            'tfidf_train': tfidf_train,
            'tfidf_test': tfidf_test}


def cross_validation(vocab_size, models, data, test_size, random_seed):
    result = []
    num_vars = ['favourites_count', 'followers_count', 'friends_count', 'listed_count', 'statuses_count',
                'avg_fav_cnt', 'avg_rt_cnt', 'max_fav_cnt', 'max_rt_cnt']
    X_train, X_test, y_train, y_test = train_test_split(data[num_vars + ['tweets', 'description']],
                                                        data.Category, test_size=test_size, random_state=random_seed,
                                                        stratify=data.Category)
    tfidf_tweets = tfidf_cat(X_train, X_test, y_train, 'tweets')
    tfidf_description = tfidf_cat(X_train,X_test, y_train, 'description')

    twscore_dict = {'Politician': tfidf_tweets['score_cat'][0],
                    'Trader': tfidf_tweets['score_cat'][1],
                    'Journalist': tfidf_tweets['score_cat'][2], }

    bioscore_dict = {'Journalist': tfidf_description['score_cat'][0],
                     'Politician': tfidf_description['score_cat'][1],
                     'Trader': tfidf_description['score_cat'][2]}

    for k in ['Politician', 'Trader', 'Journalist']:
        sorted_twind = np.argsort(-twscore_dict[k])
        remove_tw_ind = sorted_twind[int(vocab_size * len(sorted_twind)):]

        sorted_bioind = np.argsort(-bioscore_dict[k])
        remove_bio_ind = sorted_bioind[int(vocab_size * len(sorted_bioind)):]


        for model in models:
            if model != 'NB':
                train_tw = np.delete(tfidf_tweets['tfidf_train'].toarray(), remove_tw_ind, axis=1)
                train_bio = np.delete(tfidf_description['tfidf_train'].toarray(), remove_bio_ind, axis=1)

                test_tw = np.delete(tfidf_tweets['tfidf_test'].toarray(), remove_tw_ind, axis=1)
                test_bio = np.delete(tfidf_description['tfidf_test'].toarray(), remove_bio_ind, axis=1)

            else:
                terms = np.array(tfidf_description['tfidf'].get_feature_names())
                terms = np.delete(terms, remove_bio_ind)
                vec = CountVectorizer(vocabulary=terms)
                train_bio = vec.transform(X_train.description)
                test_bio = vec.transform(X_test.description)

                terms = np.array(tfidf_tweets['tfidf'].get_feature_names())
                terms = np.delete(terms, remove_tw_ind)
                vec = CountVectorizer(vocabulary=terms)
                train_tw = vec.transform(X_train.tweets)
                test_tw = vec.transform(X_test.tweets)

            X_train_dtm = hstack((X_train[num_vars], train_bio, train_tw)).toarray()
            X_test_dtm = hstack((X_test[num_vars], test_bio, test_tw)).toarray()

            clf = models[model]
            clf.fit(X_train_dtm, y_train == k)
            result.append({'model': model,
                           'category': k,
                           'vocab_size': vocab_size,
                           'test_score': clf.score(X_test_dtm, y_test==k)})
    return result

models = dict()
models['LG'] = LogisticRegression()
models['NB'] = MultinomialNB(alpha=1)
models['SVM'] = SVC(kernel='linear')
models['RF'] = RandomForestClassifier(n_estimators=200, max_depth=20)


if __name__ == '__main__':
    # write phase 1 summary
    with open(r'summary.csv', 'a', newline='') as f:
        writer = DictWriter(f, fieldnames=['model', 'category', 'vocab_size', 'test_score'])
        writer.writeheader()
        for i in range(30):
            summary = cross_validation(0.2, models, mydata, 0.2, i)
            writer.writerows(summary)
            summary = cross_validation(0.3, models, mydata, 0.2, i)
            writer.writerows(summary)
            print(i)




