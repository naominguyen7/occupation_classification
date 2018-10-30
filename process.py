import pandas as pd
from numpy import log1p
import re
from nltk.stem import SnowballStemmer

patURL = re.compile(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
patMENTION = re.compile(r'@\w*')
regex = re.compile('[^a-z ]')
stemmer = SnowballStemmer("english")

def preprocess_text(text):
    text = patURL.sub(' ', text)
    text = patMENTION.sub(' ',text)
    text = text.lower()
    text = regex.sub('', text)
    text = ' '.join(stemmer.stem(x) for x in text.split())
    return text


def process_df(df):
    num_vars = ['listed_count', 'statuses_count',
                'favourites_count', 'followers_count', 'friends_count'
                'avg_rt_cnt', 'max_rt_cnt', 'avg_fav_cnt', 'max_fav_cnt']
    df = df[(~df.followers_count.isnull())|(~df.friends_count.isnull())]
    df[num_vars] = log1p(df[num_vars]).fillna(0)
    df['tweets'] = df.tweets.fillna('').apply(preprocess_text).fillna('')
    df['description'] = df.description.fillna('').apply(preprocess_text).fillna('')
    return df

if __name__ == '__main__':
    data = pd.read_csv('raw_data.csv')
    data = process_df(data)
    data.to_csv('log_data.csv', index=False)
    print(data.shape)


