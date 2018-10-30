import time
import twitter
import pandas as pd
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('consumer_key')
parser.add_argument('consumer_secret')
parser.add_argument('access_token_key')
parser.add_argument('-access_token_secret')
twitter_keys = parser.parse_args()



# Set up Twitter Authorization
api = twitter.Api(
    consumer_key         =   twitter_keys.consumer_key,
    consumer_secret      =   twitter_keys.consumer_secret,
    access_token_key     =   twitter_keys.access_token_key,
    access_token_secret  =   twitter_keys.access_token_secret,
    tweet_mode = 'extended'
)


bio_cols = ['description', 'listed_count', 'statuses_count',
            'favourites_count', 'followers_count', 'friends_count']
tweet_cols = ['avg_rt_cnt', 'max_rt_cnt','avg_fav_cnt', 'max_fav_cnt', 'tweets']
result_limit = 200
mine_rts = False

def get_variables(account):
    user_id = account['tid']
    tweets = []

    fav_cnt = 0
    rt_cnt = 0
    max_fav_cnt = 0
    max_rt_cnt = 0

    last_tweet_id = False

    try:
        with open('backup' + str(user_id) + '.txt', 'r') as f:
            account = json.load(dict(account), f)
        return account
    except:
        pass
    try:
        user = api.GetUser(user_id=user_id, return_json=True)
        for k in bio_cols:
            account[k] = user[k] if k in user else None
    # If no user found
    except twitter.error.TwitterError:
        for k in bio_cols+tweet_cols:
            account[k] = None
        time.sleep(15)
        return account
    # Other types of error
    except Exception:
        print(self.user_id, 'Error at Bio')
        for k in bio_cols+tweet_cols:
            account[k] = None
        time.sleep(15)
        return account

    for _ in range(3):
        if last_tweet_id: # continue crawling
            statuses = api.GetUserTimeline(user_id=user_id, trim_user=True,
                                           count=result_limit,
                                           max_id=last_tweet_id - 1,
                                           include_rts=mine_rts)
            statuses = [stt.AsDict() for stt in statuses]
        else: # first crawling
            try:
                statuses = api.GetUserTimeline(user_id=user_id, trim_user=True,
                                               count=result_limit,
                                               include_rts=mine_rts)
                if len(statuses) == 0:
                    for k in tweet_cols:
                        account[k] = None
                    time.sleep(15)
                    return account
                statuses = [stt.AsDict() for stt in statuses]
            # If the account is protected
            except twitter.error.TwitterError:
                for k in tweet_cols:
                    account[k] = None
                time.sleep(15)
                return account
            # Other types of error
            except Exception:
                print(self.user_id, 'Error at Tweets')
                for k in tweet_cols:
                    account[k] = None
                time.sleep(15)
                return account

        # Extract max and average number of retweets and favorites
        for stat in statuses:
            if 'retweet_count' in stat:
                rt_cnt += stat['retweet_count']
                if stat['retweet_count'] > max_rt_cnt:
                    max_rt_cnt = stat['retweet_count']

            if 'favorite_count' in stat:
                fav_cnt += stat['favorite_count']
                if stat['favorite_count'] > max_fav_cnt:
                    max_fav_cnt = stat['favorite_count']
            tweets.append(stat['full_text'])

        last_tweet_id = stat['id']
    account['avg_rt_cnt'] = rt_cnt/len(tweets)
    account['avg_fav_cnt'] = fav_cnt/len(tweets)
    account['max_rt_cnt'] = max_rt_cnt
    account['max_fav_cnt'] = max_fav_cnt
    account['tweets'] = ' '.join(tweets)

    time.sleep(18)
    print(user_id)
    return account


if __name__ = '__main__':
    data = pd.read_csv('user_ids.csv').rename(columns={'Twitter Id':'tid'})
    data = data.apply(get_variables, axis=1)
    data.to_csv('raw_data.csv', index=False)



