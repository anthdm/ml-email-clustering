from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import linear_kernel
from helpers import parse_into_emails

import pandas as pd

def read_email_bodies():
  emails = pd.read_csv('split_emails.csv')
  email_df = pd.DataFrame(parse_into_emails(emails.message))
  email_df.drop(email_df.query("body == '' | to == '' | from_ == ''").index, inplace=True)
  email_df.drop_duplicates(inplace=True)
  return email_df['body']


class EmailDataset: 
  def __init__(self):
    stopwords = ENGLISH_STOP_WORDS.union(['ect', 'hou', 'com', 'recipient'])
    self.vec = TfidfVectorizer(analyzer='word', stop_words=stopwords, max_df=0.3, min_df=2)
    self.emails = read_email_bodies() 

    # train on the given email data.
    self.train()
  
  def train(self):
    self.vec_train = self.vec.fit_transform(self.emails)
  
  def query(self, keyword, limit):
    vec_keyword = self.vec.transform([keyword])
    cosine_sim = linear_kernel(vec_keyword, self.vec_train).flatten()
    related_email_indices = cosine_sim.argsort()[:-limit:-1]
    print(related_email_indices)
    return related_email_indices

  def find_email_by_index(self, i):
    return self.emails.as_matrix()[i]