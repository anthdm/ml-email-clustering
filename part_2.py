# Code used in part 2 of How I used machine learning to classify emails and turn them into insights.

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

from helpers import parse_into_emails
from query import EmailDataset

# Just like in part_1, read and preprocess emails
emails = pd.read_csv('split_emails.csv') 
email_df = pd.DataFrame(parse_into_emails(emails.message))
email_df.drop(email_df.query("body == '' | to == '' | from_ == ''").index, inplace=True)

stopwords = ENGLISH_STOP_WORDS.union(['ect', 'hou', 'com', 'recipient'])
vec = TfidfVectorizer(analyzer='word', stop_words=stopwords, max_df=0.3, min_df=2)
vec_train = vec.fit_transform(email_df.body)

# print out the vector of the first email
# print(vec_train[0:1])

# Find cosine similarity between the first email and all others.
cosine_sim = linear_kernel(vec_train[0:1], vec_train).flatten()
# print out the cosine similarities
# print(cosine_sim)

# Finding emails related to a query.
query = "john"

# Transform the query into the original vector
vec_query = vec.transform([query])

cosine_sim = linear_kernel(vec_query, vec_train).flatten()

# Find top 10 most related emails to the query.
related_email_indices = cosine_sim.argsort()[:-10:-1]
# print out the indices of the 10 most related emails.
print(related_email_indices)

# print out the first email 
first_email_index = related_email_indices[0]
print(email_df.body.as_matrix()[first_email_index])

# use the EmailDataset class to query for keywords.
# ds = EmailDataset()
# results = ds.query('salary', 10)

# Print out the first result.
# print(ds.find_email_by_index(results[0]))




