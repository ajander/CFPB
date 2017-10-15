# -*- coding: utf-8 -*-
"""
Application name: UMKC_research
Description: Natural language processing of CFPB complaints
Author: Adrienne Anderson
Date: 2017-10-01
"""

#%% SET-UP

# Use environment simple_nlp
import pandas as pd
from sodapy import Socrata

#%% GET COMPLAINT DATA

APP_TOKEN = 'Jhf65sZJl8oqUNCpFZ0PubeOa'

fields_to_keep = [
        'date_received',
        'product',
        'sub_product',
        'issue',
        'sub_issue',
        'complaint_what_happened',
        'company',
        'state',
        'zip_code',
        'complaint_id'
        ]

    
client = Socrata('data.consumerfinance.gov', APP_TOKEN)

results = client.get('jhzv-w97w', where='complaint_what_happened is not null', limit=200)
results_df = pd.DataFrame.from_records(results)
df = results_df[fields_to_keep]

client.close()
    
#%% SPACY SET-UP
    
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()

#%% SPACY NER

labels_of_interest = ['PERSON','ORG','PRODUCT']

def NER(doc):
    ents = [(ent,ent.label_) for ent in doc.ents]
    return [(str(e[0]),e[1]) for e in ents if e[1] == 'ORG' and len(str(e[0]).replace('X','')) > 2]
    
df['ents'] = df.complaint_what_happened.apply(lambda text: NER(text))

# Possible action plan:
# Find relevant entities in each complaint using NER function
# Find noun phrases in each complaint using find_np function
# Convert to embedding vector
# Train SVM using noun phrase embeddings as input and company as target
# A row may have more than one inputs (treat each noun phrase as a separate sample)

#%% SPACY NOUN CHUNKS

doc = nlp(df['complaint_what_happened'].str.replace('X','').iloc[3])

# Check:
# more than one token in chunk
# replace X with empty string - do this in preprocessing
# at least one token not in stopword list
# at least one token with high tfidf score?
        
def find_nc(doc):
    for nc in doc.noun_chunks:
        if len(nc) > 1 and not all(token.is_stop for token in nc):
            print(nc.text)
    return

#%% CREATE A PIPELINE
    
# Incorporating spaCy into sklearn pipeline:
# https://nicschrading.com/project/Intro-to-NLP-with-spaCy/
# But I'd rather use a spacy pipeline - more efficient processing, I assume
    
# Use textacy, a package built off of spacy?
# https://github.com/chartbeat-labs/textacy
        
#%% SENTENCE VECTORS FROM "A TOUGH TO BEAT BASELINE..."



#%% DOCUMENT VECTORS FROM GENSIM

import gensim
from gensim.models import doc2vec

docs = [doc2vec.TaggedDocument(
        words=d.split(), tags=[label]) for d, label in zip(
                df['complaint_what_happened'], df['complaint_id'])]
    
model = doc2vec.Doc2Vec(docs, size = 100, window = 300, min_count = 1, workers = 4)
v = model.docvecs

import scipy
import numpy as np

def sim(vec1,vec2):
    return 1 - scipy.spatial.distance.cosine(vec1, vec2)

def l2_dist(vec1, vec2):
    a = np.linalg.norm(vec1)
    b = np.linalg.norm(vec2)
    return scipy.spatial.distance.euclidean(a,b)

#%% create distance matrix for comparing and finding smallest & largest distances
# NEED TO MODIFY CODE HERE - also, normalize each vector first
    
from scipy.spatial.distance import pdist, squareform

d = pdist(ncoord)

# pdist just returns the upper triangle of the pairwise distance matrix. to get
# the whole (20, 20) array we can use squareform:

print(d.shape)
# (190,)

D2 = squareform(d)
print(D2.shape)
# (20, 20)