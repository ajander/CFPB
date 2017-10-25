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

import os
entry_point = r'/home/adrienne/Code/PythonProjects/CFPB/src'
os.chdir(entry_point)

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
    processed_doc = nlp(doc)
    ents = [(ent,ent.label_) for ent in processed_doc.ents]
    return [str(e[0]) for e in ents if e[1] == 'ORG' and len(str(e[0]).replace('X','')) > 2]
    
df['ents'] = df.complaint_what_happened.apply(lambda text: NER(text))

# Possible action plan:
# Find relevant entities in each complaint using NER function
# Find noun phrases in each complaint using find_np function
# Convert to embedding vector
# Train SVM using noun phrase embeddings as input and company as target
# A row may have more than one inputs (treat each noun phrase as a separate sample)

#%% SPACY NOUN CHUNKS - NOT USING RIGHT NOW...

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
        
#%% SENTENCE VECTORIZATION

# How many sentences does a complaint usually have?
num_sents = df.complaint_what_happened.apply(
        lambda text: len([s for s in nlp(text).sents]))

#%% SENTENCE VECTORS FROM "A TOUGH TO BEAT BASELINE..."

import sentence2vec.py

#%% DOCUMENT VECTORS FROM GENSIM

import gensim
from gensim.models import doc2vec
from nltk import word_tokenize
import time

def preprocess(text):
    start = time.time()
    text = text.lower()
    tokens = word_tokenize(text)
    print("nltk: %0.3f" % (time.time()-start))
    return tokens

docs = [doc2vec.TaggedDocument(
        words=d.split(), tags=[label]) for d, label in zip(
                df['complaint_what_happened'], df['complaint_id'])]
    
model = doc2vec.Doc2Vec(docs, size = 100, window = 300, min_count = 1, workers = 4)
v = model.docvecs


























