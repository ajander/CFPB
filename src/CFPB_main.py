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
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stop 
import re

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

results = client.get('jhzv-w97w', where='complaint_what_happened is not null', limit=2000)
results_df = pd.DataFrame.from_records(results)
df = results_df[fields_to_keep]

client.close()
    
#%% SPACY SET-UP
    
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()

# spacy.en.language_data.STOP_WORDS - how to access?

#%% SPACY NER

def NER(doc):
    processed_doc = nlp(doc)
    ents = [str(ent) for ent in processed_doc.ents if ent.label_ == 'ORG']
    ents = [e.replace('X','').lower() for e in ents]
    ent_tokens = [word_tokenize(e) for e in ents]
    ent_tokens = [e for e in ents if e not in stop and len(e) > 2]
    return ent_tokens
    
df['ent_tokens'] = df.complaint_what_happened.apply(lambda text: NER(text))

# Possible action plan:
# Find relevant entities in each complaint using NER function
# Find noun phrases in each complaint using find_np function
# Convert to embedding vector
# Train SVM using noun phrase embeddings as input and company as target
# A row may have more than one inputs (treat each noun phrase as a separate sample)

#%% SPACY NOUN CHUNKS

doc = df['complaint_what_happened'].iloc[3]

# Need more preprocessing (what's the best order to do these in?):
#   * Replace 'X' with empty string - DONE
#   * Convert to lowercase - DONE
#   * Remove punctuation
#   * Remove tokens with only digits
#   * Stem
# Do all steps before noun chunking? Or save some until afterwards?

# What if I use NOUNS instead of NOUN CHUNKS? Does that perform better or worse?
        
def find_nc(doc):
    tokens = []
    doc = doc.replace('X','').lower()
    for nc in nlp(doc).noun_chunks:
        if len(nc) > 1 and not all(token.is_stop for token in nc):
            tokens += [w for w in word_tokenize(nc.text) if w not in stop]
    tokens = [t for t in [re.sub(r'[^a-z]+', '', t) for t in tokens] if len(t)>2]
    tokens = [WordNetLemmatizer().lemmatize(t) for t in tokens]
    return tokens

df['nc'] = df['complaint_what_happened'].apply(lambda doc: find_nc(doc))

#%% VECTORIZE THE TOKENS DERIVED FROM NOUN CHUNKS USING ARORA METHOD

from sentence2vec import *

def create_docvec(tokens):
    words = [Word(t, nlp(t).vector) for t in tokens]
    docvec = Sentence(words)
    return docvec

docvec_input = df['nc'].apply(lambda tokens: create_docvec(tokens))

embedding_size = 300

# training
df['docvec'] = sentence_to_vec(docvec_input.tolist(), embedding_size)
# Not working...
# "ValueError: Input contains NaN, infinity or a value too large for dtype('float64')."

#%% VECTORIZE NOUN CHUNK TOKENS USING GENSIM DOC2VEC

#import gensim
from gensim.models import doc2vec

import time

#def nltk_tokenize(text):
#    text = text.lower()
#    tokens = word_tokenize(text)
#    return tokens
#
#docs = [doc2vec.TaggedDocument(
#        words=nltk_tokenize(d), tags=[label]) for d, label in zip(
#                df['complaint_what_happened'], df['complaint_id'])]
    
docs = [doc2vec.TaggedDocument(
        words=d, tags=[label]) for d, label in zip(
                df['nc'], df['complaint_id'])]
    
model = doc2vec.Doc2Vec(docs, size = 100, window = 8, min_count = 10, workers = 4)

# Find similar documents
new_doc = df['nc'][2]
new_vector = model.infer_vector(new_doc)
sims = model.docvecs.most_similar([new_vector])


























