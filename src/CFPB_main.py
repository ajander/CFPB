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

#%% GET DATA FROM CFPB

APP_TOKEN = 'Jhf65sZJl8oqUNCpFZ0PubeOa'

# Fields to retrieve:
    # date_received
    # product
    # sub_product
    # issue
    # sub_issue
    # complaint_what_happened
    # company
    # state
    # zip_code
    # consumer_consent_provided (probably need to filter on this)
    # complaint_id (unique)
    
client = Socrata("data.consumerfinance.gov", APP_TOKEN)

results = client.get("jhzv-w97w", 
                     where="complaint_what_happened is not null",
                     limit=200)
results_df = pd.DataFrame.from_records(results)
results_df = results_df[['date_received',
                         'product',
                         'sub_product',
                         'issue',
                         'sub_issue',
                         'complaint_what_happened',
                         'company',
                         'state',
                         'zip_code',
                         'complaint_id']]
    
#%% ENTITY RECOGNITION

import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()

#%% CALCULATE SENTENCE VECTORS
