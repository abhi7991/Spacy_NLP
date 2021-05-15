# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:15:00 2020

@author: abhis
"""

import pandas as pd
import datetime as datetime
from pytz import timezone
import re
from spacy.matcher import PhraseMatcher
from spacy.pipeline import EntityRuler
from spacy.matcher import Matcher
from spacy import displacy
from collections import Counter
import en_core_web_sm
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import pandas as pd
import numpy as np
import unidecode
import os
import datetime
import re
import unidecode
import re

import nltk
from nltk.corpus import stopwords
from pandas.io import sql
import os
import requests
from bs4 import BeautifulSoup
import re
import datetime as datetime
from pytz import timezone
import time
import sys
from dateutil import parser

import warnings
import json
import calendar
import numpy as np
warnings.filterwarnings('ignore')
import shutil


def ext_info(link):
    r = requests.get(link)
    print(r.status_code)
    soup1 = BeautifulSoup(r.content,'lxml')
    para = soup1.find_all("p")
    para = [x.text for x in para]
    
    txt = " ".join(str(x) for x in para)
    doc1 = nlp(txt)
    op = [(ent.text, ent.label_) for ent in doc1.ents]
    time.sleep(2)
    print(link)
    return op

def brand(string):
    doc1 = nlp(string)
    try:
#        brand = [(ent.text, ent.label_, ent.ent_id_) for ent in doc1.ents][0][0]
#        if brand not in string:
#            brand = [(ent.text, ent.label_, ent.ent_id_) for ent in doc1.ents][0][1]
        brand = [(ent.text, ent.label_) for ent in doc1.ents]
        brand = [x for x in brand if ('NEW BUSINESS' in x)]
        
    except:
        brand = " "
    return brand 


'''

Getting All the articles from a company

'''
years = ['2020','2019','2018','2017','2016']
infosys = 'https://www.infosys.com/newsroom/press-releases.html?year=2020'
r = requests.get(infosys)
print(r.status_code)
soup1 = BeautifulSoup(r.text,'html')
headlines = soup1.find_all('a',{'id':'Url'})
links = soup1.find_all('a',{'id':'Url'})
headlines = [x.text for x in headlines] 
links = [x['href'] for x in links]
df = pd.DataFrame({'Headlines':headlines,'Links':links})
df['Links'] = "https://www.infosys.com"+df['Links']
main = pd.DataFrame()
for y in years:
    infosys = 'https://www.infosys.com/newsroom/press-releases.html?year='+str(y)
    print(infosys)
    time.sleep(2)
    r = requests.get(infosys)
    print(r.status_code)
    soup1 = BeautifulSoup(r.text,'html')
    headlines = soup1.find_all('a',{'id':'Url'})
    links = soup1.find_all('a',{'id':'Url'})
    headlines = [x.text for x in headlines] 
    links = [x['href'] for x in links]
    df = pd.DataFrame({'Headlines':headlines,'Links':links})
    df['Links'] = "https://www.infosys.com"+df['Links']
    df['Year'] = y
    main = pd.concat([main,df])
#    main['Year'] = y    
#main.to_csv("Press-release_infosys.csv",index=False)    

'''

Give your own cutstom tag words
1. Im trying to search for acquisitions

'''
df = main.copy()    
  
new_business = ['partner','partners','open','joint','alliance','venture','deliver','help','tennis','atp','collaborates','collaboration','chosen','selects','contract','mou','sign','expand','agreement','signing','contracting','partnership','client','customer','selected','select','procurement','customer','new business','new client','new customer','collaborate','upgrade','select','selected','join','launch','opens','joins','adopt','adopts','build','develop','development','business development']    
acq = ['acquisition','merge','merger','acquire']
kmp = ['resigns','appoint','appointed','charge','step down','term']
#Disabling Pipelines
nlp = spacy.load("en_core_web_sm")#,disable=['parser'])    
#ruler = nlp.create_pipe("entity_ruler")
ruler = EntityRuler(nlp,overwrite_ents=True)#,validate=True)

patterns = list(nlp.tokenizer.pipe(new_business))


'''

Creating A Custom Pipeline

'''

patterns = [{"label": "NEW BUSINESS", "pattern": new_business[i],"id": new_business[i],'IS_PUNCT': True} for i in range(len(new_business))]#,"id": brands[i]} for i in range(len(brands))]
#patterns = [x.append([{"id": brands[i]} for i in range(len(brands))]) for x in patterns]
phrase_matcher_attr="POS"
ruler.add_patterns(patterns)
nlp.add_pipe(ruler)#,before="ner")

patterns1 = list(nlp.tokenizer.pipe(kmp))
patterns1 = [{"label": "KMP", "pattern": kmp[i],"id": kmp[i],'IS_PUNCT': True} for i in range(len(kmp))]#,"id": brands[i]} for i in range(len(brands))]
phrase_matcher_attr="POS"
ruler.add_patterns(patterns1)

patterns2 = list(nlp.tokenizer.pipe(acq))
patterns2 = [{"label": "ACQ", "pattern": acq[i],"id": acq[i],'IS_PUNCT': True} for i in range(len(acq))]#,"id": brands[i]} for i in range(len(brands))]
phrase_matcher_attr="POS"
ruler.add_patterns(patterns2)

df['Headlines'] = df['Headlines'].apply(lambda x: x.lower())

'''

Running a quick sample

'''

doc1 = nlp(df['Headlines'][1].tail(1)[1])
print([(ent.text, ent.label_, ent.ent_id_) for ent in doc1.ents])
a = [(ent.text, ent.label_, ent.ent_id_) for ent in doc1.ents]

   
'''

Applying for entire dataframe

'''


df['Tagging'] = df['Headlines'].apply(lambda x: brand(x))   
df['Client'] = df['Tagging'].apply(lambda x : 1 if len(x) != 0 else 0)
a = df[df['Client']==1]

test = df.copy()

test = test[test['Client']==1]

test = test.reset_index()
test = test.iloc[:,1:len(df.columns)]

link1 = test['Links'][0]
r = requests.get(link1)
print(r.status_code)
soup1 = BeautifulSoup(r.content,'lxml')
para = soup1.find_all("p")
para = [x.text for x in para]

txt = " ".join(str(x) for x in para)
doc1 = nlp(txt)



test['Info'] = test['Links'].apply(lambda x: ext_info(x))   
test['Geo_Political Entity'] = test['Info'].apply(lambda x: [y for y in x if ("GPE" in y)])
test['Monetary'] = test['Info'].apply(lambda x: [y for y in x if ("CARDINAL" in y)])
test['Organisation'] = test['Info'].apply(lambda x: [y for y in x if ("ORG" in y)])
test['Date'] = test['Info'].apply(lambda x: [y for y in x if ("DATE" in y)])
test['Monetary_2'] = test['Info'].apply(lambda x: [y for y in x if ("MONEY" in y)])
test['Product'] = test['Info'].apply(lambda x: [y for y in x if ("PRODUCT" in y)])
test['Percent'] = test['Info'].apply(lambda x: [y for y in x if ("PERCENT" in y)])
test['Tags'] = test['Info'].apply(lambda x:[ y for z, y in x]) 
tags = test['Tags'].to_list()
tags = sum(tags, [])
print(np.unique(np.array(tags))) 


os.chdir("C:\Github")

sample = test[['Headlines', 'Links', 'Year', 'Tagging',
       'Geo_Political Entity', 'Monetary','Monetary_2', 'Organisation', 'Date',
       'Product', 'Percent']]
sample.columns = ['Headlines', 'Links', 'Year', 'Overall_Tags', 'Geo_Political Entity',
       'Numeric', 'Monetary', 'Organisation', 'Date', 'Product', 'Percent']
sample.to_csv("Articles_Tagged-Infosys.csv",index=False)

df.to_csv("Press_Releases-Infosys.csv",index=False)