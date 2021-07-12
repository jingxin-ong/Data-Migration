####################################################################################
# Michael Barber, Jan2020
# Script to preprocess RAW farmpulse data dump
# Strips out Hindi entries, spell checks, lemmatizes and removes stopwords. then uploads
####################################################################################

if True:
    # first time setup, installs various packages
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    #!python -m spacy download en_core_web_sm
    # !python -m spacy download en
# logging
import sys
import logging

# base
import numpy as np
import pandas as pd
import psycopg2
from functools import wraps
from bs4 import BeautifulSoup
import decimal

#NLP
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize.toktok import ToktokTokenizer

#from nltk import word_tokenize
from nltk.corpus import wordnet
import re
from datetime import datetime
import unicodedata
import string


#from spellchecker import SpellChecker
#spell = SpellChecker(distance=1) #LV distance
#spell.word_frequency.load_words(['thrip','borer','mealy','fung']) # add words to list

tokenizer = ToktokTokenizer()

# AWS dependencies
import boto3
import botocore
import os

print("loaded")

# Key-values
# bucket_name = 'farm-pulse-pre'
log_key = "Farmpulse_log.txt"
log_path = "./{0}".format(log_key)

# S3 remote variables
s3_path_log = "logs/{0}".format(log_key)
# s3 = boto3.resource('s3')
#s3_file_key = "Static/Sample.csv"

# NLP stack
lemmatizer = WordNetLemmatizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
stopword_list.append(['farmer','crop','want','know','regard','give','plant','need'])

"""
try:
    print("test",os.environ['test'] )
except Exception:
    print("no env")
"""


def ensure_dir(file_path):
    """check for dir, and create if not found
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

'''
def get_logs(s3_path_log=s3_path_log , log_path=log_path):
    """download logs from s3
    Params:
        None
    Returns:
        None
    """
    if not os.path.exists(log_path):
        try:
            s3.Bucket(bucket_name).download_file(s3_path_log, log_path)
        except Exception:
            update_log(er='File created',upload=True)
'''

def my_logger(orig_func):
    logging.basicConfig(filename=log_key, level=logging.INFO)
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logging.info(
            'Ran with args: {}, and kwargs: {}'.format(args, kwargs))
        return orig_func(*args, **kwargs)
    return(wrapper)


def update_log(er="File created",log_path=log_path, s3_path_log = s3_path_log, upload=False):
    """update logs and upload to s3
        Params:
            (str) error message
        Returns:
            None
    """
    print(er)
    with open(log_path, 'a') as file:
        file.write(str(datetime.now()) + ',' + str(er) + '\n')
    # if upload is True:
        # s3.meta.client.upload_file(log_path, bucket_name, s3_path_log)

def get_predata():
    conn = psycopg2.connect ("dbname = postgres user = postgres host = localhost")
    cur = conn.cursor()

    try:
        connection = psycopg2.connect(user = "postgres",
                                    host = "127.0.0.1",
                                    port = "5432",
                                    database = "farmpulse")
        cursor = connection.cursor()
        print('connected')
    except (Exception, psycopg2.Error) as error :
        print("Error while connecting to PostgreSQL", error)

    postgreSQL_select_Query = "select * from call_centre.india where created_on is not null and state_name is not null and crop is not null and query_type is not null"

    cursor.execute(postgreSQL_select_Query)
    records = cursor.fetchall()
    field_names = [i[0] for i in cursor.description]
    df = pd.DataFrame(records, columns=field_names)
    print('first 5 rows of df:')
    print(df.head(5))

    return df


def log_info(strings1, strings2=""):
    logging.info('Ran with {}:{}'.format(strings1, strings2))


def save_data(dfin, outfile="./FPeng_prepped"):
    """saves file locally and to s3
    Params:
        dfin: DF to save
        outfile: path to file
    Returns:
        None
    """
    dfin.to_csv(outfile+'.csv', sep='\t', index=False)
    # s3.meta.client.upload_file(outfile+".csv", 'p3-engine', 'ETL/FPeng_prepped.csv')
    print("csv...", end=" ")

    dfin.to_pickle(outfile+'.pkl' ,protocol=4)
    # s3.meta.client.upload_file(outfile+'.pkl', 'p3-engine', 'ETL/FPeng_prepped.pkl')
    print("pkl...", end=" ")
    #dfin.to_msgpack(outfile+'.msg')
    #print("msg...", end=" ")

    #s3.meta.client.upload_file(outfile+".msg", 'p3-engine', 'ETL/FPeng_prepped.msg')

    # print("to s3 complete", end=" ")


def clean_df(dfin, top=10):
    """simple cleaning of DF
    Params:
        dfin: DF to save
        top: number of crops to retain (by most popular)
    Returns:
    """

    dfin['crop'] = dfin['crop'].astype('str')
    dfin['crop'] = dfin.crop.str.lower()

    dfin["created_on"] = dfin["created_on"].astype("datetime64")
    dfin['latitude'] = np.round(dfin.latitude.apply(pd.to_numeric),2)
    dfin['longitude'] = np.round(dfin.longitude.apply(pd.to_numeric),2)
    dfin['query_type'] = dfin['query_type'].astype('str')
    dfin['query_type'] = dfin.query_type.apply(str.lower)

    dfin['hits'] = 1

    dfin = dfin[pd.notnull(dfin.kcc_answer_raw)]
    dfin = dfin[pd.notnull(dfin['query_text_raw'])]

    dfin['query_text_raw'] = dfin.query_text_raw.str.lower()
    dfin['kcc_answer_raw'] = dfin.kcc_answer_raw.str.lower()

    dfin['state_name'] = dfin.state_name.str.lower()
    dfin['district_name'] = dfin.district_name.str.lower()

    dfin['crop_full'] = dfin.crop
    dfin['crop'] = [i.split()[0] if len(i.split())>1 else i for i in dfin.crop]
    dfin.dropna(how='all',inplace=True)

    #topcrop = dfin.crop.value_counts().head(top).index.tolist()
    topcrop = ['paddy', 'wheat', 'cotton', 'chillies', 'onion', 'brinjal', 'sugarcane', 'tomato', 'bengal', 'groundnut', 'soybean', 'potato','maize']
    dfin = dfin[dfin.crop.isin(topcrop)]
    print(dfin.crop.unique())

    dfin = dfin[['crop','created_on','latitude','longitude','query_type','query_text_raw','kcc_answer_raw','state_name','district_name','crop_full']]
    return dfin


#df = get_predata(s3key='', local_path='/Users/michaelbarber/Documents/Local_datasets/India_callcentre/FP_english_noweather.csv')


def lemmatize_text(text, fast=False):
    """
    lemmatizes text
    Params:
        text: sentence to lemmatize
        fast: use simple lemma or context aware lemma
    """
    if fast:
        return ' '.join([lemmatizer.lemmatize(i) for i in text.split()])
    else:
        return ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokenizer.tokenize(text)])


def get_wordnet_pos(word):
    """
    Map POS tag to first character lemmatize() accepts
    Params:
        word: word in
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def stem_text(text):
    """
    stem each word in sentence
    Params:
        text: sentence in
    """
    stmr = PorterStemmer() ## LancasterStemmer()
    return ' '.join([stmr.stem(i) for i in text.split()])


def strip_html_tags(text):
    """
    strip htmls
    Params:
        text: sentence in
    """
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text


def remove_special_characters(text, remove_digits=False):
    """
    remove special characters
    Params:
        text: sentence in
    """
    text = "".join([i if i not in string.punctuation else " " for i in text])
    return text


def remove_stopwords(text, is_lower_case=False):
    """
    remove stopwords from text
    Params:
        text: sentence in
    """
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def remove_fluff(text):
    """
    remove common ag-stopwords
    Params:
        text: sentence in
    returns:
    cleaner text
    """
    stopwords_list = ['farmer', 'want', 'know', 'information', 'inform', 'regard', 'give', 'crop', 'use', 'new', 'tell', 'ask']
    for wrd in stopwords_list:
        text = text.replace(wrd, "")
    crop_list = ['cotton','paddy','wheat','rice','onion','maize','tomato','potato','sugarcane' ]
    for wrd in crop_list:
        text = text.replace(wrd, "")
    return(text)


def only_numbers(text):
    """
    check text isnt just numbers and special characters
    Params:
        text: sentence in
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = remove_special_characters(text)
    text = text.strip()
    text = re.sub(r' ', '', text)
    return len(text)


def test_hindi(doc):
    """checks for hindi
    doc = list of tokens
    """
    hindi_dictionary = ['kai','hai','dhaan','dhan','jhona','pili','jankari','saaf','mela','narma','raja','brahma','jai','parbhani','sangli','jana']
    flag = any(hindi in doc for hindi in hindi_dictionary)
    return(flag)


def manual_corrections(phrase):
    phrase = phrase.replace("trip","thrip") #common typo
    phrase = phrase.replace("flythrip","fly thrip") #common typo
    phrase = phrase.replace("carl", "curl") # common typo
    phrase = phrase.replace("jassid","hopper") # translate local to global
    phrase = phrase.replace("microlla","micronutrient") # translate product to generic
    phrase = phrase.replace("mandi", "market") # hindi to english
    phrase = phrase.replace("meali", "mealy") # hindi to english
    return phrase


def normalize_text(text, skip_weather=True, defluff=False):
    """
    function to unite text processing
    Params:
        text: sentence in
        defluff: obsolite
        skip_weather: if text contains weather-related words then return one tag
    """

    if only_numbers(text) <4:
        return "dud_drop_me"

    if test_hindi(text):
        return "dud_drop_me"

    if skip_weather:
        if "weather" in text or 'rain' in text:
            return "dud_drop_me"

    if 'http' in text:
        return "dud_drop_me"

    text = manual_corrections(text)
    text = remove_special_characters(text)
    text = remove_stopwords(text)
    #text = " ".join([spell.correction(word) for word in text.split()]) # LV based spell correction
    text = lemmatize_text(text)
    text = stem_text(text)


    if defluff==True:
        text = remove_fluff(text)

    text = text.replace("  "," ") # clean up spaces
    text =  tokenizer.tokenize(text) #word_tokenize(text)
    if np.random.uniform(0,1) <0.000001: # print random snippets as progress
        print("sample:", text)
    return text


def concat_text(text):
    """
    concat tokens back into text
    Params:
        text: sentence in
    """
    textout = " ".join(text)
    return textout

def normQnA(dfin):
    """
    normalize raw Q and As
    Params:
        dfin: dataframe, must contain [query_text_raw] and [kcc_answer_raw]
    """

    update_log(er='norming Q',upload=False)
    dfin['normQ']= dfin.query_text_raw.apply(normalize_text, args=(True,True))
    dfin = dfin[dfin.normQ != 'dud_drop_me']

    #update_log(er='norming A',upload=True)
    #dfin['normA']= dfin.kcc_answer_raw.apply(normalize_text , args=(False,False))
    #dfin = dfin[dfin.normA != 'dud_drop_me']
    return(dfin)

def main(context=None, event=None): # , s3_rawdata_path='RAW/FP_english_noweather.csv'):
    """
    AWS handler for whole
    Params:
        none
    """

    # get_logs()
    # df = get_predata('sample/FP_english_noweather.csv')
    df = get_predata()
    original_size = df.shape[0]
    print(original_size)

    df = clean_df(df, top=2)
    df = normQnA(df)
    try:
        df.reset_index(inplace=True)
    except:
        pass

    save_data(df)

    #print("Original size was {0}, output size is {1}".format(original_size, df.shape))
    update_log(er="Original size was {0}, output size is {1}".format(original_size, df.shape), upload=True )


if __name__ == '__main__':
    #log_key = str(os.environ['log_name'])   # replace with your object key
    #bucket_name = os.environ['bucket_name']
    main()
