#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 13:04:32 2021

@author: zacharylazzara
"""

import pdftotext
import pandas as pd
import nltk
import nltk.data




from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import itertools

from statistics import mean
import datetime
from collections import Counter
import string
from operator import itemgetter


# Punkt is an unsupervised algorithm which is
# pre-trained on English, so we should be able to use it as is.
sent_detector = nltk.data.load("tokenizers/punkt/english.pickle")
#sent_detector.tokenize() # will need to use this to replace whatever we used before
#NOTE: We may want to train punkt on our text
#Review: https://www.nltk.org/api/nltk.tokenize.html




#reference_pdf = "insect.pdf"
#thesis_pdf = "identity.pdf"
reference_pdf = "identity.pdf"
#thesis_pdf = "identity.pdf"
thesis_pdf = "resonant.pdf"
#reference_pdf = "lorem-ipsum.pdf"
#thesis_pdf = "lorem-ipsum.pdf"




with open(thesis_pdf, "rb") as t_file:
    t_pdf = pdftotext.PDF(t_file)
    
with open(reference_pdf, "rb") as r_file:
    r_pdf = pdftotext.PDF(r_file)


def check_similarity(thesis_sentence, reference_sentence):
    # Adapted from https://stackoverflow.com/questions/46732843/compare-two-sentences-on-basis-of-grammar-using-nlp
    
    # Use stemmer 
    stm = PorterStemmer()
    
    # Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    
    ts = nltk.pos_tag(nltk.word_tokenize(thesis_sentence))
    
    ts = dict(filter(lambda x: len(x[1])>0,
                     map(lambda row: (row[0],wn.synsets(
                           stm.stem(row[0]),
                           tag_dict[row[1][0]])) if row[1][0] in tag_dict.keys() 
                         else (row[0],[]),ts)))
    
    rs = nltk.pos_tag(nltk.word_tokenize(reference_sentence))
    
    rs = dict(filter(lambda x: len(x[1])>0,
                     map(lambda row: (row[0],wn.synsets(
                              stm.stem(row[0]),
                              tag_dict[row[1][0]])) if row[1][0] in tag_dict.keys() 
                         else (row[0],[]),rs)))
    
    res = {}
    for w2,gr2 in rs.items():
        for w1,gr1 in ts.items():
            tmp = pd.Series(list(map(lambda row: row[1].path_similarity(row[0]),
                                     itertools.product(gr1,gr2)))).dropna()
            if len(tmp)>0:
                res[(w1,w2)] = tmp.max()
    #print(res)
    
    # TODO: we need to figure out why this function gives us NAN sometimes
    
    return pd.Series(res, dtype=float).groupby(level=0).max().mean(skipna = True)

cols = ["T_SENTENCE", "R_SENTENCE", "SIMILARITY"]
def compare(t_sents, r_sents):
    sim_df = pd.DataFrame(columns = cols)
    
    t_sent_num = len(t_sents)
    r_sent_num = len(r_sents)
    total_sent_num = t_sent_num * r_sent_num
    
    print("Program started on: ", datetime.datetime.now())
    progress = 0
    
    t_df = pd.DataFrame(t_sents, columns = [cols[0]])
    for t_sent in t_df[cols[0]]:
        r_df = pd.DataFrame(columns = [cols[0], cols[1], cols[2]])
        for r_sent in r_sents:
            sim = check_similarity(t_sent, r_sent)
            r_df = r_df.append({cols[0]:t_sent, cols[1]:r_sent, cols[2]:sim}, ignore_index = True)
            print("\rProgress: {:.2f}% ({}/{}), Average Similarity: {:.2f}%  ".format((progress / total_sent_num) * 100, progress, total_sent_num, sim_df[cols[2]].mean() * 100), end = "", flush = True)
            progress += 1
        t_df = t_df.append(r_df, ignore_index = True)
        sim_df = sim_df.append(t_df[t_df[cols[2]] == t_df[cols[2]].max()])
    return sim_df

def target_abstract(t_pdf, r_pdf):
    print(t_pdf[0].split("\n"))
    
    #TODO: now that we have the abstract we need to quickly compare this
    # with my thesis (or perhaps with my abstract?) to see if the reference
    # is worth pursing or not
    
    


    
    
    
    #abst_cols = ["T_SENTENCE", "R_ABSTRACT"]
    #ab_df = pd.DataFrame(columns = [abst_cols[0], abst_cols[1]])
    # TODO: we need to target the abstract paragraph somehow
    # Once we see abstract for the first time in the document, we will then take
    # the paragraph it is in (or the next one if its paragraph is smaller than a sentence).
    # Then we can start comparing against it.
    
    #for ab in ab_df:
     #   if "abstract" in r_sents:
            # read for one paragraph then stop





t_pages = [sent_detector.tokenize(page) for page in t_pdf]
r_pages = [sent_detector.tokenize(page) for page in r_pdf]

t_sents = [sent for page in t_pages for sent in page]
r_sents = [sent for page in r_pages for sent in page]

sim_df = compare(t_sents, r_sents)

t_total_sent = len(t_sents)
sent_count = 0








# print("\nSupporting Sentences:")
# for t_sent in sim_df[cols[0]]:
    
    
    
    
#     print(t_sent)
#     print("\n\n")
    
    # TODO: we need to get the r_sent associated with the highest sim (r_sent should
    # be presented as unique, showing only the highest sim for a given t_sent)
    # r_sent = 
    # sim = 
    # sent_count += 1
    
    # if sim > 0.5:
    #     print("SUPPORTING SENTENCE #{0} (Similarity: {1})\nTHESIS_{0}: {2}\nREFERENCE_{0}: {3}\n".format(sent_count, sim, t_sent, r_sent))




#TODO: we need to target abstracts first to determine paper relevance before diving in deeper




#no clue if this will work
# We want to filter out anything lower than 50% similarity so we can show only relevant passages to the user
#sim_df.groupby(cols[0])[cols[2]].transform(lambda x: float((x >= 0.5).sum()))


sep = "----------------------------------------------------------------------"
for index, row in sim_df.iterrows():
    if row[2] > 0.5:
        print("\n{4}\n{0}: SUPPORTING SENTENCE (Similarity: {1})\n{4}\n\n{0}: THESIS SENTENCE:\n\n{2}\n\n{0}: REFERENCE SENTENCE:\n\n{3}\n\n{4}".format(index, row[2], row[0], row[1], sep))

def sent_words(sents, stoplist = stopwords.words("english")):
    stoplist.extend(string.punctuation)
    stoplist.extend(["``", "''", "\"\"", "e.g.", "etal"])
    return [word for sent in sents for word in nltk.word_tokenize(sent) if word not in stoplist]

t_words = filter(str.isalpha, sent_words(t_sents))
r_words = filter(str.isalpha, sent_words(r_sents))

top_n = 10
freq_cols = ["WORD", "FREQUENCY"]

t_freq = [word for word in Counter(t_words).most_common(top_n)]
r_freq = [word for word in Counter(r_words).most_common(top_n)]

t_freq_df = pd.DataFrame(t_freq, columns = freq_cols)
r_freq_df = pd.DataFrame(r_freq, columns = freq_cols)

print("\nTOP {} MOST FREQUENT WORDS IN THESIS:\n{}\n\n{}".format(top_n, t_freq_df, sep))
print("\nTOP {} MOST FREQUENT WORDS IN REFERENCE:\n{}\n".format(top_n, r_freq_df))
print(sep)

print("\nAVERAGE SIMILARITY:\t", sim_df[cols[2]].mean())
#print("THESIS PAGES:\t\t", t_page_num)
#print("REFERENCE PAGES:\t\t", r_page_num)




#target_abstract(t_pdf, r_pdf)


# Do each thesis sentence as its own thread to speed things up!

# TODO: we need to figure out the best search terms.
# We can probably achieve this by looking for the most frequently used phrases that
# are not "common" phrases or whatnot. Essentially, we want to analyse the entire
# document and figure out which words are the most relevant and tell us about the subject,
# then from there we want to search for reference materials using these documents

# There is no database API I can use to search however, but we can just paste search terms into
# Google Scholar and load up the page for the user; if we make an extension we can help the
# user search Google Scholar while still respecting the ToS.

# For now we'll just figure out the best search terms and present that to the user, who can then
# search on their own.




# TODO: we may want some way of ignoring references when trying to find most
# relevant search terms, as we seem to only find these terms when we do that



# Algorithm:
    # Make a list of all words in the document (different words that mean the same thing
    # will be counted as the same word). Delete stop words.
    # Count the frequency of the remaining words (and words with the same meaning).
    # The highest ranked words will be our search terms
    
# We may also want to search on phrases that pop up repeatedly.





# Test run results (from 26/07/2021)
# These appear to be references to the same works between the two papers
# It may be useful to use this to our advantage somehow, perhaps we can use it as
# an optimization method, or use it to help guide our search? This will also help
# if we need to train the machine, as we can use referenced materials to figure out
# how these references are used/how they support the document

# NOTE: I've removed the test results since they take up a lot of space and don't seem to be needed anymore

