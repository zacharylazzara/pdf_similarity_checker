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
    return [word for sent in sents for word in nltk.word_tokenize(sent) if word.lower() not in stoplist]

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





# Test run results (from 02/08/2021)
# These appear to be references to the same works between the two papers
# It may be useful to use this to our advantage somehow, perhaps we can use it as
# an optimization method, or use it to help guide our search? This will also help
# if we need to train the machine, as we can use referenced materials to figure out
# how these references are used/how they support the document





'''
Journal of Neurophysiology, 71, 1959–1975.

2544342: REFERENCE SENTENCE:

Journal of Consciousness Studies, 2(3),
     200–219.

----------------------------------------------------------------------

----------------------------------------------------------------------
2544354: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2544354: THESIS SENTENCE:

Journal of Neurophysiology, 71, 1959–1975.

2544354: REFERENCE SENTENCE:

Journal of Consciousness Studies, 17,
     7–65.

----------------------------------------------------------------------

----------------------------------------------------------------------
2544357: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2544357: THESIS SENTENCE:

Journal of Neurophysiology, 71, 1959–1975.

2544357: REFERENCE SENTENCE:

Journal of Consciousness Studies,
     19(7–8), 26–44.

----------------------------------------------------------------------

----------------------------------------------------------------------
2544376: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2544376: THESIS SENTENCE:

Journal of Neurophysiology, 71, 1959–1975.

2544376: REFERENCE SENTENCE:

Journal of Artificial General Intelligence, 4(3), 130–152.

----------------------------------------------------------------------

----------------------------------------------------------------------
2544390: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2544390: THESIS SENTENCE:

Journal of Neurophysiology, 71, 1959–1975.

2544390: REFERENCE SENTENCE:

International
     Journal of Machine Consciousness, 4(1).

----------------------------------------------------------------------

----------------------------------------------------------------------
2544396: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2544396: THESIS SENTENCE:

Journal of Neurophysiology, 71, 1959–1975.

2544396: REFERENCE SENTENCE:

Journal of Philosophy, 83, 291–295.

----------------------------------------------------------------------

----------------------------------------------------------------------
2544403: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2544403: THESIS SENTENCE:

Journal of Neurophysiology, 71, 1959–1975.

2544403: REFERENCE SENTENCE:

The Journal of Neuroscience, 28(12), 2959–2964.

----------------------------------------------------------------------

----------------------------------------------------------------------
2544441: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2544441: THESIS SENTENCE:

Journal of Neurophysiology, 71, 1959–1975.

2544441: REFERENCE SENTENCE:

Journal of Trauma, 63(5), 1010–1013.

----------------------------------------------------------------------

----------------------------------------------------------------------
2544499: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2544499: THESIS SENTENCE:

Journal of Neurophysiology, 71, 1959–1975.

2544499: REFERENCE SENTENCE:

Journal of Consciousness Studies, 6(1), 49–60.

----------------------------------------------------------------------

----------------------------------------------------------------------
2555511: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2555511: THESIS SENTENCE:

The Journal of Neuroscience, 20, 6594–6611.

2555511: REFERENCE SENTENCE:

Journal of Consciousness Studies, 2(3),
     200–219.

----------------------------------------------------------------------

----------------------------------------------------------------------
2555523: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2555523: THESIS SENTENCE:

The Journal of Neuroscience, 20, 6594–6611.

2555523: REFERENCE SENTENCE:

Journal of Consciousness Studies, 17,
     7–65.

----------------------------------------------------------------------

----------------------------------------------------------------------
2555526: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2555526: THESIS SENTENCE:

The Journal of Neuroscience, 20, 6594–6611.

2555526: REFERENCE SENTENCE:

Journal of Consciousness Studies,
     19(7–8), 26–44.

----------------------------------------------------------------------

----------------------------------------------------------------------
2555545: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2555545: THESIS SENTENCE:

The Journal of Neuroscience, 20, 6594–6611.

2555545: REFERENCE SENTENCE:

Journal of Artificial General Intelligence, 4(3), 130–152.

----------------------------------------------------------------------

----------------------------------------------------------------------
2555559: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2555559: THESIS SENTENCE:

The Journal of Neuroscience, 20, 6594–6611.

2555559: REFERENCE SENTENCE:

International
     Journal of Machine Consciousness, 4(1).

----------------------------------------------------------------------

----------------------------------------------------------------------
2555565: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2555565: THESIS SENTENCE:

The Journal of Neuroscience, 20, 6594–6611.

2555565: REFERENCE SENTENCE:

Journal of Philosophy, 83, 291–295.

----------------------------------------------------------------------

----------------------------------------------------------------------
2555572: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2555572: THESIS SENTENCE:

The Journal of Neuroscience, 20, 6594–6611.

2555572: REFERENCE SENTENCE:

The Journal of Neuroscience, 28(12), 2959–2964.

----------------------------------------------------------------------

----------------------------------------------------------------------
2555610: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2555610: THESIS SENTENCE:

The Journal of Neuroscience, 20, 6594–6611.

2555610: REFERENCE SENTENCE:

Journal of Trauma, 63(5), 1010–1013.

----------------------------------------------------------------------

----------------------------------------------------------------------
2555668: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2555668: THESIS SENTENCE:

The Journal of Neuroscience, 20, 6594–6611.

2555668: REFERENCE SENTENCE:

Journal of Consciousness Studies, 6(1), 49–60.

----------------------------------------------------------------------

----------------------------------------------------------------------
2556362: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2556362: THESIS SENTENCE:

http://dx.doi.org/10.1186/1471-2202-5-42.

2556362: REFERENCE SENTENCE:

The project’s initial goal is to scan, upload, and emulate a
complete mouse brain within 5 years (https://www.humanbrainproject.eu/strategic-
mouse-brain-data).

----------------------------------------------------------------------

----------------------------------------------------------------------
2556364: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2556364: THESIS SENTENCE:

http://dx.doi.org/10.1186/1471-2202-5-42.

2556364: REFERENCE SENTENCE:

The Human Brain Project aims to use the knowledge gained from
the mouse emulation to scan and upload parts of the human brain within 10 years
(https://www.humanbrainproject.eu/strategic-human-brain-data) and the ultimate
goal of the project is to emulate the complete human brain (http://tierra.aslab.upm.

----------------------------------------------------------------------

----------------------------------------------------------------------
2556375: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2556375: THESIS SENTENCE:

http://dx.doi.org/10.1186/1471-2202-5-42.

2556375: REFERENCE SENTENCE:

2000; http://
www.fhi.ox.ac.uk/brain-emulation-roadmap-report.pdf; Hayworth 2010).

----------------------------------------------------------------------

----------------------------------------------------------------------
2556377: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2556377: THESIS SENTENCE:

http://dx.doi.org/10.1186/1471-2202-5-42.

2556377: REFERENCE SENTENCE:

Uploading and Branching Identity                                                                       19
Hayworth 2012; http://www.brainpreservation.org/content/overview).

----------------------------------------------------------------------

----------------------------------------------------------------------
2556853: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2556853: THESIS SENTENCE:

http://dx.doi.org/10.1186/1471-2202-5-42.

2556853: REFERENCE SENTENCE:

http://plato.stanford.edu/archives/spr2014/
     entries/consciousness-temporal/
Efron, R. (1970).

----------------------------------------------------------------------

----------------------------------------------------------------------
2556871: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2556871: THESIS SENTENCE:

http://dx.doi.org/10.1186/1471-2202-5-42.

2556871: REFERENCE SENTENCE:

Essay published online at http://brainpreservation.org/content/killed-bad-philosophy
Hayworth, K. (2012).

----------------------------------------------------------------------

----------------------------------------------------------------------
2556898: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2556898: THESIS SENTENCE:

http://dx.doi.org/10.1186/1471-2202-5-42.

2556898: REFERENCE SENTENCE:

http://plato.stanford.edu/archives/fall2013/entries/functionalism/
Levy, J.

----------------------------------------------------------------------

----------------------------------------------------------------------
2556917: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2556917: THESIS SENTENCE:

http://dx.doi.org/10.1186/1471-2202-5-42.

2556917: REFERENCE SENTENCE:

http://plato.stanford.edu/archives/sum2010/
      entries/qualia-knowledge/
Olson, E. (2010).

----------------------------------------------------------------------

----------------------------------------------------------------------
2556921: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2556921: THESIS SENTENCE:

http://dx.doi.org/10.1186/1471-2202-5-42.

2556921: REFERENCE SENTENCE:

http://plato.stanford.edu/archives/win2010/entries/identity-personal/
Oncel, D., Demetriades, D., Gruen, P., et al.

----------------------------------------------------------------------

----------------------------------------------------------------------
2556952: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2556952: THESIS SENTENCE:

http://dx.doi.org/10.1186/1471-2202-5-42.

2556952: REFERENCE SENTENCE:

http://plato.stanford.edu/archives/win2012/entries/dualism/
Ruhnau, E. (1995).

----------------------------------------------------------------------

----------------------------------------------------------------------
2556968: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2556968: THESIS SENTENCE:

http://dx.doi.org/10.1186/1471-2202-5-42.

2556968: REFERENCE SENTENCE:

http://www.fhi.ox.ac.uk/brain-emulation-roadmap-report.pdf
Shoemaker, S. (1984).

----------------------------------------------------------------------

----------------------------------------------------------------------
2557501: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2557501: THESIS SENTENCE:

Zoccolan, D., Kouh, M., Poggio, T., & DiCarlo, J. J.

2557501: REFERENCE SENTENCE:

Coren, S., Ward, L., & Enns, J.

----------------------------------------------------------------------

----------------------------------------------------------------------
2557550: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2557550: THESIS SENTENCE:

Zoccolan, D., Kouh, M., Poggio, T., & DiCarlo, J. J.

2557550: REFERENCE SENTENCE:

Levin, J.

----------------------------------------------------------------------

----------------------------------------------------------------------
2557555: SUPPORTING SENTENCE (Similarity: 1.0)
----------------------------------------------------------------------

2557555: THESIS SENTENCE:

Zoccolan, D., Kouh, M., Poggio, T., & DiCarlo, J. J.

2557555: REFERENCE SENTENCE:

http://plato.stanford.edu/archives/fall2013/entries/functionalism/
Levy, J.

----------------------------------------------------------------------

TOP 10 MOST FREQUENT WORDS IN THESIS:
        WORD  FREQUENCY
0  Grossberg        406
1        The        405
2     visual        264
3     object        228
4  conscious        226
5   learning        223
6        Fig        222
7      model        204
8  attention        189
9   category        179

----------------------------------------------------------------------

TOP 10 MOST FREQUENT WORDS IN REFERENCE:
            WORD  FREQUENCY
0       identity        177
1          brain        137
2  consciousness        128
3     continuity         90
4         qualia         82
5            The         77
6          would         60
7      uploading         59
8      branching         53
9       personal         46

----------------------------------------------------------------------

AVERAGE SIMILARITY:	 0.9999997260945465'''
