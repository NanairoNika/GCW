import nltk
import pymorphy2
from tqdm.notebook import tqdm
from collections import defaultdict
from nltk.tokenize import word_tokenize

nltk.download('punkt')
morph = pymorphy2.MorphAnalyzer()

def verb_extractor(dataset):
    verbs = defaultdict(list)
    texts = dataset[3].apply(word_tokenize).tolist()
    for i in tqdm(range(len(texts))):
        for word in texts[i]:
            lemm = morph.parse(word)[0]
            ps = lemm.tag.POS
            if ps == 'INFN' or ps == 'VERB':
                verbs[lemm.normal_form].append(word)
    return verbs

def features_extractor(values, fd, sd):
    genddict = defaultdict(list)
    numbdict = defaultdict(list)
    perdict = defaultdict(list)
    tensedict = defaultdict(list)

    for i in tqdm(range(len(values))):
        if values[i] in list(fd.keys()):
            for form in fd[values[i]]:
                lemm = morph.parse(form)[0]
                numbdict[values[i]].append(lemm.tag.number)
                genddict[values[i]].append(lemm.tag.gender)
                perdict[values[i]].append(lemm.tag.person)
                tensedict[values[i]].append(lemm.tag.tense)

        if values[i] in list(sd.keys()):
            for form in sd[values[i]]:
                lemm = morph.parse(form)[0]
                numbdict[values[i]].append(lemm.tag.number)
                genddict[values[i]].append(lemm.tag.gender)
                perdict[values[i]].append(lemm.tag.person)
                tensedict[values[i]].append(lemm.tag.tense)
                
    return genddict, numbdict, tensedict, perdict