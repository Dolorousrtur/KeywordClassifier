import numpy as np
from nltk import sent_tokenize, word_tokenize, RegexpTokenizer, WordNetLemmatizer, PerceptronTagger
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer

from utils_common import contains


def text2sents(text, lemmatize=False, stemmer=None):
    """
    converts a text into a list of sentences consisted of normalized words
    :param text: list of string to process
    :param lemmatize: if true, words will be lemmatized, otherwise -- stemmed
    :param stemmer: stemmer to be used, if None, PortedStemmer is used. Only applyed if lemmatize==False
    :return: list of lists of words
    """
    sents = sent_tokenize(text)

    tokenizer = RegexpTokenizer(r'\w+')

    if lemmatize:
        normalizer = WordNetLemmatizer()
        tagger = PerceptronTagger()
    elif stemmer is None:
        normalizer = PorterStemmer()
    else:
        normalizer = stemmer

    sents_normalized = []

    for sent in sents:
        sent_tokenized = tokenizer.tokenize(sent)
        if lemmatize:
            sent_tagged = tagger.tag(sent_tokenized)
            sent_normalized = [normalizer.lemmatize(w[0], get_wordnet_pos(w[1])) for w in sent_tagged]
        else:
            sent_normalized = [normalizer.stem(w) for w in sent_tokenized]

        sents_normalized.append(sent_normalized)
    return sents_normalized


def get_window(text, phrase, fr, to, outtype="list"):
    """
    returns a window of words by which a given phrase is surrounded in the text
    It is supposed that a phrase can occure no more that once in a sentence
    :param text: text string
    :param phrase: a string os words splitted by spaces
    :param fr: relative to the phrase position of windows' start (if -5, the window will contain 5 words before the phrase)
    :param to: relative to the phrase position of windows' end
    :param outtype: if 'list', a list of words in window will be returned;
    if 'text', a string of subsequent words with special tokens will be returned and sentences are splitted by '.'
    tokens:
    '\kw' -- keyword
    '\sst' -- start of a sentence
    '\sen' -- end of a sentence
    :return: list of words or a string according to outtype
    """
    if outtype not in ["list", "text"]:
        raise ValueError("outtype is either 'list' or 'text'")
    text = text.lower()
    sents = sent_tokenize(text)
    sents_tok = map(lambda x: filter(lambda y: any(c.isalpha() for c in y), word_tokenize(x)), sents)
    window = []

    ph_splitted = phrase.split()
    for sent in sents_tok:
        bounds = contains(sent, ph_splitted)
        if bounds:
            window_st = max(0, bounds[0]+fr)
            window_end = min(len(sent), bounds[1]+to)
            if bounds[0]+fr < 0:
                window.append('/SST')
            kw_added = False
            for i in range(window_st, window_end):
                if i not in range(bounds[0], bounds[1]):
                        window.append(sent[i])
                elif outtype == 'text' and not kw_added:
                    window.append('/KW')
                    kw_added = True
            if len(sent) < bounds[1]+to:
                window.append('/SEN')
    if outtype == 'list':
        return window
    elif outtype == 'text':
        return ' '.join(window)


def phvec_distance(words, target, model):
    """
    a word2vec distances between parts of a phrase (represented as a list of words) and a word in a given model
    distance(words, target)
    :param words: list of words comprising the keyword
    :param target: target word
    :param model: word2vec model
    :return: list of distances from parts of a keyword to the target.
    If the whole keyword occured in model, its length == 1
    """
    size = len(words)
    similarities = []
    for w in words:
        if w in model:
            similarities.append(model.similarity(w, target))
        else:
            similarities.append(0)
    if size == 1:
        return similarities
    else:
        for i in range(size-1):
            combined = '_'.join(words[i:i+2])
            if combined in model:
                words[i:i+2] = (combined,)
                return phvec_distance(words, target, model)
    return similarities


def phrase2word_distance(phrase, target, model):
    """
    word2vec distance between a keyword and another word
    distances from parts of a keyword are used if the whoke word didn't occured
    :param phrase: str
    :param target: str
    :param model: gensim.model.Word2Vec
    :return: distance (float)
    """
    keyword_parts = phrase.split()
    sim = np.array(phvec_distance(keyword_parts, target, model)).mean()
    return sim


def wordcount(phrase):
    """
    number of words in a phrase
    """
    return len(phrase.split())


def occurences_map(phrases):
    """
    counts occurences of each word in a list of phrases
    :param phrases: list of strings
    :return: dictionary {word : n_occurences}
    """
    word_dict = {}
    for ph in phrases:
        words = ph.lower().split()
        for w in words:
            if w not in word_dict:
                word_dict[w] = 1
            else: word_dict[w] += 1
    return word_dict


def get_wordnet_pos(treebank_tag):
    """
    converts nltk pos tag into wordnet format
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN



