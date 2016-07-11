from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag.perceptron import PerceptronTagger


def pt_convert(nltk_postag):
    if nltk_postag.startswith('J'):
        return 'a'
    if nltk_postag.startswith('V'):
        return 'v'
    if nltk_postag.startswith('R'):
        return 'r'
    return 'n'

def text2sents(text):
    sents = sent_tokenize(text)

    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()

    sents_normalized = []

    tagger = PerceptronTagger()

    for sent in sents:
        sent_tokenized = tokenizer.tokenize(sent)
        sent_tagged = tagger.tag(sent_tokenized)
        sent_normalized = [lemmatizer.lemmatize(w[0], pt_convert(w[1])) for w in sent_tagged]
        sents_normalized.append(sent_normalized)
    return sents_normalized