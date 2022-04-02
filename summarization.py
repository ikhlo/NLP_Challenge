import heapq
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')


def weight_freq(speech):
    stop_words = stopwords.words('english')
    word_frequencies = {}
    for word in word_tokenize(speech):
        if word not in stop_words:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequncy)

    return word_frequencies


def sent_score(sent_speech, word_freq):
    sentence_scores = {}
    for sent in sent_speech:
        for word in word_tokenize(sent.lower()):
            if word in word_freq.keys():
                if len(sent.split(' ')) < 50:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_freq[word]
                    else:
                        sentence_scores[sent] += word_freq[word]
    return sentence_scores


def get_summary(sent_scores, top_sent):
    summary_sentences = heapq.nlargest(top_sent, sent_scores,
                                       key=sent_scores.get)
    summary = ' '.join(summary_sentences)
    return summary


def summarize(speech, top_sent=8):
    if speech == '':
        return speech
    word_freq = weight_freq(speech)
    scores_sent = sent_score(sent_tokenize(speech), word_freq)
    summ = get_summary(scores_sent, top_sent)

    return summ
