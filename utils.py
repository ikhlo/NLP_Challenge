import json
import numpy as np


def process_json(json_data):
    """
    Function to do some basic procession on text as removing starting and
    ending whitespace, retrieve text from their list where they are stored
    and replace null text by ''.

    Args:
      (Dict) json_data: The JSON dict to be processed
    """
    for individual in json_data:
        for day_of_speech in individual['speech']:
            for bank in day_of_speech.keys():
                day_of_speech[bank] = '' if len(day_of_speech[bank]) == 0\
                    else day_of_speech[bank][0].strip('\" ')


def get_list_speeches(json_data, bank='FED'):
    """
    Get the list of unique speechs present in the JSON data

    Args:
      (Dict) json_data: The JSON dict data.
      (String) bank: The name of the bank's speeches we are working on.

    Return:
      A List of size N containing unique speechs for the chosen bank.
    """
    N = len(json_data)
    T = 20

    speech_set = set()
    speech_list = []

    for sample in range(N):
        for day in range(T):
            speech = json_data[sample]['speech'][day][bank]
            if speech not in speech_set:
                speech_set.add(speech)
                speech_list.append(speech)

    return speech_list


def get_languages(speech_list):
    """
    For a list of speechs, define the language of each speech and store it inside
    a numpy array.

    Args:
      (List) speech_list: All N-unique speeches.

    Return:
      A (N, ) numpy.ndarray with object dtype containing at index i the language
      of the i-th speech of the speeches' list.
    """
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0

    lang = []
    for speech in speech_list:
        if speech = '':
            lang.append('en')
        else:
            lang.append(detect(speech))
    return np.array(lang)


def keep_english_speeches(json_data):
    """
    A function to only retrieve speeches that are in english.

    Args:
      (Dict) json_data: The JSON dict data.

    Return:
      (List) ecb_speech: All english speeches from ECB bank.
      (List) fed_speech: All english speeches from FED bank.
    """
    ecb_speech = get_list_speeches(json_data, bank='ECB')
    lang_ecb = get_languages(ecb_speech)
    ecb_speech = np.array(ecb_speech)[lang_ecb == 'en'].tolist()

    fed_speech = get_list_speeches(json_data, bank='FED')
    lang_fed = get_languages(fed_speech)
    fed_speech = np.array(fed_speech)[lang_fed == 'en'].tolist()

    return ecb_speech, fed_speech


def build_dataset(json_data, ecb_list, fed_list, labels=True):
    """
    A function to build dataset from JSON dict and lists of speeches.

    Args:
      (Dict) json_data: The JSON dict data.
      (List) ecb_list: All unique speeches from ECB bank.
      (List) fed_list: All unique speeches from FED bank.
      (Boolean) labels: If True, return the labels of each individual.

    Return:
      (numpy.ndarray) X:
      An array of size (N, 3, 20). For each sample, there is the stock index
      values, the speech index for ECB and FED bank on 20 days.

      (numpy.ndarray) y_clf:
      An array of size (N, 1). It contains the classification label for each
      sample.

      (numpy.ndarray) y_reg:
      An array of size (N, 1). It contains the regression label for each sample.

      where N is the number of samples in the JSON dict data.
    """
    N = len(json_data)
    T = 20
    banks = ['ECB', 'FED']

    X = np.empty((N, 3, 20))
    if labels:
        y_clf = np.empty((N, 1))
        y_reg = np.empty((N, 1))

    ecb_dict = {speech: i for i, speech in enumerate(ecb_list)}
    fed_dict = {speech: i for i, speech in enumerate(fed_list)}

    for i in range(len(json_data)):
        idx_ecb = []
        idx_fed = []
        for day in range(T):
            for b in banks:
                speech = json_data[i]['speech'][day][b]
                if speech in eval(b.lower() + "_dict").keys():
                    eval(
                        "idx_" +
                        b.lower()).append(
                        eval(
                            b.lower() +
                            "_dict")[speech])
                else:
                    eval(
                        "idx_" +
                        b.lower()).append(
                        eval(
                            b.lower() +
                            "_dict")[''])

        X[i, 0, :] = np.array(json_data[i]['stock'])
        X[i, 1, :] = np.array(idx_ecb)
        X[i, 2, :] = np.array(idx_fed)

        if labels:
            y_clf[i, 0] = data[i]['target_classif']
            y_reg[i, 0] = data[i]['target_reg']

        if labels:
            return X, y_clf, y_reg
        else:
            return X
