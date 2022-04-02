from pathlib import Path
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
        day_of_speech[bank] = '' if len(day_of_speech[bank])==0 else day_of_speech[bank][0].strip('\" ')


def get_list_speeches(json_data, bank='FED'):
  """
  Get the list of unique speechs present in the JSON data
  
  Args:
    (Dict) json_data: The JSON dict data.
    (String) bank: The name of the bank's speeches we are working on.

  Return:
    A List containing unique speechs for the chosen bank. 
  """
  N = len(data)
  T = 20
    
  speech_set = set()
  speech_list = []

  for sample in range(N):
    for day in range(T):
      speech = data[sample]['speech'][day][bank]
      if speech not in speech_set:
        speech_set.add(speech)
        speech_list.append(speech)
  
  return speech_list


def load_data(path_to_data, get_dev_data = False):
    train_vix_json = Path(path_to_data) / 'train/VIX_1w.json'
    train_eur_json = Path(path_to_data) / 'train/EURUSDV1M_1w.json'

    with open(train_vix_json, 'r') as json_file:
    train_vix = json.load(json_file)

    with open(train_eur_json, 'r') as json_file:
    train_eur = json.load(json_file)

    if get_dev_data:
        val_vix_json = Path(path_to_data) / 'dev/VIX_1w.json'
        val_eur_json = Path(path_to_data) / 'dev/EURUSDV1M_1w.json'

        with open(val_vix_json, 'r') as json_file:
        valid_vix = json.load(json_file)

        with open(val_eur_json, 'r') as json_file:
        valid_eur = json.load(json_file)
        
        return train_vix, valid_vix, train_eur, valid_eur
    
    return train_vix, train_eur

def get_list_speeches(data):
    N = len(data)
    T = 20
    
    ecb_speech = ['<NA>']
    fed_speech = ['<NA>']
    banks = ['ECB', 'FED']

    for sample in range(N):
      for day in range(T):
        for b in banks:
          speech = data[sample]['speech'][day][b]
          if (len(speech) != 0) and (speech not in eval(b.lower()+"_speech")):
            eval(b.lower()+"_speech").append(speech[0])
    return fed_speech, ecb_speech

def build_dataset(data_vix, data_eur, labels=True):
    N = len(data_vix)
    T = 20
    banks = ['ECB', 'FED']
    X_stock = np.empty((N, 2))
    X_speech = np.empty((N, 2))
    y = np.empty((N, 4))
    
    fed_speech, ecb_speech = get_list_speeches(data_vix)

    for i in range(len(data_vix)):
        idx_ecb = []
        idx_fed = []
        for day in range(T):
          for b in banks:
            speech = data_vix[i]['speech'][day][b]
            if (len(speech) == 0) or (speech[0] not in eval(b.lower()+"_speech")):
              eval("idx_"+b.lower()).append(0)
            else:
              eval("idx_"+b.lower()).append(eval(b.lower()+"_speech").index(speech[0]))

        X_stock[i, 0] = np.array(data_vix[i]['stock'])
        X_stock[i, 1] = np.array(data_eur[i]['stock'])
        X_speech[i, 0] = np.array(idx_fed)
        X_speech[i, 1] = np.array(idx_ecb)
        
        if labels:
            y[i, 0] = data_vix[i]['target_reg']
            y[i, 1] = data_vix[i]['target_classif']
            y[i, 2] = data_eur[i]['target_reg']
            y[i, 3] = data_eur[i]['target_classif']
    
    return (X_stock, X_speech), y

def detect_language(list_of_speechs):
    lang = []
    for speech in speech_list:
        lang.append(detect(speech))
    lang[0] = 'en'
    return lang