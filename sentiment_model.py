from transformers import pipeline


def convert_sentiment(sentiment_outputs):
    labels = []
    for dico in sentiment_outputs:
        if dico['label'] == 'neutral':
            labels.append(0)
        elif dico['label'] == 'positive':
            labels.append(1)
        else:
            labels.append(-1)
    return labels


def get_speech_sentiment(ecb_speeches, fed_speeches, use_gpu=False):
    if use_gpu:
        device = 0
    else:
        device = -1
    model = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    clf = pipeline(
        "sentiment-analysis",
        model=model,
        truncation=True,
        device=device)

    ecb_speeches_sentiment = convert_sentiment(clf(ecb_speeches))
    fed_speeches_sentiment = convert_sentiment(clf(fed_speeches))

    return ecb_speeches_sentiment, fed_speeches_sentiment
