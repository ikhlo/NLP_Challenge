import json
import argparse
import os

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sentiment_model import get_speech_sentiment
from utils import build_dataset, concatenate_features, from_data_to_X, process_json, keep_english_speeches
from summarization import summarize

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datapath", type=str, default='data/',
                        help="Path to training data.")
    parser.add_argument("-sp", "--split", type=str, default='test',
                        help="Split: dev or test.")
    parser.add_argument("-s", "--savepath", type=str, default='res/answer/',
                        help="Path to save the results.")
    parser.add_argument("-ntop", "--n_top_sent", type=int, default=8,
                        help="Number of sentences to keep for summarization.")
    parser.add_argument(
        "-gpu",
        "--gpu",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use GPU or not for sentiment classification.")
    parser.add_argument(
        "-i",
        "--index",
        type=str,
        default='all',
        help="Which index data to evaluate. Use 'all' for all indexes \
                    in the training data directory.")

    args = parser.parse_args()

    split = args.split
    if args.index == 'all':
        indexes = [index.split('.')[0] for index in
                   os.listdir(f'{args.datapath}/{split}')]
    else:
        indexes = [args.index]

    for index in indexes:
        print(f"WORKING WITH {index}")
        split_path = f'{args.datapath}/{split}/{index}.json'
        train_path = f'{args.datapath}/train/{index}.json'
        save_path = os.path.join(args.savepath, index)
        os.makedirs(save_path, exist_ok=True)

        print("Loading JSON files...", end="")
        with open(train_path, 'r') as fp:
            train_data = json.load(fp)
            process_json(train_data)
        with open(split_path, 'r') as fp:
            split_data = json.load(fp)
            process_json(split_data)
        print("Done")

        print("Retrieving english speeches...", end="")
        train_ecb_speech, train_fed_speech = keep_english_speeches(train_data)
        split_ecb_speech, split_fed_speech = keep_english_speeches(split_data)
        print("Done")

        X_train, y_clf, y_reg = build_dataset(
            train_data, train_ecb_speech, train_ecb_speech)
        X_split = build_dataset(
            split_data,
            split_ecb_speech,
            split_fed_speech,
            labels=False)

        print("Summarize all speeches...", end="")
        train_ecb_summarized = [
            summarize(
                speech,
                args.n_top_sent) for speech in train_ecb_speech]
        train_fed_summarized = [
            summarize(
                speech,
                args.n_top_sent) for speech in train_fed_speech]

        split_ecb_summarized = [
            summarize(
                speech,
                args.n_top_sent) for speech in split_ecb_speech]
        split_fed_summarized = [
            summarize(
                speech,
                args.n_top_sent) for speech in split_fed_speech]
        print("Done")

        print("Get speeches sentiment...", end="")
        train_ecb_sentiment, train_fed_sentiment = get_speech_sentiment(
            train_ecb_summarized, train_fed_summarized, use_gpu=args.gpu)
        split_ecb_sentiment, split_fed_sentiment = get_speech_sentiment(
            split_ecb_summarized, split_fed_summarized, use_gpu=args.gpu)
        print("Done")

        print("Start modelling and prediction...", end="")
        X_train = concatenate_features(
            X_train, train_ecb_sentiment, train_fed_sentiment)
        X_split = concatenate_features(
            X_split, split_ecb_sentiment, split_fed_sentiment)

        # Classifier
        clf = LogisticRegression(
            penalty='l2',
            class_weight=None,
            solver='lbfgs',
            C=0.5,
            max_iter=10000,
            random_state=44)
        clf.fit(X_train, y_clf)
        pred_classif = clf.predict(X_split).flatten().tolist()

        # Regressor
        n_poly = 1
        poly = PolynomialFeatures(degree = n_poly, include_bias=True)
        regr = make_pipeline(StandardScaler(with_mean=True), Ridge(alpha = 1))

        X_train_reg = from_data_to_X(train_data)
        poly_features_train = poly.fit_transform(X_train_reg)
        regr.fit(poly_features_train, y_reg)

        X_split_reg = from_data_to_X(split_data)
        poly_features_test = poly.fit_transform(X_split_reg)
        pred_reg = regr.predict(poly_features_test)

        # Write outputs
        with open(os.path.join(save_path, 'pred_reg.txt'), 'w') as f:
            f.write('\n'.join(list(map(str, pred_reg))))

        with open(os.path.join(save_path, 'pred_classif.txt'), 'w') as f:
            f.write('\n'.join(list(map(str, pred_classif))))
        
        print("Done")
