import json
import argparse
import os
from utils import process_json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datapath", type=str, default = 'data/',
                help="Path to training data.")
    parser.add_argument("-sp", "--split", type=str, default = 'test',
                help="Split: dev or test.")
    parser.add_argument("-s", "--savepath", type=str, default = 'res/answer/',
                help="Path to save the results.")
    parser.add_argument("-i", "--index", type=str, default = 'all',
                help="Which index data to evaluate. Use 'all' for all indexes \
                    in the training data directory.")

    args = parser.parse_args()

    split = args.split
    if args.index =='all':
        indexes = [index.split('.')[0] for index in 
                    os.listdir(f'{args.datapath}/{split}')]
    else:
        indexes = [args.index]

    for index in indexes:
        split_path = f'{args.datapath}/{split}/{index}.json'
        train_path = f'{args.datapath}/train/{index}.json'
        save_path = os.path.join(args.savepath, index)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        with open(train_path, 'r') as fp:
            train_data = json.load(fp)
            process_json(train_data)
        with open(split_path, 'r') as fp:
            split_data = json.load(fp)
            process_json(split_data)

        pred_reg, pred_classif = create_baseline(data)

        with open(os.path.join(save_path, 'pred_reg.txt'),'w') as f:
            f.write('\n'.join(list(map(str, pred_reg))))

        with open(os.path.join(save_path, 'pred_classif.txt'),'w') as f:
            f.write('\n'.join(list(map(str, pred_classif))))