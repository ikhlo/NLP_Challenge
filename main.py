import json
import argparse
from pathlib import Path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datapath", type=str, default = 'data/',
                help="Path to training data.")
    parser.add_argument("-sp", "--split", type=str, default = 'dev',
                help="Split: dev or test.")
    parser.add_argument("-s", "--savepath", type=str, default = 'res/answer/',
                help="Path to save the results.")
    parser.add_argument("-i", "--index", type=str, default = 'all',
                help="Which index data to evaluate. Use 'all' for all indexes in the training data directory.")

    args = parser.parse_args()