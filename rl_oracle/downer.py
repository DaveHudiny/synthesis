# This script is used to decrease the value of the data in the pickle file by 1
# If the value is less than 2, it will be set to 1
# Usage: python downer.py --filename data.pkl
# Author: David Hudak

import pickle

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def downer(filename='data.pkl'):
    data_dict = load_pickle(filename)
    for key in data_dict:
        data_dict[key] = data_dict[key] - 1 if data_dict[key] >= 2 else 1
    save_pickle(data_dict, filename)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='data.pkl')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    downer(args.filename)
    print("Downer done!")