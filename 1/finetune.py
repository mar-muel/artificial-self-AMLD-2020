import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
log = logging.getLogger(__name__)

import gpt_2_simple.gpt_2 as gpt2
import os
import requests
import glob
import pickle
import pandas as pd
import re
import unicodedata
import argparse


DATA_PATH = 'data'
INPUT_FILE = 'input.txt'
INPUT_PATH = os.path.join(DATA_PATH, INPUT_FILE)

def fine_tune(run_name, model_name='124M'):
    print(f'Run fine-tuning for run {run_name} using GPT2 model {model_name}...')
    if not os.path.isdir(os.path.join("models", model_name)):
        log.info(f"Downloading {model_name} model...")
        gpt2.download_gpt2(model_name=model_name)
    sess = gpt2.start_tf_sess()
    gpt2.finetune(sess, INPUT_PATH, model_name=model_name, run_name=run_name, steps=-1, sample_every=10, save_every=10)

def remove_control_characters(s):
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    control_char_regex = r'[\r\n\t]+'
    # replace \t, \n and \r characters by a whitespace
    s = re.sub(control_char_regex, ' ', s)
    # replace HTML codes for new line characters
    s = s.replace('&#13;', '').replace('&#10;', '')
    # removes all other control characters and the NULL byte (which causes issues when parsing with pandas)
    return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")

def create_input_data(chatistics_data_folder):
    if not os.path.isdir(DATA_PATH):
        os.makedirs(DATA_PATH)
    pickles = glob.glob(os.path.join(chatistics_data_folder, '*.pkl'))
    dfs = []
    for pickle_path in pickles:
        with open(pickle_path, 'rb') as f:
            df = pickle.load(f)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    text_data = ''
    previous_was_question = False
    previous_was_you = False
    count = 0
    for conversation, g in df.groupby('conversationWithName'):
        # only consider conversations between 2 people
        if len(g['senderName'].unique()) == 2:
            for i, row in g.iterrows():
                text = remove_control_characters(row.text)
                if len(text) == 0:
                    continue
                if row.outgoing and not previous_was_you:
                    text_data += f'bot: {text}\n'
                    count += 1
                    previous_was_you = True
                elif previous_was_you:
                    text_data += f'you: {text}\n'
                    count += 1
                    previous_was_you = False
                else:
                    continue

    log.info(f'Writing total of {count:,} messages to {INPUT_PATH}...')
    with open(INPUT_PATH, 'w') as f:
        f.write(text_data)

def main():
    parser = argparse.ArgumentParser(
        description="Finetune GPT2 on chatlogs parsed with Chatistics"
    )
    parser.add_argument('-d', '--data-path', dest='data_path', help="Input data path to folder of Chatistics pickle files", default='./chatistics_data')
    parser.add_argument('-r', '--run-name', dest='run_name', help="Run name", default='run1')
    args = parser.parse_args()
    create_input_data(args.data_path)
    fine_tune(args.run_name)

if __name__ == "__main__":
    main()
