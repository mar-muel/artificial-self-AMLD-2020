import logging
import os
import glob
import pandas as pd
import unicodedata
import re
from tqdm import tqdm
import numpy as np
import random
import torch

logger = logging.getLogger(__file__)
ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens': ('<speaker1>', '<speaker2>')}
CACHE_PATH = 'cached_input_task2.txt'

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


def read_conversation_data(data_path):
    """Read conversational data from either Chatistics or other data sources (in JSON) and returns Dataframe"""
    f_path = None
    if data_path is None:
        # infer file name
        data_folder = 'data'
        input_files = glob.glob(os.path.join(data_folder, '*.json'))
        if len(input_files) == 0:
            raise Exception(f'No files found in {data_folder}')
        elif len(input_files) > 1:
            raise Exception(f'Multiple files found in {data_folder}. Specify file with chatistics_data_path argument.')
        f_path = input_files[0]
    elif data_path is not None and os.path.isfile(data_path):
        f_path = data_path
    else:
        raise FileNotFoundError(f'Input data {data_path} could not be found')
    df = pd.read_json(f_path, encoding='utf-8')
    return df

def generate_input_task2(data_path, speaker1_tag='<speaker1>', speaker2_tag='<speaker2>'):
    """Generate input data for task 2"""
    # read conversation data
    df = read_conversation_data(data_path)
    # group messages by sender and generate output text file
    min_num_interactions_per_conversation = 10
    num_interactions = 0
    prev_sender_tag = None
    output = ''
    for conversation_name, g in tqdm(df.groupby('conversationWithName'), total=len(df['conversationWithName'].unique())):
        # only consider conversations between 2 people
        if len(g['senderName'].unique()) == 2 and len(g) > min_num_interactions_per_conversation:
            # sort by time
            g = g.sort_values('timestamp', ascending=True)
            for i, row in g.iterrows():
                sender_tag = speaker1_tag if row.outgoing else speaker2_tag
                if prev_sender_tag is None:
                    # beginning of chat with person 
                    prev_sender_tag = sender_tag
                    current_messages = [remove_control_characters(row.text)]
                    continue
                if prev_sender_tag == sender_tag:
                    # concatenate/group messages by the same sender
                    current_messages.append(remove_control_characters(row.text))
                else:
                    # dump previous messsages
                    output += '{} {}\n'.format(prev_sender_tag, ' '.join(current_messages))
                    num_interactions += 1
                    # new response by other
                    prev_sender_tag = sender_tag
                    current_messages = [remove_control_characters(row.text)]
            if len(current_messages) > 0:
                output += '{} {}\n'.format(prev_sender_tag, ' '.join(current_messages))
                num_interactions += 1
    # write output data
    logger.info(f'Writing input file with {num_interactions:,} interactions to {CACHE_PATH}...')
    with open(CACHE_PATH, 'w') as f:
        f.write(output)

def get_input_task2(data_path, speaker1_tag='<speaker1>', speaker2_tag='<speaker2>', use_cache=True):
    """Load input data for task 2"""
    if not os.path.isfile(CACHE_PATH) or not use_cache:
        generate_input_task2(data_path, speaker1_tag='<speaker1>', speaker2_tag='<speaker2>')
    logger.info(f'Reading cached input file from {CACHE_PATH}...')
    with open(CACHE_PATH, 'r') as f:
        output = f.read()
    return output

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)
