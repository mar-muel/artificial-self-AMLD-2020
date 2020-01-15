import json
import logging
import os
import glob
import pandas as pd
import pickle
import unicodedata
import re
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import random
import torch

logger = logging.getLogger(__file__)
ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens': ('<speaker1>', '<speaker2>')}

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

def get_chatistics_conversation_data(chatistics_data_path, use_cache=True, cache_path='chatistics_conversation_data.json'):
    """Create conversation data from chatistics pickles"""
    if use_cache:
        if os.path.isfile(cache_path):
            logger.info('Reading cached conversation data...')
            with open(cache_path, 'r') as f:
                data = json.load(f)
            return data
    logger.info(f"Creating conversation data from Chatistics chat logs from {chatistics_data_path}")
    # load pickle files into df
    pickles = glob.glob(os.path.join(chatistics_data_path, '*.pkl'))
    dfs = []
    for pickle_path in pickles:
        with open(pickle_path, 'rb') as f:
            df = pickle.load(f)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    # generate conversations
    new_conversation_delay_hours = 24
    data = defaultdict(list)
    min_num_interactions_per_conversation = 10
    num_interactions = 0
    for conversation_name, g in tqdm(df.groupby('conversationWithName'), total=len(df['conversationWithName'].unique())):
        # only consider conversations between 2 people
        if len(g['senderName'].unique()) == 2 and len(g) > 10:
            time_last_message = datetime(1970, 1, 1)
            conversation = []
            g = g.sort_values('timestamp', ascending=True)
            for i, row in g.iterrows():
                timestamp = pd.to_datetime(row.timestamp, unit='s')
                if timestamp > timedelta(hours=new_conversation_delay_hours) + time_last_message:
                    # either beginning of chat with person or no interactions for a while (assume new conversation)
                    if len(conversation) > min_num_interactions_per_conversation:
                        data[conversation_name].append(conversation)
                        num_interactions += len(conversation)
                    # wipe previous conversation data, start new interaction
                    prevSender = row.senderName
                    current_messages = [remove_control_characters(row.text)]
                    time_last_message = timestamp
                    conversation = []
                    continue
                time_last_message = timestamp
                if prevSender == row.senderName:
                    # concatenate/group messages by the same sender
                    current_messages.append(remove_control_characters(row.text))
                else:
                    # dump previous messsages
                    prevSenderType = 'person2' if row.outgoing else 'person1' # if current is outgoing previous was person 2
                    conversation.append({'messages': current_messages, 'sender': prevSender, 'senderType': prevSenderType})
                    # response by other
                    prevSender = row.senderName
                    current_messages = [remove_control_characters(row.text)]
            # Reached end of interactions. Dump leftover interactions
            if len(conversation) > min_num_interactions_per_conversation:
                data[conversation_name].append(conversation)
                num_interactions += len(conversation)
    logger.info(f'Generated {len(data.keys()):,} conversations with a total of {num_interactions:,} interactions...')
    with open(cache_path, 'w') as f:
        json.dump(data, f)
    return data

def get_input_task2(args, speaker1_tag='<speaker1>', speaker2_tag='<speaker2>', use_cache=True):
    """Generate input data for task 2"""
    f_path = os.path.join('data', 'input_task2.txt')
    if os.path.isfile(f_path) and use_cache:
        logger.info('Input data already present.')
        with open(f_path, encoding="utf-8") as f:
            output = f.read()
        return output
    data = get_chatistics_conversation_data(args.chatistics_data_path, use_cache=use_cache)
    output = ''
    num_lines = 0
    for converation_with_name, conversations in data.items():
        for conversation in conversations:
            for interaction in conversation:
                speaker_tag = speaker1_tag
                if interaction['senderType'] == 'speaker2':
                    speaker_tag = speaker2_tag
                output += '{} {}\n'.format(speaker_tag, ' '.join(interaction['messages']))
                num_lines += 1
    # write output data
    logger.info(f'Writing input data ({num_lines:,} lines) to {f_path}...')
    with open(f_path, 'w') as f:
        f.write(output)
    return output

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

