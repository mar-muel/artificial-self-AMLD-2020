import json
import logging
import os
import tarfile
import tempfile
import socket
import glob
import torch
import pandas as pd
import pickle
import unicodedata
import re
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm
from transformers import cached_path
import random

CHATISTICS_CONV_DATA = 'chatistics_conversation_data.json'
logger = logging.getLogger(__file__)

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


def generate_chatistics_conversation_data(chatistics_data_path):
    """Create conversation data from chatistics pickles"""
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
    with open(CHATISTICS_CONV_DATA, 'w') as f:
        json.dump(data, f)

def get_chatistics_dataset(tokenizer, chatistics_data_path, create_validataion_set=False):
    """Get tokenized chatlog dataset"""
    def merge_messages(messages, max_num_words=200):
        full_message = ''
        for m in messages:
            m = m.strip()
            if len(m) > 0:
                full_message += '\n' + m
        return full_message.strip()

    # check for cached data
    cached_path = 'cached_chatistics_personachat.pkl'
    if os.path.isfile(cached_path):
        logger.info('Reading cached data...')
        with open(cached_path, 'rb') as f:
            data = pickle.load(f)
        return data
    # read data
    if not os.path.isfile(CHATISTICS_CONV_DATA):
        generate_chatistics_conversation_data(chatistics_data_path)
    with open(CHATISTICS_CONV_DATA, 'r') as f:
        conv_data = json.load(f)
    # retrieve_all_interactions by person1 in order to generate distractor messages
    all_interactions_person1 = []
    for _, conversations in conv_data.items():
        for conversation in conversations:
            for interaction in conversation:
                if interaction['senderType'] == 'person1':
                    message = merge_messages(interaction['messages'])
                    all_interactions_person1.append(message)
    num_interactions_person1 = len(all_interactions_person1)
    if num_interactions_person1 <= 20:
        raise Exception(f'Person1 needs to have at least 20 interactions')
    # create personachat data structure
    data = []
    num_utterances = 0
    for conversation_with_name, conversations in conv_data.items():
        utterances = []
        for conversation in conversations:
            history = []
            for interaction in conversation:
                # we are trying to learn person1.
                # All interactions have the shape:
                # person2, person1, ..., person2, <candidate> (candidate can be either a distractor or the true answer of person1)
                if interaction['senderType'] == 'person1':
                    # person 1
                    if len(history) == 0:
                        # skip - cannot start with person1
                        continue
                    # Generate candidates for problem
                    candidates = []
                    true_answer = merge_messages(interaction['messages'])
                    while len(candidates) < 19:
                        cand = all_interactions_person1[random.randint(0, num_interactions_person1 - 1)]
                        if cand != true_answer:
                            candidates.append(cand)
                    candidates.append(true_answer) # last candidate is true one
                    # add new problem
                    utterances.append({'history': history[-8:], 'candidates': candidates})
                    num_utterances += 1
                else:
                    # person 2
                    message = merge_messages(interaction['messages'])
                    history.append(message)
        data.append({'personality': [conversation_with_name], 'utterances': utterances})
    logger.info(f'Generated a total of {num_utterances:,} utterances from {len(data):,} conversations...')
    # split in train/validation
    random.shuffle(data)
    if create_validataion_set:
        split_int = int(len(data)*.8)
        dataset = {'train': data[:split_int], 'valid': data[split_int:]}
    else:
        dataset = {'train': data, 'valid': []}

    # tokenizing
    logger.info('Tokenizing...')
    def tokenize(obj):
        if isinstance(obj, str):
            tokens = tokenizer.tokenize(obj)
            return tokenizer.convert_tokens_to_ids(tokens)
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)
    dataset = tokenize(dataset)
    logger.info(f'Writing cached file {cached_path}...')
    with open(cached_path, 'wb') as f:
        pickle.dump(dataset, f)
    return dataset

def make_logdir(model_name: str):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    # Code copied from ignite repo
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(
        'runs', current_time + '_' + socket.gethostname() + '_' + model_name)
    return logdir
