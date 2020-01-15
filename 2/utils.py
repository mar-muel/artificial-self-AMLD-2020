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

def get_chatistics_conversation_data(chatistics_data_path, use_cache=True):
    """Create conversation data from chatistics pickles"""
    if use_cache:
        if os.path.isfile(CHATISTICS_CONV_DATA):
            logger.info('Reading cached conversation data...')
            with open(CHATISTICS_CONV_DATA, 'r') as f:
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
    with open(CHATISTICS_CONV_DATA, 'w') as f:
        json.dump(data, f)
    return data


def generate_input_task2(chatistics_data_path, speaker1_tag='<speaker1>', speaker2_tag='<speaker2>', use_cache=True):
    """Generate input data for task 2"""
    f_path = os.path.join('data', 'input.txt')
    if os.path.isfile(f_path) and use_cache:
        logger.info('Input data already present.')
        return
    data = get_chatistics_conversation_data(chatistics_data_path, use_cache=use_cache)
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
    logger.info(f'Writing task 1 input data ({num_lines:,} lines) to {f_path}...')
    with open(f_path, 'w') as f:
        f.write(output)
