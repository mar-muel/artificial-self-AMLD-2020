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
import random
import itertools
import numpy as np

CHATISTICS_CONV_DATA = 'chatistics_conversation_data.json'
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ('<speaker1>', '<speaker2>')}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
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

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)

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

def get_input_task3(args, tokenizer, use_cache=True):
    """Get tokenized chatlog dataset"""
    def merge_messages(messages):
        full_message = ''
        for m in messages:
            m = m.strip()
            if len(m) > 0:
                full_message += '\n' + m
        return full_message.strip()
    # check for cached data
    cached_path = f'cached_input_task3_{tokenizer.__module__}_{args.num_candidates}_{args.seed}.pkl'
    if use_cache:
        if os.path.isfile(cached_path):
            logger.info('Reading cached data...')
            with open(cached_path, 'rb') as f:
                data = pickle.load(f)
            return data
    # read conversation data
    conv_data = get_chatistics_conversation_data(args.chatistics_data_path, use_cache=use_cache)
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
    num_interactions = 0
    num_messages = 0
    for conversation_with_name, conversations in conv_data.items():
        for conversation in conversations:
            history = []
            for interaction in conversation:
                num_interactions += 1
                num_messages += len(interaction['messages'])
                # we are trying to learn person1.
                # Generate the following structure
                # person2, person1, ..., person2, <candidate> (candidate can be either a distractor or the true answer of person1)
                if interaction['senderType'] == 'person1':
                    # person 1
                    if len(history) == 0:
                        # skip - cannot start with person1
                        continue
                    # Generate candidates for problem
                    candidates = []
                    true_answer = merge_messages(interaction['messages'])
                    while len(candidates) < args.num_candidates-1:
                        cand = all_interactions_person1[random.randint(0, num_interactions_person1 - 1)]
                        # make sure random distractor message wasn't accidentially true answer
                        if cand != true_answer:
                            candidates.append(cand)
                    candidates.append(true_answer) # last candidate is the true one
                    # add new problem (keep a maximum of 8 interactions)
                    data.append({'history': history[-8:], 'candidates': candidates})
                else:
                    # person 2
                    message = merge_messages(interaction['messages'])
                    history.append(message)
    logger.info(f'Generated a total of {len(data):,} training examples from {len(conv_data):,} conversations consisting of {num_interactions:,} interactions and {num_messages:,} messages...')
    # shuffle examples
    random.shuffle(data)
    # tokenizing
    logger.info('Tokenizing...')
    def tokenize(obj):
        if isinstance(obj, str):
            tokens = tokenizer.tokenize(obj)
            return tokenizer.convert_tokens_to_ids(tokens)
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)
    data = tokenize(data)
    # generate training examples
    logger.info("Build inputs and labels")
    dataset = defaultdict(list)
    for interaction in data:
        history = interaction["history"][-(2*args.max_history+1):]
        instance_list = []
        for j, candidate in enumerate(interaction["candidates"]):
            lm_labels = bool(j == args.num_candidates-1)  # the last candidate is the correct reply
            instance = build_input_from_segments(history, candidate, tokenizer, lm_labels=lm_labels)
            input_length = len(instance['input_ids'])
            if input_length > args.max_input_length:
                logger.info(f'Skipping example of length {input_length} (max input length is {args.max_input_length})')
                break
            instance_list.append(instance)
        if len(instance_list) == args.num_candidates:
            # only collect examples which were below max_input_length
            for instance in instance_list:
                for input_name, input_array in instance.items():
                    dataset[input_name].append(input_array)
            dataset["mc_labels"].append(args.num_candidates - 1)
            dataset["n_candidates"] = args.num_candidates
    # pad dataset
    logger.info("Pad inputs and convert to Tensor")
    tensor_dataset = []
    dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
    for input_name in MODEL_INPUTS:
        tensor = torch.tensor(dataset[input_name])
        if input_name != "mc_labels":
            tensor = tensor.view((-1, dataset["n_candidates"]) + tensor.shape[1:])
        tensor_dataset.append(tensor)
    logger.info(f'Writing cached file {cached_path}...')
    with open(cached_path, 'wb') as f:
        pickle.dump(tensor_dataset, f)
    return tensor_dataset

def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset

def build_input_from_segments(history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 2 segments: history and last reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos]] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = list(itertools.chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-1] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
    return instance


def make_logdir(model_name: str):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    # Code copied from ignite repo
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(
        'runs', current_time + '_' + socket.gethostname() + '_' + model_name)
    return logdir
