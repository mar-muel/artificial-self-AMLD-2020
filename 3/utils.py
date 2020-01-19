import json
import logging
import os
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
from transformers import cached_path
import tempfile
import tarfile

# Special tokens to be added to tokenizer:
# - <bos> to indicate the start of the sequence
# - <eos> to indicate the end of the sequence
# - <speaker1> to indicate the beginning and the tokens of an utterance from the user
# - <speaker2> to indicate the beginning and the tokens of an utterance from the bot
# - <pad> as a padding token to build batches of sequences
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ('<speaker1>', '<speaker2>')}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
logger = logging.getLogger(__file__)
HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz"

def download_pretrained_model():
    """Download and extract finetuned model (trained on Personachat dataset)"""
    resolved_archive_file = cached_path(HF_FINETUNED_MODEL)
    tempdir = tempfile.mkdtemp()
    logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
    with tarfile.open(resolved_archive_file, 'r:gz') as archive:
        archive.extractall(tempdir)
    return tempdir

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

def set_seed(seed):
    """Set seed in random, numpy and torch/cuda"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

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

def get_grouped_conversation_data(data_path, use_cache=True, cache_path='grouped_conversation_data.json'):
    """Create grouped conversation data from input data"""
    if use_cache:
        if os.path.isfile(cache_path):
            logger.info('Reading cached conversation data...')
            with open(cache_path, 'r') as f:
                data = json.load(f)
            return data
    logger.info(f"Creating grouped conversation data...")
    # read conversational data
    df = read_conversation_data(data_path)
    # generate conversations
    new_conversation_delay_hours = 24
    data = defaultdict(list)
    min_num_interactions_per_conversation = 10
    num_interactions = 0
    for conversation_name, g in tqdm(df.groupby('conversationWithName'), total=len(df['conversationWithName'].unique()), position=0):
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

def get_input_task3(data_path, tokenizer, max_input_length=200, num_candidates=2, seed=42, max_history=2, use_cache=True):
    """Get input data for task 3"""
    def merge_messages(messages):
        """Merge multiple messages into single string"""
        full_message = ''
        for m in messages:
            m = m.strip()
            if len(m) > 0:
                full_message += '\n' + m
        return full_message.strip()
    def generate_distractor_messages():
        """Collect all messages by person1 to build a corpus of distractor messages """
        distractors = []
        for _, conversations in conv_data.items():
            for conversation in conversations:
                for interaction in conversation:
                    if interaction['senderType'] == 'person1':
                        message = merge_messages(interaction['messages'])
                        distractors.append(message)
        return distractors
    # check for cached data
    cached_path = f'cached_input_task3_{tokenizer.__module__}_{num_candidates}_{seed}.pkl'
    if use_cache:
        if os.path.isfile(cached_path):
            logger.info('Reading cached data...')
            with open(cached_path, 'rb') as f:
                data = pickle.load(f)
            return data
    # read conversation data
    conv_data = get_grouped_conversation_data(data_path, use_cache=use_cache)
    # Generate distractor messages (consisting of of all interactions by person1)
    distractors = generate_distractor_messages()
    num_distractors = len(distractors)
    if num_distractors <= 20:
        raise Exception(f'Person1 needs to have at least 20 interactions')
    num_interactions = 0
    num_messages = 0
    num_examples = 0
    dataset = defaultdict(list)
    for conversation_with_name, conversations in conv_data.items():
        for conversation in conversations:
            history = []
            for interaction in conversation:
                num_interactions += 1
                num_messages += len(interaction['messages'])
                # We aim to generate samples with the following structure
                # person2, person1, ..., person2, <candidate> (candidate can be either a distractor or the true answer of person1)
                if interaction['senderType'] == 'person1':
                    # person 1
                    if len(history) == 0:
                        # skip - cannot start with person1 (sequence needs to start with person2)
                        continue
                    # Generate candidates for problem
                    candidates = []
                    true_answer = merge_messages(interaction['messages'])
                    while len(candidates) < num_candidates-1:
                        cand = distractors[random.randint(0, num_distractors - 1)]
                        # make sure random distractor message wasn't accidentially true answer
                        if cand != true_answer:
                            candidates.append(cand)
                    candidates.append(true_answer) # last candidate is the true one
                    # generating new training example
                    history_tk = [tokenizer.encode(h) for h in history[-(2*max_history+1):]]
                    candidates_tk = [tokenizer.encode(c) for c in candidates]
                    instance_list = []
                    for j, candidate in enumerate(candidates_tk):
                        lm_labels = bool(j == num_candidates-1)  # the last candidate is the correct reply
                        # build training example (num_candidates x input_length)
                        instance = build_input_from_segments(history_tk, candidate, tokenizer, lm_labels=lm_labels)
                        input_length = len(instance['input_ids'])
                        if input_length > max_input_length:
                            # logger.info(f'Skipping example of length {input_length} (max input length is {max_input_length})')
                            break
                        instance_list.append(instance)
                    else:
                        # Only collect examples which were below max_input_length
                        for instance in instance_list:
                            for input_name, input_array in instance.items():
                                dataset[input_name].append(input_array)
                        dataset["mc_labels"].append(num_candidates - 1)  # index of true_answer
                        dataset["n_candidates"] = num_candidates
                        num_examples += 1
                else:
                    # person 2
                    message = merge_messages(interaction['messages'])
                    history.append(message)
    logger.info(f'Generated a total of {num_examples:,} training examples from {len(conv_data):,} conversations consisting of {num_interactions:,} interactions and {num_messages:,} messages...')
    # Add padding to make all input vetors the same length
    logger.info("Pad inputs and convert to Tensor")
    dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
    # Create tensors from lists
    tensor_dataset = []
    for input_name in MODEL_INPUTS:
        tensor = torch.tensor(dataset[input_name])
        if input_name != "mc_labels":
            tensor = tensor.view((-1, dataset["n_candidates"]) + tensor.shape[1:])
        tensor_dataset.append(tensor)
    # Cache data
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
    """ Build a sequence of input from 2 segments: history and candidate reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    # build input sequence as:
    # [bos, person2, person1, ..., person2, candidate, eos]
    sequence = [[bos]] + history + [reply + ([eos] if with_eos else [])]
    # Add special separator tokens between sequence:
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    # Input IDs consists of all inputs concatenated
    instance["input_ids"] = list(itertools.chain(*sequence))
    # Input types (which portions belong to speaker1 and speaker2)
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    # Index of the classification token in each input sequence.
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    # Which positions to use for language modelling (-1 is ignored)
    instance["lm_labels"] = [-1] * len(instance["input_ids"])
    if lm_labels:
        # Only train LM on the true_answer by person1 (history is filled as -1 and reply is filled with input IDs)
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
    return instance
