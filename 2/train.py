import os
import logging
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange
from transformers import (
    AdamW, 
    OpenAIGPTDoubleHeadsModel,
    OpenAIGPTTokenizer,
    GPT2DoubleHeadsModel,
    GPT2Tokenizer,
    WEIGHTS_NAME,
    CONFIG_NAME,
    WarmupLinearSchedule,
)
from utils import make_logdir, get_chatistics_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ('<speaker1>', '<speaker2>')}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-1] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
    return instance

def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    personachat = get_chatistics_dataset(tokenizer, args.chatistics_data)
    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for dataset_name, dataset in personachat.items():
        if len(dataset) == 0:
            continue
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0 and dataset_name == 'train':
            num_candidates = min(args.num_candidates, num_candidates)
        for dialog in dataset:
            persona = ''  # here we could couple the dialog to a personality
            for utterance in dialog["utterances"]:
                history = utterance["history"][-(2*args.max_history+1):]
                instance_list = []
                for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                    lm_labels = bool(j == num_candidates-1)  # the last candidate is the correct reply
                    instance = build_input_from_segments(persona, history, candidate, tokenizer, lm_labels)
                    input_length = len(instance['input_ids'])
                    if input_length > args.max_input_length:
                        logger.info(f'Skipping example of length {input_length} (max input length is {args.max_input_length})')
                        break
                    instance_list.append(instance)
                if len(instance_list) == num_candidates:
                    # only collect examples which were below max_input_length
                    for instance in instance_list:
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)
                    datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                    datasets[dataset_name]["n_candidates"] = num_candidates
    # pad dataset
    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        if len(dataset) == 0:
            continue
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        model_input_tensors = []
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            model_input_tensors.append(tensor)
        if len(model_input_tensors) == len(MODEL_INPUTS):
            tensor_datasets[dataset_name].extend(model_input_tensors)
    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False)
    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    return train_loader, valid_loader

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)

def train():
    parser = ArgumentParser()
    parser.add_argument("--chatistics_data", type=str, default="data", help="Path to chatistics data")
    parser.add_argument("--run_name", type=str, default='run1', help="The name of the run (subdirectory in ./runs)")
    parser.add_argument("--model_checkpoint", type=str, default="openai-gpt", help="Initialize model from path to checkpoint or with model name (openai-gpt/openai-gpt2)")
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every n updates steps.")
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
    parser.add_argument("--max_input_length", type=int, default=300, help="Number of tokens which will be fed into the model (reduce this number if you have memory constraints)")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()

    # Set seed
    set_seed(args)

    # Load tokenizer
    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    tokenizer_class = GPT2Tokenizer if "gpt2" in args.model_checkpoint else OpenAIGPTTokenizer # cant use Autotokenizer because checkpoint could be a Path
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    # Load model
    model_class = GPT2DoubleHeadsModel if "gpt2" in args.model_checkpoint else OpenAIGPTDoubleHeadsModel
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    # Add special tokens if they are not already added
    add_special_tokens_(model, tokenizer)

    # Get data loaders
    logger.info("Prepare datasets")
    train_loader, valid_loader = get_data_loaders(args, tokenizer)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    t_total = len(train_loader) // args.gradient_accumulation_steps * args.n_epochs
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_checkpoint):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_checkpoint.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_loader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_loader) // args.gradient_accumulation_steps)
        logger.info("Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"Continuing training from epoch {epochs_trained}")
        logger.info(f"Continuing training from global step {global_step}")
        logger.info(f"Will skip the first {steps_trained_in_current_epoch} steps in the first epoch")

    # Training loop
    model.zero_grad()
    epoch_pbar = trange(epochs_trained, int(args.n_epochs))
    for current_epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch [{current_epoch+1}/{args.n_epochs}]")
        pbar = tqdm(train_loader)
        for step, batch in enumerate(pbar):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
            (lm_loss), (mc_loss), *_ = model(input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids, mc_labels=mc_labels, lm_labels=lm_labels)
            loss = (lm_loss * args.lm_coef + mc_loss * args.mc_coef) / args.gradient_accumulation_steps
            loss.backward()
            tr_loss = loss.item()
            pbar.set_description(f"Training loss: {tr_loss:.4f}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if global_step % args.save_every == 0 and global_step > 0:
                    checkpoint_prefix = "checkpoint"
                    output_dir = os.path.join('runs', args.run_name, "{}-{}".format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    logger.info(f"Saving model checkpoint to {output_dir}")
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    logger.info(f"Saving optimizer and scheduler states to {output_dir}")
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    # save model
    output_dir = os.path.join('runs', args.run_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info(f"Saving model checkpoint to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(output_dir, "training_args.bin"))

if __name__ == "__main__":
    train()
