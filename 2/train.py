import os
import logging
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
from utils import get_input_task2, set_seed, add_special_tokens_

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')

class TextDataset(Dataset):
    def __init__(self, tokenizer, args):
        text = get_input_task2(args.data_path)
        logger.info("Tokenizing and building input...")
        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        self.examples = []
        block_size = args.max_input_length
        if block_size < 0:
            # by default use maximum possible input block size
            block_size = tokenizer.max_len_single_sentence
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, item):
        return torch.tensor(self.examples[item])

def get_data_loader(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    dataset = TextDataset(tokenizer, args)
    logger.info("Train dataset: {:,} samples".format(len(dataset)))
    logger.info("Build dataloaders")
    data_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
    return data_loader

def train():
    parser = ArgumentParser()
    parser.add_argument("--data_path", default=None, help="Path to conversational data (by default will look for single file in ./data)")
    parser.add_argument("--run_name", type=str, default='run1', help="The name of the run (subdirectory in ./runs)")
    parser.add_argument("--model", type=str, default="openai-gpt", choices=['openai-gpt', 'gpt2'], help="Initialize model from path to checkpoint or with model name (openai-gpt/openai-gpt2)")
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every n updates steps.")
    parser.add_argument("--max_input_length", type=int, default=400, help="Number of tokens which will be fed into the model (reduce this number if you have memory constraints)")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Load tokenizer
    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    tokenizer_class = GPT2Tokenizer if "gpt2" in args.model else OpenAIGPTTokenizer # cant use Autotokenizer because checkpoint could be a Path
    tokenizer = tokenizer_class.from_pretrained(args.model)
    # Load model
    model_class = GPT2LMHeadModel if "gpt2" in args.model else OpenAIGPTLMHeadModel
    model = model_class.from_pretrained(args.model)
    model.to(args.device)
    # Add special tokens if they are not already added
    add_special_tokens_(model, tokenizer)

    # Get data loaders
    logger.info("Prepare datasets")
    data_loader = get_data_loader(args, tokenizer)

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
    t_total = len(data_loader) // args.gradient_accumulation_steps * args.n_epochs
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(data_loader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(data_loader) // args.gradient_accumulation_steps)
        logger.info("Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"Continuing training from epoch {epochs_trained}")
        logger.info(f"Continuing training from global step {global_step}")
        logger.info(f"Will skip the first {steps_trained_in_current_epoch} steps in the first epoch")

    # Training loop
    model.zero_grad()
    epoch_pbar = trange(epochs_trained, int(args.n_epochs))
    av_loss = 0
    for current_epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch [{current_epoch+1}/{args.n_epochs}]")
        pbar = tqdm(data_loader)
        for step, batch in enumerate(pbar):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            inputs, labels = (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            loss, *_ = model(inputs, labels=labels)
            loss.backward()
            tr_loss = loss.item()
            av_loss = (step*av_loss + tr_loss)/(step + 1)
            pbar.set_description(f"Average loss: {av_loss:.4f}")
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
