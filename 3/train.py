import os
import logging
from argparse import ArgumentParser
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
    get_linear_schedule_with_warmup,
)
from utils import get_input_task3, add_special_tokens_, set_seed, download_pretrained_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')


def get_data_loader(args, tokenizer, use_cache=True):
    """ Prepare the dataset for training and evaluation """
    # get dataset of tensors
    data = get_input_task3(args.data_path, tokenizer, max_input_length=args.max_input_length, num_candidates=args.num_candidates, seed=args.seed, max_history=args.max_history, use_cache=use_cache)
    logger.info("Building training data loader")
    train_dataset = TensorDataset(*data)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    logger.info("Train dataset input shape: (Batch size, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    return train_loader

def train():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None, help="Path to conversational data (by default will look for single file in ./data)")
    parser.add_argument("--run_name", type=str, default='run1', help="The name of the run (subdirectory in ./runs)")
    parser.add_argument("--model", type=str, default="openai-gpt", help="Initialize model from path to checkpoint or with model name (openai-gpt/openai-gpt2)")
    parser.add_argument("--save_every", type=int, default=100, help="Save checkpoint every n updates steps.")
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
    parser.add_argument("--max_input_length", type=int, default=200, help="Number of tokens which will be fed into the model (reduce this number if you have memory constraints)")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
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
    parser.add_argument("--use_huggingface_model", action='store_true', help="Start training from pre-trained model by Huggingface")
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    if args.use_huggingface_model:
        args.model = download_pretrained_model()
        logger.info(f'Using pre-trained Personachat model {args.model}')

    # Load tokenizer
    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    tokenizer_class = GPT2Tokenizer if "gpt2" in args.model else OpenAIGPTTokenizer # cant use Autotokenizer because checkpoint could be a Path
    tokenizer = tokenizer_class.from_pretrained(args.model)
    # Load model
    model_class = GPT2DoubleHeadsModel if "gpt2" in args.model else OpenAIGPTDoubleHeadsModel
    model = model_class.from_pretrained(args.model)
    model.to(args.device)
    # Add special tokens if they are not already added
    add_special_tokens_(model, tokenizer)

    # Get data loaders
    logger.info("Prepare datasets")
    train_loader = get_data_loader(args, tokenizer, use_cache=True)

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
        try:
            global_step = int(args.model.split("-")[-1].split("/")[0])
        except:
            global_step = 0
        epochs_trained = global_step // (len(train_loader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_loader) // args.gradient_accumulation_steps)
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
            # caclulate exponential moving average
            av_loss = (step*av_loss + loss)/(step + 1)
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
