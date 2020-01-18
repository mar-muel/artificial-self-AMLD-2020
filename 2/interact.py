import os
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings
import torch
import torch.nn.functional as F
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import set_seed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def sample_sequence(conversation, model, args, num_samples=1):
    context = torch.tensor(conversation, dtype=torch.long, device=args.device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in range(args.max_length):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)
            # scale by temperature
            next_token_logits = outputs[0][:, -1, :] / (args.temperature if args.temperature > 0 else 1.)
            # filter by top-k/top-p
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.top_k, top_p=args.top_p)
            if args.temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated


def run():
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, default='run1', help="The name of the run (subdirectory in ./runs)")
    parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=80, help="Maximum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=1.0, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.8, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--no_info", action='store_true', default=False, help="Only show conversation output")
    args = parser.parse_args()

    # set seed
    set_seed(args)

    logger.info("Get pretrained model and tokenizer")
    model_path = os.path.join('runs', args.run_name)
    tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args.model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(model_path)
    model = model_class.from_pretrained(model_path)
    model.to(args.device)
    history = []
    personality = []
    speaker1_tag = '<speaker1>'
    speaker2_tag = '<speaker2>'
    speaker1_tag_id = tokenizer.convert_tokens_to_ids(speaker1_tag)
    speaker2_tag_id = tokenizer.convert_tokens_to_ids(speaker2_tag)
    history = f"""
{speaker2_tag} Hi!
{speaker1_tag} Hello
{speaker2_tag} Are you ready?
{speaker1_tag} Yes!
{speaker2_tag} Ok let's start chatting
{speaker1_tag} Sure, what do you want to talk about?"""
    print(history)
    print('\n[Chat with the model! Send "h" to see the full history]\n')
    history = history.split('\n')
    while True: 
        message = None
        while not message:
            message = input(f'{speaker2_tag} ')
            if message == 'h':
                print('\n'.join(history))
                message = None
        # add new message to history
        history.append(f'{speaker2_tag} {message}')
        # keep only most recent conversation as input to the model
        recent_history = history[-(2*args.max_history):]
        # concatenate history into single string and add trigger word "bot:"
        history_str = '{}\n{}'.format('\n'.join(recent_history), speaker1_tag)
        # tokenize text and convert into vocabulary ids (input ids)
        history_enc = tokenizer.encode(history_str, add_special_tokens=True)
        with torch.no_grad():
            out_ids = sample_sequence(history_enc, model, args)
        out_ids = out_ids[:, len(history_enc):].tolist()[0]
        if not args.no_info:
            print(20*'-')
            print('Output of model:')
            full_output = tokenizer.decode(out_ids, clean_up_tokenization_spaces=True)
            print(full_output)
            print('\nInput to the model:')
            print(history_str)
            print(20*'-' + '\n')
        # Select part before speaker tags as answer
        for i, out_id in enumerate(out_ids):
            if out_id in [speaker1_tag_id, speaker2_tag_id]:
                break
        answer = '{} {}'.format(speaker1_tag, tokenizer.decode(out_ids[:i]))
        print(answer)
        # add answer to history
        history.append(answer)

if __name__ == "__main__":
    run()
