import gpt_2_simple as gpt2
import json
import os
import sys
import numpy as np
import argparse

def generate(args):
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, run_name=args.run_name)
    while True: 
        message = None
        while not message:
            message = input('>> ')
        text = gpt2.generate(sess, run_name=args.run_name, prefix=message, length=args.length, temperature=args.temperature, top_k=args.top_k, return_as_list=True)
        print(text[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interact with fine tuned model")
    parser.add_argument('-r', '--run_name', help="Run name", default='run1')
    parser.add_argument('-t', '--temperature', dest='temperature', type=float, help="Temperature", default=1)
    parser.add_argument('--top-k', dest='top_k', type=float, help="Top k", default=0)
    parser.add_argument('--length', type=int, help="Length of output sequence", default=200)
    args = parser.parse_args()
    generate(args)
