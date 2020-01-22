import gpt_2_simple as gpt2
import os
import requests
import glob
import pickle
import pandas as pd
import re
import unicodedata
import argparse
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
log = logging.getLogger(__name__)


def fine_tune(args, model_name='124M'):
    print(f'Run fine-tuning for run {args.run_name} using GPT2 model {model_name}...')
    if not os.path.isdir(os.path.join("models", model_name)):
        log.info(f"Downloading {model_name} model...")
        gpt2.download_gpt2(model_name=model_name)
    sess = gpt2.start_tf_sess()
    gpt2.finetune(sess, args.data_path, model_name=model_name, run_name=args.run_name, steps=-1, sample_every=10, save_every=10)

def main():
    parser = argparse.ArgumentParser(
        description="Finetune GPT2 on various text data sets"
    )
    parser.add_argument('-d', '--data_path', required=True, help="Path to input data (txt file)")
    parser.add_argument('-r', '--run_name', help="Run name", default='run1')
    args = parser.parse_args()
    fine_tune(args)

if __name__ == "__main__":
    main()
