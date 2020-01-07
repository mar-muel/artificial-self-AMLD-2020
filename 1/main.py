# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Task 1: Finetuning a GPT-2 language model

# We will learn
# * What it means to fine-tune a language model
# * How to control the text generation process
# * What are the limits of text generation
# Note that we will ignore the complexity of the model as well as the mechanics of the fine-tuning process in this task (more of this at a later stage!).

%tensorflow_version 1.x
!pip install -r requirements.txt
import gpt_2_simple as gpt2
