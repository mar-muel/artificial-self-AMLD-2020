# Task 1

For this task we will use OpenAI's GPT-2 model to fine-tune their language model on arbitrary input text. For this we will use the library [gpt-2-simple](https://github.com/minimaxir/gpt-2-simple/) by Max Woolf.

## Task
In this task we play around with text generation to get a feeling for style transfer. You can fine-tune a GPT-2 model through

## Data
You can use the data sets provided (see **Sample data**) or use your own data sets (see **Custom data**).

### Sample data
Raw data sets are included in this repository for this and the following tasks. In order to use some of them for this task, we have to compile them into files usable by gpt-2-simple.
To find out about which data sets you can compile and a few options (eg. sample size) run `python prepare.py --help`. To compile all data sets available for task 1 with the standard option run
```
python prepare.py all
```
The input data files are then generated in the data folder of this task (1/data).

### Custom data
You can fine-tune the model for any kind of data that can be represented in text form. Just create a plain text file with one sample per line and put it into the data folder of this task.

## Usage
You can either run this code locally or use the Colab notebook.

### Local
1. Clone this repository using
```
git clone git@github.com:mar-muel/artificial-self-AMLD-2020.git && cd artificial-self-AMLD-2020
```
2. Install the dependencies
```
cd 1
pip install -r requirements.txt
```
3. Fine-tune the GPT-2 model for the data file of your choice by running
```
python train.py --run_name run1 --data_path data/<data-filename>
```
4. Generate text with the same style as your training data by running
```
python generate.py --run_name run1
```
Run `python generate.py --help` to see parameters available to influence the text generation.

### Colab notebook

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lk9iZnD5mkAf29FCN3QmcSssFDrWjE8W)
