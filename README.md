# Meet your Artificial Self: Generate text that sounds like you
This repository contains all resources for the [Applied Machine Learning Days](https://appliedmldays.org/) workshop [Meet your Artificial Self: Generate text that sounds like you](https://appliedmldays.org/workshops/meet-your-artificial-self-generate-text-that-sounds-like-you).

In this workshop, participants are tasked to download their own chat logs and build a chat bot that generates text similar to their writing. As an alternative to using chat logs, we provide a number of other conversational (and non-conversational datasets) datasets in this repository.

## Gitter
Feel free to join our Gitter during the workshop:

[![Gitter](https://badges.gitter.im/artificial-self-AMLD-2020/community.svg)](https://gitter.im/artificial-self-AMLD-2020/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

## Slides
Find the workshop slides [here](https://docs.google.com/presentation/d/1-aU5fSWyQN4GwP3KFDy5KorM7c-FJJFjiRp3bJ2sqIY/edit?usp=sharing).

# Usage
The workshop is split in 3 tasks. You can run each task locally (by cloning this repository) or by running the Colab notebook (see links below). If you run locally, make sure you have access to GPU(s) and you are running Python 3.6+ (also make sure you have sufficient storage space). More detailed instructions are provided in the different subfolders.

## Task 1
Fine-tune GPT-2 on various [datasets](datasets) (including tweets, poetry, programming code, chess, music and more!). Thanks to [@manueth](https://github.com/manueth) for compiling the datasets! 

:arrow_right: [Read more](1) 

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lk9iZnD5mkAf29FCN3QmcSssFDrWjE8W)

## Task 2
We use the same approach of style transfer to train a conversational model from our chat logs. You can either use [Chatistics](https://github.com/MasterScrat/Chatistics) to parse your own chat logs or you can use some of the provided resources. Thanks to [@MasterScrat](https://github.com/MasterScrat) for compiling the conversational datasets!

:arrow_right: [Read more](2) 

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iHcQ8_K0cfRE3v8QX6FMKAzdSSGtf5IX)

## Task 3
We extend the approach in task 2 by introducing multi-task learning, improving data preprocessing, and adding token types.

:arrow_right: [Read more](3) 

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XYNef9zcHhTjt6kM6ydL9oXTshoRknIV)

# Credits
* [@manueth](https://github.com/manueth) and [@MasterScrat](https://github.com/MasterScrat)
* [minimaxir/gpt-2-simple](https://github.com/minimaxir/gpt-2-simple)
* [hunggingface/transformers](https://github.com/huggingface/transformers)
* [huggingface/transfer-learning-conv-ai](https://github.com/huggingface/transfer-learning-conv-ai)
* [MasterScrat/Chatistics](https://github.com/MasterScrat/Chatistics)
