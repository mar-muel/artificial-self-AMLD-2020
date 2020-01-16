# Task 2

In task 1 we learnt how to fine-tune a language model and we saw how style transfer works. Conversations are a different beast however! In this task we will try our first approach at training a conversational model.

## Task
In this task we will try a naive approach to getting conversational style by simply feeding the model "raw" conversation data of the form:
```
<speaker1> Hi
<speaker2> Hey - how are you?
<speaker1> Great, thanks!
...
```
Our hope is that the model will simply learn this structure and we will be able to query the model with an input of the form:

```
<speaker2> Am I speaking to a bot?
<speaker1>
```
We then expect the model to extend the text from this prefix.

## Data

## Usage

### Code

### Notebook
