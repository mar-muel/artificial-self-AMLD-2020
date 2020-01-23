# Task 3

In task 2 we tried to use a "hack" to get our model to speak. In this task we will add two more ingredients:
* Multi-task learning
* Specifying token types

As you will (hopefully) see this greatly improves the model.

**Note:** This task is heavily influenced by [this](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313) blog post on building dialog models based on the Personachat dataset which was the winning approach in the ConvAI2 challenge in 2018.

# Usage
Again you can use the local version or the Colab for this task.

## Locally
1. If you haven't already, clone the repository
2. Make sure you have PyTorch with GPU-support installed. Follow the instructions [here](https://pytorch.org/get-started/locally/) here to install the proper version depending on your OS. Also make sure you are using Python 3.6+. Then run:
```
pip install -r 3/requirements.txt
```
3. Make sure you have placed your conversational dataset (JSON file) into the folder `3/data/`
4. Use `python train.py --run_name run1` to train your first model
5. Use `python interact.py --run_name run1` to interact

## Colab
[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XYNef9zcHhTjt6kM6ydL9oXTshoRknIV)
  
## Slack integration
If you want to talk to your bot in Slack (instead of through the command line) do the following:
1. [Create a Slack app](https://slack.com/intl/en-ch/help/articles/115005265703-Create-a-bot-for-your-workspace) and get your API token
2. Find your user ID (member ID) from your Slack 
3. Set two environment variables
```
export SLACK_API_TOKEN=<API token>
export SLACK_USER=<your user id>
```
4. Run `python interact_slack.py --run_name run1`
5. You should see the bot is now online and you can start talking to it.
