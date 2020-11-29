import json
import os
import sys
from dotenv import load_dotenv
load_dotenv()
import random
import nltk
import re
import heapq
import numpy as np



class DataProcessor:

    # TODO: Make this explanation better
    # Milliseconds between messages under which we consider messages be a single message
    CONCAT_THRESHOLD = int(os.getenv("CONCAT_THRESHOLD"))

    def __init__(self, simulacrum_name):
        # The full name of the recipient as shown in the various message_X.json files
        self.simulacrum_name = simulacrum_name
        self.sent = []
        self.received = []
        self.pairs = []

    def wordBagFromList(self, listArgument):
        newList = listArgument

        for i in range(len(newList)):
            newList[i] = newList[i].lower()
            newList[i] = re.sub(r'\W', ' ', newList[i])
            newList[i] = re.sub(r'\s+', ' ', newList[i])

        wordfreq = {}
        for sentence in newList:
            tokens = nltk.word_tokenize(sentence)
            for token in tokens:
                if token not in wordfreq.keys():
                    wordfreq[token] = 1
                else:
                    wordfreq[token] += 1
        #             Put number of words in ENV later
        most_freq = heapq.nlargest(200, wordfreq, key=wordfreq.get)
        sentence_vectors = []
        for sentence in newList:
            sentence_tokens = nltk.word_tokenize(sentence)
            sent_vec = []
            for token in most_freq:
                if token in sentence_tokens:
                    sent_vec.append(1)
                else:
                    sent_vec.append(0)
            sentence_vectors.append(sent_vec)
        return np.asarray(sentence_vectors)
        

    def process(self):
        disallowed_list = ["You are now connected on Messenger.", "http", "www.", "@", "You joined the video chat.", "joined the video chat."]
        path = os.getcwd() + "/data/messages/inbox/"
        conversations = []
        # for file_name in [file for file in os.listdir(path) if file.endswith('.json')]:
        for (dirpath, folder, file) in os.walk(path):
            if len(file) > 0 and file[0].endswith('.json'):
                for message in file:
                    conversations.append(dirpath + "/" + message)
        for conversation in conversations:
            with open(conversation) as json_file:
                data = json.load(json_file)
                last_timestamp = 0
                last_sender = ""
                for message in data["messages"]:
                    if "content" in message and not any(disallowed in message["content"] for disallowed in disallowed_list) and message["type"] == "Generic":
                        if self.simulacrum_name == message["sender_name"]:
                            if len(self.sent) != 0 and last_timestamp - message["timestamp_ms"] < DataProcessor.CONCAT_THRESHOLD and last_sender == self.simulacrum_name:
                                self.sent[-1] = message["content"] + " " + self.sent[-1]
                            else:
                                self.sent.append(message["content"])
                        else:
                            if len(self.received) != 0 and message["sender_name"] == last_sender and last_timestamp - message["timestamp_ms"] < DataProcessor.CONCAT_THRESHOLD:
                                self.received[-1] = message["content"] + " " + self.received[-1]
                            else:
                                self.received.append(message["content"])
                        if last_sender == self.simulacrum_name and message["sender_name"] != self.simulacrum_name:
                            self.pairs.append((len(self.sent) - 1, len(self.received) - 1))
                        last_sender = message["sender_name"]
                        last_timestamp = message["timestamp_ms"]
        return self


processed = DataProcessor(os.getenv("SIMULACRUM_NAME")).process()
# print(processed.simulacrum_name)
# print(processed.sent)
# print(processed.received)

# print(len(processed.pairs))
temp = random.randint(0, len(processed.pairs)-1)
print(temp)
x, y = processed.pairs[random.randint(0, temp)]
# x, y = processed.pairs[42981]
print(processed.received[y])
print(processed.sent[x])
