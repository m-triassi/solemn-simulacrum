import json
import os
from dotenv import load_dotenv
load_dotenv()
import random
import nltk
import re
import heapq
import numpy as np
from sklearn.model_selection import train_test_split


class DataProcessor:

    # Threshold of time accepted between messages from the same user.
    # Used to concatenate messages that were sent in rapid succession.
    CONCAT_THRESHOLD = int(os.getenv("CONCAT_THRESHOLD"))

    def __init__(self, simulacrum_name):
        # The full name of the recipient as shown in the various message_X.json files
        self.simulacrum_name = simulacrum_name
        self.sent = []
        self.received = []
        self.pairs = []
        self.sent_vector = []
        self.received_vector = []
        self.vocabulary_size = 0
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def extract(self):
        disallowed_list = ["You are now connected on Messenger.", "http", "www.", "@", "You joined the video chat.", "joined the video chat.", "set the nickname", "set his own nickname", "set your nickname to"]
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
                # 4 possibilities of the following loop:
                # 1 the message was sent by the person we wish to impersonate.
                # 1.1 It is the first message and does not need to be concatenated. So we add it to the sent list
                # 1.2 It is a message sent in rapid succession and needs to be concatenated to the previous sent message
                # 2 the message is sent by another person.
                # 2.1 This is this persons first message and can be added to the list of received messages.
                # 2.2 This is a message sent in rapid succession and needs to be concatenated to the previous received message.
                for message in data["messages"]:
                    if "content" in message and not any(disallowed in message["content"] for disallowed in disallowed_list) and message["type"] == "Generic":
                        if self.simulacrum_name == message["sender_name"]: #1
                            if len(self.sent) != 0 and last_timestamp - message["timestamp_ms"] < DataProcessor.CONCAT_THRESHOLD and last_sender == self.simulacrum_name:
                                self.sent[-1] = message["content"] + " " + self.sent[-1] #1.2
                            else:
                                self.sent.append(message["content"]) #1.1
                        else: #2
                            if len(self.received) != 0 and message["sender_name"] == last_sender and last_timestamp - message["timestamp_ms"] < DataProcessor.CONCAT_THRESHOLD:
                                self.received[-1] = message["content"] + " " + self.received[-1] #2.2
                            else:
                                self.received.append(message["content"]) #2.1
                        #If this message is from the person we wish to impersonate and the previous message was not,
                        #than it is a response and needs to be recorded.
                        if last_sender == self.simulacrum_name and message["sender_name"] != self.simulacrum_name:
                            self.pairs.append((len(self.sent) - 1, len(self.received) - 1))
                        last_sender = message["sender_name"]
                        last_timestamp = message["timestamp_ms"]

        self.sent = self.clean_sentences(self.sent)
        self.received = self.clean_sentences(self.received)
        return self

    def clean_sentences(self, sentences):
        for i in range(len(sentences)):
            sentences[i] = sentences[i].lower()
            # sentences[i] = re.sub(r'\W', ' ', sentences[i])
            sentences[i] = re.sub(r'\s+', ' ', sentences[i])
        sentences = [x for x in sentences if x and x != " "]
        return sentences

    def create_vocabulary(self, sentences):
        wordfreq = {}
        #count the appearance of each word in each sentencce.
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            for token in tokens:
                if token not in wordfreq.keys():
                    wordfreq[token] = 1
                else:
                    wordfreq[token] += 1
        #             Put number of words in ENV later
        return heapq.nlargest(200, wordfreq, key=wordfreq.get)

    # no longer being used.
    # Was the original method used to converd sentences to vectors.
    # Used the top X words in the vocabulary and set a value to 1 if the word appeared in the sentence.
    def create_sentence_vectors(self, sentences, vocabulary):
        sentence_vectors = []
        for sentence in sentences:
            sentence_tokens = nltk.word_tokenize(sentence)
            sent_vec = []
            for token in vocabulary:
                if token in sentence_tokens:
                    sent_vec.append(1)
                else:
                    sent_vec.append(0)
            sentence_vectors.append(sent_vec)
        return np.asarray(sentence_vectors)

    def process(self):
        self.extract()
        all_messages = self.sent
        all_messages.extend(self.received)
        vocabulary = self.create_vocabulary(all_messages)
        self.vocabulary_size = len(vocabulary)
        self.sent_vector = self.create_sentence_vectors(self.sent, vocabulary)
        self.received_vector = self.create_sentence_vectors(self.received, vocabulary)
        X = self.sent_vector
        y = np.full(shape=len(self.sent_vector), fill_value=1, dtype=np.int)
        X = np.concatenate((X, self.received_vector))
        y = np.concatenate((y, np.full(shape=len(self.received_vector), fill_value=0, dtype=np.int)))
        # print(self.sent_vector.shape)
        # print(self.received_vector.shape)
        print("Processed!")
        return train_test_split(X, y, test_size=0.5)

    def plain_label(self):
        self.extract()
        X = np.array(self.sent)
        y = np.full(shape=len(self.sent), fill_value=1, dtype=np.int)
        X = np.concatenate((X, np.array(self.received)))
        y = np.concatenate((y, np.full(shape=len(self.received), fill_value=0, dtype=np.int)))

        return train_test_split(X, y, test_size=0.5)

    def get_random_pair(self):
        temp = random.randint(0, len(self.pairs) - 1)
        print(temp)
        x, y = self.pairs[random.randint(0, temp)]
        print(self.received[y])
        print(self.sent[x])

    def cache_results(self, train_X, test_X, train_y, test_y, simulacrum_name=os.getenv("SIMULACRUM_NAME")):
        if (bool(os.getenv("ENABLE_CACHING"))):
            print("Caching Results...")
            np.savetxt(f"data/train_X_{simulacrum_name}.csv", train_X, delimiter=",")
            np.savetxt(f"data/test_X_{simulacrum_name}.csv", test_X, delimiter=",")
            np.savetxt(f"data/train_y_{simulacrum_name}.csv", train_y, delimiter=",")
            np.savetxt(f"data/test_y_{simulacrum_name}.csv", test_y, delimiter=",")
            print("Cached!")
        else:
            print("Caching data is not enabled in the .env.")

    def load_cache(self, simulacrum_name=os.getenv("SIMULACRUM_NAME")):
        print("Loading from cache")
        if bool(os.getenv("ENABLE_CACHING")):
            return np.genfromtxt(f"data/train_X_{simulacrum_name}.csv", delimiter=","), \
                   np.genfromtxt(f"data/test_X_{simulacrum_name}.csv", delimiter=","), \
                   np.genfromtxt(f"data/train_y_{simulacrum_name}.csv", delimiter=","), \
                   np.genfromtxt(f"data/test_y_{simulacrum_name}.csv", delimiter=",")
        else:
            print("Caching data is not enabled in the .env.")


# processed = DataProcessor(os.getenv("SIMULACRUM_NAME")).extract()

# train_X, test_X, train_y, test_y = DataProcessor(os.getenv("SIMULACRUM_NAME")).process()
# print(train_X)
# print(train_X.shape)
# print(train_y.shape)
#
# print(test_X.shape)
# print(test_y.shape)
