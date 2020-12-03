from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings(action='ignore')
import gensim
from gensim.models import Word2Vec
from processor import DataProcessor
import os
from dotenv import load_dotenv
load_dotenv()

class Aggregator:

    def __init__(self):
        self.vocabulary = []
        self.labels = []

    def train(self, input=None):
        if input == None:
            data = self.vocabulary
        else:
            data = self.build_vocabulary(input)

        # Create CBOW model
        return gensim.models.Word2Vec(data, min_count=1, size=100)


    def build_vocabulary(self, input, label=None):
        data = []
        labels = []
        for f in input:
            # iterate through each sentence in the file
            for i in sent_tokenize(f.replace("\n", " ")):
                temp = []

                # tokenize the sentence into words
                for j in word_tokenize(i):
                    temp.append(j.lower())

                if label != None: labels.append(label)
                data.append(temp)

        self.labels = labels
        self.vocabulary = self.vocabulary + data
        return data


# processor = DataProcessor(os.getenv("SIMULACRUM_NAME")).extract()
# vocab = processor.sent + processor.received
# aggr = Aggregator()
# aggr.build_vocabulary(vocab)
#
# vec_model = aggr.train()
