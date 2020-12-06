import os
from processor import DataProcessor
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from dotenv import load_dotenv
load_dotenv()

# Potentially extend nn.Module ?
# Word to vec pytorch use as embedding layer
class SimulacrumClassifier():

    def __init__(self):
        # super().__init__()
        processor = DataProcessor(os.getenv("SIMULACRUM_NAME"))
        self.train_X, self.test_X, self.train_y, self.test_y = processor.process()
        self.num_epoch = int(os.getenv("CLASSIFIER_NUM_EPOCH"))
        self.model = nn.Sequential(
            # nn.Embedding(200, 1),
            # nn.ReLU(),
            # nn.MaxPool1d(1),
            # nn.Flatten(),
            nn.Linear(200, 1)
        )
        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.001)


    def train(self):
        temp_X = Variable(torch.Tensor(self.train_X))
        temp_y =  torch.unsqueeze(Variable(torch.Tensor(self.train_y)), 1)
        for epoch in range(self.num_epoch):
            y_predict = self.model(temp_X)
            # print(y_predict[0], temp_y[0])
            loss_value = self.loss(y_predict, temp_y)
            self.model.zero_grad()
            loss_value.backward()
            self.optimizer.step()
            if epoch == 1 or epoch % 50 == 0:
                print(f"Epoch {epoch} had training loss {loss_value}")


# classifier = SimulacrumClassifier()
# classifier.train()
# torch.save(classifier.model, "data/model.pt")