import json
import os
import sys
from dotenv import load_dotenv
load_dotenv()


class DataProcessor:

    def __init__(self, simulacrum_name):
        # The full name of the recipient as shown in the various message_X.json files
        self.simulacrum_name = simulacrum_name
        self.sent = []
        self.received = []


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
                for message in data["messages"]:
                    if "content" in message and not any(disallowed in message["content"] for disallowed in disallowed_list) and message["type"] == "Generic":
                        if self.simulacrum_name == message["sender_name"]:
                            self.sent.append(message["content"])
                        else:
                            self.received.append(message["content"])
        return self


processed = DataProcessor(os.getenv("SIMULACRUM_NAME")).process()
print(processed.simulacrum_name)
# print(processed.sent)
# print(processed.received)