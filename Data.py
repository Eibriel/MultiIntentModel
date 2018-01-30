import json
import numpy as np


class Data:
    def __init__(self):
        characters = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
        signs = "¿ ? ¡ ! . , ; : @ _ - + * < > ( ) / $ 0 1 2 3 4 5 6 7 8 9"
        accents = "` ´ ¨ ^"
        space = " "
        uppermark = "^"

        padding = 0
        unknown = 1

        self.relations = [
            "have",
            "is",
            "do",
            "model_is",
            "size",
            "want",
            "problem_with",
            "cant"
        ]

        self.relations_ = [
            "have",
            "drink"
        ]

        self.items = [
            "sender",
            "iphone",
            "screen",
            "receiver",
            "batery_change",
            "screen_change",
            "6",
            "6s",
            "5s",
            "apple_watch",
            "42mm",
            "sound",
            "6plus",
            "estimate_cost",
            "service",
            "turn_off",
            "spare_info",
            "cost_info"
        ]

        self.items_ = [
            "i",
            "ball",
            "water"
        ]

        self.vocab = characters.split(" ") + signs.split(" ") + accents.split(" ") + [space] + [uppermark]

    def message_to_ints(self, text, batch_count):
        text = text.replace("Á", "´^a")
        text = text.replace("É", "´^e")
        text = text.replace("Í", "´^i")
        text = text.replace("Ó", "´^o")
        text = text.replace("Ú", "´^u")
        #
        text = text.replace("á", "´a")
        text = text.replace("é", "´e")
        text = text.replace("í", "´i")
        text = text.replace("ó", "´o")
        text = text.replace("ú", "´u")
        #
        text = text.replace("À", "`^a")
        text = text.replace("È", "`^e")
        text = text.replace("Ì", "`^i")
        text = text.replace("Ò", "`^o")
        text = text.replace("Ù", "`^u")
        #
        text = text.replace("à", "`a")
        text = text.replace("è", "`e")
        text = text.replace("ì", "`i")
        text = text.replace("ò", "`o")
        text = text.replace("ù", "`u")
        #
        text = text.replace("Ä", "¨^a")
        text = text.replace("Ë", "¨^e")
        text = text.replace("Ï", "¨^i")
        text = text.replace("Ö", "¨^o")
        text = text.replace("Ü", "¨^u")
        #
        text = text.replace("ä", "¨a")
        text = text.replace("ë", "¨e")
        text = text.replace("ï", "¨i")
        text = text.replace("ö", "¨o")
        text = text.replace("ü", "¨u")
        #
        text = text.replace("ñ", "~n")
        text = text.replace("Ñ", "~^N")
        #
        text = text.replace("A", "^a")
        text = text.replace("B", "^b")
        text = text.replace("C", "^c")
        text = text.replace("D", "^d")
        text = text.replace("E", "^e")
        text = text.replace("F", "^f")
        text = text.replace("G", "^g")
        text = text.replace("H", "^h")
        text = text.replace("I", "^i")
        text = text.replace("J", "^j")
        text = text.replace("K", "^k")
        text = text.replace("L", "^l")
        text = text.replace("M", "^m")
        text = text.replace("N", "^n")
        text = text.replace("O", "^o")
        text = text.replace("P", "^p")
        text = text.replace("Q", "^q")
        text = text.replace("R", "^r")
        text = text.replace("S", "^s")
        text = text.replace("T", "^t")
        text = text.replace("U", "^u")
        text = text.replace("V", "^v")
        text = text.replace("W", "^w")
        text = text.replace("X", "^x")
        text = text.replace("Y", "^y")
        text = text.replace("Z", "^z")
        #
        ints = np.zeros(batch_count, dtype=np.int32)
        for char in range(len(text)):
            if text[char] in self.vocab:
                ints[char] = self.vocab.index(text[char]) + 2
            else:
                ints[char] = 1
        return ints

    def message_to_batch(self, message, batch_length):
        start = 1
        end = 1 + batch_length
        for n in range(batch_length):
            message = [0] + message
            message = message + [0]
        batches = []
        for n in range(len(message) - batch_length - 1):
            batches.append(message[start + n:end + n])
        return batches

    def run(self, with_batches=True):
        with open("selected_data.json") as data_file:
            selected_data = json.load(data_file)
        selected_data = selected_data[:6]
        selected_data_ = [
            [
                "i have a ball",
                [
                    "i have ball"
                ]
            ],
            [
                "i drink water",
                [
                    "i drink water"
                ]
            ],
            [
                "i have a ball and drink water",
                [
                    "i drink water",
                    "i have ball"
                ]
            ]
        ]
        #
        questions_all = []
        for message in selected_data:
            for question in message[1]:
                if question not in questions_all:
                    questions_all.append(question)
        self.questions_all = questions_all
        #
        int_messages = []
        batch_length = 5
        batch_count = 500
        messages_x_y = []
        # np_size = (len(selected_data), (batch_count * batch_length) + len(questions_all))
        np_size_x = (len(selected_data), batch_count)
        np_size_y = (len(selected_data), len(questions_all) * 2)
        data_np_x = np.zeros(np_size_x, dtype=np.int32)
        data_np_y = np.zeros(np_size_y, dtype=np.int32)
        for message in range(len(selected_data)):
            int_message = self.message_to_ints(selected_data[message][0], batch_count)
            # int_message_batch = self.message_to_batch(int_message, batch_length)
            for message_char in range(len(int_message)):
                data_np_x[message][message_char] = int_message[message_char]
            for question in range(0, len(questions_all * 2), 2):
                question_value_true = 0
                question_value_false = 1
                if questions_all[int(question * 0.5)] in selected_data[message][1]:
                    question_value_true = 1
                    question_value_false = 0
                data_np_y[message][question] = question_value_true
                data_np_y[message][question + 1] = question_value_false
        return data_np_x, data_np_y
