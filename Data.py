import sys
import json
import random
import numpy as np

from simple_data import simple_data


class Data:
    def __init__(self, data_source=0):
        characters = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
        signs = "¿ ? ¡ ! . , ; : @ _ - + * \" ' < > ( ) / $ 0 1 2 3 4 5 6 7 8 9"
        accents = "` ´ ¨ ^"
        space = " "
        uppermark = "^"

        self.special = {
            "padding": 0,
            "unknown": 1,
            "start": 2,
            "end": 3
        }
        self.vocab = characters.split(" ") + signs.split(" ") + accents.split(" ") + [space] + [uppermark]
        if data_source == 0:
            self.relations = [
                "have",
                "is",
                "do",
                "model_is",
                "size",
                "want",
                "problem_with",
                "cant",
                "says",
                "hello"
            ]
            self.items = [
                "sender",
                "iphone",
                "fake",
                "ipad",
                "ipod",
                "macbook",
                "screen",
                "receiver",
                "batery_change",
                "screen_change",
                "speaker_change",
                "iphone_7_plus",
                "iphone_7",
                "iphone_6",
                "iphone_6s",
                "iphone_6_plus",
                "iphone_5",
                "iphone_5s",
                "ipad_3rdgen",
                "ipod_touch",
                "apple_watch",
                "42mm",
                "sound",
                "service",
                "turn_on",
                "spare_info",
                "cost_info",
                "time_info",
                "charge_pin",
                "charge",
                "hello",
                "thanks",
                "opening_hours_info",
                "address_info",
                "payment_info",
                "unlock_service",
                "finishing_info",
                "wet",
                "accesory_info",
                "back_camera_change",
                "ipad_2",
                "ipod_null",
                "iphone_4s",
                "ipad_air",
                "ipad_mini_2",
                "wifi_issues",
                "proximity_change",
                "ipad_mini",
                "ipod_nano",
                "burned",
                "iphone_5c",
                "flash_issues",
                "appointment_info",
                "screen_protector",
                "iphone_se",
                "volume_issues",
                "iphone_4",
                "iphone_8",
                "shell_change",
                "fixing_issues",
                "iphone_8_plus",
                "ipad_air_2",
                "ipad_4thgen",
                "ipad_1stgen",
                "charge_pin_change",
                "warranty_info",
                "home_button_change",
                "iphone_6s_plus"
            ]
            with open("train_data.json") as data_file:
                all_data = json.load(data_file)
            # Collect all questions
            self.questions_count = {}
            for message in all_data:
                for question in message[1]:
                    if question not in self.questions_count:
                        self.questions_count[question] = 0
                    else:
                        self.questions_count[question] += 1
            self.questions_all = []
            for message in all_data:
                for question in message[1]:
                    if self.questions_count[question] < 500:
                        continue
                    if question not in self.questions_all:
                        self.questions_all.append(question)
            # Spliting data into Training and Dev
            self.selected_data = all_data[:-100]
            self.dev_data = all_data[-100:]
            # Augmenting data
            tmp_data = []
            for data in self.selected_data:
                tmp_data.append(data)
                if random.random() > 0.9:
                    tmp_data.append([data[0].lower(), data[1]])
                    # print(tmp_data[-1][0])
                if random.random() > 0.9:
                    tmp_data.append([data[0].upper(), data[1]])
                    # print(tmp_data[-1][0])
                if random.random() > 0.9:
                    temp_text = ""
                    for char in data[0]:
                        if random.random() > 0.7:
                            char = char.upper()
                        elif random.random() > 0.7:
                            char = char.lower()
                        temp_text = "{}{}".format(temp_text, char)
                    tmp_data.append([temp_text, data[1]])
                    # print(tmp_data[-1][0])
                if random.random() > 0.9:
                    temp_text = ""
                    for char in data[0]:
                        if char == " " and random.random() > 0.91:
                            continue
                        if random.random() > 0.95:
                            char = "{}{}".format(" ", char)
                        temp_text = "{}{}".format(temp_text, char)
                    tmp_data.append([temp_text, data[1]])
                    # print(tmp_data[-1][0])
                if random.random() > 0.9:
                    temp_text = ""
                    for n_char in range(random.randint(0, 700)):
                        temp_text = "{}{}".format(temp_text, random.choice(self.vocab))
                    tmp_data.append([temp_text, []])
            self.selected_data = tmp_data
        elif data_source == 1:
            self.relations = [
                "have",
                "drink"
            ]
            self.items = [
                "i",
                "ball",
                "water"
            ]
            self.selected_data = simple_data


    def message_to_ints(self, text, max_sentence_length=None):
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
        if max_sentence_length is None:
            max_sentence_length = len(text) + 2  # Text length + start + end
        ints = np.zeros(max_sentence_length, dtype=np.int32)
        for char in range(len(text)):
            char_key = -char - 2
            chat_text = len(text) - char - 1
            if text[chat_text] in self.vocab:
                ints[char_key] = self.vocab.index(text[chat_text]) + len(self.special.keys())
            else:
                ints[char_key] = self.special["unknown"]
        ints[-len(text) - 1 - 1] = self.special["start"]
        ints[-1] = self.special["end"]
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

    def prepare_data(self, data_to_prepare):
        # Some item or relation is not in the lists?
        # then exit.
        missing_thing = False
        for question in self.questions_all:
            q_list = question.split(" ")
            if q_list[0] not in self.items:
                print("item", q_list[0])
                missing_thing = True
            if q_list[2] not in self.items:
                print("item", q_list[2])
                missing_thing = True
            if q_list[1] not in self.relations:
                print("relation", q_list[1])
                missing_thing = True
        if missing_thing:
            sys.exit()
        # Convert messages to ints (Only to know the length)
        int_data = []
        for data in data_to_prepare:
            int_message = self.message_to_ints(data[0])
            int_data.append(int_message)
        # Separate data into buckets
        buckets_config = [50, 100, 300, 600, 1000]
        data_bk_x = []
        data_bk_y = []
        batches_x = []
        batches_y = []
        for bucket_key in range(len(buckets_config)):
            # Get min and max message length for the bucket
            if bucket_key == 0:
                min_length = 0
            else:
                min_length = buckets_config[bucket_key - 1]
            max_length = buckets_config[bucket_key]
            # Mask current data acording to min and max length
            masked_data = []
            for data_key in range(len(int_data)):
                if int_data[data_key].shape[0] >= min_length and int_data[data_key].shape[0] < max_length:
                    masked_data.append(data_to_prepare[data_key])
                    if random.random() > 0.6:
                        masked_data.append(["", []])
            print("Buckets: ", buckets_config[bucket_key], len(masked_data))
            # Create numpy arrays for X and Y
            np_size_x = (len(masked_data), max_length)
            np_size_y = (len(masked_data), len(self.questions_all))
            data_np_x = np.zeros(np_size_x, dtype=np.int32)
            data_np_y = np.zeros(np_size_y, dtype=np.int32)
            for message in range(len(masked_data)):
                int_message = self.message_to_ints(masked_data[message][0], max_length)
                for message_char in range(len(int_message)):
                    data_np_x[message][message_char] = int_message[message_char]
                for question in range(0, len(self.questions_all), 1):
                    question_value_true = 0
                    question_value_false = 1
                    if self.questions_all[question] in masked_data[message][1]:
                        question_value_true = 1
                        question_value_false = 0
                    data_np_y[message][question] = question_value_true
                data_bk_x.append(data_np_x)
                data_bk_y.append(data_np_y)
        # Generate mixed buckets data
        batch_sizes = [100, 50, 10, 3, 1]
        for bk in range(len(data_bk_x)):
            x = data_bk_x[bk]
            y = data_bk_y[bk]
            for batch_size in batch_sizes:
                if float(len(x)) / batch_size > 1.0:
                    break
            for nn in range(0, len(x) - batch_size, batch_size):
                batches_x.append(x[nn:nn + batch_size])
                batches_y.append(y[nn:nn + batch_size])
            batches_x.append(x[-batch_size - 1:-1])
            batches_y.append(y[-batch_size - 1:-1])
        # Dev data

        return batches_x, batches_y

    def run(self):
        batches_x, batches_y = self.prepare_data(self.selected_data)
        batches_dev_x, batches_dev_y = self.prepare_data(self.dev_data)
        return batches_x, batches_y, batches_dev_x, batches_dev_y
