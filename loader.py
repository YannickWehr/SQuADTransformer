import json
import numpy as np

np.set_printoptions(suppress=True,
   formatter={'float_kind':'{:.5f}'.format})

class Loader():
    def __init__(self, path="./train-v2.0.json"):
        dataset = open(path)
        dataset = dataset.read()
        dataset = json.loads(dataset)
        dataset = dataset["data"]
        self.dataset = dataset
        
        self.topic_no = 0
        self.paragraph_no = 0
        self.question_no = 0

    def load_we_from_file(self, path="./glove6B300d.txt"):
        we_dict = {}
        with open(path, 'r') as f:
            l = f.readlines()
            for i in range(len(l)):
                k = l[i].split()
                we_dict[k[0]] = np.array([int(float(h)*100000) for h in k[1:]])
        self.word_embeddings = we_dict

    def load_question(self, topic_no, paragraph_no, question_no):
        topic = self.dataset[topic_no]
        paragraph = topic['paragraphs'][paragraph_no]
        question = paragraph['qas'][question_no]['question']
        is_impossible = paragraph['qas'][question_no]['is_impossible']
        if not is_impossible:
            answer = paragraph['qas'][question_no]['answers'][0]['text'] #TODO Multiple answers
        else:
            answer = '<IMPOSSIBLE>' #TODO Is there a better way to do this?
        context = paragraph['context']
        return (question, answer, context)

    def load_next_question(self):
        question, answer, context = self.load_question(self.topic_no, self.paragraph_no, self.question_no)
        if (self.paragraph_no + 1) >= len(self.dataset[self.topic_no]['paragraphs']):
            self.topic_no += 1
            self.paragraph_no = 0
            self.question_no = 0
        elif (self.question_no + 1) >= len(self.dataset[self.topic_no]['paragraphs'][self.paragraph_no]['qas']):
            self.paragraph_no += 1
            self.question_no = 0
        else:
            self.question_no += 1

        return question, answer, context
    
    def calc_average(self, path="./glove6B300d.txt", write_to_file=True):
        avg = []
        with open(path, 'r') as f:
            l = f.readlines()
            for i in range(len(l)):
                k = l[i].split()
                avg.append([float(h) for h in k[1:]])
        avg = np.array(avg)
        avg = np.mean(avg, axis=0)
        avg = "<UNK> " + np.array2string(
            avg, precision=5, suppress_small=True, max_line_width=99999999)[1:-1] #There seems to be no option to set max_line_width to infinity
        if write_to_file:
            with open(path, 'a') as f:
                f.write(avg+"\n")
        else:
            return avg

    def lookup_we(self, word):
        if word in self.word_embeddings:
            return self.word_embeddings[word]
        else:
            try:
                return self.word_embeddings["<UNK>"]
            except KeyError:
                print("Warning! UNK is not defined in the Word Embeddings file.")
                return self.calc_average(write_to_file=False)
            
    def return_we(self, words, uncase=True):
        word_vectors = []
        words = words.lower()
        words = words.split()
        for word in words:
            word_vectors.append(self.lookup_we(word))
        return np.array(word_vectors)
