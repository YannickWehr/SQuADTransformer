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
        self.has_reverse_we = False

        self.topic_no = 0
        self.paragraph_no = 0
        self.question_no = 0

    def load_we_from_file(self, path="./glove6B300d.txt", use_int=False):
        we_dict = {}
        with open(path, 'r') as f:
            l = f.readlines()
            for i in range(len(l)):
                k = l[i].split()
                if use_int == True: 
                    we_dict[k[0]] = np.array([int(float(h)*100000) for h in k[1:]])
                else:
                    we_dict[k[0]] = np.array([float(h) for h in k[1:]])
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

    def load_random_question(self): 
        total_topics = len(self.dataset)
        topic_num = np.random.randint(0, total_topics)

        total_paragraphs = len(self.dataset[topic_num]['paragraphs'])
        paragraph_num = np.random.randint(0, total_paragraphs)

        total_questions = len(self.dataset[topic_num]['paragraphs'][paragraph_num]['qas'])
        question_num = np.random.randint(0, total_questions)

        question, answer, context = self.load_question(topic_num, paragraph_num, question_num)
      
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
    
    def add_average(self, path="./glove.6B.50d.txt"):
        text = "<UNK>" + " -0.12920076 -0.28866628 -0.01224866 -0.05676644 -0.20210965 -0.08389011 0.33359843  0.16045167  0.03867431  0.17833012  0.04696583 -0.00285802 0.29099807  0.04613704 -0.20923874 -0.06613114 -0.06822549  0.07665912 0.3134014   0.17848536 -0.1225775  -0.09916984 -0.07495987  0.06413227 0.14441176  0.60894334  0.17463093  0.05335403 -0.01273871  0.03474107 -0.8123879  -0.04688699  0.20193407  0.2031118  -0.03935686  0.06967544 -0.01553638 -0.03405238 -0.06528071  0.12250231  0.13991883 -0.17446303 -0.08011883  0.0849521  -0.01041659 -0.13705009  0.20127155  0.10069408 0.00653003  0.01685157"
        with open(path, 'a') as f:
              f.write(text)

    def add_eos(self):
        self.word_embeddings["<EOS>"] = np.random.rand(len(list(self.word_embeddings.values())[0]))    
    
    def lookup_we(self, word, inverse=False):
        if inverse == False:
            word_embeddings = self.word_embeddings
        else:
            word_embeddings = self.inverse_we
        if word in word_embeddings:
            return word_embeddings[word]
        else:
            try:
                return word_embeddings["<UNK>"]
            except KeyError:
                print("Warning! UNK is not defined in the Word Embeddings file.")
                return self.calc_average(write_to_file=False)
            
    def return_we(self, words, uncase=True):
        word_vectors = []
        if uncase==True:
          words = words.lower()
        words = words.split()  
        
        for word in words:
            word_vectors.append(self.lookup_we(word))
        return np.array(word_vectors)

    def create_inverse_we(self):
        inv_we = {v.tobytes(): k for k, v in self.word_embeddings.items()}
        self.inverse_we = inv_we
        self.has_reverse_we = True
      
    def return_inverse_we(self, arrays):
      words = []
      for array in arrays:
        words.append(self.lookup_we(array.tobytes(), inverse=True))
      return words