import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from sentencepiece import SentencePieceProcessor

def accuracy_graph(path, start=2):
    f = open(path)
    text = f.readlines()
    text = text[start:]
    acc = []
    for i in range(len(text)):
        if i % 14 == 0:
            acc.append(float(text[i].split()[-1]))
    f.close()
    return acc

def moving_average(numbers, window_size=3):
    i = 0
    moving_averages = []
    while i < len(numbers) - window_size + 1:
        this_window = numbers[i : i + window_size]
        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1
    return moving_averages

def graph():
    logs = ['./logs/squad_ed.txt']
    start = 2
    for log in logs:
        plt.plot(moving_average(accuracy_graph(log, start=start)))
    plt.show()

graph()

#ds = tfds.load('cnn_dailymail', split='train')
#ds = ds.take(1)
#for example in tfds.as_numpy(ds):
#    input(example)