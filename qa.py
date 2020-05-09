import numpy as np
import trax
import os
import loader

#Dataset structure:
#[Topic1, Topic2, ...] --> {title, paragraphs: [Paragraph1, Paragraph2, ...]} --> {qas:[{question, id, answers: [{text, answer_start}], is_impossible}], context}
squad_loader = loader.Loader()
squad_loader.load_we_from_file(path="./glove6B300d.txt")
print("SQuAD Data and Word Embeddings imported...")

def encode(question, answer, context):
  question_we = squad_loader.return_we(question)
  answer_we = squad_loader.return_we(answer)
  context_we = squad_loader.return_we(context)
  return question_we, answer_we, context_we

def input_function(n_devices):
  while True:
    inputs = []
    outputs = []

    for i in range(n_devices):
      question, answer, context = squad_loader.load_next_question()
      question_id, answer_id, context_id = encode(question, answer, context)
      inputs.append(np.concatenate((question_id, context_id)))
      outputs.append(answer_id)
    inputs = np.concatenate(inputs)
    outputs = np.concatenate(outputs)
    yield (inputs, outputs)

def lm_input_function(n_devices):
  while True:
    inputs = []
    outputs = []

    for i in range(n_devices):
      question, answer, context = squad_loader.load_next_question()
      question_id, answer_id, context_id = encode(question, answer, context)
      inputs.append(np.concatenate((question_id, context_id)))
      outputs.append(answer_id)
    inputs = np.concatenate(inputs)
    outputs = np.concatenate(outputs)
    values = np.concatenate((inputs, outputs))
    mask = np.concatenate((np.zeros_like(inputs), np.ones_like(outputs)))
    yield (values, values, mask)

def my_reformer(mode):
  return trax.models.Reformer(input_vocab_size=400000, d_model=300, mode=mode)

def my_transformer(mode):
  return trax.models.Transformer(input_vocab_size=400000, d_model=300, mode=mode, n_heads=5)

def my_transformerlm(mode):
  return trax.models.TransformerLM(vocab_size=400000, d_model=300, n_heads=5, mode=mode, n_layers=4)


output_dir = os.path.expanduser('~/train_dir/')
trainer = trax.supervised.Trainer(
    model=my_transformerlm,
    loss_fn=trax.layers.CrossEntropyLoss(),
    optimizer=trax.optimizers.Adam,  # Change optimizer params here.
    lr_schedule=trax.lr.MultifactorSchedule,  # Change lr schedule here.
    inputs=trax.supervised.Inputs(lm_input_function),
    output_dir=output_dir)

print("Model created...")


# print("Starting Traning...")
# for _ in range(30):
#   trainer.train_epoch(n_steps=10, n_eval_steps=1)

# print("Staring Evaluation...")
# trainer.evaluate(10)
