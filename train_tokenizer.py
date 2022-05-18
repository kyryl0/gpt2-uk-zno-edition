import datasets
from src.make_datasets import make_sentence_files
import cfg
from src.make_datasets import sample_and_make_tempfile
import sentencepiece as spm
import time
dataset_uk = datasets.load_dataset("path/to/dataset")
dataset_uk["train"][961563]
make_sentence_files(dataset_uk["train"])
vocab_sizes = [8000, 16000, 32000, 48000]

def train_uk(vocab_size):
    start = time.time()
    model_prefix = "cc100_uk" + "_vocab_" + str(vocab_size)
    spm.SentencePieceTrainer.train(
    input = tempfile_path,
    model_prefix = model_prefix,
    vocab_size = vocab_size,
    character_coverage = 0.9995,
    num_threads = 60,
    train_extremely_large_corpus = True
    )
    print("Trained {} in {} seconds".format(model_prefix,time.time()-start))

for vocab_size in vocab_sizes:
    train_uk(vocab_size)
