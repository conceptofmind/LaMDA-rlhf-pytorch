from datasets import load_dataset
from tokenizers import SentencePieceBPETokenizer

dataset = load_dataset('the_pile/all', split='train', streaming=True)

tokenizer = SentencePieceBPETokenizer()

def batch_iterator(dataset):
    for i in dataset:
        yield i["text"]

tokenizer.train_from_iterator(
    text='',
    vocab_size=32_000,
    min_frequency=2,
    show_progress=True,
)

tokenizer.save()
