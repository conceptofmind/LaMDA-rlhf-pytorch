from datasets import load_dataset
#from tokenizers import SentencePieceBPETokenizer
import io
import sentencepiece as spm

dataset = load_dataset('conceptofmind/pile_wikipedia_en', split='train', streaming=True)

# tokenizer = SentencePieceBPETokenizer()

def batch_iterator(dataset):
    for i in dataset:
        yield i["text"]
        
model = io.BytesIO()

spm.SentencePieceTrainer.train(
    sentence_iterator = batch_iterator(dataset), 
    model_writer=model, 
    vocab_size=32000,
    model_type='bpe',
)

# tokenizer.train_from_iterator(
#     text='',
#     vocab_size=32_000,
#     min_frequency=2,
#     show_progress=True,
# )

# tokenizer.save()
