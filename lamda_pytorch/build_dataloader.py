import copy
from itertools import chain
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import get_world_size
from .config.config import CFG
from transformers import AutoTokenizer, default_data_collator


def build_dataloaders(args: CFG, tokenizer: AutoTokenizer):
    """
    Build streaming dataloaders for the PaLM model.
    Useful for low RAM and storage environments.
    Requires stable internet connection.
    """

    # Load training dataset
    load_train_data = load_dataset(args.train_dataset_name, "enron_emails", split = args.choose_train_split)

    # Remove unused columns from the training dataset
    load_train_data = load_train_data.remove_columns(args.remove_train_columns)

    # Load validation dataset
    load_eval_data = load_dataset(args.eval_dataset_name, "enron_emails", split = args.choose_eval_split)

    # Remove unused columns from the validation dataset
    load_eval_data = load_eval_data.remove_columns(args.remove_eval_columns)

    # Shuffle the training input files. Set a buffer size to
    shuffled_train_files = load_train_data.shuffle(seed = args.seed)

    # Shuffle the validation input files. Set a buffer size to
    shuffled_eval_files = load_eval_data.shuffle(seed = args.seed)

    """
    A sequence length of 2048 is used for the model. Input examples are concatenated
    together and then split into sequences of exactly 2048 tokens, so that there are 
    no padding tokens, but examples may be split in the middle.

    PaLM: Scaling Language Modeling with Pathways:
    https://arxiv.org/pdf/2204.02311.pdf

    Tokenize function reference:
    https://github.com/hpcaitech/PaLM-colossalai/blob/main/data/wikitext.py
    """

    def tokenize(examples):
        seq_length = args.tokenizer_seq_length
        examples = tokenizer(examples[args.select_input_string])
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= seq_length:
            total_length = (total_length // seq_length) * seq_length

        result = {
            k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
            for k, t in concatenated_examples.items()
        }

        result["labels"] = copy.deepcopy(result["input_ids"])

        return result
    
    """
    Map the tokenization function to the shuffled training files to create an 
    Iterable training dataset of batched input sequences of 2048 tokens.
    Remove columns from the the shuffled training files so that you are left with 
    only the input_ids, attention_mask, and labels columns.
    """
    
    tokenized_train_dataset = shuffled_train_files.map(tokenize, batched = True, remove_columns = [args.select_input_string])

    """
    Map the tokenization function to the shuffled validation files to create an 
    Iterable validation dataset of batched input sequences of 2048 tokens.
    Remove columns from the the shuffled training files so that you are left with 
    only the input_ids, attention_mask, and labels columns.
    """
    
    tokenized_eval_dataset = shuffled_eval_files.map(tokenize, batched = True, remove_columns = [args.select_input_string])

    # Convert the format of the tokenized train dataset to PyTorch Tensors
    #train_with_torch = tokenized_train_dataset.set_format(type = "torch")

    # Convert the format of the tokenized validation dataset to PyTorch Tensors
    #eval_with_torch = tokenized_eval_dataset.set_format(type = "torch")

    # Train dataset used for sampling.
    sample_train_dataset = DistributedSampler(tokenized_train_dataset) if get_world_size() > 1 else None

    # Validation dataset used for sampling.
    sample_eval_dataset = DistributedSampler(tokenized_eval_dataset) if get_world_size() > 1 else None

    # Create the train dataloader. If the length of a tokenized input sequence is less than 2048 drop it.
    train_dataloader = DataLoader(tokenized_train_dataset, shuffle = True, sampler = sample_train_dataset, drop_last = True, collate_fn = default_data_collator, batch_size = args.batch_size)

    # Create the validation dataloader. If the length of a tokenized input sequence is less than 2048 drop it.
    eval_dataloader = DataLoader(tokenized_eval_dataset, sampler = sample_eval_dataset, drop_last = True, collate_fn = default_data_collator, batch_size = args.batch_size)

    # Return the training and validation dataloaders to be used in the model
    print('Done building dataloaders')
    return train_dataloader, eval_dataloader

if __name__ == '__main__':
    
    # Get Dataloader Configuration Arguments
    data_loader_args = CFG()

    # Get Tokenizer Configuration Arguments
    tokenizer_args = 'gpt2'

    # Load the pretrained tokenizer of your choosing
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_args)

    # Test Build Dataloaders
    train_loader, eval_loader = build_dataloaders(args = data_loader_args, tokenizer = tokenizer)

    print(next(iter(train_loader))['input_ids'])
    print(next(iter(train_loader))['input_ids'].shape)