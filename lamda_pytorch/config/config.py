from typing import Optional, ClassVar
from dataclasses import dataclass, field

@dataclass
class CFG:

    """
    Configuration for ZeRO
    """

    use_zero: bool = field(
        default = False,
        metadata = {'help': 'whether to use zero'}
    )

    """
    Configuration for optimizer
    """

    lr: float = field(
        default = 0.001,
        metadata = {'help': 'learning rate'}
    )

    """
    Configuration class for LaMDA model.
    """

    num_tokens: int = field(
        default = 50257,
        metadata = {'help': 'number of tokens'}
    )

    dim: int = field(
        default = 1024,
        metadata = {'help': 'dimension of the embedding'}
    )

    depth: int = field(
        default = 6,
        metadata = {'help': 'depth of the transformer'}
    )

    heads: int = field(
        default = 4,
        metadata = {'help': 'number of heads in the transformer'}
    )

    dim_head: int = field(
        default = 64,
        metadata = {'help': 'dimension of the head'}
    )

    """
    Configuration for data loader.
    """

    use_huggingface: bool = field(
        default = True,
        metadata = {'help': 'Whether to use huggingface datasets'}
    )

    train_dataset_name: Optional[str] = field(
        default="the_pile", 
        metadata={"help": "Path to Hugging Face training dataset."}
    )

    eval_dataset_name: Optional[str] = field(
        default="the_pile", 
        metadata={"help": "Path to Hugging Face validation dataset."}
    )

    choose_train_split: Optional[str] = field(
        default="train", 
        metadata={"help": "Choose Hugging Face training dataset split."}
    )

    choose_eval_split: Optional[str] = field(
        default="train", 
        metadata={"help": "Choose Hugging Face validation dataset split."}
    )

    train_columns: ClassVar[list[str]] = field(
        default = ['meta'], 
        metadata={"help": "Train dataset columns to remove."}
    )

    eval_columns: ClassVar[list[str]] = field(
        default = ['meta'], 
        metadata={"help": "Validation dataset columns to remove."}
    )

    seed: Optional[int] = field(
        default=42, 
        metadata={"help": "Random seed used for reproducibility."}
    )

    tokenizer_seq_length: Optional[int] = field(
        default=1024, 
        metadata={"help": "Sequence lengths used for tokenizing examples."}
    )

    select_input_string: Optional[str] = field(
        default="text", 
        metadata={"help": "Select the key to used as the input string column."}
    )

    set_format: Optional[str] = field(
        default="torch", 
        metadata={"help": "Convert the format to PyTorch Tensors"}
    )
    
    batch_size: Optional[int] = field(
        default=4, 
        metadata={"help": "Batch size for training and validation."}
    )

    save_to_path: Optional[str] = field(
        default="''", 
        metadata={"help": "Save the dataset to local disk."}
    )

    """
    Configuration for Weights and Biases
    """

    use_wandb: bool = field(
        default = False,
        metadata = {'help': 'Whether to use Weights and Biases for logging'}
    )

    project_name: Optional[str] = field(
        default="LaMDA pre-training",
        metadata = {'help': 'Name of the project'}
    )