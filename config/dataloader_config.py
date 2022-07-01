from typing import Optional, ClassVar
from dataclasses import dataclass, field

@dataclass
class DataLoaderArguments:
    """
    Configuration for data loader.
    """

    train_dataset_name: Optional[str] = field(
        default="", 
        metadata={"help": "Path to Hugging Face training dataset."}
    )

    eval_dataset_name: Optional[str] = field(
        default="", 
        metadata={"help": "Path to Hugging Face validation dataset."}
    )

    choose_train_split: Optional[str] = field(
        default="", 
        metadata={"help": "Choose Hugging Face training dataset split."}
    )

    choose_eval_split: Optional[str] = field(
        default="", 
        metadata={"help": "Choose Hugging Face validation dataset split."}
    )

    train_columns: ClassVar[list[str]] = field(
        default = [], 
        metadata={"help": "Train dataset columns to remove."}
    )

    eval_columns: ClassVar[list[str]] = field(
        default = [], 
        metadata={"help": "Validation dataset columns to remove."}
    )

    train_buffer: Optional[int] = field(
        default=10000, 
        metadata={"help": "Size of buffer used to shuffle streaming dataset."}
    )

    eval_buffer: Optional[int] = field(
        default=1000, 
        metadata={"help": "Size of buffer used to shuffle streaming dataset."}
    )

    seed: Optional[int] = field(
        default=42, 
        metadata={"help": "Random seed used for reproducibility."}
    )

    tokenizer_seq_length: Optional[int] = field(
        default=512, 
        metadata={"help": "Sequence lengths used for tokenizing examples."}
    )

    select_input_string: Optional[str] = field(
        default="content", 
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