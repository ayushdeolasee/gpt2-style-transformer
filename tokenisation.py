from datasets import load_dataset
import numpy as np
import tiktoken
import os
from tqdm import tqdm # pip install tqdm
from rich import print
import click
import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# batch_size = 16
# shuffled_dataset_num = 1
# max_lenght = 2048 
# shard_size = int(1e8)
# DATA_CACHE_DIR = "./fineweb10B-edu"

@click.command()
@click.option("--batch_size", default=int(16), help="Batch size to be created")
@click.option("--directory", default="./fineweb10B-edu", help="Path to directory")
@click.option("--max_lenght", default=2048, help="Max lenght of tokens")
@click.option("--dataset-type", default="fineweb-edu", help="Shard size")
@click.option("--name", default="sample-10BT", help="Name of the dataset")
def download(directory, dataset_type, name, batch_size, max_lenght):
    print(f"Staring download of [bold magenta]dataset[/bold magenta]: HuggingFaceFW/{dataset_type} | in [bold blue]directory[/bold blue]: {directory} | with [bold red]max lenght[/bold red]: {max_lenght}")
    shard_size = int(1e8)

    # DATA_CACHE_DIR = "./fineweb10B-edu"
    DATA_CACHE_DIR = directory
    if os.path.exists(directory):
        print(f"[bold green]Directory {directory} already exists[/bold green] :heavy_check_mark:")
    else:
        os.makedirs(directory, exist_ok=True)
        print(f":warning: [bold yellow]Directory {directory} created[/bold yellow]")

    fw = load_dataset(f"HuggingFaceFW/{dataset_type}", name=name, split="train")
    enc = tiktoken.encoding_for_model("gpt2")
    eot = enc._special_tokens['<|endoftext|>']

    def tokenize(doc): 
        tokens = [eot]
        tokens.extend(enc.encode_ordinary(doc["text"]))
        tokens_np = np.array(tokens)
        # assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
        tokens_np_uint16 = tokens_np.astype(np.uint16)
        return tokens_np_uint16

    def write_datafile(filename, tokens_np): 
        np.save(filename, tokens_np)
    
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    # Direct iteration over the dataset
    for doc in fw:
        tokens = tokenize(doc)
        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

# write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])


if __name__ == "__main__":
   download() 