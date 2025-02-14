from datasets import load_dataset
import numpy as np
import tiktoken
import os
from tqdm import tqdm # pip install tqdm

batch_size = 16
shuffled_dataset_num = 1
max_lenght = 2048 
shard_size = int(1e8)
DATA_CACHE_DIR = "./fineweb10B-edu"

fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")

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

import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

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


# shard_index = 0
#     # preallocate buffer to hold current shard
# all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
# token_count = 0
# progress_bar = None

# for tokens in ds:
#     if token_count + len(tokens) < shard_size:
#         # simply append tokens to current shard
#         print(f"Lenght of tokens {len(tokens)}")
#         all_tokens_np[token_count:int(token_count+len(tokens))] = tokens
#         token_count += len(tokens)
#         # update progress bar
#         if progress_bar is None:
#             progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
#         progress_bar.update(len(tokens))
#     else:
#             # write the current shard and start a new one
#         split = "val" if shard_index == 0 else "train"
#         filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")            # split the document into whatever fits in this shard; the remainder goes to next one
#         remainder = shard_size - token_count
#         progress_bar.update(remainder)
#         all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
#         write_datafile(filename, all_tokens_np)
#         shard_index += 1
#         progress_bar = None
#         # populate the next shard with the leftovers of the current doc
#         all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
#         token_count = len(tokens)-remainder

#     # write any remaining tokens as the last shard
# if token_count != 0:
#     split = "val" if shard_index == 0 else "train"
#     filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
#     write_datafile(filename, all_tokens_np[:token_count])

# for i in shuffled_dataset:
#     encoded_text = enc.encode(str(i["text"])) 

#     if len(encoded_text) < max_lenght:
#           print("Adding padding tokens")
#           for _ in range(max_lenght - len(encoded_text)):
#               encoded_text.append(200019)
#     else:
#         encoded_text = encoded_text[:max_lenght]

#     data.append({
#                 "text": str(i["text"]),
#                 "tokenized_text": encoded_text
#                 })

# print("Data tokenized")

# if not os.path.exists(f"./data/{shuffled_dataset_num}.json"):
#     open(f"./data/{shuffled_dataset_num}.json", "w").close()
#     print("Create file")

# print("Checked if file exists")

# with open(f"./data/{shuffled_dataset_num}.json", "a") as f:
#             print("Writing to file")
#             json.dump(data, f)

# print("Data written to file")