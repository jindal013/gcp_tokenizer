import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from google.cloud.storage import Client, transfer_manager
import ray
import time
import socket
from collections import Counter
from ray.util.multiprocessing import Pool

BLUE = '\033[34m'
RESET = '\033[0m'

ray.init() # for main node
assert ray.is_initialized()

# checking if ray works before script

print('''This cluster consists of
    {} nodes in total
    {} CPU resources in total
'''.format(len(ray.nodes()), ray.cluster_resources()['CPU']))

@ray.remote
def f():
    time.sleep(0.001)
    return socket.gethostbyname("localhost")

object_ids = [f.remote() for _ in range(10000)]
ip_addresses = ray.get(object_ids)

print('Tasks executed')
for ip_address, num_tasks in Counter(ip_addresses).items():
    print('    {} tasks on {}'.format(num_tasks, ip_address))

# ------------------------------------------
# code for script

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)

BUCKET_NAME = "test2_10bt_gpt4"
WORKERS = int(os.cpu_count() / 1.5)
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
FILE_NAMES = os.listdir(DATA_CACHE_DIR)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True)

enc = tiktoken.encoding_for_model("gpt-4") # 'cl100k_base'

eot = enc._special_tokens['<|endoftext|>'] 

@ray.remote
def tokenize(doc):
  tokens = [eot]
  tokens.extend(enc.encode_ordinary(doc["text"]))
  tokens_np = np.array(tokens)
  assert (0 <= tokens_np).all() and (tokens_np < 2**32).all(), "token dictionary too large for uint32"
  tokens_np_uint32 = tokens_np.astype(np.uint32)
  return tokens_np_uint32

@ray.remote
def write_datafile(filename, tokens_np):
  np.save(filename, tokens_np)

print(f'{BLUE}hf dataset has been accessed {RESET}') # not downloaded now as we are streaming the dataset


cpu_count = os.cpu_count()

storage_client = Client()
bucket = storage_client.bucket(BUCKET_NAME)

@ray.remote
def upload_file():
  
  @ray.remote
  def upload_many_blobs_with_transfer_manager(bucket_name, filenames, source_directory="", workers=8):

    results = transfer_manager.upload_many_from_filenames(
      bucket, filenames, source_directory=source_directory, max_workers=workers, worker_type=transfer_manager.THREAD
    )

    for name, result in zip(filenames, results):

      if isinstance(result, Exception):
        print("Failed to upload {} due to exception: {}".format(name, result))
      else:
        print("Uploaded {} to {}.".format(name, bucket.name))

  FILE_NAMES = os.listdir(DATA_CACHE_DIR)
  print('------------')
  print(f'starting upload to GCP bucket {BUCKET_NAME}')
  upload_many_blobs_with_transfer_manager.remote(BUCKET_NAME, FILE_NAMES, DATA_CACHE_DIR, WORKERS)
  print(f'done upload to GCP bucket {BUCKET_NAME} for {FILE_NAMES}')
  print('cleaning up files....')
  for file in FILE_NAMES:
    full_path = DATA_CACHE_DIR + '/' + file
    os.remove(full_path)
    print(f'removed {file} successfully')

# def main():
with Pool(ray_address="auto") as pool:
  shard_index = 0
  all_tokens_np = np.empty((shard_size,), dtype=np.uint32)
  token_count = 0
  progress_bar = None
  
  for tokens in pool.map(tokenize.remote, fw, chunksize=32):

    if token_count + len(tokens) < shard_size:
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
      write_datafile.remote(filename, all_tokens_np)
      upload_file.remote()
      shard_index += 1
      progress_bar = None
      # populate the next shard with the leftovers of the current doc
      all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
      token_count = len(tokens)-remainder

  # write any remaining tokens as the last shard
  if token_count != 0:
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
    write_datafile.remote(filename, all_tokens_np[:token_count])
    upload_file.remote()

# if __name__ == '__main__':
#   main()