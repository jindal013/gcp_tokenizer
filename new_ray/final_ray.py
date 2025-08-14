import os
import shutil
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from google.cloud.storage import Client, transfer_manager
import argparse
import ray

# -----------------------------
# Connect to Ray cluster
# -----------------------------
# Run: ray start --head on VM1, ray start --address='HEAD_IP:6379' on VM2
ray.init(address="auto")

BLUE = '\033[34m'
RESET = '\033[0m'

local_dir = "data_dir"
remote_name = "sample-350BT"
shard_size = int(1e8)  # 100M tokens per shard

VAL_SPLIT = 350
TEST_SPLIT = 700

BUCKET_NAME = "ray_jaxformer"
WORKERS = int(os.cpu_count())
print('using', WORKERS, 'cpus')

storage_client = Client()
bucket = storage_client.bucket(BUCKET_NAME)

def setup_argument_parser():
    parser = argparse.ArgumentParser(description='Process the 350BT dataset')
    parser.add_argument('--continue', dest='continue_processing', action='store_true',
                        help='Continue processing from a checkpoint')
    parser.set_defaults(continue_processing=False)
    return parser

parser = setup_argument_parser()
args = parser.parse_args()
continue_processing = args.continue_processing
checkpoint_to_resume = None
shard_to_resume = 0

if continue_processing:
    print(f'{BLUE}Continuing processing from checkpoint{RESET}')
    blobs = bucket.list_blobs(prefix="checkpoints/")
    checkpoint_blobs = [b for b in blobs if str(b.name).endswith(".txt")]
    if not checkpoint_blobs:
        print(f'{BLUE}No checkpoints found, starting new processing{RESET}')
    else:
        latest_checkpoint = max(checkpoint_blobs, key=lambda b: b.updated)
        print(f'{BLUE}Found latest checkpoint: {latest_checkpoint.name}{RESET}')
        checkpoint_to_resume = latest_checkpoint.name[len("checkpoints/"):-4]
        shard_to_resume = int(latest_checkpoint.download_as_bytes().decode('utf-8'))
        print(f'{BLUE}Resuming from checkpoint {checkpoint_to_resume} at shard {shard_to_resume}{RESET}')
else:
    print(f'{BLUE}Starting new process... no checkpoint given{RESET}')

# ------------------------------------------
# Create local dirs
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True)

# Init tokenizer
enc = tiktoken.encoding_for_model("gpt-4")
eot = enc._special_tokens['<|endoftext|>']

@ray.remote
def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.uint32)
    return tokens_np, doc["id"]

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

def upload_file(split):
    def upload_many_blobs_with_transfer_manager(split, filenames, source_directory="", workers=8):
        blob_names = [split + name for name in filenames]
        blob_file_pairs = [(os.path.join(source_directory, f), bucket.blob(b))
                           for f, b in zip(filenames, blob_names)]
        results = transfer_manager.upload_many(
            blob_file_pairs, skip_if_exists=True, max_workers=workers,
            worker_type=transfer_manager.THREAD
        )
        for name, result in zip(filenames, results):
            if isinstance(result, Exception):
                print(f"Failed to upload {name}: {result}")
            else:
                print(f"Uploaded {name} to {bucket.name}")

    FILE_NAMES = os.listdir(DATA_CACHE_DIR)
    print('------------')
    print(f'starting upload to GCP bucket {BUCKET_NAME}')
    upload_many_blobs_with_transfer_manager(split, FILE_NAMES, DATA_CACHE_DIR, WORKERS)
    print(f'done upload to GCP bucket {BUCKET_NAME} for {FILE_NAMES}')
    print('cleaning up files....')
    for file in FILE_NAMES:
        os.remove(os.path.join(DATA_CACHE_DIR, file))
        print(f'removed {file} successfully')

def upload_checkpoint():
    checkpoint_files = os.listdir(checkpoint_dir)
    for filename in checkpoint_files:
        blob = bucket.blob(f"checkpoints/{filename}")
        blob.upload_from_filename(os.path.join(checkpoint_dir, filename))
        print('saved checkpoint!!')
    for filename in checkpoint_files:
        os.remove(os.path.join(checkpoint_dir, filename))
        print(f'removed {filename} successfully')

# -----------------------------
# Skip to resume point
# -----------------------------
if continue_processing:
    skipped = 0
    print('starting streaming to skip to desired location')
    for doc in fw:
        if doc["id"] == checkpoint_to_resume:
            print(f'{BLUE}Resuming from document {doc["id"]} at shard {shard_to_resume}. Skipped {skipped} docs.{RESET}')
            break
        if skipped % 10000 == 0:
            print(skipped)
        skipped += 1

# -----------------------------
# Main processing with Ray
# -----------------------------
shard_index = shard_to_resume + 1 if continue_processing else 0
all_tokens_np = np.empty((shard_size,), dtype=np.uint32)
token_count = 0
progress_bar = None

BATCH_SIZE = 16
doc_iter = iter(fw)

while True:
    # Collect next batch
    batch = []
    try:
        for _ in range(BATCH_SIZE):
            batch.append(next(doc_iter))
    except StopIteration:
        pass
    if not batch:
        break

    # Submit tokenization to Ray workers
    futures = [tokenize.remote(doc) for doc in batch]
    results = ray.get(futures)

    # Process results in order
    for tokens, doc_id in results:
        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            checkpoint_filename = os.path.join(checkpoint_dir, f"{doc_id}.txt")
            with open(checkpoint_filename, "w") as f:
                f.write(str(shard_index))

            if shard_index < VAL_SPLIT:
                split = 'val/'
                shard_index_number = shard_index
            elif shard_index < TEST_SPLIT:
                split = 'test/'
                shard_index_number = shard_index - VAL_SPLIT
            else:
                split = 'train/'
                shard_index_number = shard_index - TEST_SPLIT
            split_name = split[:-1]

            filename = os.path.join(DATA_CACHE_DIR, f"{split_name}_{shard_index_number:04d}")
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            upload_file(split)
            upload_checkpoint()
            shard_index += 1
            progress_bar = None
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

# Write last shard
if token_count != 0:
    if shard_index < VAL_SPLIT:
        split = 'val/'
        shard_index_number = shard_index
    elif shard_index < TEST_SPLIT:
        split = 'test/'
        shard_index_number = shard_index - VAL_SPLIT
    else:
        split = 'train/'
        shard_index_number = shard_index - TEST_SPLIT
    split_name = split[:-1]

    filename = os.path.join(DATA_CACHE_DIR, f"{split_name}_{shard_index_number:04d}")
    write_datafile(filename, all_tokens_np[:token_count])
    upload_file(split)
    upload_checkpoint()

# Cleanup
if os.path.exists(DATA_CACHE_DIR):
    shutil.rmtree(DATA_CACHE_DIR)
