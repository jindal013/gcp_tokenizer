import os
from ray.util.multiprocessing import Pool
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from google.cloud.storage import Client, transfer_manager
import ray

# ---------- constants / helpers (safe for child import) ----------
LOCAL_DIR   = "edu_fineweb10B"
REMOTE_NAME = "sample-10BT"
SHARD_SIZE  = int(1e8)          # 100 M tokens
BUCKET_NAME = "test2_10bt_gpt4"

# make the tokenizer once per worker ------------------------------
def _worker_init():
    global enc, EOT                 # each worker gets its own copy
    enc = tiktoken.encoding_for_model("gpt-4")
    EOT = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    toks = [EOT]
    toks.extend(enc.encode_ordinary(doc["text"]))
    out = np.asarray(toks, dtype=np.uint32)
    return out

def write_datafile(fname, arr):
    np.save(fname, arr)

def upload_shard(dir_path, bucket):
    # uploads *all* files currently in dir_path then deletes them
    names = os.listdir(dir_path)
    results = transfer_manager.upload_many_from_filenames(
        bucket, names, source_directory=dir_path,
        max_workers=8, worker_type=transfer_manager.THREAD
    )
    for n, r in zip(names, results):
        print("Uploaded" if not isinstance(r, Exception) else "FAILED", n)
    for n in names:
        os.remove(os.path.join(dir_path, n))

# ------------------ everything else runs only once ----------------
if __name__ == "__main__":
    ray.init()
    print("Ray driver initialised")

    # local cache dir
    DATA_CACHE_DIR = os.path.join(os.getcwd(), LOCAL_DIR)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # streaming dataset – iterate in the driver, pass docs to workers
    fw = load_dataset("HuggingFaceFW/fineweb-edu",
                      name=REMOTE_NAME, split="train", streaming=True)

    # pool backed by Ray
    pool = Pool(initializer=_worker_init)
    storage_client = Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    shard_idx   = 0
    buf         = np.empty((SHARD_SIZE,), dtype=np.uint32)
    filled      = 0
    pbar        = None

    for toks in pool.imap_unordered(tokenize, fw):
        # enough room in this shard?
        if filled + len(toks) < SHARD_SIZE:
            buf[filled:filled+len(toks)] = toks
            filled += len(toks)
            if pbar is None:
                pbar = tqdm(total=SHARD_SIZE,
                            unit="tokens", desc=f"Shard {shard_idx}")
            pbar.update(len(toks))
            continue

        # flush current shard
        remainder = SHARD_SIZE - filled
        pbar.update(remainder)
        buf[filled:filled+remainder] = toks[:remainder]
        split = "val" if shard_idx == 0 else "train"
        fname = f"edufineweb_{split}_{shard_idx:06d}.npy"
        write_datafile(os.path.join(DATA_CACHE_DIR, fname), buf)
        upload_shard(DATA_CACHE_DIR, bucket)

        # start next shard with leftover tokens
        shard_idx += 1
        pbar.close()
        pbar   = None
        buf[:len(toks)-remainder] = toks[remainder:]
        filled = len(toks) - remainder

    # flush final partial shard
    if filled:
        split = "val" if shard_idx == 0 else "train"
        fname = f"edufineweb_{split}_{shard_idx:06d}.npy"
        write_datafile(os.path.join(DATA_CACHE_DIR, fname), buf[:filled])
        upload_shard(DATA_CACHE_DIR, bucket)

    pool.close()
    pool.join()
