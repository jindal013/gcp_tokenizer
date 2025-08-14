import os
import shutil
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm
from google.cloud.storage import Client, transfer_manager
import argparse

remote_name = "sample-350BT"

# fineweb = load_dataset("tiiuae/falcon-refinedweb", split="train", streaming=True)
fineweb = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True)

# skip the first 10 samples and then take only the first 10, resulting in [10:20]
fineweb = fineweb.skip(0)

for a in fineweb: 
  print("Subset of fineweb dataset:", a['id'])
  break

# <urn:uuid:9b0b7a97-4882-4e06-adc0-44f4bbdc3349>
# 


# <urn:uuid:e2300ad5-01dd-4e80-92b3-7ec88785cc9d>