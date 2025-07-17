# Data for Jaxformer

Collection of scripts to efficiently upload tokenized shards to GCP buckets. Work in progress.

Setting up pipelines
- [X] Locally save on the laptop with Karpathy script (10B, GPT-2)
- [X] Locally save on the laptop with GPT-4 tokens (10B, GPT-4)
- [ ] Figure out the multiprocessing bugs for the GCP upload (macOS, works fine on Linux)
- [X] Figure out how to take from huggingface to upload to GCP directly --> not needed, use streaming
- [X] Naive solution to upload to GCP Bucket (300B, GPT-4)
- [ ] Optimized version to upload to GCP Bucket with parallelism across nodes (300B, GPT-4)

GCP
- [X] Create the GCP bucket
- [X] Upload data to bucket remotely