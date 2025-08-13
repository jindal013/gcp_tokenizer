# GCP Tokenizer

Collection of scripts to efficiently upload tokenized shards to GCP buckets. Work in progress.

### Setting up pipelines

  - [X] Locally save on the laptop with Karpathy script (10B, GPT-2)
  - [X] Locally save on the laptop with GPT-4 tokens (10B, GPT-4)
  - [ ] Figure out the multiprocessing bugs for the GCP upload (macOS, works fine on Linux)
  - [X] Figure out how to take from huggingface to upload to GCP directly --\> not needed, use streaming
  - [X] Naive solution to upload to GCP Bucket (300B, GPT-4)
  - [ ] Optimized version to upload to GCP Bucket with parallelism across nodes (300B, GPT-4)

### GCP

  - [X] Create the GCP bucket
  - [X] Upload data to bucket remotely

-----

### Setup and Execution

To get started, follow these steps to prepare your Google Cloud environment and configure the scripts for execution.

1.  **Initialize `gcloud`**: First, you need to set up the Google Cloud SDK and link it to your project. Run the `gcloud init` command in your terminal and follow the prompts to select your project.

    ```bash
    gcloud init
    ```

2.  **Authenticate Credentials**: Authenticate your application-default credentials. This step is crucial as it allows the Python scripts to securely access and interact with your Google Cloud Storage bucket.

    ```bash
    gcloud auth application-default login
    ```

3.  **Create a GCP Bucket**: Create a new Google Cloud Storage bucket to store your tokenized data. Although Google Cloud Storage has a flat namespace, you can simulate a hierarchical structure (like folders) by using prefixes in your object names. The scripts will handle this automatically.

    ```bash
    gcloud storage buckets create gs://[YOUR_BUCKET_NAME]
    ```

4.  **Configure Scripts**: Open the Python scripts and change any placeholder names to match your specific setup, such as the `BUCKET_NAME` and `DATA_CACHE_DIR`.

5.  **Run `make_folder.py`**: Execute the `make_folder.py` script to create the necessary local directories for temporary data storage.

    ```bash
    python make_folder.py
    ```

6.  **Run `350bt.py`**: Finally, run the main `350bt.py` script. This will start the data streaming, tokenization, and upload process.

    ```bash
    python 350bt.py
    ```