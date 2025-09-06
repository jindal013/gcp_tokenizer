# GCP Tokenizer

Collection of scripts to efficiently upload tokenized shards to GCP buckets. This is not the right repo you should be looking at though. See this: [Jaxformer](https://github.com/divyamakkar0/Jaxformer) or visit the blog website at [jaxformer.com](https://jaxformer.com).

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