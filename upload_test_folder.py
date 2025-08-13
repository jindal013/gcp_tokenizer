from google.cloud.storage import Client, transfer_manager
import os

# The bucket name should be just the bucket, not including any sub-paths.
BUCKET_NAME = "350bt_gpt4"
# The remote folder is specified as a prefix to the blob's name.
REMOTE_FOLDER_PREFIX = "test_folder/"
DATA_CACHE_DIR = "/home/chinmay/gcp_tokenizer/edu_fineweb10B"
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
FILE_NAMES = os.listdir(DATA_CACHE_DIR)
WORKERS = os.cpu_count() // 2

def main():

    def upload_many_blobs_with_transfer_manager(
        bucket_name, filenames, source_directory="", workers=8
    ):

        storage_client = Client()
        bucket = storage_client.bucket(bucket_name)

        # Create the full list of blob names with the prefix for the remote folder
        # The transfer_manager handles the actual file paths from the source_directory
        blob_names = [REMOTE_FOLDER_PREFIX + name for name in filenames]

        # Use upload_many instead of upload_many_from_filenames for more control over blob names
        # The `upload_many` function expects a list of (file_path, blob_object) pairs.
        # We need to construct the blob objects with the correct names.
        blob_file_pairs = [(os.path.join(source_directory, f), bucket.blob(b)) for f, b in zip(filenames, blob_names)]

        results = transfer_manager.upload_many(
            blob_file_pairs, max_workers=workers, worker_type=transfer_manager.THREAD
        )

        for name, result in zip(filenames, results):

            if isinstance(result, Exception):
                print("Failed to upload {} due to exception: {}".format(name, result))
            else:
                # The result is the uploaded Blob object
                print("Uploaded")

    upload_many_blobs_with_transfer_manager(BUCKET_NAME, FILE_NAMES, DATA_CACHE_DIR, WORKERS)

if __name__ == '__main__':
    main()