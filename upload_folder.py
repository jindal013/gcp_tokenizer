from google.cloud import storage_control_v2


def create_folder(bucket_name: str, folder_name: str) -> None:
    storage_control_client = storage_control_v2.StorageControlClient()
    # The storage bucket path uses the global access pattern, in which the "_"
    # denotes this bucket exists in the global namespace.
    project_path = storage_control_client.common_project_path("_")
    bucket_path = f"{project_path}/buckets/{bucket_name}"

    request = storage_control_v2.CreateFolderRequest(
        parent=bucket_path,
        folder_id=folder_name,
    )
    response = storage_control_client.create_folder(request=request)

    print(f"Created folder: {response.name}")

if __name__ == '__main__':
   # The ID of your GCS bucket
  bucket_name = "350bt_gpt4"

    # The name of the folder to be created
  for folder_name in ['train', 'val', 'test']:
    create_folder(bucket_name, folder_name)