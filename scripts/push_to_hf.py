from huggingface_hub import HfApi

repo_id = "yadhuvtk/YD-Vector-Model"
folder_path = "outputs/exports/YD-Vector-Model"

api = HfApi()
api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
api.upload_folder(folder_path=folder_path, repo_id=repo_id, repo_type="model")

print("Uploaded:", repo_id)