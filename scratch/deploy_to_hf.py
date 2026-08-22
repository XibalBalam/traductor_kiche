import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
USERNAME = "XibalBalam"
REPO_NAME = "traductor-kiche"
REPO_ID = f"{USERNAME}/{REPO_NAME}"

api = HfApi(token=HF_TOKEN)

try:
    print(f"Creating Space {REPO_ID}...")
    api.create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="docker", exist_ok=True)
    print("Space created or already exists.")
except Exception as e:
    print(f"Error creating space: {e}")
    exit(1)

secrets = {
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
    "AWS_REGION": os.getenv("AWS_REGION", "us-east-1"),
    "S3_BUCKET_NAME": os.getenv("S3_BUCKET_NAME", ""),
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY", "")
}

print("Setting secrets...")
for k, v in secrets.items():
    if v:
        try:
            api.add_space_secret(repo_id=REPO_ID, key=k, value=v)
            print(f"Set secret: {k}")
        except Exception as e:
            print(f"Error setting secret {k}: {e}")
    else:
        print(f"Warning: Secret {k} was empty in local .env")

print("Uploading code to Hugging Face...")
# We ignore the venv directory, scratch, and .git
ignore_patterns = ["venv/*", "scratch/*", ".git/*", "__pycache__/*", "*.mp3", "*.wav", "outputs/*", "uploads/*"]

try:
    api.upload_folder(
        folder_path=".",
        repo_id=REPO_ID,
        repo_type="space",
        ignore_patterns=ignore_patterns,
        commit_message="Update for background loading and mobile UI"
    )
    print("Code uploaded successfully!")
except Exception as e:
    print(f"Error uploading code: {e}")
