from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="dyn2024/3DGS-QA",
    repo_type="dataset",
    local_dir="./3DGS_QA_Database"
)
