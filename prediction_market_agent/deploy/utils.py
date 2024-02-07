import os
import subprocess


def export_requirements_file(output_dir: str, extra_deps: list[str] = []):
    if not os.path.exists(output_dir):
        raise ValueError(f"Directory {output_dir} does not exist")
    output_file = f"{output_dir}/requirements.txt"
    subprocess.run(
        f"poetry export -f requirements.txt --output {output_file}", shell=True
    )
    if extra_deps:
        with open(output_file, "w") as f:
            for dep in extra_deps:
                f.write(f"{dep}\n")
    print(f"Saved requirements to {output_dir}/requirements.txt")


def gcloud_deploy_cmd(
    gcp_function_name: str, source: str, entry_point: str, api_keys: dict[str, str]
) -> str:
    api_keys_str = " ".join([f"{k}={v}" for k, v in api_keys.items()])
    return (
        f"gcloud functions deploy {gcp_function_name} "
        f"--runtime python310 "
        f"--trigger-http "
        f"--allow-unauthenticated "
        f"--gen2 "
        f"--region europe-west2 "  # London
        f"--source {source} "
        f"--entry-point {entry_point} "
        f"--set-env-vars {api_keys_str}"
    )
