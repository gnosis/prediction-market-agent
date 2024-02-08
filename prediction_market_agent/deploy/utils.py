import os
import subprocess
import sys
from google.cloud.functions_v2.services.function_service.client import (
    FunctionServiceClient,
)
from google.cloud.functions_v2.types.functions import Function


def export_requirements_from_toml(output_dir: str, extra_deps: list[str] = []):
    if not os.path.exists(output_dir):
        raise ValueError(f"Directory {output_dir} does not exist")
    output_file = f"{output_dir}/requirements.txt"
    subprocess.run(
        f"poetry export -f requirements.txt --without-hashes --output {output_file}",
        shell=True,
    )
    if extra_deps:
        with open(output_file, "a") as f:
            for dep in extra_deps:
                f.write(f"{dep}\n")
    print(f"Saved requirements to {output_dir}/requirements.txt")


def gcloud_deploy_cmd(
    gcp_function_name: str,
    source: str,
    entry_point: str,
    api_keys: dict[str, str],
    memory: int,  # in MB
) -> str:
    api_keys_str = " ".join([f"{k}={v}" for k, v in api_keys.items()])
    cmd = (
        f"gcloud functions deploy {gcp_function_name} "
        f"--runtime {get_gcloud_python_runtime_str()} "
        f"--trigger-http "
        f"--gen2 "
        f"--region {get_gcloud_region()} "
        f"--source {source} "
        f"--entry-point {entry_point} "
        f"--memory {memory}MB "
        f"--no-allow-unauthenticated "
    )
    if api_keys:
        cmd += f"--set-env-vars {api_keys_str} "

    return cmd


def gcloud_schedule_cmd(function_name: str, cron_schedule: str) -> str:
    return (
        f"gcloud scheduler jobs create http {function_name} "
        f"--schedule '{cron_schedule}' "
        f"--uri {get_gcloud_function_uri(function_name)} "
        f"--http-method POST "
        f"--location {get_gcloud_region()}"
    )


def gcloud_delete_function_cmd(fname: str) -> None:
    return f"gcloud functions delete {fname} --region={get_gcloud_region()} --quiet"


def get_gcloud_project_id() -> str:
    return (
        subprocess.run(
            "gcloud config get-value project", shell=True, capture_output=True
        )
        .stdout.decode()
        .strip()
    )


def get_gcloud_parent() -> str:
    return f"projects/{get_gcloud_project_id()}/locations/{get_gcloud_region()}"


def get_gcloud_id_token() -> str:
    return (
        subprocess.run(
            "gcloud auth print-identity-token", shell=True, capture_output=True
        )
        .stdout.decode()
        .strip()
    )


def get_gcloud_region() -> str:
    return "europe-west2"  # London


def get_gcloud_python_runtime_str():
    return f"python{sys.version_info.major}{sys.version_info.minor}"


def get_gcloud_function_uri(function_name: str) -> str:
    return (
        subprocess.run(
            f"gcloud functions describe {function_name} --region {get_gcloud_region()} --format='value(url)'",
            shell=True,
            capture_output=True,
        )
        .stdout.decode()
        .strip()
    )


def api_keys_to_str(api_keys: dict[str, str]) -> str:
    return " ".join([f"{k}={v}" for k, v in api_keys.items()])


def get_gcp_function(fname: str) -> Function:
    client = FunctionServiceClient()
    response = client.list_functions(parent=get_gcloud_parent())
    for function in response:
        if function.name.split("/")[-1] == fname:
            return function

    fnames = [f.name.split("/")[-1] for f in response]
    raise ValueError(f"Function {fname} not found in function list {fnames}")


def gcp_function_is_active(fname: str) -> bool:
    return get_gcp_function(fname).state == Function.State.ACTIVE
