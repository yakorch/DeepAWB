import ast
import pathlib

import paramiko
from loguru import logger as console_logger

from src.deep_awb.model_inference import InferenceStats


@console_logger.catch
def sftp_upload(hostname, port, username, password, local_file_path, remote_file_path):
    # Create an SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(hostname, port=port, username=username, password=password)

        sftp = ssh.open_sftp()

        sftp.put(local_file_path, remote_file_path)
        console_logger.debug(f"Successfully uploaded {local_file_path} to {remote_file_path}")

        # Close the SFTP session
        sftp.close()

    finally:
        ssh.close()


def get_env_edge_credentials():
    """
    Get the edge device credentials from the environment: `EDGE_DEVICE_IP`, `SSH_PORT`, `EDGE_DEVICE_USERNAME`, `EDGE_DEVICE_PASSWORD`, `EDGE_DEVICE_REPO_PATH`.
    """
    import os

    from dotenv import load_dotenv

    load_dotenv()

    return os.environ["EDGE_DEVICE_IP"], 22, os.environ["EDGE_DEVICE_USERNAME"], os.environ["EDGE_DEVICE_PASSWORD"], os.environ["EDGE_DEVICE_REPO_PATH"]


def sftp_upload_model_to_edge_device(model_path: pathlib.Path):
    hostname, port, username, password, repo_path = get_env_edge_credentials()
    sftp_upload(
        hostname,
        port,
        username,
        password,
        str(model_path),
        str(pathlib.Path(repo_path) / "src" / "deep_awb" / "models" / model_path.name),
    )


def send_job_via_ssh(hostname, port, username, password, command):
    ssh = paramiko.SSHClient()

    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(hostname, port=port, username=username, password=password)

        _, stdout, stderr = ssh.exec_command(command)

        output = stdout.read().decode()
        error = stderr.read().decode()

        if output:
            console_logger.info(output)
        if error:
            console_logger.error(error)
        return output
    finally:
        ssh.close()


def send_job_to_edge_device(command):
    hostname, port, username, password, _ = get_env_edge_credentials()
    return send_job_via_ssh(hostname, port, username, password, command)


def evaluate_remote_model_inference(model_name: str, image_scale: float):
    inference_result = send_job_to_edge_device(
        f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate ai && cd DeepAWB && python -m src.deep_awb.model_inference --script_module_path src/deep_awb/models/{model_name} --n_runs 25 --image_scale {image_scale} --optimize 1"
    )
    inference_result = ast.literal_eval(inference_result)
    inference_result = InferenceStats(**inference_result)
    return inference_result


if __name__ == "__main__":
    from src.deep_awb import FINAL_DEEPAWB_MODEL

    evaluate_remote_model_inference(FINAL_DEEPAWB_MODEL.name, 2)
