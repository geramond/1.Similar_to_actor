import os
import logging
import sys

import click
from s3_storage import download_file

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@click.command("download")
@click.option("--s3-bucket")
@click.option("--remote-path")
@click.option("--output-path")
def download_dataset(s3_bucket: str, remote_path: str, output_path: str):
    remote_path = f"{remote_path}/sampled_train_50k.csv"
    output_local_path = f"{output_path}/sampled_train_50k.csv"

    logger.info(f"remote_path: {remote_path}")
    logger.info(f"output_local_path: {output_local_path}")

    os.makedirs(output_path, exist_ok=True)
    download_file(
        bucket_name=s3_bucket, remote_path=remote_path, local_path=output_local_path,
    )


if __name__ == "__main__":
    download_dataset()
