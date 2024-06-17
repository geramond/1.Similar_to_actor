import os
import boto3
from dotenv import load_dotenv


load_dotenv()
aws_access_key_id = os.getenv("aws_access_key_id")
aws_secret_access_key = os.getenv("aws_secret_access_key")

session = boto3.session.Session()
s3_client = session.client(
    service_name="s3",
    region_name="ru-msk",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    endpoint_url="https://hb.ru-msk.vkcs.cloud",
)


def download_file(
    bucket_name: str, remote_path: str, local_path: str,
):
    s3_client.download_file(bucket_name, remote_path, local_path)


def upload_file(bucket_name: str, local_path: str, remote_path: str):
    s3_client.upload_file(local_path, bucket_name, remote_path)
