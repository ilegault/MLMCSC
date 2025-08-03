#!/usr/bin/env python3
"""
S3 Client for MLMCSC Object Storage

This module provides a unified interface for interacting with S3-compatible
object storage services (AWS S3, MinIO, etc.) for storing images and models.
"""

import os
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, BinaryIO
from io import BytesIO

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config

logger = logging.getLogger(__name__)

class S3Client:
    """S3-compatible object storage client for MLMCSC."""
    
    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        region: str = 'us-east-1'
    ):
        """
        Initialize S3 client.
        
        Args:
            endpoint_url: S3 endpoint URL (for MinIO or other S3-compatible services)
            access_key: S3 access key
            secret_key: S3 secret key
            bucket_name: Default bucket name
            region: AWS region
        """
        self.endpoint_url = endpoint_url or os.getenv('S3_ENDPOINT')
        self.access_key = access_key or os.getenv('S3_ACCESS_KEY')
        self.secret_key = secret_key or os.getenv('S3_SECRET_KEY')
        self.bucket_name = bucket_name or os.getenv('S3_BUCKET', 'mlmcsc-data')
        self.region = region
        
        # Configure boto3 client
        config = Config(
            region_name=self.region,
            retries={'max_attempts': 3, 'mode': 'adaptive'},
            max_pool_connections=50
        )
        
        try:
            if self.endpoint_url:
                # For MinIO or other S3-compatible services
                self.client = boto3.client(
                    's3',
                    endpoint_url=self.endpoint_url,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    config=config
                )
            else:
                # For AWS S3
                self.client = boto3.client(
                    's3',
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    config=config
                )
            
            # Test connection
            self._test_connection()
            logger.info(f"S3 client initialized successfully (bucket: {self.bucket_name})")
            
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise
    
    def _test_connection(self):
        """Test S3 connection and create bucket if it doesn't exist."""
        try:
            # Check if bucket exists
            self.client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket {self.bucket_name} exists and is accessible")
        except ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                # Bucket doesn't exist, create it
                try:
                    if self.region == 'us-east-1':
                        self.client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.region}
                        )
                    logger.info(f"Created bucket {self.bucket_name}")
                except ClientError as create_error:
                    logger.error(f"Failed to create bucket {self.bucket_name}: {create_error}")
                    raise
            else:
                logger.error(f"Cannot access bucket {self.bucket_name}: {e}")
                raise
    
    def upload_file(
        self,
        file_path: Path,
        key: Optional[str] = None,
        bucket: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Upload a file to S3.
        
        Args:
            file_path: Path to the file to upload
            key: S3 object key (if None, generated from file path)
            bucket: Bucket name (if None, uses default)
            metadata: Additional metadata to store with the object
            
        Returns:
            S3 object key
        """
        bucket = bucket or self.bucket_name
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Generate key if not provided
        if key is None:
            # Create a hierarchical key based on file type and date
            file_ext = file_path.suffix.lower()
            date_prefix = datetime.now().strftime('%Y/%m/%d')
            
            if file_ext in ['.jpg', '.jpeg', '.png', '.tiff']:
                key = f"images/{date_prefix}/{file_path.name}"
            elif file_ext in ['.pkl', '.joblib', '.pt', '.pth']:
                key = f"models/{date_prefix}/{file_path.name}"
            else:
                key = f"data/{date_prefix}/{file_path.name}"
        
        # Calculate file hash for integrity checking
        file_hash = self._calculate_file_hash(file_path)
        
        # Prepare metadata
        upload_metadata = {
            'upload_timestamp': datetime.utcnow().isoformat(),
            'file_size': str(file_path.stat().st_size),
            'file_hash': file_hash,
            'original_filename': file_path.name
        }
        
        if metadata:
            upload_metadata.update(metadata)
        
        try:
            # Upload file
            self.client.upload_file(
                str(file_path),
                bucket,
                key,
                ExtraArgs={
                    'Metadata': upload_metadata,
                    'ServerSideEncryption': 'AES256'
                }
            )
            
            logger.info(f"Uploaded {file_path} to s3://{bucket}/{key}")
            return key
            
        except ClientError as e:
            logger.error(f"Failed to upload {file_path}: {e}")
            raise
    
    def upload_data(
        self,
        data: bytes,
        key: str,
        bucket: Optional[str] = None,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Upload binary data to S3.
        
        Args:
            data: Binary data to upload
            key: S3 object key
            bucket: Bucket name (if None, uses default)
            content_type: MIME type of the data
            metadata: Additional metadata to store with the object
            
        Returns:
            S3 object key
        """
        bucket = bucket or self.bucket_name
        
        # Calculate data hash
        data_hash = hashlib.sha256(data).hexdigest()
        
        # Prepare metadata
        upload_metadata = {
            'upload_timestamp': datetime.utcnow().isoformat(),
            'data_size': str(len(data)),
            'data_hash': data_hash
        }
        
        if metadata:
            upload_metadata.update(metadata)
        
        # Prepare extra args
        extra_args = {
            'Metadata': upload_metadata,
            'ServerSideEncryption': 'AES256'
        }
        
        if content_type:
            extra_args['ContentType'] = content_type
        
        try:
            # Upload data
            self.client.put_object(
                Bucket=bucket,
                Key=key,
                Body=data,
                **extra_args
            )
            
            logger.info(f"Uploaded data to s3://{bucket}/{key}")
            return key
            
        except ClientError as e:
            logger.error(f"Failed to upload data to {key}: {e}")
            raise
    
    def download_file(
        self,
        key: str,
        file_path: Optional[Path] = None,
        bucket: Optional[str] = None
    ) -> bytes:
        """
        Download a file from S3.
        
        Args:
            key: S3 object key
            file_path: Local path to save the file (if None, returns bytes)
            bucket: Bucket name (if None, uses default)
            
        Returns:
            File content as bytes
        """
        bucket = bucket or self.bucket_name
        
        try:
            if file_path:
                # Download to file
                file_path.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(bucket, key, str(file_path))
                logger.info(f"Downloaded s3://{bucket}/{key} to {file_path}")
                return file_path.read_bytes()
            else:
                # Download to memory
                response = self.client.get_object(Bucket=bucket, Key=key)
                data = response['Body'].read()
                logger.info(f"Downloaded s3://{bucket}/{key} to memory")
                return data
                
        except ClientError as e:
            logger.error(f"Failed to download s3://{bucket}/{key}: {e}")
            raise
    
    def delete_file(self, key: str, bucket: Optional[str] = None) -> bool:
        """
        Delete a file from S3.
        
        Args:
            key: S3 object key
            bucket: Bucket name (if None, uses default)
            
        Returns:
            True if successful
        """
        bucket = bucket or self.bucket_name
        
        try:
            self.client.delete_object(Bucket=bucket, Key=key)
            logger.info(f"Deleted s3://{bucket}/{key}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to delete s3://{bucket}/{key}: {e}")
            return False
    
    def list_files(
        self,
        prefix: str = '',
        bucket: Optional[str] = None,
        max_keys: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        List files in S3 bucket.
        
        Args:
            prefix: Key prefix to filter by
            bucket: Bucket name (if None, uses default)
            max_keys: Maximum number of keys to return
            
        Returns:
            List of file information dictionaries
        """
        bucket = bucket or self.bucket_name
        
        try:
            response = self.client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    files.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'],
                        'etag': obj['ETag'].strip('"')
                    })
            
            logger.info(f"Listed {len(files)} files with prefix '{prefix}'")
            return files
            
        except ClientError as e:
            logger.error(f"Failed to list files with prefix '{prefix}': {e}")
            raise
    
    def get_file_metadata(
        self,
        key: str,
        bucket: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get metadata for an S3 object.
        
        Args:
            key: S3 object key
            bucket: Bucket name (if None, uses default)
            
        Returns:
            Object metadata dictionary
        """
        bucket = bucket or self.bucket_name
        
        try:
            response = self.client.head_object(Bucket=bucket, Key=key)
            
            metadata = {
                'key': key,
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'etag': response['ETag'].strip('"'),
                'content_type': response.get('ContentType', ''),
                'metadata': response.get('Metadata', {})
            }
            
            return metadata
            
        except ClientError as e:
            logger.error(f"Failed to get metadata for s3://{bucket}/{key}: {e}")
            raise
    
    def generate_presigned_url(
        self,
        key: str,
        bucket: Optional[str] = None,
        expiration: int = 3600,
        method: str = 'get_object'
    ) -> str:
        """
        Generate a presigned URL for S3 object access.
        
        Args:
            key: S3 object key
            bucket: Bucket name (if None, uses default)
            expiration: URL expiration time in seconds
            method: HTTP method ('get_object' or 'put_object')
            
        Returns:
            Presigned URL
        """
        bucket = bucket or self.bucket_name
        
        try:
            url = self.client.generate_presigned_url(
                method,
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expiration
            )
            
            logger.info(f"Generated presigned URL for s3://{bucket}/{key}")
            return url
            
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL for s3://{bucket}/{key}: {e}")
            raise
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def cleanup_old_files(
        self,
        prefix: str,
        days_old: int = 30,
        bucket: Optional[str] = None
    ) -> int:
        """
        Clean up old files from S3.
        
        Args:
            prefix: Key prefix to filter by
            days_old: Delete files older than this many days
            bucket: Bucket name (if None, uses default)
            
        Returns:
            Number of files deleted
        """
        bucket = bucket or self.bucket_name
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        try:
            files = self.list_files(prefix=prefix, bucket=bucket)
            deleted_count = 0
            
            for file_info in files:
                if file_info['last_modified'].replace(tzinfo=None) < cutoff_date:
                    if self.delete_file(file_info['key'], bucket):
                        deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old files with prefix '{prefix}'")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old files: {e}")
            return 0