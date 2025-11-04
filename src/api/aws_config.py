import os
import aioboto3
from botocore.exceptions import ClientError
from typing import Optional

class AWSConfig:
    """AWS service configuration and clients."""
    
    def __init__(self):
        # Load from environment variables
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        self.dynamodb_table = os.getenv('DYNAMODB_TABLE_NAME', 'trustworthy-model-registry')
        self.s3_bucket = os.getenv('S3_BUCKET_NAME', 'trustworthy-models')
        
        # AWS credentials (handled automatically by EC2 IAM role or env vars)
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        # Create session
        self.session = aioboto3.Session(
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            region_name=self.aws_region
        )
    
    def get_dynamodb_client(self):
        """Get DynamoDB client context manager."""
        return self.session.client('dynamodb')
    
    def get_dynamodb_resource(self):
        """Get DynamoDB resource context manager."""
        return self.session.resource('dynamodb')
    
    def get_s3_client(self):
        """Get S3 client context manager."""
        return self.session.client('s3')

# Global instance
aws_config = AWSConfig()