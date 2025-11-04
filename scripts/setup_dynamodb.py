"""
Script to create DynamoDB table for the Trustworthy Model Registry.
Run this once to set up your database.
"""
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

def create_table():
    """Create the DynamoDB table with proper schema."""
    dynamodb = boto3.resource(
        'dynamodb',
        region_name=os.getenv('AWS_REGION', 'us-east-1'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )
    
    table_name = os.getenv('DYNAMODB_TABLE_NAME', 'trustworthy-model-registry')
    
    try:
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {
                    'AttributeName': 'id',
                    'KeyType': 'HASH'  # Partition key
                }
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': 'id',
                    'AttributeType': 'S'
                },
                {
                    'AttributeName': 'url',
                    'AttributeType': 'S'
                }
            ],
            GlobalSecondaryIndexes=[
                {
                    'IndexName': 'url-index',
                    'KeySchema': [
                        {
                            'AttributeName': 'url',
                            'KeyType': 'HASH'
                        }
                    ],
                    'Projection': {
                        'ProjectionType': 'ALL'
                    },
                    'ProvisionedThroughput': {
                        'ReadCapacityUnits': 5,
                        'WriteCapacityUnits': 5
                    }
                }
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        )
        
        # Wait for table to be created
        print(f"Creating table {table_name}...")
        table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
        print(f"Table {table_name} created successfully!")
        
        return table
        
    except dynamodb.meta.client.exceptions.ResourceInUseException:
        print(f"Table {table_name} already exists!")
        return dynamodb.Table(table_name)
    except Exception as e:
        print(f"Error creating table: {e}")
        raise

if __name__ == "__main__":
    create_table()