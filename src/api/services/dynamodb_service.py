from typing import Dict, List, Optional
from datetime import datetime
from decimal import Decimal
from botocore.exceptions import ClientError
import json
import asyncio

from api.aws_config import aws_config
from api.models.artifact import ArtifactType

def convert_floats_to_decimal(obj):
    """
    Recursively convert all float values to Decimal for DynamoDB compatibility.
    """
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: convert_floats_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats_to_decimal(item) for item in obj]
    return obj

def convert_decimal_to_float(obj):
    """
    Recursively convert all Decimal values back to float for JSON serialization.
    """
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimal_to_float(item) for item in obj]
    return obj

class DynamoDBService:
    """Service for DynamoDB operations."""
    
    def __init__(self):
        self.table_name = aws_config.dynamodb_table
    
    async def create_artifact(self, artifact_data: Dict) -> Dict:
        """Create a new artifact in DynamoDB."""
        async with aws_config.get_dynamodb_resource() as dynamodb:
            table = await dynamodb.Table(self.table_name)
            
            # Convert floats to Decimal before storing
            item = convert_floats_to_decimal({
                'id': artifact_data['id'],
                'name': artifact_data['name'],
                'type': artifact_data['type'],
                'url': artifact_data['url'],
                'created_at': artifact_data['created_at'],
                'updated_at': artifact_data.get('updated_at', artifact_data['created_at'])
            })
            
            # Add rating if present
            if artifact_data.get('rating'):
                item['rating'] = convert_floats_to_decimal(artifact_data['rating'])
            
            await table.put_item(Item=item)
            return convert_decimal_to_float(item)
    
    async def get_artifact(self, artifact_id: str) -> Optional[Dict]:
        """Get artifact by ID."""
        async with aws_config.get_dynamodb_resource() as dynamodb:
            table = await dynamodb.Table(self.table_name)
            
            try:
                response = await table.get_item(Key={'id': artifact_id})
                item = response.get('Item')
                return convert_decimal_to_float(item) if item else None
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    return None
                raise
    
    async def update_artifact(self, artifact_id: str, updates: Dict) -> Dict:
        """Update an artifact."""
        async with aws_config.get_dynamodb_resource() as dynamodb:
            table = await dynamodb.Table(self.table_name)
            
            # Convert updates to Decimal
            updates = convert_floats_to_decimal(updates)
            
            update_expr = "SET "
            expr_attr_values = {}
            expr_attr_names = {}
            
            for key, value in updates.items():
                if key != 'id':  # Don't update the primary key
                    placeholder = f"#{key}"
                    value_placeholder = f":{key}"
                    update_expr += f"{placeholder} = {value_placeholder}, "
                    expr_attr_values[value_placeholder] = value
                    expr_attr_names[placeholder] = key
            
            # Add updated_at timestamp
            update_expr += "#updated_at = :updated_at"
            expr_attr_values[':updated_at'] = datetime.utcnow().isoformat()
            expr_attr_names['#updated_at'] = 'updated_at'
            
            response = await table.update_item(
                Key={'id': artifact_id},
                UpdateExpression=update_expr,
                ExpressionAttributeValues=expr_attr_values,
                ExpressionAttributeNames=expr_attr_names,
                ReturnValues="ALL_NEW"
            )
            
            return convert_decimal_to_float(response.get('Attributes', {}))
    
    async def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact."""
        async with aws_config.get_dynamodb_resource() as dynamodb:
            table = await dynamodb.Table(self.table_name)
            
            try:
                await table.delete_item(Key={'id': artifact_id})
                return True
            except ClientError:
                return False
    
    async def list_artifacts(
        self,
        name_filter: Optional[str] = None,
        type_filter: Optional[List[str]] = None,
        limit: int = 10,
        last_key: Optional[Dict] = None
    ) -> tuple[List[Dict], Optional[Dict]]:
        """List artifacts with optional filters."""
        async with aws_config.get_dynamodb_resource() as dynamodb:
            table = await dynamodb.Table(self.table_name)
            
            scan_kwargs = {'Limit': limit}
            
            # Add pagination
            if last_key:
                scan_kwargs['ExclusiveStartKey'] = last_key
            
            # Build filter expression
            filter_parts = []
            expr_attr_values = {}
            expr_attr_names = {}
            
            if name_filter and name_filter != "*":
                filter_parts.append("#name = :name")
                expr_attr_values[':name'] = name_filter
                expr_attr_names['#name'] = 'name'
            
            if type_filter:
                filter_parts.append("#type IN (" + ", ".join([f":type{i}" for i in range(len(type_filter))]) + ")")
                expr_attr_names['#type'] = 'type'
                for i, t in enumerate(type_filter):
                    expr_attr_values[f':type{i}'] = t
            
            if filter_parts:
                scan_kwargs['FilterExpression'] = " AND ".join(filter_parts)
                scan_kwargs['ExpressionAttributeValues'] = expr_attr_values
                scan_kwargs['ExpressionAttributeNames'] = expr_attr_names
            
            response = await table.scan(**scan_kwargs)
            
            items = response.get('Items', [])
            next_key = response.get('LastEvaluatedKey')
            
            # Convert Decimal back to float
            items = [convert_decimal_to_float(item) for item in items]
            
            return items, next_key
    
    async def get_artifact_by_url(self, url: str) -> Optional[Dict]:
        """Find artifact by URL (requires GSI or scan)."""
        async with aws_config.get_dynamodb_resource() as dynamodb:
            table = await dynamodb.Table(self.table_name)
            
            # Using scan for simplicity - consider adding GSI for production
            response = await table.scan(
                FilterExpression='#url = :url',
                ExpressionAttributeNames={'#url': 'url'},
                ExpressionAttributeValues={':url': url}
            )
            
            items = response.get('Items', [])
            if items:
                return convert_decimal_to_float(items[0])
            return None
    
    # async def clear_all_artifacts(self):
    #     """Clear all artifacts (for reset endpoint)."""
    #     async with aws_config.get_dynamodb_resource() as dynamodb:
    #         table = await dynamodb.Table(self.table_name)
            
    #         # Scan and delete all items
    #         response = await table.scan()
    #         items = response.get('Items', [])
            
    #         async with table.batch_writer() as batch:
    #             for item in items:
    #                 await batch.delete_item(Key={'id': item['id']})
    
    async def clear_all_artifacts(self):
        """Clear all artifacts with verification and retry logic."""
        async with aws_config.get_dynamodb_resource() as dynamodb:
            table = await dynamodb.Table(self.table_name)
            
            max_retries = 3
            for attempt in range(max_retries):
                deleted_count = 0
                scan_kwargs = {
                    'ProjectionExpression': '#id', 
                    'ExpressionAttributeNames': {'#id': 'id'}
                }
                
                # Phase 1: Scan and collect all IDs
                all_ids = []
                while True:
                    try:
                        response = await table.scan(**scan_kwargs)
                        items = response.get('Items', [])
                        
                        for item in items:
                            all_ids.append(item['id'])
                        
                        last_key = response.get('LastEvaluatedKey')
                        if not last_key:
                            break
                        
                        scan_kwargs['ExclusiveStartKey'] = last_key
                        
                    except Exception as e:
                        print(f"Error scanning table: {e}")
                        raise
                
                print(f"Attempt {attempt + 1}: Found {len(all_ids)} artifacts to delete")
                
                if not all_ids:
                    print("No artifacts found - table is clear")
                    return 0
                
                # Phase 2: Delete all items in batches
                batch_size = 25  # DynamoDB batch write limit
                for i in range(0, len(all_ids), batch_size):
                    batch = all_ids[i:i + batch_size]
                    
                    try:
                        async with table.batch_writer() as writer:
                            for artifact_id in batch:
                                await writer.delete_item(Key={'id': artifact_id})
                                deleted_count += 1
                    except Exception as e:
                        print(f"Error deleting batch: {e}")
                        raise
                
                print(f"Deleted {deleted_count} items in attempt {attempt + 1}")
                
                # Phase 3: Verify deletion with eventual consistency delay
                await asyncio.sleep(1.0)  # Wait for DynamoDB consistency
                
                # Check if any items remain
                verify_response = await table.scan(
                    ProjectionExpression='#id',
                    ExpressionAttributeNames={'#id': 'id'},
                    Limit=1
                )
                
                remaining = len(verify_response.get('Items', []))
                print(f"Verification: {remaining} items remaining")
                
                if remaining == 0:
                    print(f"✓ Successfully cleared all artifacts (deleted {deleted_count} total)")
                    return deleted_count
                
                # If items remain, retry
                print(f"⚠ Items still present after attempt {attempt + 1}, retrying...")
            
            # If we get here, deletion failed after all retries
            raise Exception(f"Failed to clear all artifacts after {max_retries} attempts")
    
    # async def clear_all_artifacts(self):
    #     """Clear all artifacts (for reset endpoint) - Correctly handles pagination."""
    #     async with aws_config.get_dynamodb_resource() as dynamodb:
    #         table = await dynamodb.Table(self.table_name)
            
    #         # Use ProjectionExpression to only retrieve the primary key ('id') for efficiency
    #         scan_kwargs = {
    #             'ProjectionExpression': '#id', 
    #             'ExpressionAttributeNames': {'#id': 'id'}
    #         }
            
    #         # Scan and delete all items with pagination
    #         while True:
    #             response = await table.scan(**scan_kwargs)
    #             items = response.get('Items', [])
                
    #             # Use batch_writer for efficient mass deletion
    #             async with table.batch_writer() as batch:
    #                 for item in items:
    #                     # Assuming 'id' is the primary key
    #                     await batch.delete_item(Key={'id': item['id']})
                
    #             # Check for pagination (LastEvaluatedKey)
    #             last_key = response.get('LastEvaluatedKey')
    #             if not last_key:
    #                 break  # No more pages
                
    #             # Update scan_kwargs for the next page
    #             scan_kwargs['ExclusiveStartKey'] = last_key

# Global instance
db_service = DynamoDBService()

