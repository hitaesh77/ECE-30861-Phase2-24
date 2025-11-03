from typing import Dict, List, Optional
from datetime import datetime
from botocore.exceptions import ClientError

from api.aws_config import aws_config
from api.models.artifact import ArtifactType

class DynamoDBService:
    """Service for DynamoDB operations."""
    
    def __init__(self):
        self.table_name = aws_config.dynamodb_table
    
    async def create_artifact(self, artifact_data: Dict) -> Dict:
        """Create a new artifact in DynamoDB."""
        async with aws_config.get_dynamodb_resource() as dynamodb:
            table = await dynamodb.Table(self.table_name)
            
            item = {
                'id': artifact_data['id'],
                'name': artifact_data['name'],
                'type': artifact_data['type'],
                'url': artifact_data['url'],
                'created_at': artifact_data['created_at'],
                'updated_at': artifact_data.get('updated_at', artifact_data['created_at'])
            }
            
            # Add rating if present
            if artifact_data.get('rating'):
                item['rating'] = artifact_data['rating']
            
            await table.put_item(Item=item)
            return item
    
    async def get_artifact(self, artifact_id: str) -> Optional[Dict]:
        """Get artifact by ID."""
        async with aws_config.get_dynamodb_resource() as dynamodb:
            table = await dynamodb.Table(self.table_name)
            
            try:
                response = await table.get_item(Key={'id': artifact_id})
                return response.get('Item')
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    return None
                raise
    
    async def update_artifact(self, artifact_id: str, updates: Dict) -> Dict:
        """Update an artifact."""
        async with aws_config.get_dynamodb_resource() as dynamodb:
            table = await dynamodb.Table(self.table_name)
            
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
            
            return response.get('Attributes', {})
    
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
            return items[0] if items else None
    
    async def clear_all_artifacts(self):
        """Clear all artifacts (for reset endpoint)."""
        async with aws_config.get_dynamodb_resource() as dynamodb:
            table = await dynamodb.Table(self.table_name)
            
            # Scan and delete all items
            response = await table.scan()
            items = response.get('Items', [])
            
            async with table.batch_writer() as batch:
                for item in items:
                    await batch.delete_item(Key={'id': item['id']})

# Global instance
db_service = DynamoDBService()