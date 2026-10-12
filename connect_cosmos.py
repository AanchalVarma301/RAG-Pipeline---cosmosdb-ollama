
import os
from dotenv import load_dotenv
from azure.cosmos import CosmosClient

# Load settings from the .env file
load_dotenv()

endpoint = os.getenv("COSMOS_ENDPOINT")
key = os.getenv("COSMOS_KEY")
database_name = os.getenv("COSMOS_DATABASE")
container_name = os.getenv("COSMOS_CONTAINER")

# Connect to the local Cosmos DB Emulator
client = CosmosClient(
    endpoint,
    credential=key,
    connection_verify=False
)

# Access the database and container
database = client.get_database_client(database_name)
container = database.get_container_client(container_name)

print("Connected to Cosmos DB!")

# Read the test record you created earlier
item = container.read_item(
    item="chunk-001",
    partition_key="chunk-001"
)

print("Test record retrieved successfully:")
print(item)