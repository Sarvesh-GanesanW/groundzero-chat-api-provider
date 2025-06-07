class Constants:
    DEFAULT_REGION_NAME = "us-east-1"
    DEFAULT_MODEL_ID = "arn:aws:bedrock:us-east-1::inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0"
    RAG_CHUNK_SIZE = 1000
    RAG_CHUNK_OVERLAP = 50
    EMBEDDING_DIMENSION = 1024
    EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
    DB_CREDENTIALS = 'db_credentials'