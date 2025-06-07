import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from utils.dbUtils import getSiteDB
from constants import Constants
from utils.credProvider import getCredentials
import os
import json

site = os.getenv("ENV")

credential = getCredentials()
user = credential['username']
password = credential['password']
host = credential['host']
port = credential['port']
embeddingDimension = Constants.EMBEDDING_DIMENSION

def getDbConnection(siteName):
    """Gets a connection to the PostgreSQL database and registers pgvector type."""
    conn = psycopg2.connect(
        dbname = getSiteDB(siteName), user=user, password=password, host=host, port = port
    )
    register_vector(conn)
    return conn


def createConversationsTable(siteName):
    """Creates the conversations table in the database if it does not already exist."""
    conn = getDbConnection(siteName)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            conversation_id UUID NOT NULL,
            user_id UUID NOT NULL,
            conversation_name VARCHAR(255) DEFAULT 'New Chat',
            s3_file_location TEXT NOT NULL
        );
    """)
    cur.execute("""
        ALTER TABLE conversations
        ADD CONSTRAINT unique_conversation_id UNIQUE (conversation_id);
    """)
    conn.commit()
    cur.close()
    conn.close()

def createEmbeddingsTable(siteName):
    """Creates the document_embeddings table for RAG if it doesn't exist."""
    conn = getDbConnection(siteName)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS document_embeddings (
            id SERIAL PRIMARY KEY,
            source_document_name TEXT,
            chunk_text TEXT NOT NULL,
            embedding vector({embeddingDimension})
        );
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS hnsw_index
        ON document_embeddings
        USING hnsw (embedding vector_cosine_ops);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS ivfflat_index
        ON document_embeddings
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """)
    conn.commit()
    cur.close()
    conn.close()


def insertConversationMetadata(conversationId, userId, s3FileLocation, siteName, conversationName='New Chat'):
    """Inserts a new conversation metadata record into the conversations table."""
    conn = getDbConnection(siteName)
    cur = conn.cursor()
    cur.execute("""
            INSERT INTO conversations (conversation_id, user_id, s3_file_location, conversation_name)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (conversation_id) DO NOTHING;
        """, (conversationId, userId, s3FileLocation, conversationName))
    conn.commit()
    cur.close()
    conn.close()

def insertEmbeddingChunks(chunks, siteName):
    """Inserts multiple embedding chunks into the document_embeddings table.

    Args:
        chunks: A list of tuples, where each tuple is
                (source_document_name, chunk_text, embedding_vector).
                embedding_vector should be a list or numpy array.
    """
    if not chunks:
        return

    conn = getDbConnection(siteName)
    cur = conn.cursor()
    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO document_embeddings (source_document_name, chunk_text, embedding)
           VALUES %s""",
        chunks,
        template=f"(%s, %s, %s::vector({embeddingDimension}))"
    )
    conn.commit()
    cur.close()
    conn.close()


def getConversationMetadata(conversationId, siteName):
    """Gets metadata for a conversation from the database."""
    conn = getDbConnection(siteName)
    cur = conn.cursor()
    cur.execute("""
        SELECT conversation_id, user_id, s3_file_location, conversation_name
        FROM conversations
        WHERE conversation_id = %s
    """, (conversationId,))

    row = cur.fetchone()
    cur.close()
    conn.close()

    if row:
        metadata = {
            "conversationId": row[0],
            "userId": row[1],
            "s3FileLocation": row[2],
            "conversationName": row[3]
        }
    else:
        metadata = None

    return metadata

def findSimilarChunks(queryEmbedding, siteName, limit=5):
    """Finds similar text chunks based on cosine similarity to the query embedding.

    Args:
        queryEmbedding: The vector embedding of the user's query (list or numpy array).
        limit: The maximum number of similar chunks to return.

    Returns:
        A list of strings, where each string is a relevant text chunk.
    """
    conn = getDbConnection(siteName)
    cur = conn.cursor()
    cur.execute(
        f"""SELECT chunk_text
           FROM document_embeddings
           ORDER BY embedding <=> %s::vector({embeddingDimension})
           LIMIT %s""",
        (queryEmbedding, limit)
    )
    results = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return results


def createAgentsTable(siteName):
    """Creates the agents table in the database if it does not already exist."""
    conn = getDbConnection(siteName)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS agents (
            id SERIAL PRIMARY KEY,
            agent_id UUID NOT NULL,
            user_id UUID NOT NULL,
            agent_name VARCHAR(255) NOT NULL,
            agent_description TEXT,
            agent_config JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("""
        ALTER TABLE agents
        ADD CONSTRAINT unique_agent_id UNIQUE (agent_id);
    """)
    conn.commit()
    cur.close()
    conn.close()


def insertAgent(agentId, userId, agentName, agentDescription, agentConfig, siteName):
    """Inserts a new agent record into the agents table."""
    conn = getDbConnection(siteName)
    cur = conn.cursor()
    cur.execute("""
            INSERT INTO agents (agent_id, user_id, agent_name, agent_description, agent_config)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (agent_id) DO NOTHING;
        """, (agentId, userId, agentName, agentDescription, json.dumps(agentConfig)))
    conn.commit()
    cur.close()
    conn.close()


def getAgentMetadata(agentId, siteName):
    """Gets metadata for an agent from the database."""
    conn = getDbConnection(siteName)
    cur = conn.cursor()
    cur.execute("""
        SELECT agent_id, user_id, agent_name, agent_description, agent_config, created_at, updated_at
        FROM agents
        WHERE agent_id = %s
    """, (agentId,))

    row = cur.fetchone()
    cur.close()
    conn.close()

    if row:
        metadata = {
            "agentId": row[0],
            "userId": row[1],
            "agentName": row[2],
            "agentDescription": row[3],
            "agentConfig": json.loads(row[4]) if row[4] else {},
            "createdAt": row[5].isoformat() if row[5] else None,
            "updatedAt": row[6].isoformat() if row[6] else None
        }
    else:
        metadata = None

    return metadata


def updateAgent(agentId, agentName=None, agentDescription=None, agentConfig=None, siteName=None):
    """Updates an existing agent record in the agents table."""
    conn = getDbConnection(siteName)
    cur = conn.cursor()
    
    updateFields = []
    params = []
    
    if agentName is not None:
        updateFields.append("agent_name = %s")
        params.append(agentName)
    
    if agentDescription is not None:
        updateFields.append("agent_description = %s")
        params.append(agentDescription)
    
    if agentConfig is not None:
        updateFields.append("agent_config = %s")
        params.append(json.dumps(agentConfig))
    
    if updateFields:
        updateFields.append("updated_at = CURRENT_TIMESTAMP")
        
        query = f"""
            UPDATE agents
            SET {', '.join(updateFields)}
            WHERE agent_id = %s
        """
        params.append(agentId)
        
        cur.execute(query, params)
    
    conn.commit()
    cur.close()
    conn.close()


def deleteAgent(agentId, siteName):
    """Deletes an agent record from the agents table."""
    conn = getDbConnection(siteName)
    cur = conn.cursor()
    cur.execute("""
        DELETE FROM agents
        WHERE agent_id = %s
    """, (agentId,))
    conn.commit()
    cur.close()
    conn.close()


def getUserAgents(userId, siteName):
    """Gets all agents for a specific user from the database."""
    conn = getDbConnection(siteName)
    cur = conn.cursor()
    cur.execute("""
        SELECT agent_id, user_id, agent_name, agent_description, agent_config, created_at, updated_at
        FROM agents
        WHERE user_id = %s
        ORDER BY created_at DESC
    """, (userId,))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    agents = []
    for row in rows:
        agent = {
            "agentId": row[0],
            "userId": row[1],
            "agentName": row[2],
            "agentDescription": row[3],
            "agentConfig": json.loads(row[4]) if row[4] else {},
            "createdAt": row[5].isoformat() if row[5] else None,
            "updatedAt": row[6].isoformat() if row[6] else None
        }
        agents.append(agent)

    return agents
