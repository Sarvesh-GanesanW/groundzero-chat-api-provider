import json
import uvicorn
import os
import time
from typing import Optional, AsyncGenerator, Dict, Any, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import boto3
import uuid
from constants import Constants
import traceback
import jwt
from appCache import promptCache
from utils.bedrockUtils import interactWithBedrock

try:
    from memory.conversationMemory import ConversationMemory
    from utils.s3Utils import getS3FileContent
    from utils.databaseUtils import (getConversationMetadata, createConversationsTable,
                                           createEmbeddingsTable, insertEmbeddingChunks, insertConversationMetadata,
                                           createAgentsTable, insertAgent, getAgentMetadata, updateAgent, 
                                           deleteAgent, getUserAgents)
    from utils.bedrockUtils import getEmbeddingsBedrock, generate as originalGenerate
    from utils.ragUtils import parsePdf, parseCsv, parseExcel, chunkText
    from GZGraph.gzState import GZAgentState, GZMessage
    from GZGraph.gzAgentFactory import createToolUsingAgent
    from utils.dbUtils import getSiteDB
except ImportError as e:
    print(f"Error importing modules: {e}.")
    exit(1)

defaultModelId = Constants.DEFAULT_MODEL_ID
app = FastAPI(title="GZChat LLM and Agent Framework")

origins = [
    "*",
    "null"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

gzToolAgentExecutor = createToolUsingAgent(useAgenticRAG=True)

class ChatRequest(BaseModel):
    userInput: str
    conversationId: str
    userId: str
    conversationName: Optional[str] = 'New Chat'
    modelId: Optional[str] = defaultModelId
    useRag: Optional[bool] = True
    isAgent: Optional[bool] = False
    agentConfig: Optional[Dict[str, Any]] = Field(default_factory=dict)
    useCache: Optional[bool] = True


class GetConversationRequest(BaseModel):
    conversationId: str


class CacheControlRequest(BaseModel):
    action: str  # clear, stats, invalidate
    pattern: Optional[str] = None

class CreateAgentRequest(BaseModel):
    agentName: str
    agentDescription: Optional[str] = None
    agentConfig: Dict[str, Any] = Field(default_factory=dict)

class GetAgentRequest(BaseModel):
    agentId: str

class UpdateAgentRequest(BaseModel):
    agentId: str
    agentName: Optional[str] = None
    agentDescription: Optional[str] = None
    agentConfig: Optional[Dict[str, Any]] = None

class DeleteAgentRequest(BaseModel):
    agentId: str

class GetUserAgentsRequest(BaseModel):
    userId: Optional[str] = None

async def getUserAndRole(request: Request) -> Dict[str, Any]:
    """
    This dependency extracts user information from the Authorization header (JWT).
    """
    try:
        token = request.headers.get('Authorization')
        if not token:
            raise HTTPException(status_code=401, detail="Unauthorized: Missing Authorization header")

        decoded_jwt = jwt.decode(token, options={"verify_signature": False})
        user = decoded_jwt['cognito:username']
        group = ''
        allGroups = []
        if 'cognito:groups' in decoded_jwt:
            allGroups = decoded_jwt['cognito:groups']
            if 'Admin' in allGroups:
                group = 'ADMIN'
            elif 'Modify' in allGroups:
                group = 'MODIFY'
            elif 'Readonly' in allGroups:
                group = 'READONLY'
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized: Token does not have a user")
        return {'user': user, 'group': group, 'allGroups': allGroups}
    except Exception as ex:
        print(ex)
        raise HTTPException(status_code=401, detail="Unauthorized")

def getSubdomain(request):
    methodName = "app.getSubdomain"
    subdomain = ""
    try:
        origin = request.headers.get("origin")
        if origin is None:
            subdomain = request.headers.get("gz-site")
        else:
            origin = origin.replace("http://", "").replace("https://", "")
            urlParts = origin.split(".")
            if len(urlParts) == 4:
                subdomain = urlParts[0]
    except Exception as ex:
        stack_trace = traceback.format_exc()
        print({"method": methodName, "type": type(ex).__name__, "message": str(ex), "stack_trace": stack_trace})
        raise HTTPException(status_code=401, detail="Unauthorized")
    return subdomain

@app.middleware("http")
async def addRequestAttributes(request: Request, call_next):
    request.state.subdomain = None
    request.state.username = None
    request.state.group = None
    request.state.groups = None

    user_info = await getUserAndRole(request)
    request.state.subdomain = getSubdomain(request)
    request.state.username = user_info["user"]
    request.state.group = user_info["group"]
    request.state.groups = user_info["allGroups"]

    response = await call_next(request)
    return response

@app.on_event("startup")
async def startupEvent():
    print("Checking/Creating database tables...")
    try:
        print("Database tables checked/created.")
    except Exception as dbError:
        print(f"FATAL: Could not initialize database tables: {dbError}")

async def ensureSiteTables(siteName):
    """
    Ensures site-specific tables exist. Call this in routes that need them.
    """
    try:
        createConversationsTable(siteName)
        createEmbeddingsTable(siteName)
        createAgentsTable(siteName)
    except Exception as e:
        print(f"Error ensuring site tables for {siteName}: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.post('/upload_doc')
async def uploadDoc(request: Request, file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail='No file part')
    if not file.filename:
        raise HTTPException(status_code=400, detail='No selected file')
    subdomain = request.state.subdomain
    siteName = getSiteDB(subdomain)
    await ensureSiteTables(siteName)
    filename = file.filename
    fileContent = await file.read()
    fileMimeType = file.content_type
    text = ""
    if fileMimeType == 'application/pdf':
        text = parsePdf(fileContent)
    elif fileMimeType == 'text/csv':
        text = parseCsv(fileContent)
    elif fileMimeType in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
        text = parseExcel(fileContent)
    elif fileMimeType and fileMimeType.startswith('text/'):
        try:
            text = fileContent.decode('utf-8', errors='ignore')
        except Exception as e:
             raise HTTPException(status_code=500, detail=f"Error decoding text file: {e}")
    else:
         raise HTTPException(status_code=400, detail=f"Unsupported file type: {fileMimeType}")

    if not text:
         raise HTTPException(status_code=500, detail='Could not extract text from file')

    textChunks = chunkText(text)
    if not textChunks:
         raise HTTPException(status_code=500, detail='Could not chunk text')

    embeddings = getEmbeddingsBedrock(textChunks)
    chunksToInsert = [(filename, textChunks[i], emb) for i, emb in enumerate(embeddings) if emb]

    if not chunksToInsert:
         raise HTTPException(status_code=500, detail='Failed to generate any embeddings for the file')

    try:
        insertEmbeddingChunks(chunksToInsert, siteName)
        return JSONResponse(content={'message': f'Successfully processed and stored {len(chunksToInsert)} chunks from {filename}, subdomain: {request.state.subdomain}'}, status_code=200) # Access it here
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Database insertion error: {str(e)}')


@app.post('/cache_control')
async def cacheControl(request: CacheControlRequest):
    """Control the prompt cache with operations like clear, stats, and invalidate."""
    if request.action == "clear":
        count = promptCache.clear()
        return JSONResponse(content={'message': f'Cache cleared, {count} entries removed'})
    elif request.action == "stats":
        stats = promptCache.getStats()
        return JSONResponse(content={'cache_stats': stats})
    elif request.action == "invalidate" and request.pattern:
        count = promptCache.invalidateByPattern(request.pattern)
        return JSONResponse(content={'message': f'Invalidated {count} cache entries with pattern "{request.pattern}"'})
    else:
        raise HTTPException(status_code=400, detail="Invalid cache control action")


@app.get('/test_cache')
async def testCache(prompt: str, modelId: Optional[str] = None, useRag: Optional[bool] = False, simulate: Optional[bool] = False):
    """
    Test the prompt cache functionality.
    
    Parameters:
    - prompt: The test prompt to cache and retrieve
    - modelId: The model ID to use (defaults to the system default if not provided)
    - useRag: Whether to simulate RAG being enabled
    - simulate: If true, don't actually call the model but simulate a response
    
    This endpoint helps verify that caching is working correctly:
    1. First call: Cache miss, generates new response
    2. Second call with same parameters: Cache hit, returns cached response
    """
    testModelId = modelId or Constants.DEFAULT_MODEL_ID
    cacheKey = f"test_{prompt}"
    
    cachedResponse = promptCache.get(cacheKey, testModelId, useRag)
    
    if cachedResponse:
        return JSONResponse(content={
            'cache_status': 'hit',
            'response': cachedResponse.get('fullResponse', 'No content'),
            'cached_at': cachedResponse.get('timestamp', 0),
            'metadata': {
                'inputTokens': cachedResponse.get('inputTokens', 0),
                'outputTokens': cachedResponse.get('outputTokens', 0)
            }
        })
    
    if simulate:
        simulatedResponse = f"This is a simulated response for: {prompt}"
        simulatedData = {
            'fullResponse': simulatedResponse,
            'inputTokens': len(prompt.split()),
            'outputTokens': len(simulatedResponse.split()),
            'timestamp': time.time(),
            'metrics': {
                'inputTokens': len(prompt.split()),
                'outputTokens': len(simulatedResponse.split()),
                'stop_reason': 'simulated'
            }
        }
        
        promptCache.set(cacheKey, testModelId, simulatedData, useRag, ttl=3600)
        
        return JSONResponse(content={
            'cache_status': 'miss',
            'response': simulatedResponse,
            'simulated': True,
            'stored_in_cache': True
        })
    else:
        try:
            promptText = f"Human: {prompt}\n\nAssistant:"
            responseStream = interactWithBedrock(promptText, testModelId, enableCaching=False)
            
            if responseStream is None:
                return JSONResponse(content={
                    'cache_status': 'miss',
                    'error': 'Failed to get response from model'
                }, status_code=500)
            
            fullResponse = ""
            inputTokens = 0
            outputTokens = 0
            
            for event in responseStream:
                chunk = event.get('chunk')
                if chunk:
                    chunkData = json.loads(chunk.get('bytes').decode())
                    chunkType = chunkData.get('type')
                    
                    if chunkType == 'content_block_delta':
                        text = chunkData.get('delta', {}).get('text', '')
                        fullResponse += text
                    elif chunkType == 'message_delta':
                        usage = chunkData.get('usage', {})
                        if 'output_tokens' in usage:
                            outputTokens = usage['output_tokens']
                    elif chunkType == 'message_stop':
                        bedrock_metadata = chunkData.get('amazon-bedrock-invocationMetrics', {})
                        inputTokens = bedrock_metadata.get('inputTokenCount', 0)
                        outputTokens = bedrock_metadata.get('outputTokenCount', outputTokens or 0)
            
            cacheData = {
                'fullResponse': fullResponse,
                'inputTokens': inputTokens,
                'outputTokens': outputTokens,
                'timestamp': time.time(),
                'metrics': {
                    'inputTokens': inputTokens,
                    'outputTokens': outputTokens,
                    'stop_reason': 'normal'
                }
            }
            promptCache.set(cacheKey, testModelId, cacheData, useRag, ttl=3600)
            
            return JSONResponse(content={
                'cache_status': 'miss',
                'response': fullResponse,
                'stored_in_cache': True,
                'metadata': {
                    'inputTokens': inputTokens,
                    'outputTokens': outputTokens
                }
            })
        except Exception as e:
            return JSONResponse(content={
                'cache_status': 'miss',
                'error': f'Error generating response: {str(e)}'
            }, status_code=500)


@app.post('/chat')
async def chat(chatRequest: ChatRequest, request: Request):
    subdomain = request.state.subdomain
    siteName = getSiteDB(subdomain)
    await ensureSiteTables(siteName)

    if chatRequest.isAgent:
        loaded_messages_for_agent: List[GZMessage] = []
        s3_bucket_name_for_agent = os.getenv("DEFAULT_BUCKET_NAME", "genailogs1")
        agent_s3_object_key_prefix = "agent_conversations"

        if chatRequest.conversationId:
            try:
                metadata = getConversationMetadata(chatRequest.conversationId, siteName)
                if metadata and metadata.get('s3FileLocation'):
                    s3_file_key_from_meta = metadata['s3FileLocation']
                    file_key_to_load = s3_file_key_from_meta

                    if not file_key_to_load.startswith(agent_s3_object_key_prefix + "/"):
                         print(f"Agent Mode: s3FileLocation '{file_key_to_load}' in metadata does not match expected agent prefix. Treating as new conversation for agent.")
                    else:
                        s3_content = getS3FileContent(s3_bucket_name_for_agent, file_key_to_load)
                        if s3_content:
                            try:
                                history_from_s3 = json.loads(s3_content)
                                if isinstance(history_from_s3, list):
                                    for msg_dict in history_from_s3:
                                        msg_dict.setdefault('id', str(uuid.uuid4()))
                                        msg_dict.setdefault('contentType', 'text')
                                        msg_dict.setdefault('isError', False)
                                        loaded_messages_for_agent.append(GZMessage(**msg_dict))
                                    print(f"Agent Mode: Loaded {len(loaded_messages_for_agent)} messages from S3 for {chatRequest.conversationId}")
                            except json.JSONDecodeError:
                                print(f"Agent Mode: JSONDecodeError for S3 history, {chatRequest.conversationId}")
                            except Exception as e_load:
                                print(f"Agent Mode: Error converting S3 to GZMessage, {chatRequest.conversationId}, {e_load}")
                        else:
                            print(f"Agent Mode: No S3 content for {file_key_to_load}, {chatRequest.conversationId}")
                else:
                    print(f"Agent Mode: No S3 metadata for {chatRequest.conversationId}, starting fresh.")
            except Exception as e:
                print(f"Agent Mode: Error S3 history load, {chatRequest.conversationId}, {str(e)}")

        initialAgentState = GZAgentState(
            userInput=chatRequest.userInput,
            conversationId=chatRequest.conversationId,
            userId=chatRequest.userId,
            modelId=chatRequest.modelId or Constants.DEFAULT_MODEL_ID,
            messages=loaded_messages_for_agent
        )

        agentInvocationConfig = chatRequest.agentConfig or {}
        if chatRequest.useRag is False:
            agentInvocationConfig["disableRag"] = True

        async def streamAgentResponse() -> AsyncGenerator[str, None]:
            finalStateFromAgent = None
            fullAiTextMessage = ""
            try:
                async for event in gzToolAgentExecutor.invokeStream(initialAgentState, agentInvocationConfig):
                    yield "data: " + json.dumps(event) + "\n\n"
                    if event.get("type") == "llm_chunk" and "text" in event:
                        fullAiTextMessage += event.get("text", "")
                    if event.get("type") == "graph_complete":
                        finalStateFromAgent = GZAgentState(**event.get("finalStateSnapshot", {}))
                        break
                    if event.get("type") == "graph_error" or event.get("type") == "node_error_unhandled":
                        finalStateFromAgent = initialAgentState
                        finalStateFromAgent.errorOccurred = True
                        finalStateFromAgent.errorMessage = event.get("detail")
                        break
            except Exception as e:
                yield "data: " + json.dumps({"type": "critical_error", "detail": str(e)}) + "\n\n"

            if finalStateFromAgent:
                try:
                    messages_to_save_dicts = [msg.model_dump(exclude_none=True) for msg in finalStateFromAgent.messages]
                    agent_s3_file_key = f"{agent_s3_object_key_prefix}/{finalStateFromAgent.conversationId}.json"
                    s3Client = boto3.client('s3')
                    s3Client.put_object(
                        Bucket=s3_bucket_name_for_agent,
                        Key=agent_s3_file_key,
                        Body=json.dumps(messages_to_save_dicts, indent=2)
                    )
                    print(f"Agent Mode: State saved to S3/{agent_s3_file_key} for {finalStateFromAgent.conversationId}")
                    insertConversationMetadata(
                        finalStateFromAgent.conversationId,
                        finalStateFromAgent.userId,
                        agent_s3_file_key,
                        siteName,
                        chatRequest.conversationName,
                    )
                except Exception as e_save:
                    print(f"Agent Mode: Error saving GZAgent state for {finalStateFromAgent.conversationId}: {str(e_save)}")
                    yield "data: " + json.dumps({"type": "system_warning", "detail": f"Failed to save agent history: {str(e_save)}"}) + "\n\n"

            yield "data: [DONE]\n\n"
        return StreamingResponse(streamAgentResponse(), media_type='text/event-stream')
    else:
        conversationId = chatRequest.conversationId
        userId = chatRequest.userId
        userInput = chatRequest.userInput
        conversationName = chatRequest.conversationName
        modelIdToUse = chatRequest.modelId
        useRagFlag = chatRequest.useRag

        conversationMemoryInstance = ConversationMemory()
        conversationMemoryInstance.conversationId = conversationId
        s3_bucket_name_non_agent = os.getenv("DEFAULT_BUCKET_NAME", "genailogs1")
        non_agent_s3_object_key_prefix = "standard_conversations"

        try:
            metadata = getConversationMetadata(conversationId, siteName)
            if metadata and metadata.get('s3FileLocation'):
                s3_file_key_from_meta = metadata['s3FileLocation']
                file_key_to_load = s3_file_key_from_meta

                if not file_key_to_load.startswith(non_agent_s3_object_key_prefix + "/"):
                     print(f"Non-Agent Mode: s3FileLocation '{file_key_to_load}' in metadata does not match expected non-agent prefix. History might be incompatible or from agent.")
                s3Content = getS3FileContent(s3_bucket_name_non_agent, file_key_to_load)
                if s3Content:
                    try:
                        history = json.loads(s3Content)
                        if isinstance(history, list):
                            conversationMemoryInstance.memory = history
                    except json.JSONDecodeError:
                        print(f"Error decoding S3 history for conversation {conversationId}")
        except Exception as e:
            print(f"Error loading previous conversation state for {conversationId}: {str(e)}")
        
        # Check if we should use cache
        useCache = chatRequest.useCache

        return StreamingResponse(originalGenerate(
            userInput=userInput,
            conversationMemoryInstance=conversationMemoryInstance,
            userId=userId,
            modelId=modelIdToUse,
            useRag=useRagFlag,
            conversationId=conversationId,
            conversationName=conversationName,
            siteName=siteName,
            useCache=useCache
        ), media_type='text/event-stream')


@app.post('/get_conversation')
async def getConversation(requestBody: GetConversationRequest, request: Request):
    conversationIdParam = requestBody.conversationId
    subdomain = request.state.subdomain
    siteName = getSiteDB(subdomain)
    await ensureSiteTables(siteName)
    if not conversationIdParam:
         raise HTTPException(status_code=400, detail='Missing conversationId')

    metadata = getConversationMetadata(conversationIdParam, siteName)
    if not metadata:
         raise HTTPException(status_code=404, detail='No conversation found')

    s3FileLocation = metadata.get('s3FileLocation')
    if not s3FileLocation:
        raise HTTPException(status_code=500, detail='Internal server error: Missing S3 location in metadata')

    try:
        if '/' not in s3FileLocation:
             raise ValueError("Invalid S3 location format")
        s3BucketName = os.getenv("DEFAULT_BUCKET_NAME", "genailogs1")
        bucketNameFromFile, fileKey = s3FileLocation.split('/', 1)
    except ValueError:
        raise HTTPException(status_code=500, detail='Internal server error: Invalid S3 location')

    fileContent = getS3FileContent(s3BucketName, fileKey)
    if fileContent is None:
         raise HTTPException(status_code=500, detail='Error retrieving conversation from S3')

    try:
        chatMessages = json.loads(fileContent)
    except json.JSONDecodeError:
         raise HTTPException(status_code=500, detail='Invalid Conversation Format')
    return JSONResponse(content={'chatMessages': chatMessages})


@app.post('/new_chat')
async def newChat():
    return JSONResponse(content={'message': 'New chat session initiated. Previous server-side state for this client might be cleared depending on client-side conversationId management.'})


@app.post('/agents')
async def createAgent(agentRequest: CreateAgentRequest, request: Request):
    """Create a new agent."""
    subdomain = request.state.subdomain
    siteName = getSiteDB(subdomain)
    await ensureSiteTables(siteName)
    
    userId = request.state.username
    if not userId:
        raise HTTPException(status_code=401, detail="Unauthorized: Missing user ID")
    
    agentId = str(uuid.uuid4())
    
    try:
        insertAgent(
            agentId=agentId,
            userId=userId,
            agentName=agentRequest.agentName,
            agentDescription=agentRequest.agentDescription,
            agentConfig=agentRequest.agentConfig,
            siteName=siteName
        )
        
        return JSONResponse(content={
            'agentId': agentId,
            'message': 'Agent created successfully'
        }, status_code=201)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating agent: {str(e)}")


@app.get('/agents/{agent_id}')
async def getAgent(agent_id: str, request: Request):
    """Get an agent by ID."""
    subdomain = request.state.subdomain
    siteName = getSiteDB(subdomain)
    await ensureSiteTables(siteName)
    
    if not agent_id:
        raise HTTPException(status_code=400, detail='Missing agent ID')
    
    try:
        agent = getAgentMetadata(agent_id, siteName)
        if not agent:
            raise HTTPException(status_code=404, detail='Agent not found')
        
        return JSONResponse(content={'agent': agent})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving agent: {str(e)}")


@app.put('/agents/{agent_id}')
async def updateAgentById(agent_id: str, agentRequest: UpdateAgentRequest, request: Request):
    """Update an existing agent."""
    subdomain = request.state.subdomain
    siteName = getSiteDB(subdomain)
    await ensureSiteTables(siteName)
    
    userId = request.state.username
    if not userId:
        raise HTTPException(status_code=401, detail="Unauthorized: Missing user ID")
    
    if not agent_id:
        raise HTTPException(status_code=400, detail='Missing agent ID')
    
    try:
        # Check if agent exists and belongs to user
        existing_agent = getAgentMetadata(agent_id, siteName)
        if not existing_agent:
            raise HTTPException(status_code=404, detail='Agent not found')
        
        if existing_agent['userId'] != userId:
            raise HTTPException(status_code=403, detail='Forbidden: Agent does not belong to user')
        
        updateAgent(
            agentId=agent_id,
            agentName=agentRequest.agentName,
            agentDescription=agentRequest.agentDescription,
            agentConfig=agentRequest.agentConfig,
            siteName=siteName
        )
        
        return JSONResponse(content={'message': 'Agent updated successfully'})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating agent: {str(e)}")


@app.delete('/agents/{agent_id}')
async def deleteAgentById(agent_id: str, request: Request):
    """Delete an agent by ID."""
    subdomain = request.state.subdomain
    siteName = getSiteDB(subdomain)
    await ensureSiteTables(siteName)
    
    userId = request.state.username
    if not userId:
        raise HTTPException(status_code=401, detail="Unauthorized: Missing user ID")
    
    if not agent_id:
        raise HTTPException(status_code=400, detail='Missing agent ID')
    
    try:
        # Check if agent exists and belongs to user
        existing_agent = getAgentMetadata(agent_id, siteName)
        if not existing_agent:
            raise HTTPException(status_code=404, detail='Agent not found')
        
        if existing_agent['userId'] != userId:
            raise HTTPException(status_code=403, detail='Forbidden: Agent does not belong to user')
        
        deleteAgent(agent_id, siteName)
        
        return JSONResponse(content={'message': 'Agent deleted successfully'})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting agent: {str(e)}")


@app.get('/agents')
async def getUserAgentsList(request: Request):
    """Get all agents for the current user."""
    subdomain = request.state.subdomain
    siteName = getSiteDB(subdomain)
    await ensureSiteTables(siteName)
    
    userId = request.state.username
    if not userId:
        raise HTTPException(status_code=401, detail="Unauthorized: Missing user ID")
    
    try:
        agents = getUserAgents(userId, siteName)
        return JSONResponse(content={'agents': agents})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving user agents: {str(e)}")


if __name__ == '__main__':
    portToUse = 5000
    print("Starting server on port 5000...")
    uvicorn.run("app:app", host="0.0.0.0", port=portToUse, log_level="info")
    print(f"Server started on port {portToUse}.")
