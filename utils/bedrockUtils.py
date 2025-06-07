import json
from typing import List, AsyncGenerator
from botocore.exceptions import ClientError
from utils.databaseUtils import findSimilarChunks
from memory.conversationMemory import ConversationMemory
from utils.databaseUtils import embeddingDimension
from dotenv import load_dotenv
import os
import boto3
from utils.s3Utils import uploadConversationToS3
from utils.databaseUtils import insertConversationMetadata
from constants import Constants
from utils.bodyGenerator import BedrockModelBodies
from appCache import promptCache

BodyGenerator = BedrockModelBodies()

s3BucketName = os.getenv("S3_BUCKET_NAME", "genailogs1")
load_dotenv()
embeddingModelId = Constants.EMBEDDING_MODEL_ID
ragTopK = 3

try:
    bedrockRuntime = boto3.client(service_name='bedrock-runtime', region_name=Constants.DEFAULT_REGION_NAME)
    print("Bedrock client initialized successfully.")
except Exception as e:
    print(f"Error initializing Bedrock client: {e}")
    bedrockRuntime = None

def getEmbeddingsBedrock(texts: List[str]) -> List[List[float]]:
    if not bedrockRuntime:
        print("Bedrock client not initialized.")
        return [[] for _ in texts]

    embeddings = []
    for text in texts:
        try:
            body = json.dumps({
                "inputText": text,
                "dimensions": embeddingDimension,
                "normalize": True
            })
            response = bedrockRuntime.invoke_model_with_response_stream(
                body=body,
                modelId=embeddingModelId,
                accept="application/json",
                contentType="application/json"
            )
            responseBody = json.loads(response.get('body').read())
            embeddings.append(responseBody.get("embedding"))
        except ClientError as e:
            print(f"Bedrock embedding error: {e}")
            embeddings.append([])
        except Exception as e:
            print(f"Error generating embedding: {e}")
            embeddings.append([])
    return embeddings

def interactWithBedrock(formattedPrompt: str, modelId: str, enableCaching: bool = True):
    """
    Invoke Bedrock model with caching support.

    Note: For streaming responses, we don't cache the response itself but return the stream.
    The actual caching of the complete response happens in the generate function after
    collecting the full stream.
    """
    if not bedrockRuntime:
        print("Bedrock client not initialized.")
        return None

    # We don't do caching at this level for streaming responses
    # Caching happens after we collect the full response in the generate function

    body_string = None
    try:
        if "anthropic.claude" in modelId:
            body_string = BodyGenerator.getAnthropicClaudeBody(promptText=formattedPrompt)
        elif "amazon.nova" in modelId:
            body_string = BodyGenerator.getAwsNovaBody(promptText=formattedPrompt)
        elif "meta.llama" in modelId:
            body_string = BodyGenerator.getMetaLlamaBody(promptText=formattedPrompt)
        elif "deepseek.r1" in modelId:
            body_string = BodyGenerator.getDeepseekBody(promptText=formattedPrompt)
        elif "mistral.pixtral" in modelId:
            body_string = BodyGenerator.getMistralPixtralBody(promptText=formattedPrompt, imageUrl="")
        else:
            error_message = f"Unsupported model ID for interactWithBedrock's direct body generation: {modelId}. Please add a corresponding method in BedrockModelBodies or extend this function."
            print(error_message)
            raise ValueError(error_message)

        response = bedrockRuntime.invoke_model_with_response_stream(
            modelId=modelId,
            body=body_string,
            accept="application/json",
            contentType="application/json"
        )

        return response.get('body', None)

    except ClientError as e:
        print(f"Bedrock invocation error: {e} (Model ID: {modelId})")
        return None
    except ValueError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in interactWithBedrock: {e} (Model ID: {modelId})")
        return None

def getResponse(userInput: str, conversationMemoryInstance: ConversationMemory, userId: str, modelId: str, siteName, useRag: bool = False, useCache: bool = True):
    """
    Generate a response from the model, with optional RAG and caching support.

    This function builds the prompt, retrieves RAG content if enabled, and gets the response
    from Bedrock. Caching is primarily handled at the generate function level (where the
    complete response is collected) rather than at this level.
    """
    retrievedContext = ""
    prompt = ""

    # Get conversation context
    conversationId = conversationMemoryInstance.conversationId
    currentContext = conversationMemoryInstance.getCurrentContext()

    # Check if we can use a cached response
    if useCache:
        cacheKey = f"{conversationId}_{userInput}"
        cachedResponse = promptCache.get(cacheKey, modelId, useRag)

        if cachedResponse:
            print(f"Cache hit: Using cached response for conversation {conversationId}")
            return cachedResponse

    if useRag:
        try:
            queryEmbedding = getEmbeddingsBedrock([userInput])[0]
            if queryEmbedding:
                similarChunks = findSimilarChunks(queryEmbedding, limit=ragTopK, siteName=siteName)
                if similarChunks:
                    retrievedContext = "\n\nRelevant information found:\n" + "\n---\n".join(similarChunks)
                    print(f"--- RAG Context Retrieved ---\n{retrievedContext}\n---------------------------")
                else:
                    print("No similar chunks found for RAG.")
            else:
                print("Could not generate query embedding for RAG.")
        except Exception as e:
            print(f"Error during RAG retrieval: {e}")

    if useRag and retrievedContext:
        prompt = f"""Human: You are an AI assistant designed to provide accurate and helpful responses.
            Use the following pieces of information to answer the user's query. If the provided 'Relevant Information Found' directly addresses the user's query, prioritize using it for your answer. Refer to the 'Conversation History' for context on the ongoing dialogue.

            Conversation History:
            {currentContext}

            Relevant Information Found:
            {retrievedContext}

            User Query: {userInput}

            Guidelines for Response:
            1.  **Prioritize Accuracy:** Base your response primarily on the 'Relevant Information Found'. Only use your general knowledge if the retrieved information is insufficient or irrelevant to the user's query.
            2.  **Integrate Naturally:** Weave the relevant information into your response smoothly. Do *not* explicitly state "Based on the retrieved information..." or refer to the retrieval process.
            3.  **Be Comprehensive:** Provide a thorough and detailed response. Elaborate on points where helpful, even if the user's input isn't a direct question.
            4.  **Maintain Context:** Ensure your response fits logically within the 'Conversation History'.
            5.  **Avoid Hallucination:** Do NOT invent information or make assumptions beyond what is provided in the context or your verified knowledge base. If you cannot answer accurately based on the provided information, state that you don't have the necessary details.
            6.  **Structure for Clarity:** Use numbered steps or bullet points *only if* it genuinely improves the clarity of instructions, lists, or multi-part answers. Otherwise, use standard paragraph format.
            7.  **Respond Directly:** Address the user's query directly and clearly.
            8.  **Date and time:** If you are unsure of the date and time, use provided tools to get the date. Always search for current date and then proceed.
            9.  **Code output and any other requests:** Always use MDX format to output code or any clipboard type requests the user makes.
            Assistant:"""
    else:
        prompt = f"""Human: You are an AI assistant designed to provide helpful and engaging responses.
            Refer to the 'Conversation History' below for context on the ongoing dialogue. Respond to the user's latest statement or question.

            Conversation History:
            {currentContext}

            User Input: {userInput}

            Guidelines for Response:
            1.  **Be Helpful and Engaging:** Provide a meaningful and elaborate response, even if the user's input isn't a direct question. Continue the conversation naturally.
            2.  **Maintain Context:** Ensure your response fits logically within the 'Conversation History'.
            3.  **Rely on Knowledge:** Base your response on your general knowledge and the conversation history.
            4.  **Avoid Hallucination:** Do NOT invent facts or details. Stick to information you know or that is present in the conversation history. If you are unsure about something, it's better to state that.
            5.  **Structure for Clarity:** Use numbered steps or bullet points *only if* it genuinely improves the clarity of instructions, lists, or multi-part answers. Otherwise, use standard paragraph format.
            6.  **Respond Directly:** Address the user's input clearly and appropriately.
            7.  **Code output and any other requests:** Always use MDX format to output code or any clipboard type requests the user makes.
            Assistant:"""
    print(f"--- Prompt Sent to Bedrock ---\n{prompt}\n---------------------------")
    conversationMemoryInstance.updateMemory(conversationMemoryInstance.conversationId, userId, 'Human', userInput, outputTokens=0)
    if useCache:
        print(f"Cache miss: Fetching new response for conversation {conversationId}")

    responseStream = interactWithBedrock(prompt, modelId)

    return responseStream

async def generate(userInput: str, conversationMemoryInstance: 'ConversationMemory', userId: str, modelId: str, useRag: bool, conversationId: str, conversationName: str, siteName, useCache: bool = True) -> AsyncGenerator[str, None]:
    """
    Generate a streaming response from the model, with caching support.

    This function:
    1. Checks the cache for an existing response if caching is enabled
    2. If found, replays the cached response as a simulated stream
    3. If not found, gets a new response from the model
    4. Collects the full response and caches it for future use
    """
    fullResponse = ""
    inputTokens = 0
    outputTokens = 0

    # Only check cache if caching is enabled
    if useCache:
        cacheKey = f"{conversationId}_{userInput}"
        cachedResponseData = promptCache.get(cacheKey, modelId, useRag)

        if cachedResponseData and 'fullResponse' in cachedResponseData:
            # Replay the cached response as if it was streaming
            print(f"Cache replay: Streaming cached response for conversation {conversationId}")
            cached_full_response = cachedResponseData['fullResponse']

            # Send cache hit status event
            cache_hit_event = {
                "type": "cache_status",
                "status": "hit",
                "source": "prompt_cache"
            }
            yield "data: " + json.dumps(cache_hit_event) + "\n\n"

            # Send metadata event
            metrics_event = {
                "type": "llm_metrics",
                "source": "llm_direct_cached",
                "metrics": {
                    "inputTokens": cachedResponseData.get('inputTokens', 0),
                    "outputTokens": cachedResponseData.get('outputTokens', 0),
                    "stop_reason": "cached"
                }
            }
            yield "data: " + json.dumps(metrics_event) + "\n\n"

            # Stream the cached response in chunks to simulate streaming
            chunk_size = 20  # Characters per chunk
            for i in range(0, len(cached_full_response), chunk_size):
                chunk = cached_full_response[i:i+chunk_size]
                text_chunk_event = {
                    "type": "llm_chunk",
                    "source": "llm_direct_cached",
                    "text": chunk
                }
                yield "data: " + json.dumps(text_chunk_event) + "\n\n"

            yield "data: [DONE]\n\n"
            return

    # If we get here, it means we had a cache miss or caching is disabled
    if useCache:
        cache_miss_event = {
            "type": "cache_status",
            "status": "miss",
            "source": "prompt_cache"
        }
        yield "data: " + json.dumps(cache_miss_event) + "\n\n"

    responseStream = getResponse(userInput, conversationMemoryInstance, userId, modelId, siteName, useRag, useCache)

    if responseStream is None:
        error_event = {
            "type": "error",
            "source": "llm_direct",
            "detail": "An error occurred while processing your request with the backend model."
        }
        yield "data: " + json.dumps(error_event) + "\n\n"
        yield "data: [DONE]\n\n"
        return

    try:
        for event in responseStream:
            chunk = event.get('chunk')
            if chunk:
                chunkData = json.loads(chunk.get('bytes').decode())
                chunkType = chunkData.get('type')

                if chunkType == 'content_block_delta':
                    text = chunkData.get('delta', {}).get('text', '')
                    fullResponse += text
                    text_chunk_event = {
                        "type": "llm_chunk",
                        "source": "llm_direct",
                        "text": text
                    }
                    yield "data: " + json.dumps(text_chunk_event) + "\n\n"
                elif chunkType == 'message_delta':
                    usage = chunkData.get('usage', {})
                    if 'output_tokens' in usage:
                        outputTokens = usage['output_tokens']
                elif chunkType == 'message_stop':
                    bedrock_metadata = chunkData.get('amazon-bedrock-invocationMetrics', {})
                    inputTokens = bedrock_metadata.get('inputTokenCount', 0)
                    outputTokens = bedrock_metadata.get('outputTokenCount', outputTokens or 0)
                    stop_reason = chunkData.get('stop_reason', 'unknown')
                    metrics_event = {
                        "type": "llm_metrics",
                        "source": "llm_direct",
                        "metrics": {
                             "inputTokens": inputTokens,
                             "outputTokens": outputTokens,
                             "stop_reason": stop_reason
                        }
                    }
                    yield "data: " + json.dumps(metrics_event) + "\n\n"

            elif any(errKey in event for errKey in ['internalServerException', 'modelStreamErrorException', 'validationException', 'throttlingException', 'modelTimeoutException']):
                errorKey = next(iter(event))
                errorMessageContent = event[errorKey].get('message', 'Unknown LLM streaming error')
                error_event = {
                    "type": "error",
                    "source": "llm_direct",
                    "detail": f"{errorKey}: {errorMessageContent}"
                }
                yield "data: " + json.dumps(error_event) + "\n\n"
    except Exception as e:
        error_event = {
            "type": "error",
            "source": "llm_direct",
            "detail": f"Error processing response stream: {str(e)}"
        }
        yield "data: " + json.dumps(error_event) + "\n\n"
    finally:
        if fullResponse:
            conversationMemoryInstance.updateMemory(
                conversationId, userId, 'AI', fullResponse, inputTokens, outputTokens
            )

            # Cache the complete response for future use if caching is enabled
            if useCache:
                cacheKey = f"{conversationId}_{userInput}"
                cache_data = {
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

                # Store in cache with 1 hour TTL
                promptCache.set(
                    cacheKey,
                    modelId,
                    cache_data,
                    useRag,
                    ttl=3600  # Cache for 1 hour by default
                )

                # Log cache storage
                print(f"Cache store: Saved response for conversation {conversationId}, {len(fullResponse)} chars")

                # Send cache storage event to client
                cache_store_event = {
                    "type": "cache_status",
                    "status": "stored",
                    "source": "prompt_cache",
                    "detail": {
                        "size": len(fullResponse),
                        "ttl": 3600
                    }
                }
                yield "data: " + json.dumps(cache_store_event) + "\n\n"

            try:
                s3FileLocation = uploadConversationToS3(conversationMemoryInstance, s3BucketName)
                if s3FileLocation:
                    insertConversationMetadata(conversationId, userId, s3FileLocation, siteName, conversationName)
            except Exception as e_save:
                print(f"!!! [generate] Error saving conversation state for {conversationId}: {e_save}")
                system_warning_event = {
                    "type": "system_warning",
                    "source": "llm_direct_save",
                    "detail": f"Failed to save conversation history: {str(e_save)}"
                }
                yield "data: " + json.dumps(system_warning_event) + "\n\n"
        yield "data: [DONE]\n\n"
