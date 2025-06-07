##Upload Document for RAG (/upload_doc)

Replace /path/to/your/document.pdf with the actual path to your file (PDF, CSV, XLSX, TXT).

```bash
curl -X POST \
     -F "file=@/home/saga/LLMChat/Sarvesh_Resume.pdf" \
     http://127.0.0.1:5000/upload_doc
```

##Start a Chat (/chat) - Basic
This sends a simple message using default settings. Add -N flag if you want curl to stay connected for the streaming response.

```bash
curl -X POST -N \
     -H "Content-Type: application/json" \
     -d '{
           "userInput": "Hello, how are you today?",
           "conversationId": "1739804a-819f-46e5-8e67-2ae74d51f2e4",
           "userId": "1739804a-819f-46e5-8e67-2ae74d51f2e4"
         }' \
     http://127.0.0.1:5000/chat
```
##Start a Chat (/chat) - With RAG and Specific Model
This example uses RAG and specifies the Claude 3 Haiku model.
```bash
curl -X POST -N \
     -H "Content-Type: application/json" \
     -d '{
           "userInput": "Summarize the key points from the document about project alpha.",
           "modelId": "arn:aws:bedrock:ap-south-1::inference-profile/apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
           "useRag": true
         }' \
     http://127.0.0.1:5000/chat
```

##Get Conversation History (/get_conversation)
Replace YOUR_CONVERSATION_UUID with an actual conversation ID obtained from a previous chat response or database record.
```bash
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"conversationId": "YOUR_CONVERSATION_UUID"}' \
     http://127.0.0.1:5000/get_conversation
```
##Start a New Chat Session (/new_chat)

This endpoint doesn't require a body; it signals the server to prepare for a logically new conversation context (though the current implementation uses Depends for state, so this might just return a confirmation).
Bash

curl -X POST http://127.0.0.1:5000/new_chat