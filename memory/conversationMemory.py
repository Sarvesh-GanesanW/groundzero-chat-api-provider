import datetime
import json

class ConversationMemory:

    def __init__(self):
        """Initializes a conversationMemory instance to track conversation history."""
        self.conversationId = None
        self.memory = []

    def updateMemory(self, conversationId, userId, speaker, message, inputTokens=0, outputTokens=0):
        """Updates the conversation memory with a new message and its token counts.

        Args:
            conversationId: The ID of the conversation.
            userId: The ID of the user.
            speaker: 'Human' or 'AI'.
            message: The text of the message.
            inputTokens: Number of tokens in the input (relevant for AI message, is the prompt tokens).
            outputTokens: Number of tokens in the output (relevant for AI message).
                           For Human message, this can represent the tokens in their input message.
        """
        timestamp = datetime.datetime.now().isoformat()
        tokenCount = inputTokens + outputTokens

        entry = {
            'conversationId': conversationId,
            'userId': userId,
            'timestamp': timestamp,
            'speaker': speaker,
            'message': message,
            'tokenCount': tokenCount
        }
        self.memory.append(entry)

    def traverseMemory(self, speakerFilter=None, lastNMessages=None): 
        """Traverses the conversation memory."""
        filteredMemory = self.memory 

        if speakerFilter:
            filteredMemory = [msg for msg in filteredMemory if msg['speaker'] == speakerFilter]

        if lastNMessages:
            filteredMemory = filteredMemory[-lastNMessages:]

        for message in filteredMemory:
            timestamp = message['timestamp']
            convId = message['conversationId']
            user = message['userId']
            speaker = message['speaker']
            text = message['message']
            print(f"ConvID: {convId}, UserID: {user}, {timestamp} - {speaker}: {text}")


    def getCurrentContext(self): 
        """Returns the current context from the conversation memory."""
        context = [f"{entry['speaker']}: {entry['message']}" for entry in self.memory]
        return "\n".join(context)


    def toJson(self): 
        """Converts the conversation memory to JSON format."""
        return json.dumps(self.memory, indent=4)


    def clearMemory(self): 
        """Clears the conversation memory."""
        self.memory = []