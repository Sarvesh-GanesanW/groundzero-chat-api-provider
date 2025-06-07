import boto3


def generateFilename(conversationMemoryInstance):
    """
    Generates a filename for storing a conversationMemory object in S3.

    The filename is generated based on the conversationId property of the
    conversationMemory object, with a .json extension.
    """
    return f"conversation_{conversationMemoryInstance.conversationId}.json"


def uploadConversationToS3(conversationMemoryInstance, bucketName):
    """Uploads a conversationMemory object to S3.

    Generates a filename based on the conversation ID, converts the
    conversationMemory object to JSON, uploads it to the provided S3 bucket,
    and returns the S3 file location if successful or None if there was an error.
    """
    s3Client = boto3.client('s3')
    filename = generateFilename(conversationMemoryInstance)
    jsonData = conversationMemoryInstance.toJson()
    try:
        s3Client.put_object(Bucket=bucketName, Key=filename, Body=jsonData)
        print(f"Conversation uploaded to S3 with filename: {filename}")
        return f"{bucketName}/{filename}"
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def getS3FileContent(bucketName, fileKey):
    """
Retrieves the content of an S3 object and returns it as a string.

Args:
    bucketName: The name of the S3 bucket.
    fileKey: The key of the S3 object to retrieve.

Returns:
    The content of the S3 object as a string, or None if there was an error.
    """
    s3Client = boto3.client('s3')
    try:
        response = s3Client.get_object(Bucket=bucketName, Key=fileKey)
        return response['Body'].read().decode('utf-8')
    except Exception as e:
        print(f"Error retrieving file from S3: {e}")
        return None