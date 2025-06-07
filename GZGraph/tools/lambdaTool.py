import boto3
import json
from typing import Dict, Any, Optional
from pydantic import Field, validator

from GZGraph.gzToolBase import GZTool, GZToolInputSchema

class LambdaInvokeInput(GZToolInputSchema):
    """Input schema for AWS Lambda invocation."""
    functionName: str = Field(..., description="The name or ARN of the Lambda function to invoke.")
    payload: Dict[str, Any] = Field(..., description="The JSON payload to send to the Lambda function.")
    region: Optional[str] = Field(default="us-east-1", description="AWS region where the Lambda function is deployed.")
    invocationType: str = Field(default="RequestResponse", description="Lambda invocation type: RequestResponse, Event, or DryRun.")

    @validator('invocationType')
    def validateInvocationType(cls, v):
        allowed_types = ["RequestResponse", "Event", "DryRun"]
        if v not in allowed_types:
            raise ValueError(f"invocationType must be one of {allowed_types}")
        return v

class AWSLambdaTool(GZTool):
    """Tool for invoking AWS Lambda functions."""

    def __init__(self):
        super().__init__(
            toolName="invokeLambda",
            description="Invokes an AWS Lambda function with the provided payload and returns the response. Use this to execute serverless functions.",
            inputSchema=LambdaInvokeInput
        )

    async def executeTool(self, validatedInput: LambdaInvokeInput, **kwargs) -> Dict[str, Any]:
        """Execute the Lambda function with the provided input."""
        try:
            lambdaClient = boto3.client(
                'lambda',
                region_name=validatedInput.region
            )

            payloadJson = json.dumps(validatedInput.payload)

            response = lambdaClient.invoke(
                FunctionName=validatedInput.functionName,
                InvocationType=validatedInput.invocationType,
                Payload=payloadJson
            )
            if validatedInput.invocationType == "RequestResponse":
                if 'Payload' in response:
                    payloadStream = response['Payload']
                    payloadBytes = payloadStream.read()
                    payloadStr = payloadBytes.decode('utf-8')

                    try:
                        result = json.loads(payloadStr)
                    except json.JSONDecodeError:
                        result = {"raw_response": payloadStr}

                    statusCode = response.get('StatusCode')
                    functionError = response.get('FunctionError')

                    return {
                        "statusCode": statusCode,
                        "functionError": functionError,
                        "result": result
                    }
                else:
                    return {"error": "No payload returned from Lambda function"}

            elif validatedInput.invocationType == "Event":
                return {
                    "statusCode": response.get('StatusCode'),
                    "message": "Lambda function invoked asynchronously"
                }
            elif validatedInput.invocationType == "DryRun":
                return {
                    "statusCode": response.get('StatusCode'),
                    "message": "Lambda function dry run successful"
                }
        except boto3.exceptions.ClientError as e:
            errorCode = e.response.get('Error', {}).get('Code', 'UnknownError')
            errorMessage = e.response.get('Error', {}).get('Message', str(e))
            return {
                "error": f"Lambda invocation failed with error: {errorCode}",
                "message": errorMessage
            }
        except Exception as e:
            return {
                "error": "Unexpected error during Lambda invocation",
                "message": str(e)
            }
