import json

class ModelARN:
    """
    Contains ARNs for various Bedrock models.
    """
    CLAUDE_3_7_SONNET_V2 = "arn:aws:bedrock:us-east-1::inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    CLAUDE_4_0_SONNET_V1 = "arn:aws:bedrock:us-east-1::inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0"
    CLAUDE_4_0_OPUS_V1 = "arn:aws:bedrock:us-east-1::inference-profile/us.anthropic.claude-opus-4-20250514-v1:0"
    AWS_NOVA_LITE = "arn:aws:bedrock:us-east-1::inference-profile/us.amazon.nova-lite-v1:0"
    AWS_NOVA_MICRO = "arn:aws:bedrock:us-east-1::inference-profile/us.amazon.nova-micro-v1:0"
    AWS_NOVA_PRO = "arn:aws:bedrock:us-east-1::inference-profile/us.amazon.nova-pro-v1:0"
    AWS_NOVA_PREMIER = "arn:aws:bedrock:us-east-1::inference-profile/us.amazon.nova-premier-v1:0"
    DEEPSEEK_V1 = "arn:aws:bedrock:us-east-1::inference-profile/us.deepseek.r1-v1:0"
    LLAMA_4_MAV = "arn:aws:bedrock:us-east-1::inference-profile/us.meta.llama4-maverick-17b-instruct-v1:0"
    LLAMA_4_SCOUT = "arn:aws:bedrock:us-east-1::inference-profile/us.meta.llama4-scout-17b-instruct-v1:0"
    MISTRAL_PIXTRAL_LARGE = "arn:aws:bedrock:us-east-1::inference-profile/us.mistral.pixtral-large-2502-v1:0"


class BedrockModelBodies:
    """
    Provides methods to get pre-configured JSON string bodies for various Bedrock models.
    Inference parameters are fixed within each method.
    """

    def __init__(self):
        pass

    def getAnthropicClaudeBody(self, promptText: str, messages: list = None) -> str:
        """
        Returns the JSON string body for Anthropic Claude models with fixed parameters.
        Optionally accepts a pre-defined messages list.
        """
        if messages is None:
            actualMessages = [{"role": "user", "content": promptText}]
        else:
            actualMessages = messages

        bodyDict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200_000,
            "messages": actualMessages,
            "temperature": 0.7,
            "top_k": 300,
            "top_p": 0.999,
            "stop_sequences": []
        }
        return json.dumps(bodyDict)

    def getAwsNovaBody(self, promptText: str) -> str:
        """
        Returns the JSON string body for AWS Nova models with fixed parameters.
        """
        bodyDict = {
            "inferenceConfig": {
                "max_new_tokens": 300_000,
                "temperature": 0.5,
                "top_p": 0.9,
                "stop_sequences": []
            },
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": promptText
                        }
                    ]
                }
            ]
        }
        return json.dumps(bodyDict)

    def getMetaLlamaBody(self, promptText: str) -> str:
        """
        Returns the JSON string body for Meta Llama models with fixed parameters.
        """
        bodyDict = {
            "prompt": promptText,
            "max_gen_len": 128_000,
            "temperature": 0.7,
            "top_p": 0.9
        }
        return json.dumps(bodyDict)

    def getDeepseekBody(self, promptText: str) -> str:
        """
        Returns the JSON string body for Deepseek models with fixed parameters.
        """
        bodyDict = {
            "inferenceConfig": {
                "max_tokens": 128_000,
                "temperature": 0.7,
                "top_p": 0.9,
                "stop_sequences": []
            },
            "messages": [
                {
                    "role": "user",
                    "content": promptText
                }
            ]
        }
        return json.dumps(bodyDict)

    def getMistralPixtralBody(self, promptText: str = None, imageUrl: str = None) -> str:
        """
        Returns the JSON string body for Mistral Pixtral, supporting multimodal input
        (text and/or image URL) with fixed inference parameters.
        """
        if not promptText and not imageUrl:
            raise ValueError("Either promptText or imageUrl must be provided for Mistral Pixtral.")

        content_parts = []
        if promptText:
            print("Warning: For Mistral Pixtral text input, ensure `promptText` is formatted with any required instruction tokens if applicable.")
            content_parts.append({
                "type": "text",
                "text": promptText
            })

        if imageUrl:
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": imageUrl
                }
            })

        bodyDict = {
            "messages": [
                {
                    "role": "user",
                    "content": content_parts
                }
            ],
            "max_tokens": 128_000,
            "temperature": 0.7,
        }
        return json.dumps(bodyDict)
