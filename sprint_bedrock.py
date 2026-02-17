import boto3
from langchain_aws import ChatBedrock, BedrockEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

def get_llm():
    return ChatBedrock(
        model_id="us.anthropic.claude-3-haiku-20240307-v1:0",
        client=boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-east-1")
        ),
        model_kwargs={"max_tokens": 4096, "temperature": 0.1}
    )

def get_embeddings():
    return BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        client=boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
    )

if __name__ == "__main__":
    print("Testing Bedrock...")
    llm = get_llm()
    response = llm.invoke("Say hello!")
    print("LLM works:", response.content)
    embeddings = get_embeddings()
    vec = embeddings.embed_query("test")
    print("Embeddings work:", len(vec), "dimensions")