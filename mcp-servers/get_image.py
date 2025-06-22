from mcp.server.fastmcp import FastMCP
import os
import requests
import base64
from typing import Dict, Optional, List
from dotenv import load_dotenv
import logging

# Initialize FastMCP app
app = FastMCP("text_to_image_mcp")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Azure OpenAI configurations
# Azure OpenAI configurations
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "dall-e-3")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

if not AZURE_OPENAI_API_KEY:
    raise ValueError("AZURE_OPENAI_API_KEY environment variable must be set")
    
if not AZURE_OPENAI_ENDPOINT:
    raise ValueError("AZURE_OPENAI_ENDPOINT environment variable must be set")

# Ensure the endpoint URL doesn't have a trailing slash
AZURE_OPENAI_ENDPOINT = AZURE_OPENAI_ENDPOINT.rstrip('/')

def generate_image_request(
    prompt: str, 
    size: str = "1024x1024", 
    n: int = 1,
    response_format: str = "url", 
    quality: str = "standard"
) -> Dict:
    """Send request to Azure OpenAI to generate image from text."""
    # Fix: Use the base URL without specific paths
    base_url = AZURE_OPENAI_ENDPOINT.split("/openai/deployments")[0]
    url = f"{base_url}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/images/generations?api-version={AZURE_OPENAI_API_VERSION}"
    
    # Alternative if deployment name is already in the endpoint
    # url = f"{AZURE_OPENAI_ENDPOINT}?api-version={AZURE_OPENAI_API_VERSION}"
    
    logger.info(f"Requesting image generation from: {url}")
    
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY
    }
    
    payload = {
        "prompt": prompt,
        "size": size,
        "n": n,
        "response_format": response_format,
        "quality": quality
    }
    
    try:
        logger.info(f"Sending request to Azure OpenAI DALL-E endpoint")
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Successfully generated image")
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during API request: {str(e)}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"Response content: {e.response.text}")
        raise RuntimeError(f"Failed to generate image: {str(e)}")

@app.tool()
def text_to_image(
    prompt: str,
    size: str = "1024x1024",
    n: int = 1,
    output_format: str = "url",
    quality: str = "standard",
    save_locally: bool = False,
    output_dir: str = "output_images"
) -> Dict:
    """
    Generate an image from text using Azure OpenAI DALL-E.
    
    Args:
        prompt: Text description of the desired image
        size: Image size (256x256, 512x512, 1024x1024, 1792x1024, or 1024x1792)
        n: Number of images to generate
        output_format: Format of the image in the response ("url" or "b64_json")
        quality: Image quality ("standard" or "hd")
        save_locally: Whether to save images locally
        output_dir: Directory to save images if save_locally is True
        
    Returns:
        Dict with image URLs or base64 data and metadata
    """
    try:
        # Validate inputs
        if not prompt:
            raise ValueError("Prompt cannot be empty")
            
        valid_sizes = ["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]
        if size not in valid_sizes:
            raise ValueError(f"Size must be one of {', '.join(valid_sizes)}")
            
        if output_format not in ["url", "b64_json"]:
            raise ValueError("Output format must be 'url' or 'b64_json'")
            
        if quality not in ["standard", "hd"]:
            raise ValueError("Quality must be 'standard' or 'hd'")
        
        # Generate image
        response_data = generate_image_request(
            prompt=prompt,
            size=size,
            n=n,
            response_format=output_format,
            quality=quality
        )
        
        # Handle local saving if requested
        saved_paths = []
        if save_locally and response_data.get("data"):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            for i, img_data in enumerate(response_data["data"]):
                file_path = f"{output_dir}/image_{i+1}.png"
                
                if "b64_json" in img_data:
                    # Save base64 encoded image
                    with open(file_path, "wb") as f:
                        f.write(base64.b64decode(img_data["b64_json"]))
                    saved_paths.append(file_path)
                elif "url" in img_data:
                    # Download and save image from URL
                    image_response = requests.get(img_data["url"])
                    image_response.raise_for_status()
                    with open(file_path, "wb") as f:
                        f.write(image_response.content)
                    saved_paths.append(file_path)
        
        # Prepare response
        result = {
            "images": response_data.get("data", []),
            "created": response_data.get("created"),
            "model": AZURE_OPENAI_DEPLOYMENT,
            "metadata": {
                "prompt": prompt,
                "size": size,
                "quality": quality
            }
        }
        
        if saved_paths:
            result["saved_paths"] = saved_paths
            
        return result
    
    except Exception as e:
        logger.error(f"Error in text_to_image: {str(e)}")
        raise RuntimeError(f"Failed to generate image from text: {str(e)}")

if __name__ == "__main__":
    # For debugging/testing
    result = text_to_image(
        prompt="A beautiful sunset over mountains",
        save_locally=True
    )
    print(result)