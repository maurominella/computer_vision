import os, json, base64
from dotenv import load_dotenv  # requires python-dotenv
from openai import AzureOpenAI        
from .overlay_bboxes import draw_bboxes
from pathlib import Path
from .utils import compose_filename

CURRENT_DIR = Path(__file__).parent

def to_data_uri(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def hotspots_analysis(
        azure_endpoint: str,
        api_key: str,
        api_version: str,
        deployment_name: str,
        original_image_path: str, 
        hotspots_image_path: str,
        ):
    """
    Analyze art collision in an image using Azure OpenAI multimodal capabilities
    comparing the original image to the hotspots heatmap image.
    """

    message_text = "Analyze the art collision in the image. Identify the areas with the most color distortion due to overlapping paint layers."
    
    original_uri = to_data_uri(original_image_path)
    hotspots_uri = to_data_uri(hotspots_image_path)

    with open(CURRENT_DIR / "llm_data/system_message_multimodal.txt", "r", encoding="utf-8") as f:
        system_text = f.read()

    with open(CURRENT_DIR / "llm_data/multimodal_output_schema.json", "r", encoding="utf-8") as f:
        schema  = json.loads(f.read())

    messages = [
        {"role":"system","content": system_text},
        {"role":"user","content": [
            {"type":"text","text":message_text},
            {"type":"image_url","image_url":{"url": original_uri}},
            {"type":"image_url","image_url":{"url": hotspots_uri}}
        ]}
    ]

    client = AzureOpenAI(
        azure_endpoint = azure_endpoint, # Azure OpenAI resource
        api_key        = api_key,  
        api_version    = api_version ,# at least 2024-02-15-preview,
    )

    response = client.chat.completions.create(
        model=deployment_name,
        temperature=0.2,
        top_p=0.9,
        seed=7,  # balance results
        response_format={ "type": "json_schema", "json_schema": { "name": "art_collision_schema", "schema": schema } },
        messages=messages
    )

    # response = client.responses.create(
    #     model=deployment_name,
    #     temperature=0.2,
    #     response_format={
    #         "type":"json_schema",
    #         "json_schema":{
    #             "name":"art_collision_schema",
    #             "schema": schema 
    #         }
    #     },
    #     input=[
    #         {"role":"system","content":[{"type":"text","text":system_text}]},
    #         {"role":"user","content":[
    #             {"type":"input_text","text":message_text},
    #             {"type":"input_image","image_url":{"url": original_uri}},
    #             {"type":"input_image","image_url":{"url": hotspots_uri}}
    #         ]}
    #     ]
    # )

    # payload = json.loads(response.output_text)

    payload = json.loads(response.choices[0].message.content)
    
    print(json.dumps(payload, indent=2))

    out = draw_bboxes(original_image_path, payload, compose_filename(original_image_path, "04_llm_bboxes"))
    return out


def main():
    # Environment variables loading
    if not load_dotenv("./../../../config/credentials_my.env"):
        print("Environment variables not loaded, execution stopped")
        return
    else:
        print("Environment variables have been loaded ;-)")

    hostspotted_image = hotspots_analysis(
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_MULTIMODEL_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), # Azure OpenAI resource
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),# at least 2024-02-15-preview,
        original_image_path="G6YK54W3653-MCDM.png", # "G6YH19W3643-G6O3.png", 
        hotspots_image_path="G6YK54W3653-hotspots_heat.png", # "G6YH19W3643-hotspots_heat.png"
    )
    
    print(f"Hotspots analysis completed into <{hostspotted_image}>.")

if __name__ == "__main__":
    main()