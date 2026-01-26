import os
import shutil
from dotenv import load_dotenv

from common.utils import compose_filename, copy_file # requires python-dotenv


if not load_dotenv("./../config/credentials_my.env"):
    print("Environment variables not loaded, cell execution stopped")
else:
    print("Environment variables have been loaded ;-)")

VISION_ENDPOINT = os.getenv('VISION_ENDPOINT')
VISION_KEY = os.getenv('VISION_KEY')
images_path = os.getenv('IMAGES_PATH', './images/1. ARTWORK COLLISION/')

def main():
    from common.roi_identification import roi_identification
    from common.roi_highlighting import roi_overlay
    from common.roi_hotspots import roi_hotspots
    from common.image_genai import genai_analysis

    payloads = []

    images_to_process = [] # [] # leave empty to process all images in the folder
    if not images_to_process:
        images_to_process = os.listdir(images_path)

    # iterate over images in the images folder
    for image_file_name in images_to_process:
        if os.path.isfile(os.path.join(images_path, image_file_name)):
            print(f"Duplicating source image {image_file_name}...")
            image_source_path=os.path.join(images_path, image_file_name)
            image_target_path=compose_filename(os.path.join(images_path, image_file_name), "00_original")
            copy_file(
                source_file=image_source_path,
                target_file=image_target_path
            )

            print(f"Analyzing image {image_file_name}...")
            roi_payload = roi_identification(
                image_path=image_source_path,
                VISION_ENDPOINT=VISION_ENDPOINT,
                VISION_KEY=VISION_KEY,
                features="Caption,Objects,Tags,DenseCaptions",
                limit_proposals=0, # get all proposals
            )            
            payloads.append(roi_payload)

            print(f"Overlaying ROI for image {image_file_name}...")
            roi_overlay(
                image_path=image_source_path,
                payload=roi_payload,
            )

            print(f"Creating hotspots for image {image_file_name}...")
            hotspots = roi_hotspots(
                image_path=image_source_path,
                create_edge_map=True,
                create_local_variance_map=True,
                create_high_freq_map=True,
                save_hotspots_heat=True,
                save_hotspots_overlay=True
            )

            print(f"Analyzing with GENAI hotspots for image {image_file_name}...")
            hostspotted_image = genai_analysis(
                deployment_name=os.getenv("AZURE_OPENAI_CHAT_MULTIMODEL_DEPLOYMENT_NAME"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), # Azure OpenAI resource
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),# at least 2024-02-15-preview,
                original_image_path=image_source_path, # "G6YH19W3643-G6O3.png", 
                hotspots_image_path=hotspots["hotspots_heat_path"], # "G6YH19W3643-hotspots_heat.png"
                save_payload=True,
            )

    print(f"{len(payloads)} images processed.") 

if __name__ == "__main__":
    main()