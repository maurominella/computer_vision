import os
from dotenv import load_dotenv # requires python-dotenv

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

    payloads = []

    # iterate over images in the images folder
    for image_file_name in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, image_file_name)):
            print(f"Analyzing image: {image_file_name}")
            roi_payload = roi_identification(
                image_path=os.path.join(images_path, image_file_name),
                VISION_ENDPOINT=VISION_ENDPOINT,
                VISION_KEY=VISION_KEY,
                features="Caption,Objects,Tags,DenseCaptions",
                limit_proposals=0, # get all proposals
            )            
            payloads.append(roi_payload)

            print(f"Creating overlay for image: {image_file_name}")
            roi_overlay(
                image_path=os.path.join(images_path, image_file_name), 
                payload=roi_payload,
            )

            print(f"Creating hotspots for image: {image_file_name}")
            roi_hotspots(
                image_path=os.path.join(images_path, image_file_name),
                create_edge_map=True,
                create_local_variance_map=True,
                create_high_freq_map=True,
                save_hotspots_heat=True,
                save_hotspots_overlay=True
            )

    print(f"{len(payloads)} images processed.") 

if __name__ == "__main__":
    main()