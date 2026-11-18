import os, json

img_dir = "data/raw_images"
out_file = "data/captions/metadata.jsonl"

with open(out_file, "w") as f:
    for img in os.listdir(img_dir):
        if img.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Dynamically writes perfectly formatted JSON for every image
            entry = {"file_name": f"{img_dir}/{img}", "text": "A portrait of prism_target"}
            f.write(json.dumps(entry) + "\n")
            
print("Captions perfectly formatted and saved!")
