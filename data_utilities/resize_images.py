"""
Script that downscales the dimensions of the WSIs.
"""

# %% IMPORTS
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from datetime import datetime

# %% LOAD PATHS
img_path = "/local/data3/chrsp39/DPD-AI/TMA"
save_path = "/local/data3/chrsp39/DPD-AI/TMA_RESIZED_IMAGES"

if not os.path.exists(save_path):
    os.makedirs(save_path)

# %% RESIZE IMAGES 
basewidth = 500

slide_ids = os.listdir(img_path)
print(f"Total number of slides to be resized: {len(slide_ids)}.")

resized_slide_ids = os.listdir(save_path)
print(f"Total number of slides already resized: {len(resized_slide_ids)}.")

slide_ids_count = 0

start_time = datetime.now()

for slide_id in slide_ids:
    slide_name = os.path.splitext(slide_id)[0]
    resized_slide_id_path = os.path.join(save_path, slide_name + ".png")
    if os.path.exists(resized_slide_id_path):
        print(f"Resized slide already exist: {slide_id}. Skipping...")
        slide_ids_count += 1
        continue

    slide_ids_count += 1
    print(f"Working on slide: {slide_id}.")

    slide_id_path = os.path.join(img_path, slide_id)

    # Open image with PIL
    image = Image.open(slide_id_path)
    # Convert to RGB if necessary
    if image.mode in ('RGBA', 'P', 'L'):
        image = image.convert('RGB')
    print(f"Opened with PIL: {slide_id}")

    # resize the image
    wpercent = (basewidth/float(image.size[0]))
    hsize = int((float(image.size[1])*float(wpercent)))
    image = image.resize((basewidth, hsize), Image.Resampling.LANCZOS)

    # get the slide_id name without format the extension
    slide_id = slide_id.split(".")[0]
    
    save_filename = os.path.join(save_path, slide_id + ".png")    
    image.save(save_filename, format='PNG', optimize=True, quality=90)


    print(f"Slides that are resized: {slide_ids_count}/{len(slide_ids)}.")
    print("------------------------------------------------------------")

end_time = datetime.now()

print(f"Total time to resize all the slides: {end_time-start_time}")
print("Resizing slides finished!")

# %%