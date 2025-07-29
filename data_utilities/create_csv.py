# %%IMPORTS
import os
import pandas as pd

# %% LOAD PATHS
csv = pd.read_csv("/local/data3/chrsp39/DPD-AI/CSVs/PDAC_data_IMT_AI_project(Blad1).csv", delimiter=';')
img_path = "/local/data3/chrsp39/DPD-AI/TMA"
save_path = "/local/data3/chrsp39/DPD-AI/CSVs"

# %% RENAMING
csv = csv.rename(columns={'Score': 'label'}) 

csv = csv.rename(columns={'Pat id': 'case_id'}) 
csv['case_id'] = csv['case_id'].astype(str).str.strip().str.replace('.0', '', regex=False)
csv['case_id'] = csv['case_id'].replace({
    'lever': 'liver', 'Lever': 'liver',
    'njure': 'kidney', 'Njure': 'kidney',
    'Colon': 'colon',
    'Pancreas': 'pancreas',
})

organs = ['liver', 'kidney', 'colon', 'pancreas']
csv['case_id'] = csv['case_id'].where(csv['case_id'].isin(organs), csv['case_id'].apply(lambda x: f'S{x}'))

csv['Block'] = csv['Block'].astype(str).str.strip().str.replace('.0', '', regex=False)
csv['Block'] = csv['Block'].apply(lambda x: f'LKPG{x}')

csv['Position'] = csv['Position'].astype(str).str.strip().str.replace('.0', '', regex=False)
csv['Position'] = csv['Position'].apply(lambda x: f'P{x}')

csv['label'] = csv['label'].astype(str).str.strip().str.replace('.0', '', regex=False)

csv['label'] = csv['label'].replace({
    '-': 'Not Available',})

organs_df = csv[csv['case_id'].isin(organs)]

organs_df['case_id'] = organs_df['case_id'].astype(str) + '_' + organs_df['Position'].astype(str) + '_' + organs_df['Block'].astype(str)
organs_df['slide_id'] = organs_df['case_id'].astype(str)

organs_df.drop(columns=['Position', 'Block'], inplace=True)

organs_df = organs_df[['case_id', 'slide_id', 'label']]

csv = csv[~csv['case_id'].isin(organs)]

csv['slide_id'] = csv['case_id'].astype(str) + '_SC' + csv['label'].astype(str) + '_' + csv['Position'].astype(str) + '_' + csv['Block'].astype(str)


csv['case_id'] = csv['case_id'].astype(str) + '_SC' + csv['label'].astype(str)

csv.drop(columns=['Position', 'Block'], inplace=True)
csv = csv[['case_id', 'slide_id', 'label']]

curated_data = pd.concat([organs_df, csv], ignore_index=True)

# save the modified DataFrame to a new csv file for data visualization
curated_data.to_csv(os.path.join(save_path, "curated_data.csv"), index=False)

# %% csv for CLAM
# drop 'Not Available' label
csv = csv[csv['label'] != 'Not Available']

# %%

# remove entries not in image directory
files = os.listdir(img_path)
imgs = []

for file in files:
    img_name = file.split(".")[0]
    imgs.append(img_name)

non_existent_slides = []
i = 0
for slide_id in csv['slide_id']:
    if slide_id not in imgs:
        i += 1
        print(f"Slide ID {slide_id} not found in image directory.")
        non_existent_slides.append(slide_id)
print(f"Total non-existent slides: {i}")

csv = csv[csv['slide_id'].isin(imgs)]

# save csv for CLAM
csv.to_csv(os.path.join(save_path, "4_class.csv"), index=False)

# %%
# drop label score 3
csv = csv[csv['label'] != '3']

# save csv for CLAM
csv.to_csv(os.path.join(save_path, "0_1_2_classes.csv"), index=False)

# %%
