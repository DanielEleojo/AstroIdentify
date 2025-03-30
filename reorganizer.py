import os
import shutil

# Define the path to the root directory
root_dir = r"C:\Users\DBABA\OneDrive\Documents\Data Science Projects\AstroIdentify\Constalations-Classification-1"

# Define dataset folders
dataset_folders = ['train', 'test', 'valid']

# Define the constellation classes
classes = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", 
    "Libra", "Scorpius", "Sagittarius", "Capricornus", "Aquarius", "Pisces"
]

# Process each dataset folder
for folder in dataset_folders:
    folder_path = os.path.join(root_dir, folder)
    
    # Loop through all files in the dataset folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Check if file starts with a known class name
            for cls in classes:
                if filename.startswith(cls + "_"):
                    # Create the class subfolder if it doesn't exist
                    class_folder_path = os.path.join(folder_path, cls)
                    os.makedirs(class_folder_path, exist_ok=True)
                    
                    # Move the file to the class folder
                    src_path = os.path.join(folder_path, filename)
                    dst_path = os.path.join(class_folder_path, filename)
                    shutil.move(src_path, dst_path)
                    break  # No need to check other classes once matched

print("Image reorganization complete.")
