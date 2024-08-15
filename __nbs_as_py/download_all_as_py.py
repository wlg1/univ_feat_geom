#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('capture', '', "import os\n\n# Step 1: Mount Google Drive\nfrom google.colab import drive\ndrive.mount('/content/drive')\n\n# Step 2: Install nbconvert if not already installed\n!pip install nbconvert\n")


# In[2]:


# Step 3: Define the source folder containing your .ipynb files
source_folder_path = '/content/drive/My Drive/Colab Notebooks/__convert_nb_to_py/'

# Step 4: Define the target folder where you want to save the .py files
target_folder_path = '/content/drive/My Drive/Colab Notebooks/__convert_nb_to_py/__nbs_as_py'

# Create the target folder if it doesn't exist
os.makedirs(target_folder_path, exist_ok=True)

# Step 5: Convert all .ipynb files in the source folder and subfolders to .py files in the target folder
import os
import nbformat
from nbconvert import PythonExporter

for dirpath, _, filenames in os.walk(source_folder_path):
    for filename in filenames:
        if filename.endswith(".ipynb"):
            # Read the notebook content
            with open(os.path.join(dirpath, filename), 'r', encoding='utf-8') as f:
                notebook_content = nbformat.read(f, as_version=4)

            # Convert to Python script
            python_exporter = PythonExporter()
            script, _ = python_exporter.from_notebook_node(notebook_content)

            # Determine the subfolder structure
            relative_path = os.path.relpath(dirpath, source_folder_path)
            target_subfolder_path = os.path.join(target_folder_path, relative_path)

            # Create subfolders in the target path if they don't exist
            os.makedirs(target_subfolder_path, exist_ok=True)

            # Save the Python script in the target subfolder
            script_filename = filename.replace('.ipynb', '.py')
            with open(os.path.join(target_subfolder_path, script_filename), 'w', encoding='utf-8') as f:
                f.write(script)

print("Conversion completed. Scripts saved in:", target_folder_path)


# In[ ]:




