# OntoMatcher
This is the README file for the Project OntoMatcher.

OntoMatcher is a project that focuses on enhancing biomedical ontology alignment using context-aware Siamese networks, LLMs (Language Models), and BioBERT. It leverages state-of-the-art techniques to improve the alignment of biomedical ontologies, facilitating better interoperability and knowledge integration in the field.

If you are interested in learning more about the details and methodology of OntoMatcher, we recommend referring to the paper titled "OntoMatcher: Leveraging Context-Aware Siamese Networks, LLMs and BioBERT for Enhanced Biomedical Ontology Alignment." The paper is published in the Semantic Web Journal and can be accessed at the following link:

[OntoMatcher: Leveraging Context-Aware Siamese Networks, LLMs and BioBERT for Enhanced Biomedical Ontology Alignment](https://www.semantic-web-journal.net/content/ontomatcher-leveraging-context-aware-siamese-networks-llms-and-biobert-enhanced-biomedical)

Please note that the README file is still a work in progress, and we will be updating it with more complete information soon.
## Status: In Progress

**Note: This README is still being updated. Please check back later for the complete version.**
```bash
pip install -r requirements.txt
```

## Downloading and Extracting Files from Google Drive

This guide provides instructions on how to download a .rar file from Google Drive and extract it using Python.

### Prerequisites

The Python modules `gdown` and `rarfile` are required. If they are not installed, use the following command to install them:

```bash
pip install gdown rarfile
```

Note: The rarfile module uses the unrar utility, which must be installed on your system.

For Unix-like systems, you can usually install it with your package manager. For example, on Debian or Ubuntu, run:
```bash
sudo apt install unrar
```

For Windows, the `rarfile` module will use the `UnRAR.exe` utility from the RARLAB's RAR for Windows package, which must be installed and available in the system PATH.

## Steps
1. Download the .rar file from Google Drive
Use the following Python script to download the .rar file:

```python
import os
import gdown

# Set the base directory (usually the project directory)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Set the path for the downloaded file
output_path_data = os.path.join(base_dir, 'extracted_data.rar')
output_path_model = os.path.join(base_dir, 'model.rar')

# Download the file from Google Drive
url_data = 'https://drive.google.com/uc?id=1h4PRX5ykvYYTq-YXJIxuNul3sLvYPlU8'
gdown.download(url_data, output_path_data, quiet=False)

url_model = 'https://drive.google.com/file/d/1v9a19SbL9_SR1Ta-vRNut56C613RKumH'
gdown.download(url_model, output_path_model, quiet=False)
```
Remember to adjust the url variable with the actual Google Drive URL of the .rar file you want to download.

2. Extract the `.rar` file
Use the following Python script to extract the .rar file:
```python
import rarfile

# Set the path for the extracted files
extraction_path = base_dir

# Extract .rar file
with rarfile.RarFile(output_path_data) as rf:
    rf.extractall(path=extraction_path)
with rarfile.RarFile(output_path_model) as rf:
    rf.extractall(path=extraction_path)
```
This script will extract the contents of the .rar file to the project directory.
That's it! Follow these steps to download a .rar file from Google Drive and extract it using Python.
If you have any further questions, feel free to ask.
