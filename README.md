## installation

Clone the repository :

git clone https://github.com/leenasoowq/tumor_detection.git
cd tumor_detection

Set up environment:

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

Install dependencies :

pip install -r requirements.txt

running UI:

streamlit run app.py 

The UI uses a pretrained model to analyze brain scan images and identify tumor locations.


Background of UI:

- Supports loading 4 different imaging modalities (e.g., T1, T2, FLAIR, etc.)

- dataset for testing https://www.kaggle.com/datasets/awsaf49/brats2020-training-data

- Displays a 360-degree view of the brain with tumor visualization

- Automatically highlights tumor regions across slices



