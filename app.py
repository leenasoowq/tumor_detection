import streamlit as st
import numpy as np
import os
from io import BytesIO
import nibabel as nib
import tempfile
from skimage.transform import resize
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# load my trained model
@st.cache_resource
def load_resunet_model():
    return load_model('resunet_model.h5', compile=False)

model = load_resunet_model()

def load_nii(file):
    import gzip
    import shutil
    
    if isinstance(file, str):
        return np.asarray(nib.load(file).dataobj)

    # Check real extension
    fname = file.name
    if fname.endswith(".nii.gz"):
        suffix = ".nii.gz"
    elif fname.endswith(".nii"):
        suffix = ".nii"
    else:
        raise ValueError("Unsupported file format")

    # Save uploaded file to disk with correct suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    try:
        # Load the image data into memory first
        img = nib.load(tmp_path)
        data = np.asarray(img.dataobj)
        
        # Close the image file explicitly
        img.uncache()
        
        return data
    finally:
        # Add a small delay and retry mechanism for file deletion
        import time
        for _ in range(3):  # Try up to 3 times
            try:
                os.remove(tmp_path)
                break
            except PermissionError:
                time.sleep(0.1)  # Wait a bit before retrying

def preprocess_slice(slice_img):
    slice_resized = resize(slice_img, (128, 128), preserve_range=True)
    return (slice_resized - np.min(slice_resized)) / (np.max(slice_resized) - np.min(slice_resized) + 1e-8)

st.title("Brain Tumor Segmentation")
st.write("Upload 4 MRI modalities (NIfTI files) of the **same patient scan**, select a slice, and see predicted tumor mask.")

# Upload files
flair_file = st.file_uploader("FLAIR (.nii)", type=["nii", "nii.gz"])
t1_file = st.file_uploader("T1 (.nii)", type=["nii", "nii.gz"])
t1ce_file = st.file_uploader("T1CE (.nii)", type=["nii", "nii.gz"])
t2_file = st.file_uploader("T2 (.nii)", type=["nii", "nii.gz"])

if flair_file and t1_file and t1ce_file and t2_file:
    # Load data
    flair_img = load_nii(flair_file)
    t1_img = load_nii(t1_file)
    t1ce_img = load_nii(t1ce_file)
    t2_img = load_nii(t2_file)

    # Check shape consistency
    if not (flair_img.shape == t1_img.shape == t1ce_img.shape == t2_img.shape):
        st.error("All modalities must have the same shape!")
    else:
        # Select slice index
        slice_idx = st.slider("Select slice index", 0, flair_img.shape[2]-1, flair_img.shape[2]//2)

        # Preprocess exactly like in training
        flair_slice = preprocess_slice(flair_img[:, :, slice_idx])
        t1_slice = preprocess_slice(t1_img[:, :, slice_idx])
        t1ce_slice = preprocess_slice(t1ce_img[:, :, slice_idx])
        t2_slice = preprocess_slice(t2_img[:, :, slice_idx])

        # Stack into 4 channels
        multi_modal_img = np.stack([flair_slice, t1_slice, t1ce_slice, t2_slice], axis=-1)
        multi_modal_img = np.expand_dims(multi_modal_img, axis=0)

        # Predict
        pred_mask = model.predict(multi_modal_img)[0, :, :, 0]
        binary_mask = (pred_mask > 0.5).astype(np.uint8)

        # Show plots
        st.subheader("Prediction Results")

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(flair_slice, cmap='gray')
        axs[0].set_title("FLAIR slice")
        axs[0].axis("off")

        axs[1].imshow(binary_mask, cmap='gray')
        axs[1].set_title("Predicted Tumor Mask")
        axs[1].axis("off")

        # Overlay
        overlay = np.stack([flair_slice]*3, axis=-1)
        overlay[binary_mask==1] = [1, 0, 0]  # red highlight
        axs[2].imshow(overlay)
        axs[2].set_title("Overlay on FLAIR")
        axs[2].axis("off")

        st.pyplot(fig)
