import os
import numpy as np
import nibabel as nib
from skimage.transform import resize

def load_nii(file_path):
    img = nib.load(file_path)
    return np.asarray(img.dataobj)

def extract_slices_for_segmentation(data_path, num_slices=155, stride=5, target_size=(128, 128)):
    images, masks, metadata = [], [], []

    for patient_id in os.listdir(data_path):
        patient_path = os.path.join(data_path, patient_id)
        if not os.path.isdir(patient_path):
            continue

        modalities = {}
        for modality in ['flair', 't1', 't1ce', 't2']:
            file_path = os.path.join(patient_path, f"{patient_id}_{modality}.nii")
            if not os.path.exists(file_path):
                break
            modalities[modality] = load_nii(file_path)
        else:
            mask_path = os.path.join(patient_path, f"{patient_id}_seg.nii")
            if not os.path.exists(mask_path):
                continue
            mask_volume = load_nii(mask_path)

            for slice_idx in range(0, num_slices, stride):
                slices = []
                skip = False
                for modality in ['flair', 't1', 't1ce', 't2']:
                    slice_img = modalities[modality][:, :, slice_idx]
                    if np.max(slice_img) - np.min(slice_img) == 0:
                        skip = True
                        break
                    slice_img = resize(slice_img, target_size, preserve_range=True)
                    slice_img = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img))
                    slices.append(slice_img)

                if skip:
                    continue

                multi_modal_img = np.stack(slices, axis=-1)
                slice_mask = mask_volume[:, :, slice_idx]
                slice_mask = resize(slice_mask, target_size, preserve_range=True, order=0)
                slice_mask = (slice_mask > 0).astype(np.uint8)

                images.append(multi_modal_img)
                masks.append(slice_mask[..., np.newaxis])
                metadata.append((patient_id, slice_idx))

    return np.array(images), np.array(masks), metadata
