import numpy as np
from src.data_loader import extract_slices_for_segmentation
from src.model import build_resunet
from src.train import train_model
from src.evaluate import dice_score, plot_prediction

# Set your data path here
data_path = "brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"


# Load data
images, masks, metadata = extract_slices_for_segmentation(data_path, stride=5)

# Build and train model
model = build_resunet((128, 128, 4))
model, X_val, y_val = train_model(model, images, masks, epochs=10)

# Make predictions
pred_mask = model.predict(X_val[0:1])[0]
binary_mask = (pred_mask > 0.5).astype(np.uint8)

# Plot results
plot_prediction(X_val, y_val, binary_mask, i=0)

# Print Dice Score
print("Dice Score:", dice_score(y_val[0][:,:,0], binary_mask[:,:,0]))

