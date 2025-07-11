import numpy as np
import matplotlib.pyplot as plt

def dice_score(y_true, y_pred):
    smooth = 1e-6
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def plot_prediction(X_val, y_val, pred_mask, i=0):
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(X_val[i][:,:,0], cmap='gray')
    plt.title("Input MRI (FLAIR)")
    plt.subplot(1,3,2)
    plt.imshow(y_val[i][:,:,0], cmap='gray')
    plt.title("Ground Truth Mask")
    plt.subplot(1,3,3)
    plt.imshow(pred_mask[:,:,0], cmap='gray')
    plt.title("Predicted Mask")
    plt.show()
