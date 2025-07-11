from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

def train_model(model, images, masks, validation_split=0.2, batch_size=16, epochs=10):
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=validation_split, random_state=42)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Save only if val_accuracy improves
    checkpoint = ModelCheckpoint("best.h5", 
                                monitor="val_accuracy",   # or "val_loss"
                                mode="max",               # "max" for accuracy, "min" for loss
                                save_best_only=True,
                                verbose=1
    )   
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs)
    return model, X_val, y_val
