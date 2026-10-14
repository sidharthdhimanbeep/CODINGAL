
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Create a folder to save visualizations
os.makedirs("visualizations", exist_ok=True)

# 1. Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# 2. Normalize the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3. Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Add a channel dimension for image augmentation
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

datagen.fit(x_train)

# 4. Build a deeper neural network
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Flatten(),

    layers.Dense(128),
    layers.LeakyReLU(negative_slope=0.1),

    layers.Dense(64),
    layers.LeakyReLU(negative_slope=0.1),

    layers.Dense(10, activation="softmax")
])

# 5. Compile the model using RMSprop
model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 6. Train the model for 10 epochs
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=10,
    validation_data=(x_test, y_test)
)

# 7. Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# 8. Plot training and validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("visualizations/accuracy.png")
plt.show()

# 9. Plot training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("visualizations/loss.png")
plt.show()

# 10. Make predictions
predictions = model.predict(x_test[:10])
predicted_digits = predictions.argmax(axis=1)

# 11. Visualize predictions against actual digits
plt.figure(figsize=(12, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    plt.imshow(x_test[i].squeeze(), cmap="binary")

    predicted = predicted_digits[i]
    actual = y_test[i]

    plt.title(
        f"Predicted: {predicted}\nActual: {actual}",
        color="green" if predicted == actual else "red"
    )

    plt.axis("off")

plt.tight_layout()
plt.savefig("visualizations/predictions.png")
plt.show()

# 12. Save the trained model
model.save("digit_classifier.keras")

print("\nModel saved successfully!")
print("Visualizations saved in the visualizations folder.")
