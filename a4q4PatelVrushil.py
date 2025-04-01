import keras
import numpy as np
import cv2  # OpenCV for resizing
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras import datasets
from sklearn.model_selection import train_test_split





# Load the MINST dataset
(x_train_full, y_train_full), (x_test, y_test) = datasets.mnist.load_data()

# Split the training set into training (80%) and validation (20%) sets
x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)

# Normalize images
x_train, x_valid, x_test = x_train / 255.0, x_valid / 255.0, x_test / 255.0

def show_samples(x, y, num_samples=10):
    plt.figure(figsize=(12, 5))
    for i in range(num_samples):
        idx = np.random.randint(len(x))  # Select a random index
        plt.subplot(2, 5, i + 1)
        plt.imshow(x[idx])
        plt.title(f"Label: {y[idx]}")
        plt.axis('off')
    plt.savefig(f'Samples{num_samples}.png')
    plt.show()

# visualizing random MNIST samples
show_samples(x_train, y_train)

# Define a "bad" feature extractor (adds noise)
def bad_feature_extractor(x):
    noise = np.random.normal(0, 0.1, x.shape)  # Add Gaussian noise
    x_noisy = np.clip(x + noise, 0, 1)  # Ensure values stay in valid range
    return x_noisy

# Apply the bad feature extractor
x_train_bad = np.array([bad_feature_extractor(img) for img in x_train])
x_valid_bad = np.array([bad_feature_extractor(img) for img in x_valid])
x_test_bad = np.array([bad_feature_extractor(img) for img in x_test])

# Define a simple neural network model
# https://keras.io/api/datasetsdatasets/mnist/ - figured out that images have 28x28 pixels
def create_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to plot validation error
def plot_error(history, feature_type):
    plt.figure()
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Validation Error Over Epochs {feature_type}')
    plt.legend()
    plt.savefig(f'Val_error{feature_type}.png')
    plt.show()


# One-hot encode the labels
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
y_train_one_hot = to_categorical(y_train, num_classes=10)
y_valid_one_hot = to_categorical(y_valid, num_classes=10)
y_test_one_hot = to_categorical(y_test, num_classes=10)

# Train the model with noisy images
model_bad = create_model()
history_bad = model_bad.fit(x_train_bad, y_train_one_hot , epochs=10, batch_size=64, validation_data=(x_valid_bad, y_valid_one_hot))

# Plot validation error for the bad feature extractor
plot_error(history_bad, "-Bad_Feature_Extractor")

# Evaluate the bad model on the test set
loss, accuracy = model_bad.evaluate(x_test_bad, y_test_one_hot, verbose=2)
print(f"Bad - Final Test Loss: {loss:.4f}")
print(f"Bad - Final Test Accuracy: {accuracy:.4f}")


def good_feature_extractor(x):
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    # Applying the sharpening filter
    x_sharp = cv2.filter2D(x, -1, sharpening_kernel)

    # Normalize the values to be between 0 and 1
    x_sharp = np.clip(x_sharp, 0, 1)

    return x_sharp
# training with the same CNN but with sharp images.
x_train_good = np.array([good_feature_extractor(img) for img in x_train])
x_valid_good = np.array([good_feature_extractor(img) for img in x_valid])
x_test_good = np.array([good_feature_extractor(img) for img in x_test])

# good model
model_good = create_model()
history_good = model_good.fit(x_train_good, y_train_one_hot,
                              epochs=10, batch_size=64,
                              validation_data=(x_valid_good, y_valid_one_hot))

# Plot validation error for the good feature extractor
plot_error(history_good, "-Good_Feature_Extractor")

loss, accuracy = model_good.evaluate(x_test_good, y_test_one_hot, verbose=2)

print(f"Good - Final Test Loss: {loss:.4f}")
print(f"Good - Final Test Accuracy: {accuracy:.4f}")

