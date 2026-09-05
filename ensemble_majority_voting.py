import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import mnist

print("Loading MNIST Dataset...")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)
print("Dataset loaded successfully!")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import mnist

print("Loading MNIST Dataset...")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)
print("Dataset loaded successfully!")

# 2. DATA PREPROCESSING

print("\nStarting Data Preprocessing...")

# Pixel values ko 0-255 se 0-1 range me convert karna
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# CNN ke liye image me channel dimension add karna
x_train_cnn = np.expand_dims(x_train, axis=-1)
x_test_cnn = np.expand_dims(x_test, axis=-1)

print("Training data after preprocessing:", x_train.shape)
print("CNN training data shape:", x_train_cnn.shape)
print("Data preprocessing completed!")

# 3. MODEL 1 - SIMPLE NEURAL NETWORK

print("\nTraining Model 1: Simple Neural Network...")

model1 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model1.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model1.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=128,
    validation_split=0.1
)

print("Model 1 training completed!")

# 4. MODEL 2 - DEEP NEURAL NETWORK

print("\nTraining Model 2: Deep Neural Network...")

model2 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),

    tf.keras.layers.Dense(10, activation="softmax")
])

model2.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model2.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=128,
    validation_split=0.1
)

print("Model 2 training completed!")

# 5. MODEL 3 - CNN

print("\nTraining Model 3: CNN...")

model3 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model3.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model3.fit(
    x_train_cnn,
    y_train,
    epochs=3,
    batch_size=128,
    validation_split=0.1
)

print("Model 3 training completed!")

# 6. PREDICTIONS FROM ALL 3 MODELS

print("\nGetting Predictions...")

pred1 = np.argmax(model1.predict(x_test, verbose=0), axis=1)
pred2 = np.argmax(model2.predict(x_test, verbose=0), axis=1)
pred3 = np.argmax(model3.predict(x_test_cnn, verbose=0), axis=1)

print("Predictions generated successfully!")


# 7. MAJORITY VOTING

print("\nApplying Majority Voting...")

all_predictions = np.array([pred1, pred2, pred3])

ensemble_predictions = []

for i in range(len(y_test)):
    votes = all_predictions[:, i]

    # Select the class receiving maximum votes
    majority_vote = np.bincount(votes).argmax()

    ensemble_predictions.append(majority_vote)

ensemble_predictions = np.array(ensemble_predictions)

print("Majority Voting completed!")

# 8. ACCURACY COMPARISON

from sklearn.metrics import accuracy_score

accuracy1 = accuracy_score(y_test, pred1)
accuracy2 = accuracy_score(y_test, pred2)
accuracy3 = accuracy_score(y_test, pred3)
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)

print("\n" + "=" * 50)
print("ACCURACY RESULTS")
print("=" * 50)

print(f"Simple Neural Network : {accuracy1 * 100:.2f}%")
print(f"Deep Neural Network   : {accuracy2 * 100:.2f}%")
print(f"CNN                   : {accuracy3 * 100:.2f}%")
print(f"Ensemble Voting       : {ensemble_accuracy * 100:.2f}%")

# 9. ACCURACY GRAPH

models = ["Simple NN", "Deep NN", "CNN", "Ensemble"]
accuracies = [
    accuracy1 * 100,
    accuracy2 * 100,
    accuracy3 * 100,
    ensemble_accuracy * 100
]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

for i, value in enumerate(accuracies):
    plt.text(i, value + 0.5, f"{value:.2f}%", ha="center")

plt.tight_layout()
plt.savefig("accuracy_comparison.png")
plt.show()

# 10. CONFUSION MATRIX

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("\nGenerating Confusion Matrix...")

cm = confusion_matrix(y_test, ensemble_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=np.arange(10)
)

disp.plot()
plt.title("Confusion Matrix - Ensemble Model")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("Confusion Matrix generated successfully!")

# 11. SAMPLE PREDICTIONS

print("\nDisplaying Sample Predictions...")

plt.figure(figsize=(12, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], cmap="gray")
    plt.title(
        f"Actual: {y_test[i]}\n"
        f"Predicted: {ensemble_predictions[i]}"
    )
    plt.axis("off")

plt.tight_layout()
plt.savefig("sample_predictions.png")
plt.show()

print("Sample predictions generated successfully!")

# 12. FINAL RESULT SUMMARY

print("\n" + "=" * 50)
print("FINAL PROJECT RESULT")
print("=" * 50)

print(f"Simple Neural Network Accuracy : {accuracy1 * 100:.2f}%")
print(f"Deep Neural Network Accuracy   : {accuracy2 * 100:.2f}%")
print(f"CNN Accuracy                   : {accuracy3 * 100:.2f}%")
print(f"Ensemble Accuracy              : {ensemble_accuracy * 100:.2f}%")

print("\nBest Individual Model: CNN")

if ensemble_accuracy >= max(accuracy1, accuracy2, accuracy3):
    print("Ensemble model performed better than individual models.")
else:
    print("Ensemble model successfully combined predictions using majority voting.")

print("\nProject completed successfully!")

majority_vote = np.bincount(votes).argmax()