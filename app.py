import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# --------------------------
# Load trained models
# --------------------------
perceptron_model = tf.keras.models.load_model("perceptron_model.h5")
ann_model = tf.keras.models.load_model("ann_model.h5")
cnn_model = tf.keras.models.load_model("cnn_model.h5")

# --------------------------
# Page config
st.set_page_config(page_title="Digit Recognition App", layout="centered")
st.title("Digit Recognition App")
st.write("Upload a 28x28 grayscale image of a digit and see the predictions of three models.")

# --------------------------
# Sidebar description
st.sidebar.header("About This App")
st.sidebar.write("""
This app uses three deep learning models to recognize handwritten digits:

- **Perceptron**: Simple neural network with single layer.
- **ANN**: Multi-layer neural network with higher accuracy.
- **CNN**: Convolutional Neural Network with best performance for image recognition.

**Accuracy of Models (approx):**
- Perceptron: ~92%
- ANN: ~97%
- CNN: ~99%

Upload a 28x28 grayscale image to see predictions.
""")

# --------------------------
def preprocess_image(img):
    """Convert to 28x28 grayscale, invert, resize, and center the digit"""
    img = img.convert('L')
    img = ImageOps.invert(img)
    img_array = np.array(img)
    coords = np.column_stack(np.where(img_array > 0))

    if coords.size == 0:
        return np.zeros((28,28))

    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0)
    img = img.crop((y0, x0, y1, x1))

    w, h = img.size
    if w > h:
        new_w = 20
        new_h = int(h * 20 / w)
    else:
        new_h = 20
        new_w = int(w * 20 / h)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    new_img = Image.new('L', (28,28), 0)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    new_img.paste(img, (left, top))

    return np.array(new_img)/255.0

# --------------------------
# Upload image
uploaded_file = st.file_uploader("Upload a 28x28 grayscale image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L')
    img_array = preprocess_image(img)
    image = Image.fromarray((img_array*255).astype('uint8'))

    st.subheader("Processed Image Preview")
    st.image(image, width=150)

    # --------------------------
    # Prepare input for models
    img_perceptron = img_array.reshape(1,28,28)
    img_ann = img_array.reshape(1,28,28)
    img_cnn = img_array.reshape(1,28,28,1)

    # --------------------------
    # Make predictions
    pred_p = perceptron_model.predict(img_perceptron)
    pred_a = ann_model.predict(img_ann)
    pred_c = cnn_model.predict(img_cnn)

    class_p = np.argmax(pred_p)
    class_a = np.argmax(pred_a)
    class_c = np.argmax(pred_c)

    # --------------------------
    # Show predictions
    st.subheader("Predictions")
    st.write(f"Perceptron Prediction: {class_p}")
    st.write(f"ANN Prediction: {class_a}")
    st.write(f"CNN Prediction: {class_c}")

    # --------------------------
    st.subheader("Final Prediction")
    st.success(f"Final Predicted Digit: {class_c}")