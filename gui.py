import gradio as gr
import tensorflow as tf
import numpy as np

# Load the digit recognition model
digit_model = tf.keras.models.load_model("/home/hamad/gui_digit_recognizer/model.h5")

def recognize_digit(image):
    if image is not None:
        image = image.reshape((1, 28, 28, 1)).astype("float32") / 255
        prediction = digit_model.predict(image)
        return np.argmax(prediction)
    else:
        return None

def perform_arithmetic_operation(image1, operator, image2):
    digit1 = recognize_digit(image1)
    digit2 = recognize_digit(image2)
    
    if digit1 is None or digit2 is None or operator not in ["+", "-", "*", "/"]:
        return "Invalid input"

    if operator == "+":
        result = digit1 + digit2
    elif operator == "-":
        result = digit1 - digit2
    elif operator == "*":
        result = digit1 * digit2
    elif operator == "/":
        if digit2 == 0:
            return "Division by zero error"
        result = digit1 / digit2
    else:
        return "Invalid operator"

    return f"{digit1} {operator} {digit2} = {result}"

iface = gr.Interface(
    fn=perform_arithmetic_operation,
    inputs=[
        gr.Image(image_mode="L", source="canvas", shape=(28, 28), invert_colors=True, label="Digit 1"),
        gr.Radio(choices=["+", "-", "*", "/"], label="Operator"),
        gr.Image(image_mode="L", source="canvas", shape=(28, 28), invert_colors=True, label="Digit 2")
    ],
    outputs="text",
    live=True,
)

iface.launch()