# digit-recognition-calculator

This project is a digit recognition calculator that uses a convolutional neural network (CNN) to recognize handwritten digits and perform arithmetic operations based on user input.

## Prerequisites

Before you begin, ensure you have the following installed on your machine:

- Python 3.6 or higher
- pip (Python package installer)

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/Hamadn/digit-recognition-calculator.git
    cd digit-recognition-calculator
    ```

2. **Create a virtual environment** (optional but recommended):
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Training the Model

If you need to train the model from scratch, follow these steps:

1. **Run the training script**:
    ```sh
    python main.py
    ```

    This script will:
    - Load the MNIST dataset.
    - Train a CNN model on the training data.
    - Evaluate the model on the test data.
    - Save the trained model to 'model.h5'.

2. **Check the output**:
    - The script will print the test accuracy and display a confusion matrix plot.
    - A classification report will also be printed to the command line.

## Running the GUI

To run the graphical user interface (GUI) for the digit recognition calculator, follow these steps:

1. **Ensure the trained model is available**:
    - The trained model should be saved as 'model.h5' in the directory.
    - If you have trained the model using 'main.py', the model will be saved automatically.

2. **Run the GUI script**:
    ```sh
    python gui.py
    ```

    This script will:
    - Load the trained digit recognition model.
    - Launch a Gradio interface where you can draw digits and select an operator to perform arithmetic operations.

3. **Interact with the GUI**:
    - Open the provided URL in your web browser.
    - Draw digits on the canvas for "Digit 1" and "Digit 2".
    - Select an operator (`+`, `-`, `*`, `/`) using the radio buttons.
    - The result of the arithmetic operation will be displayed.