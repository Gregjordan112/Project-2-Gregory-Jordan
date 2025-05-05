import os
import warnings
import datetime  # For timestamp
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)
import time
import serial

port = "/dev/cu.usbmodem1101"  # Replace with your serial port
baud_rate = 9600

from tensorflow.keras.layers import DepthwiseConv2D

class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)  # Ignore 'groups' argument
        super().__init__(*args, **kwargs)

# Load the model
model = load_model(
    "/Users/gregjordan1/Downloads/Model2/keras_model.h5",
    compile=False,
    custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D}
)

# Load the labels
class_names = open("/Users/gregjordan1/Downloads/Model2/labels.txt", "r").readlines()

# Function to log detections
def log_detection(class_label, action):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("detection_log.txt", "a") as log_file:
        log_file.write(f"{timestamp} - Detected: {class_label}, Action: {action}\n")
    
    print(f"Logged: {timestamp} - {class_label} - {action}")  # Print for debug purposes

# CAMERA can be 0 or 1 based on default camera of your computer
try:
    camera = cv2.VideoCapture(1)

    # Set camera resolution for better quality (1280x720 or 1920x1080)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Warm up the camera
    for _ in range(10):
        ret, _ = camera.read()

    # Confirm camera resolution
    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera resolution: {width}x{height}")

    arduino = serial.Serial(port, baud_rate)
    time.sleep(2)

    while True:
        # Grab the webcamera's image.
        ret, image = camera.read()

        if not ret or image is None:
            print("Failed to grab frame from webcam.")
            continue  # Skip this iteration and try again

        # Resize the raw image into (224-height,224-width) pixels
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # Show the image in a window
        cv2.imshow("Webcam Image", image)

        # Make the image a numpy array and reshape it to the model's input shape
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Predict using the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        class_label = class_name[2:].strip().lower()  # Clean up the class label

        print(f"Class Label Detected: '{class_label}'")  # Debugging line

        # Decide what message to send
        message_to_send = " "  # Default to empty

        if class_label == 'face':
            message_to_send = "ON\n"
            print("Detected Face - Turning ON Green LED")
            log_detection(class_label, "Turning ON Green LED")  # Log the detection
        elif class_label == 'noface': 
            message_to_send = "OFF\n"
            print("Detected NotFace - Turning ON Red LED")
            log_detection(class_label, "Turning ON Red LED")  # Log the detection
        else:
            print("No matching class label found.")  # Debugging line
            log_detection(class_label, "No matching class")  # Log the detection

        # Only send if message_to_send is not empty
        if not message_to_send.strip():
            print("No message to send.")
        else:
            arduino.write(message_to_send.encode('utf-8'))
            arduino.flush()
            print(f"Sent: {message_to_send.strip()}")
        time.sleep(1)  # delay between messages

        # Listen to the keyboard for presses.
        keyboard_input = cv2.waitKey(1)
        if keyboard_input == 27:  # Exit if 'ESC' is pressed
            break

except serial.SerialException as e:
    print(f"Serial Error: {e}")

finally:
    # Close serial connection once, at the end
    arduino.close()
    print("Serial port closed")

    # Release the camera and close OpenCV windows
    camera.release()
    cv2.destroyAllWindows()