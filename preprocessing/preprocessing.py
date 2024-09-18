# Cropping out the face using opencv model (Face detcetion)
import cv2
import torch
# Load the pre-trained Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
import numpy as np

def image_crop(image):
    # Convert image to numpy array if it's not already
    image = np.array(image)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image.astype(np.uint8)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if at least one face is detected
    if len(faces) > 0:
        # Crop the face from the image (taking the first detected face)
        x, y, w, h = faces[0]
        cropped_face = image[y:y+h, x:x+w]
        return cropped_face

    # Return None if no face is detected
    return image

# Function for cropping faces
def find_faces(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_list = []
    co_ordinates = []
    for (x, y, w, h) in faces:
        face_list.append(image[y:y+h, x:x+w])
        co_ordinates.append([x, y, w, h])
    return face_list, co_ordinates

# CLAHE function for histogram equalization and normalization
def clahe_func_normalization(image, clip_limit=2.0, grid_size=4):
    # If the input is a torch Tensor, convert it to a NumPy array
    if isinstance(image, torch.Tensor):
        # Convert torch Tensor (CHW) to numpy (HWC)
        image = image.permute(1, 2, 0).detach().numpy()
    
    # Convert the image to LAB color space using OpenCV
    lab_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB)

    # Split the LAB image into L, A, B channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    cl_l_channel = clahe.apply(l_channel)

    # Merge the CLAHE enhanced L channel back with A and B channels
    merged_lab_image = cv2.merge((cl_l_channel, a_channel, b_channel))

    # Convert the LAB image back to RGB color space
    clahe_image = cv2.cvtColor(merged_lab_image, cv2.COLOR_LAB2RGB)

    # Normalize the image to 0-1 range
    final_normalized_image = clahe_image / 255.0

    return final_normalized_image

# Function to resize the image
def resize_image(image, size=64):
    # Resize the image to the specified size
    resized_image = cv2.resize(image, (size, size))
    return resized_image

# Function to convert an image to a PyTorch tensor
def to_tensor(image):
    tensor_image = torch.tensor(image).float()
    return tensor_image



# Preprocessing pipeline
def preprocessing_pipeline(image):
  image = image_crop(image)
  image = clahe_func_normalization(image)
  image = resize_image(image)
  image = to_tensor(image)
  return image



# Preprocessing pipeline for images
def preprocessing_pipeline_v2(image):
    # Perform CLAHE and normalization
    image = clahe_func_normalization(image)
    
    # Resize the image and convert to tensor
    image = resize_image(image)
    image = to_tensor(image)
    
    return image