import torch
from preprocessing.preprocessing import find_faces, preprocessing_pipeline_v2
from prediction.prediction import predict_happiness_v2
import cv2
import matplotlib.pyplot as plt
image_path = "test_image_3.jpg"
test_image = cv2.imread(image_path)

images, location = find_faces(test_image)

# Loop through detected faces and check if they are happy
for i, img in enumerate(images):
    # Predict happiness for each face
    is_happy = predict_happiness_v2(img)
    
    # Get the coordinates of the detected face
    x, y, w, h = location[i]
    
    # If the model predicts happiness, draw a rectangle around the face
    if is_happy:
        cv2.rectangle(test_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the result
plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()