import cv2
from src.params import resize_h, resize_w, n_classes,data_dir
from src.model import CardClassifierCNN
import torch
import torchvision.transforms as transforms
from src.eval import *
from src.set_up import PlayingCardDataset
import numpy as np


dataset = PlayingCardDataset(data_dir)
class_names = dataset.classes 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CardClassifierCNN(num_classes=n_classes)
model.load_state_dict(torch.load('./params/model.pt'))
# model.to(device)
model.eval()
# Create a VideoCapture object
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((resize_h, resize_w)),
    transforms.ToTensor(),
])
cap = cv2.VideoCapture(0)  # 0 for default camera, or specify camera index

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set camera properties (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    # cv2.imshow('Camera Feed', frame)
    
    image_tensor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image_tensor = transform(image_tensor).unsqueeze(0)

    probabilities = predict(model, image_tensor, "cpu")
    prob_index = np.argmax(probabilities)

# Get the index of the maximum element

    visualize_predictions(frame, probabilities, class_names)
    print(class_names[prob_index])

    

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



