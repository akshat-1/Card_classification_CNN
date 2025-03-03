import numpy as np
from glob import glob
import torchvision.transforms as transforms
from eval import *
from model import CardClassifierCNN
from params import *
from set_up import PlayingCardDataset



test_images = glob('./dataset/test/*/*')
test_examples = np.random.choice(test_images, 10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CardClassifierCNN(num_classes=n_classes)
model.load_state_dict(torch.load('./params/model.pt'))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((resize_h, resize_w)),
    transforms.ToTensor(),
])

dataset = PlayingCardDataset(data_dir, transform)
for example in test_examples:
    original_image, image_tensor = preprocess_image(example, transform)
    probabilities = predict(model, image_tensor, device)

    # Assuming dataset.classes gives the class names
    class_names = dataset.classes 
    visualize_predictions(original_image, probabilities, class_names)