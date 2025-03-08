import torch.nn as nn
import torch.optim as optim 

#first 2 CNN layers for extracting gabor like (high level) features, next two for extracting low level features
class CardClassifierCNN(nn.Module):
    def __init__(self, num_classes=53):
        super(CardClassifierCNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU() # Activation functions
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(64, 100, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(100, 120, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(120, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(100)
        self.bn3 = nn.BatchNorm2d(120)
        self.bn4 = nn.BatchNorm2d(128)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))


        self.dropout = nn.Dropout(0.25)
        self.dropout1 = nn.Dropout(0.5)
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # Adjust input size based on image size
        self.relu5 = nn.ReLU() # Activation functions
        self.fc2 = nn.Linear(512, num_classes)
    def forward(self,x):
        
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
       
        
        x = self.dropout1(x)
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
       
        x = self.dropout(x)
        x = self.pool3(self.relu3(self.bn3(self.conv3(x)))) 
       
        x =self.relu4(self.bn4(self.conv4(x)))
       
        # print(x.size())
        # x = self.global_pool(x)
        x = x.view(x.size(0),-1)  # Adjust based on input image size
        # print(x.size())
        x = self.relu5(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # no activation function in the end
        return x
    
# Loss function
