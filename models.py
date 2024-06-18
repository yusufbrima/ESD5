import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchaudio

class CustomResNet18(nn.Module):
    """
    A custom ResNet-18 model that accepts a single input channel and outputs a specified number of classes.

    Attributes:
        base_model (nn.Module): The base ResNet-18 model with modifications.
    """
    
    def __init__(self, num_classes, weights=None, modelstr='resnet18'):
        """
        Initializes the custom ResNet-18 model with a single input channel and a custom number of output classes.

        Parameters:
            num_classes (int): The number of output classes for the final classification layer.
            weights (str, optional): The type of pre-trained weights to use (e.g., 'IMAGENET1K_V1').
        """
        super(CustomResNet18, self).__init__()
        
        # Load the ResNet-18 model, optionally with pre-trained weights
        if modelstr == 'resnet18':
            self.base_model = models.resnet18(weights=weights)
            # Modify the first convolutional layer to accept 1 channel instead of 3
            self.base_model.conv1 = nn.Conv2d(1, self.base_model.conv1.out_channels,
                                          kernel_size=self.base_model.conv1.kernel_size,
                                          stride=self.base_model.conv1.stride,
                                          padding=self.base_model.conv1.padding,
                                          bias=False)
        
            # Modify the final fully connected layer to output the specified number of classes
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        else:
            self.base_model = models.densenet121(weights=weights)
            # Modify the first convolutional layer to accept 1 channel instead of 3
            self.base_model.features.conv0 = nn.Conv2d(1, self.base_model.features.conv0.out_channels,
                                          kernel_size=self.base_model.features.conv0.kernel_size,
                                          stride=self.base_model.features.conv0.stride,
                                          padding=self.base_model.features.conv0.padding,
                                          bias=False)
            # Modify the final fully connected layer to output the specified number of classes
            self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, num_classes)
    

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the model.
        """
        return self.base_model(x)


class ESC50Model(nn.Module):
  def __init__(self, input_shape, num_cats=5):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 32, kernel_size = 3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.bn3 = nn.BatchNorm2d(64)
    self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.bn4 = nn.BatchNorm2d(64)
    self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.bn5 = nn.BatchNorm2d(128)
    self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.bn6 = nn.BatchNorm2d(128)
    self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.bn7 = nn.BatchNorm2d(256)
    self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.bn8 = nn.BatchNorm2d(256)
    self.dense1 = nn.Linear(256*(((input_shape[1]//2)//2)//2)*(((input_shape[2]//2)//2)//2),500)
    self.dropout = nn.Dropout(0.5)
    self.dense2 = nn.Linear(500, num_cats)
  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(self.bn1(x))
    x = self.conv2(x)
    x = F.relu(self.bn2(x))
    x = F.max_pool2d(x, kernel_size=2) 
    x = self.conv3(x)
    x = F.relu(self.bn3(x))
    x = self.conv4(x)
    x = F.relu(self.bn4(x))
    x = F.max_pool2d(x, kernel_size=2)
    x = self.conv5(x)
    x = F.relu(self.bn5(x))
    x = self.conv6(x)
    x = F.relu(self.bn6(x))
    x = F.max_pool2d(x, kernel_size=2)
    x = self.conv7(x)
    x = F.relu(self.bn7(x))
    x = self.conv8(x)
    x = F.relu(self.bn8(x))
    x = x.view(x.size(0),-1)
    x = F.relu(self.dense1(x))
    x = self.dropout(x)
    x = self.dense2(x)
    return x
# Example usage
if __name__ == "__main__":
    pass
