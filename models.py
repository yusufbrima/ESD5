import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchaudio

class CustomCNNModel(nn.Module):
    """
    A customized CNN model that accepts a single input channel and outputs a specified number of classes.

    Attributes:
        base_model (nn.Module): The base CNN model with modifications.
    """
    
    def __init__(self, num_classes, weights=None, modelstr='resnet18'):
        """
        Initializes the custom model with a single input channel and a custom number of output classes.

        Parameters:
            num_classes (int): The number of output classes for the final classification layer.
            weights (str, optional): The type of pre-trained weights to use (e.g., 'IMAGENET1K_V1').
        """
        super(CustomCNNModel, self).__init__()
        
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



# Example usage
if __name__ == "__main__":
    pass
