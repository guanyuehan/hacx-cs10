import torch
import torchvision.models as models

model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)


torch.save(model)

# Set the model to evaluation mode if you are only doing inference
model.eval()

# You can then use this 'model' object for inference or fine-tuning
# For example, to print the model architecture:
print(model)