import torch
import torchvision.models as models
import os


def load_model(model_path="model.pth"):
    # Define the model architecture (must match your trained model)
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(in_features=2048, out_features=102)  # Ensure output matches 102 classes

    # Load the fine-tuned model weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    
    # Set model to evaluation mode
    model.eval()
    print("✅ Model loaded successfully and ready for inference!")
    
    return model

# If the file is run directly, load the model
if __name__ == "__main__":
    model = load_model()

