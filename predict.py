"""
LAVE: Model Prediction Script

This module provides functionality for making predictions on new images
using trained image classification models.

Author: Purushothaman Natarajan, Athira Nambiar
License: MIT
"""

import os
import argparse
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from train import CustomModel


def load_image(image_path: str, image_size: int) -> torch.Tensor:
    """
    Load and preprocess an image for prediction.
    
    Args:
        image_path (str): Path to the image file.
        image_size (int): Target size for resizing.
        
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image


def predict(
    model: nn.Module, 
    image: torch.Tensor, 
    class_names: list
) -> Tuple[int, float]:
    """
    Make a prediction on an image.
    
    Args:
        model (nn.Module): The trained model.
        image (torch.Tensor): Preprocessed image tensor.
        class_names (list): List of class names.
        
    Returns:
        tuple: (predicted_class_index, probability)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = [torch.nn.functional.softmax(output, dim=1) for output in outputs]
        
        avg_probabilities = torch.mean(torch.stack(probabilities), dim=0)
        top_prob, top_class = avg_probabilities.topk(1, dim=1)
        
        return top_class.item(), top_prob.item()


def main(
    model_path: str, 
    img_path: str, 
    train_dir: str, 
    base_model_names: list,
    image_size: int = 224
) -> None:
    """
    Main function for making predictions.
    
    Args:
        model_path (str): Path to the saved model.
        img_path (str): Path to the image to predict.
        train_dir (str): Directory containing training data for class names.
        base_model_names (list): List of base model names.
        image_size (int): Target image size.
    """
    class_names = [d.name for d in os.scandir(train_dir) if d.is_dir()]
    num_classes = len(class_names)
    
    model = CustomModel(base_model_names, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    image = load_image(img_path, image_size)
    
    pred, prob = predict(model, image, class_names)
    predicted_label = class_names[pred]
    
    print(f"Predicted Label: {predicted_label}, Probability: {prob:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a pre-trained model and make a prediction on a new image"
    )
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model')
    parser.add_argument('--img_path', type=str, required=True,
                        help='Path to the image to be predicted')
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Directory containing training dataset for inferring class names')
    parser.add_argument('--base_model_names', type=str, nargs='+', required=True,
                        help='List of base model names')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size (default: 224)')
    
    args = parser.parse_args()
    
    main(
        args.model_path, 
        args.img_path, 
        args.train_dir, 
        args.base_model_names,
        args.image_size
    )
