"""
LAVE: Model Testing Script

This module provides functionality for testing trained image classification models
and generating evaluation metrics including classification reports and confusion matrices.

Author: Purushothaman Natarajan, Athira Nambiar
License: MIT
"""

import os
import argparse
from typing import Tuple, Union, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from train import CustomModel, CustomDataset


def load_test_data(test_path: str, image_size: int) -> Union[Tuple[DataLoader, int], Tuple[torch.Tensor, None]]:
    """
    Load test data from a directory or single image.
    
    Args:
        test_path (str): Path to test directory or single image file.
        image_size (int): Target size for image resizing.
        
    Returns:
        tuple: (test_data, num_classes) where test_data is either a DataLoader or Tensor.
        
    Raises:
        ValueError: If no subdirectories are found in the test directory.
    """
    if os.path.isfile(test_path):
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = Image.open(test_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        return image, None
    else:
        image_files = []
        image_labels = []
        labels = [d.name for d in os.scandir(test_path) if d.is_dir()]
        
        if not labels:
            raise ValueError(
                "No subdirectories found in the test directory. "
                "Please ensure the directory contains labeled subfolders."
            )
        
        label_map = {label: idx for idx, label in enumerate(labels)}
        
        for label in labels:
            label_dir = os.path.join(test_path, label)
            for image_file in os.listdir(label_dir):
                image_files.append(os.path.join(label_dir, image_file))
                image_labels.append(label_map[label])
        
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        test_dataset = CustomDataset(image_files, image_labels, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
        
        return test_loader, len(labels)


def evaluate_model(
    model: nn.Module,
    test_data: Union[torch.Tensor, DataLoader],
    criterion: nn.Module,
    class_names: List[str],
    log_dir: str,
    model_name: str
) -> Union[int, None]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model (nn.Module): The trained model to evaluate.
        test_data: Test data as Tensor or DataLoader.
        criterion (nn.Module): Loss function.
        class_names (list): List of class names.
        log_dir (str): Directory to save evaluation results.
        model_name (str): Name of the model for logging.
        
    Returns:
        int or None: Predicted class index for single image, None for batch evaluation.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    if isinstance(test_data, torch.Tensor):
        test_data = test_data.to(device)
        with torch.no_grad():
            outputs = model(test_data)
            _, preds = torch.max(outputs[0], 1)
        return preds.item()
    else:
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_data, desc="Testing"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs[0], 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        report = classification_report(all_labels, all_preds, target_names=class_names)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        print(f"Model: {model_name}")
        print(report)
        
        os.makedirs(log_dir, exist_ok=True)
        
        report_file = os.path.join(log_dir, f"classification_report_{model_name}.txt")
        conf_matrix_file = os.path.join(log_dir, f"confusion_matrix_{model_name}.txt")
        conf_matrix_plot_file = os.path.join(log_dir, f"confusion_matrix_{model_name}.png")
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        np.savetxt(
            conf_matrix_file, 
            conf_matrix, 
            fmt='%d', 
            delimiter=',', 
            header=','.join(class_names)
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.savefig(conf_matrix_plot_file)
        plt.close()
        
        print(f"Confusion matrix and classification report saved to {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Custom Model')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the test image or directory containing test data')
    parser.add_argument('--shape', type=int, default=224,
                        help='Image size (default: 224)')
    parser.add_argument('--base_model_name', type=str, required=True,
                        help='Base model name')
    parser.add_argument('--model_path', type=str,
                        help='Path to the saved model')
    parser.add_argument('--models_folder_path', type=str,
                        help='Path to the folder containing multiple saved models')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Directory to save logs and reports')
    
    args = parser.parse_args()
    
    if args.model_path and args.models_folder_path:
        raise ValueError("Provide either model_path or models_folder_path, not both.")
    if not args.model_path and not args.models_folder_path:
        raise ValueError("Provide either model_path or models_folder_path.")
    
    test_data, num_classes = load_test_data(args.data_path, args.shape)
    
    if os.path.isdir(args.data_path):
        class_names = [d.name for d in os.scandir(args.data_path) if d.is_dir()]
    else:
        class_names = []
    
    if args.model_path:
        model = CustomModel([args.base_model_name], num_classes)
        model.load_state_dict(torch.load(args.model_path))
        criterion = nn.CrossEntropyLoss()
        
        if isinstance(test_data, torch.Tensor):
            pred = evaluate_model(model, test_data, criterion, class_names, args.log_dir, args.base_model_name)
            print(f"Predicted Label: {pred}")
        else:
            evaluate_model(model, test_data, criterion, class_names, args.log_dir, args.base_model_name)
    
    elif args.models_folder_path:
        model_files = [f for f in os.listdir(args.models_folder_path) if f.endswith('.pth')]
        
        for model_file in model_files:
            model_path = os.path.join(args.models_folder_path, model_file)
            model = CustomModel([args.base_model_name], num_classes)
            model.load_state_dict(torch.load(model_path))
            criterion = nn.CrossEntropyLoss()
            
            if isinstance(test_data, torch.Tensor):
                pred = evaluate_model(model, test_data, criterion, class_names, args.log_dir, model_file)
                print(f"Model: {model_file}, Predicted Label: {pred}")
            else:
                evaluate_model(model, test_data, criterion, class_names, args.log_dir, model_file)
