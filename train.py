"""
LAVE: Model Training Script

This module provides functionality for training image classification models
using transfer learning with various pre-trained backbone architectures.

Author: Purushothaman Natarajan, Athira Nambiar
License: MIT
"""

import os
import argparse
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for loading images and labels.
    
    Attributes:
        image_files (list): List of image file paths.
        image_labels (list): List of corresponding labels.
        transform (callable, optional): Transform to apply to images.
    """
    
    def __init__(self, image_files: List[str], image_labels: List[int], transform=None):
        """
        Initialize the dataset.
        
        Args:
            image_files (list): List of image file paths.
            image_labels (list): List of corresponding labels.
            transform (callable, optional): Transform to apply to images.
        """
        self.image_files = image_files
        self.image_labels = image_labels
        self.transform = transform
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (image tensor, label)
        """
        image = Image.open(self.image_files[idx]).convert('RGB')
        label = self.image_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_data(data_dir: str, image_size: int, batch_size: int) -> Tuple[DataLoader, DataLoader, int]:
    """
    Load and prepare train and validation data loaders.
    
    Args:
        data_dir (str): Path to the data directory with train/val/test subfolders.
        image_size (int): Target size for image resizing.
        batch_size (int): Batch size for data loaders.
        
    Returns:
        tuple: (train_loader, val_loader, num_classes)
        
    Raises:
        ValueError: If no subdirectories are found in the train directory.
    """
    subfolders = ['train', 'val', 'test']
    image_files = {subfolder: [] for subfolder in subfolders}
    image_labels = {subfolder: [] for subfolder in subfolders}
    
    labels = [d.name for d in os.scandir(os.path.join(data_dir, 'train')) if d.is_dir()]
    if not labels:
        raise ValueError(
            "No subdirectories found in the train directory. "
            "Please ensure the directory contains labeled subfolders."
        )
    
    label_map = {label: idx for idx, label in enumerate(labels)}
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(data_dir, subfolder)
        for label in labels:
            label_dir = os.path.join(subfolder_path, label)
            if os.path.exists(label_dir):
                for image_file in os.listdir(label_dir):
                    image_files[subfolder].append(os.path.join(label_dir, image_file))
                    image_labels[subfolder].append(label)
        
        image_labels[subfolder] = [label_map[label] for label in image_labels[subfolder]]
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = CustomDataset(image_files['train'], image_labels['train'], transform=transform)
    val_dataset = CustomDataset(image_files['val'], image_labels['val'], transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    num_classes = len(labels)
    return train_loader, val_loader, num_classes


class CustomModel(nn.Module):
    """
    A custom model that uses pre-trained backbones with modified classifiers.
    
    Supports multiple architectures including ResNet, VGG, MobileNet, and more.
    The base model layers are frozen and only the classifier head is trained.
    
    Attributes:
        model_dict (dict): Dictionary mapping model names to (model_fn, weights) tuples.
        models (nn.ModuleList): List of base models with modified classifiers.
        base_model_names (list): Names of the base models used.
    """
    
    model_dict = {
        'alexnet': (models.alexnet, models.AlexNet_Weights.DEFAULT),
        'convnext_tiny': (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT),
        'densenet121': (models.densenet121, models.DenseNet121_Weights.DEFAULT),
        'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
        'efficientnet_v2_s': (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.DEFAULT),
        'googlenet': (models.googlenet, models.GoogLeNet_Weights.DEFAULT),
        'inception_v3': (models.inception_v3, models.Inception_V3_Weights.DEFAULT),
        'mnasnet1_0': (models.mnasnet1_0, models.MNASNet1_0_Weights.DEFAULT),
        'mobilenet_v2': (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
        'mobilenet_v3_small': (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
        'regnet_y_400mf': (models.regnet_y_400mf, models.RegNet_Y_400MF_Weights.DEFAULT),
        'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT),
        'resnext50_32x4d': (models.resnext50_32x4d, models.ResNeXt50_32X4D_Weights.DEFAULT),
        'shufflenet_v2_x1_0': (models.shufflenet_v2_x1_0, models.ShuffleNet_V2_X1_0_Weights.DEFAULT),
        'squeezenet1_0': (models.squeezenet1_0, models.SqueezeNet1_0_Weights.DEFAULT),
        'vgg16': (models.vgg16, models.VGG16_Weights.DEFAULT),
        'wide_resnet50_2': (models.wide_resnet50_2, models.Wide_ResNet50_2_Weights.DEFAULT),
    }
    
    def __init__(self, base_model_names: List[str], num_classes: int):
        """
        Initialize the custom model.
        
        Args:
            base_model_names (list): List of base model names to use.
            num_classes (int): Number of output classes.
            
        Raises:
            ValueError: If an unsupported model name is provided.
            NotImplementedError: If model modification is not implemented for the architecture.
        """
        super(CustomModel, self).__init__()
        
        self.models = nn.ModuleList()
        self.base_model_names = base_model_names
        
        for base_model_name in base_model_names:
            if base_model_name not in self.model_dict:
                raise ValueError(f"Unsupported model name: {base_model_name}")
            
            base_model_func, weight_func = self.model_dict[base_model_name]
            base_model = base_model_func(weights=weight_func)
            
            for param in base_model.parameters():
                param.requires_grad = False
            
            if hasattr(base_model, 'classifier'):
                if isinstance(base_model.classifier, nn.Sequential):
                    num_ftrs = base_model.classifier[0].in_features
                else:
                    num_ftrs = base_model.classifier.in_features
                
                base_model.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(num_ftrs, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(512, num_classes)
                )
            elif hasattr(base_model, 'fc'):
                num_ftrs = base_model.fc.in_features
                base_model.fc = nn.Sequential(
                    nn.Linear(num_ftrs, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(512, num_classes)
                )
            elif hasattr(base_model, 'head'):
                num_ftrs = base_model.head.in_features
                base_model.head = nn.Sequential(
                    nn.Linear(num_ftrs, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(512, num_classes)
                )
            else:
                raise NotImplementedError(f"Modification for {base_model_name} not implemented")
            
            self.models.append(base_model)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through all models.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            list: List of outputs from each model.
        """
        outputs = [model(x) for model in self.models]
        return outputs


def create_and_train_model(
    base_model_names: List[str],
    num_classes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    log_dir: str,
    model_dir: str,
    epochs: int,
    optimizer_name: str,
    learning_rate: float,
    step_gamma: float,
    alpha: float,
    batch_size: int,
    patience: int
) -> None:
    """
    Train a model using transfer learning.
    
    Args:
        base_model_names (list): List of base model names to train.
        num_classes (int): Number of output classes.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        log_dir (str): Directory for logging.
        model_dir (str): Directory to save trained models.
        epochs (int): Number of training epochs.
        optimizer_name (str): Optimizer name ('adam' or 'sgd').
        learning_rate (float): Learning rate.
        step_gamma (float): Learning rate scheduler gamma.
        alpha (float): Momentum for SGD.
        batch_size (int): Batch size.
        patience (int): Early stopping patience.
        
    Raises:
        ValueError: If an unsupported optimizer is specified.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for base_model_name in base_model_names:
        print(f"Training with base model: {base_model_name}")
        
        model = CustomModel([base_model_name], num_classes)
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        
        if optimizer_name.lower() == "adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=alpha)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        scheduler = ExponentialLR(optimizer, gamma=step_gamma)
        
        best_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs[0], labels)
                _, preds = torch.max(outputs[0], 1)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)
            print(f"Training Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
            
            model.eval()
            val_loss = 0.0
            val_corrects = 0
            
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc="Validation"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs[0], labels)
                    _, preds = torch.max(outputs[0], 1)
                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data)
            
            val_loss /= len(val_loader.dataset)
            val_acc = val_corrects.double() / len(val_loader.dataset)
            print(f"Validation Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                model_path = os.path.join(model_dir, f"{base_model_name}_best_model.pth")
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch + 1} epochs")
                    break
            
            scheduler.step()
        
        print(f"Finished Training {base_model_name} with best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Custom Models for Image Classification')
    parser.add_argument('--data_path', type=str, required=True, 
                        help='Directory containing the data')
    parser.add_argument('--shape', type=int, default=224, 
                        help='Image size (default: 224)')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--optimizer', type=str, default='adam', 
                        help='Optimizer (default: adam)')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--step_gamma', type=float, default=0.9, 
                        help='Learning rate scheduler gamma (default: 0.9)')
    parser.add_argument('--alpha', type=float, default=0.9, 
                        help='Momentum for SGD (default: 0.9)')
    parser.add_argument('--patience', type=int, default=5, 
                        help='Patience for early stopping (default: 5)')
    parser.add_argument('--log_dir', type=str, default='logs', 
                        help='Directory for TensorBoard logs (default: logs)')
    parser.add_argument('--model_dir', type=str, default='models', 
                        help='Directory to save models (default: models)')
    parser.add_argument('--base_model_names', type=str, nargs='+', required=True, 
                        help='Base model names')
    args = parser.parse_args()
    
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    train_loader, val_loader, num_classes = load_data(args.data_path, args.shape, args.batch_size)
    create_and_train_model(
        args.base_model_names, 
        num_classes, 
        train_loader, 
        val_loader,
        args.log_dir, 
        args.model_dir, 
        args.epochs, 
        args.optimizer, 
        args.learning_rate,
        args.step_gamma, 
        args.alpha, 
        args.batch_size, 
        args.patience
    )
