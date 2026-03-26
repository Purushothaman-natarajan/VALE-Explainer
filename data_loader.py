"""
LAVE: Data Loading and Preprocessing

This module provides functionality for loading, preprocessing, augmenting,
and splitting image datasets for training, validation, and testing.

Author: Purushothaman Natarajan, Athira Nambiar
License: MIT
"""

import os
import sys
import argparse
import uuid
from typing import Tuple, Optional

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from tqdm import tqdm


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Image Data Loader with Augmentation and Splits'
    )
    parser.add_argument('--path', type=str, required=True,
                        help='Path to the folder containing images')
    parser.add_argument('--dim', type=int, default=224,
                        help='Required image dimension')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--target_folder', type=str, required=True,
                        help='Folder to store the train, test, and val splits')
    parser.add_argument('--augment_data', action='store_true',
                        help='Apply data augmentation')
    return parser.parse_args()


def create_transforms(
    image_size: int, 
    augment_data: bool
) -> Tuple[transforms.Compose, Optional[transforms.Compose]]:
    """
    Create transforms for image preprocessing and augmentation.
    
    Args:
        image_size (int): Target image size.
        augment_data (bool): Whether to create augmentation transforms.
        
    Returns:
        tuple: (base_transform, augmentation_transform)
    """
    base_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    if augment_data:
        augment_transform = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)], 
                p=0.5
            ),
            transforms.ToTensor(),
        ])
        return base_transform, augment_transform
    
    return base_transform, None


class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for loading images with labels.
    
    Attributes:
        image_paths (list): List of image file paths.
        labels (list): List of corresponding label indices.
        label_to_name (dict): Mapping from label indices to names.
        transform (callable, optional): Transform to apply to images.
    """
    
    def __init__(
        self, 
        image_paths: list, 
        labels: list, 
        label_to_name: dict, 
        transform=None
    ):
        """
        Initialize the dataset.
        
        Args:
            image_paths (list): List of image file paths.
            labels (list): List of corresponding label indices.
            label_to_name (dict): Mapping from label indices to names.
            transform (callable, optional): Transform to apply to images.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.label_to_name = label_to_name
        self.transform = transform
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (image tensor, label name)
        """
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label_idx = self.labels[idx]
        label_name = self.label_to_name[label_idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label_name


def save_image(image: torch.Tensor, file_path: str) -> None:
    """
    Save an image tensor to a file.
    
    Args:
        image (torch.Tensor or PIL.Image): Image to save.
        file_path (str): Path to save the image.
    """
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    image.save(file_path)


def load_data(
    path: str, 
    image_size: int, 
    batch_size: int, 
    augment_data: bool, 
    num_workers: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load data from a directory and create data loaders.
    
    Args:
        path (str): Path to the data directory.
        image_size (int): Target image size.
        batch_size (int): Batch size for data loaders.
        augment_data (bool): Whether to apply data augmentation.
        num_workers (int): Number of workers for data loading.
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
        
    Raises:
        ValueError: If no images are found in the specified path.
    """
    all_images = []
    labels = []
    label_to_idx = {}
    idx_to_label = {}
    
    for subdir, _, files in os.walk(path):
        label = os.path.basename(subdir)
        if label not in label_to_idx:
            idx = len(label_to_idx)
            label_to_idx[label] = idx
            idx_to_label[idx] = label
        
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(subdir, fname))
                labels.append(label_to_idx[label])
    
    unique_labels = set(labels)
    print(f"Found {len(all_images)} images in {path}\n")
    print(f"Labels found ({len(unique_labels)}): {', '.join(idx_to_label[i] for i in unique_labels)}\n")
    
    if len(all_images) == 0:
        raise ValueError(f"No images found in the specified path: {path}")
    
    dataset = CustomDataset(
        all_images, 
        labels, 
        idx_to_label, 
        transform=transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    )
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    val_size = int(0.5 * len(test_dataset))
    test_size = len(test_dataset) - val_size
    val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])
    
    print(f"Data split into {len(train_dataset)} train, {len(val_dataset)} validation, and {len(test_dataset)} test images.\n")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def save_datasets_to_folders(
    dataloader: DataLoader, 
    folder_path: str, 
    transform: Optional[transforms.Compose] = None
) -> int:
    """
    Save datasets to organized folders by class.
    
    Args:
        dataloader (DataLoader): DataLoader containing the dataset.
        folder_path (str): Path to save the images.
        transform (callable, optional): Transform to apply before saving.
        
    Returns:
        int: Number of images saved.
    """
    os.makedirs(folder_path, exist_ok=True)
    
    count = 0
    
    for images, labels in tqdm(dataloader, desc=f"Saving to {folder_path}"):
        for i in range(images.shape[0]):
            image = images[i]
            label_name = labels[i]
            label_folder = os.path.join(folder_path, label_name)
            
            os.makedirs(label_folder, exist_ok=True)
            
            file_path = os.path.join(label_folder, f"{uuid.uuid4().hex}.jpg")
            save_image(image, file_path)
            count += 1
            
            if transform:
                pil_image = transforms.ToPILImage()(image)
                aug_image = transform(pil_image)
                file_path = os.path.join(label_folder, f"{uuid.uuid4().hex}_aug.jpg")
                save_image(aug_image, file_path)
                count += 1
    
    print(f"Saved {count} images to {folder_path}\n")
    return count


def main() -> None:
    """Main function for data loading and preprocessing."""
    args = parse_arguments()
    
    if not os.path.exists(args.target_folder):
        os.makedirs(args.target_folder)
    
    train_folder = os.path.join(args.target_folder, 'train')
    val_folder = os.path.join(args.target_folder, 'val')
    test_folder = os.path.join(args.target_folder, 'test')
    
    base_transform, augment_transform = create_transforms(args.dim, args.augment_data)
    
    train_loader, val_loader, test_loader = load_data(
        args.path,
        args.dim,
        args.batch_size,
        args.augment_data,
        args.num_workers
    )
    
    train_count = save_datasets_to_folders(train_loader, train_folder, augment_transform)
    val_count = save_datasets_to_folders(val_loader, val_folder)
    test_count = save_datasets_to_folders(test_loader, test_folder)
    
    print(f"Train dataset saved to: {train_folder}\n")
    print(f"Validation dataset saved to: {val_folder}\n")
    print(f"Test dataset saved to: {test_folder}\n")
    
    print('-' * 20)
    
    print(f"Number of images in training set: {train_count}\n")
    print(f"Number of images in validation set: {val_count}\n")
    print(f"Number of images in test set: {test_count}\n")


if __name__ == "__main__":
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)
    main()
