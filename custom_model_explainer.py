"""
LAVE: Language-Aware Visual Explanations - Custom Model Explainer

This module provides a wrapper for explaining custom-trained PyTorch models
using SHAP for feature importance, SAM for segmentation, and TinyLLaVA for
textual explanations.

Author: Purushothaman Natarajan, Athira Nambiar
License: MIT
"""

import os
import urllib.request
import json

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import shap

from segment_anything import SamPredictor, sam_model_registry
from tinyllava.model.builder import load_pretrained_model
from tinyllava.mm_utils import get_model_name_from_path
from tinyllava.eval.run_tiny_llava import eval_model


class CustomModel(nn.Module):
    """
    A custom model wrapper that loads a pre-trained backbone and modifies
    the classifier head for custom classification tasks.
    
    Attributes:
        model_dict (dict): Dictionary mapping model names to their constructors.
        models (nn.ModuleList): List of base models with modified classifiers.
        base_model_names (list): Names of the base models used.
    """
    
    model_dict = {
        'alexnet': (torchvision.models.alexnet, torchvision.models.AlexNet_Weights.DEFAULT),
        'convnext_tiny': (torchvision.models.convnext_tiny, torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT),
        'densenet121': (torchvision.models.densenet121, torchvision.models.DenseNet121_Weights.DEFAULT),
        'efficientnet_b0': (torchvision.models.efficientnet_b0, torchvision.models.EfficientNet_B0_Weights.DEFAULT),
        'googlenet': (torchvision.models.googlenet, torchvision.models.GoogLeNet_Weights.DEFAULT),
        'mobilenet_v2': (torchvision.models.mobilenet_v2, torchvision.models.MobileNet_V2_Weights.DEFAULT),
        'resnet18': (torchvision.models.resnet18, torchvision.models.ResNet18_Weights.DEFAULT),
        'resnet50': (torchvision.models.resnet50, torchvision.models.ResNet50_Weights.DEFAULT),
        'vgg16': (torchvision.models.vgg16, torchvision.models.VGG16_Weights.DEFAULT),
    }
    
    def __init__(self, base_model_names, num_classes):
        """
        Initialize the custom model.
        
        Args:
            base_model_names (list): List of base model names to use.
            num_classes (int): Number of output classes.
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
    
    def forward(self, x):
        """
        Forward pass through all models.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            list: List of outputs from each model.
        """
        outputs = [model(x) for model in self.models]
        return outputs


class PyTorchExplainableWrapper:
    """
    A wrapper class for generating multimodal explanations for custom-trained
    PyTorch image classification models.
    
    This class combines SHAP for computing feature importance, SAM for 
    generating visual segmentation masks, and TinyLLaVA for generating
    natural language descriptions of the model's predictions.
    
    Attributes:
        device (str): Device to run computations on ('cuda' or 'cpu').
        model (torch.nn.Module): The custom-trained PyTorch model.
        sam_predictor: The SAM predictor for image segmentation.
        tiny_llava_tokenizer: Tokenizer for TinyLLaVA model.
        tiny_llava_model: TinyLLaVA language model.
        image_processor: Image processor for TinyLLaVA.
        context_len: Maximum context length for TinyLLaVA.
    
    Example:
        >>> explainer = PyTorchExplainableWrapper(
        ...     custom_model_path='model.pth',
        ...     sam_checkpoint='sam_vit_h.pth',
        ...     tiny_llava_model_path='bczhou/TinyLLaVA-3.1B',
        ...     num_classes=10,
        ...     base_model_name='resnet50'
        ... )
        >>> explainer.run_pipeline('image.jpg', class_names)
    """
    
    def __init__(self, custom_model_path, sam_checkpoint, tiny_llava_model_path, num_classes, 
                 base_model_name='resnet50', device=None):
        """
        Initialize the explainer wrapper.
        
        Args:
            custom_model_path (str): Path to the custom-trained model weights.
            sam_checkpoint (str): Path to the SAM model checkpoint.
            tiny_llava_model_path (str): Path or identifier for TinyLLaVA model.
            num_classes (int): Number of output classes for the model.
            base_model_name (str): Name of the base model architecture.
            device (str, optional): Device to use ('cuda' or 'cpu'). 
                Defaults to CUDA if available.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_name = base_model_name
        
        self.model = self._load_custom_model(custom_model_path, num_classes)
        self.model.eval()
        
        self.sam_predictor = self._load_sam_model(sam_checkpoint)
        
        self.tiny_llava_tokenizer, self.tiny_llava_model, self.image_processor, self.context_len = self._load_tiny_llava(
            tiny_llava_model_path
        )
    
    def _load_custom_model(self, model_path, num_classes):
        """
        Load the custom-trained model from the given path.
        
        Args:
            model_path (str): Path to model weights.
            num_classes (int): Number of output classes.
            
        Returns:
            nn.Module: Loaded model.
        """
        model = CustomModel([self.base_model_name], num_classes=num_classes).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
    
    def _load_sam_model(self, sam_checkpoint):
        """
        Load the SAM (Segment Anything Model) for image segmentation.
        
        Args:
            sam_checkpoint (str): Path to SAM checkpoint file.
            
        Returns:
            SamPredictor: Configured SAM predictor.
        """
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        return SamPredictor(sam)
    
    def _load_tiny_llava(self, model_path):
        """
        Load the TinyLLaVA model for text generation.
        
        Args:
            model_path (str): Path or identifier for TinyLLaVA model.
            
        Returns:
            tuple: Tokenizer, model, image processor, and context length.
        """
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path)
        )
        return tokenizer, model, image_processor, context_len
    
    def preprocess_image(self, image_path):
        """
        Load and preprocess an image for model inference.
        
        Args:
            image_path (str): Path to the input image.
            
        Returns:
            tuple: Preprocessed tensor and original PIL image.
        """
        image = Image.open(image_path).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(image).unsqueeze(0).to(self.device)
        return input_tensor, image
    
    def predict_label(self, input_tensor, class_names):
        """
        Make predictions using the custom model.
        
        Args:
            input_tensor (torch.Tensor): Preprocessed image tensor.
            class_names (list): List of class names.
            
        Returns:
            tuple: Top 3 predicted labels and their scores.
        """
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        top3_prob, top3_catid = torch.topk(outputs[0], 3)
        top3_labels = [class_names[i] for i in top3_catid[0]]
        top3_scores = top3_prob[0].tolist()
        
        return top3_labels, top3_scores
    
    def explain_with_shap(self, input_tensor):
        """
        Generate SHAP explanations for the model prediction.
        
        Args:
            input_tensor (torch.Tensor): Preprocessed image tensor.
            
        Returns:
            numpy.ndarray: SHAP values for the input.
        """
        background = torch.randn((1, 3, 224, 224)).to(self.device)
        explainer = shap.DeepExplainer(self.model, background)
        shap_values = explainer.shap_values(input_tensor)
        
        return shap_values
    
    def segment_image(self, shap_values, input_tensor, image):
        """
        Generate a segmentation mask using SAM based on SHAP values.
        
        Args:
            shap_values (numpy.ndarray): SHAP values from the explainer.
            input_tensor (torch.Tensor): Preprocessed image tensor.
            image (PIL.Image): Original image.
            
        Returns:
            numpy.ndarray: Masked image with important regions highlighted.
        """
        input_image_np = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        self.sam_predictor.set_image(input_image_np)
        
        shap_values_abs = np.abs(shap_values[0])
        top_indices = np.unravel_index(
            np.argsort(shap_values_abs, axis=None)[::-1], 
            shap_values_abs.shape
        )
        top_coordinates = list(zip(top_indices[1], top_indices[2]))
        
        input_point = np.array([top_coordinates[0]])
        input_label = np.array([1])
        
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=input_point, 
            point_labels=input_label, 
            multimask_output=True
        )
        
        best_mask = masks[np.argmax(scores)]
        
        masked_image = np.array(image).copy()
        masked_image[best_mask == 0] = 0
        
        return masked_image
    
    def caption_image(self, masked_image_path, prompt="Explain the object in the image:"):
        """
        Generate a textual caption for the masked image using TinyLLaVA.
        
        Args:
            masked_image_path (str): Path to the masked image.
            prompt (str): Prompt to guide the caption generation.
            
        Returns:
            str: Generated caption for the image.
        """
        args = type('Args', (), {
            "model_path": "bczhou/TinyLLaVA-3.1B",
            "model_base": None,
            "model_name": get_model_name_from_path("bczhou/TinyLLaVA-3.1B"),
            "query": prompt,
            "conv_mode": "phi",
            "image_file": masked_image_path,
            "sep": ",",
            "temperature": 0,
            "top_p": 1,
            "num_beams": 1,
            "max_new_tokens": 1024
        })()
        return eval_model(args)
    
    def run_pipeline(self, image_path, class_names, output_path="masked_image.jpg"):
        """
        Run the complete explanation pipeline on an image.
        
        This method performs the following steps:
        1. Preprocess the input image
        2. Make predictions using the custom model
        3. Generate SHAP explanations
        4. Create segmentation masks using SAM
        5. Generate textual captions using TinyLLaVA
        
        Args:
            image_path (str): Path to the input image.
            class_names (list): List of class names.
            output_path (str): Path to save the masked image.
            
        Returns:
            tuple: Top predictions, SHAP values, masked image, and caption.
        """
        input_tensor, image = self.preprocess_image(image_path)
        
        top3_labels, top3_scores = self.predict_label(input_tensor, class_names)
        print(f"Top 3 Predictions: {[(label, score) for label, score in zip(top3_labels, top3_scores)]}")
        
        shap_values = self.explain_with_shap(input_tensor)
        shap.image_plot([shap_values], [input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()])
        
        masked_image = self.segment_image(shap_values, input_tensor, image)
        
        plt.imshow(masked_image)
        plt.title("Masked Image")
        plt.show()
        plt.imsave(output_path, masked_image)
        
        caption = self.caption_image(output_path)
        
        return top3_labels, top3_scores, shap_values, masked_image, caption


if __name__ == "__main__":
    import argparse
    import torchvision.models
    
    parser = argparse.ArgumentParser(description='LAVE Custom Model Explainer')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the custom-trained model')
    parser.add_argument('--img_path', type=str, required=True, 
                        help='Path to the input image')
    parser.add_argument('--sam_checkpoint', type=str, required=True,
                        help='Path to SAM checkpoint')
    parser.add_argument('--tiny_llava_model_path', type=str, default='bczhou/TinyLLaVA-3.1B',
                        help='Path to TinyLLaVA model')
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Directory containing training data for class names')
    parser.add_argument('--base_model_name', type=str, default='resnet50',
                        help='Base model architecture')
    parser.add_argument('--output_path', type=str, default='masked_image.jpg',
                        help='Output path for masked image')
    
    args = parser.parse_args()
    
    class_names = [d.name for d in os.scandir(args.train_dir) if d.is_dir()]
    num_classes = len(class_names)
    
    pipeline = PyTorchExplainableWrapper(
        custom_model_path=args.model_path,
        sam_checkpoint=args.sam_checkpoint,
        tiny_llava_model_path=args.tiny_llava_model_path,
        num_classes=num_classes,
        base_model_name=args.base_model_name
    )
    
    pipeline.run_pipeline(args.img_path, class_names, args.output_path)
