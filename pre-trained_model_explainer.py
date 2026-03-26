"""
LAVE: Language-Aware Visual Explanations - Pre-trained Model Explainer

This module provides a wrapper for explaining pre-trained PyTorch models
using SHAP for feature importance, SAM for segmentation, and TinyLLaVA for
textual explanations. Optimized for edge device deployment.

Author: Purushothaman Natarajan, Athira Nambiar
License: MIT
"""

import urllib.request

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import shap

from segment_anything import SamPredictor, sam_model_registry
from tinyllava.model.builder import load_pretrained_model
from tinyllava.mm_utils import get_model_name_from_path
from tinyllava.eval.run_tiny_llava import eval_model


class PyTorchExplainableWrapper:
    """
    A wrapper class for generating multimodal explanations for pre-trained
    PyTorch image classification models.
    
    This class provides lazy loading for memory efficiency and supports
    edge device optimizations including FP16 computation and model quantization.
    
    Attributes:
        device (str): Device to run computations on ('cuda' or 'cpu').
        model (torch.nn.Module): The pre-trained PyTorch model.
        sam_predictor: The SAM predictor for image segmentation (lazy loaded).
        tiny_llava_tokenizer: Tokenizer for TinyLLaVA model (lazy loaded).
        tiny_llava_model: TinyLLaVA language model (lazy loaded).
        fp16_enabled (bool): Whether FP16 computation is enabled.
    
    Example:
        >>> explainer = PyTorchExplainableWrapper(
        ...     model_name_or_path="densenet121",
        ...     sam_checkpoint="sam_vit_h.pth",
        ...     tiny_llava_model_path="bczhou/TinyLLaVA-3.1B"
        ... )
        >>> explainer.run_pipeline("image.jpg")
    """
    
    SUPPORTED_MODELS = {
        'densenet121': ('torchvision.models.densenet121', 'DenseNet121_Weights.DEFAULT'),
        'resnet50': ('torchvision.models.resnet50', 'ResNet50_Weights.DEFAULT'),
        'resnet18': ('torchvision.models.resnet18', 'ResNet18_Weights.DEFAULT'),
        'vgg16': ('torchvision.models.vgg16', 'VGG16_Weights.DEFAULT'),
        'mobilenet_v2': ('torchvision.models.mobilenet_v2', 'MobileNet_V2_Weights.DEFAULT'),
    }
    
    def __init__(self, model_name_or_path, sam_checkpoint, tiny_llava_model_path, device=None):
        """
        Initialize the explainer wrapper with lazy loading.
        
        Args:
            model_name_or_path (str): Name of pre-trained model or path to custom model.
            sam_checkpoint (str): Path to the SAM model checkpoint.
            tiny_llava_model_path (str): Path or identifier for TinyLLaVA model.
            device (str, optional): Device to use ('cuda' or 'cpu'). 
                Defaults to CUDA if available.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16_enabled = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7
        
        self.model = self._load_model(model_name_or_path)
        self.model.eval()
        
        self.sam_checkpoint = sam_checkpoint
        self.sam_predictor = None
        
        self.tiny_llava_model_path = tiny_llava_model_path
        self.tiny_llava_tokenizer = None
        self.tiny_llava_model = None
        self.image_processor = None
        self.context_len = None
    
    def _load_model(self, model_name_or_path):
        """
        Load the pre-trained model.
        
        Args:
            model_name_or_path (str): Model name or path to model file.
            
        Returns:
            nn.Module: Loaded model.
        """
        if model_name_or_path in self.SUPPORTED_MODELS:
            model_path, weights_name = self.SUPPORTED_MODELS[model_name_or_path]
            module_path, class_name = model_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            model_class = getattr(module, class_name)
            weights_class = getattr(module, weights_name)
            model = model_class(weights=weights_class).to(self.device)
            
            if self.device == "cpu":
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
        else:
            model = torch.load(model_name_or_path, map_location=self.device)
        
        return model
    
    def _load_sam_model(self):
        """
        Lazily load the SAM (Segment Anything Model) for image segmentation.
        
        Returns:
            SamPredictor: Configured SAM predictor.
        """
        if self.sam_predictor is None:
            model_type = "vit_h"
            sam = sam_model_registry[model_type](checkpoint=self.sam_checkpoint)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
    
    def _load_tiny_llava(self):
        """
        Lazily load the TinyLLaVA model for text generation.
        
        Returns:
            tuple: Tokenizer, model, image processor, and context length.
        """
        if self.tiny_llava_tokenizer is None:
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=self.tiny_llava_model_path,
                model_base=None,
                model_name=get_model_name_from_path(self.tiny_llava_model_path)
            )
            self.tiny_llava_tokenizer = tokenizer
            self.tiny_llava_model = model
            self.image_processor = image_processor
            self.context_len = context_len
    
    def preprocess_image(self, image_path, image_size=128):
        """
        Load and preprocess an image for model inference.
        
        Args:
            image_path (str): Path to the input image.
            image_size (int): Target image size (default: 128 for edge optimization).
            
        Returns:
            tuple: Preprocessed tensor and original PIL image.
        """
        image = Image.open(image_path).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(image).unsqueeze(0).to(self.device)
        
        if self.fp16_enabled:
            input_tensor = input_tensor.half()
            
        return input_tensor, image
    
    def predict_label(self, input_tensor):
        """
        Make predictions using the pre-trained model.
        
        Args:
            input_tensor (torch.Tensor): Preprocessed image tensor.
            
        Returns:
            tuple: Top 3 predicted labels and their scores.
        """
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        with urllib.request.urlopen(LABELS_URL) as url:
            class_labels = json.loads(url.read().decode())
        
        top3_prob, top3_catid = torch.topk(outputs, 3)
        top3_labels = [class_labels[i] for i in top3_catid[0]]
        top3_scores = top3_prob[0].tolist()
        
        return top3_labels, top3_scores
    
    def explain_with_shap(self, input_tensor, image_size=128):
        """
        Generate SHAP explanations for the model prediction.
        
        Args:
            input_tensor (torch.Tensor): Preprocessed image tensor.
            image_size (int): Size of the input image.
            
        Returns:
            numpy.ndarray: SHAP values for the input.
        """
        background = torch.randn((1, 3, image_size, image_size)).to(self.device)
        if self.fp16_enabled:
            background = background.half()
        
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
        self._load_sam_model()
        
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
        self._load_tiny_llava()
        
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
    
    def run_pipeline(self, image_path, output_path="masked_image.jpg"):
        """
        Run the complete explanation pipeline on an image.
        
        This method performs the following steps:
        1. Preprocess the input image
        2. Make predictions using the pre-trained model
        3. Generate SHAP explanations
        4. Create segmentation masks using SAM
        5. Generate textual captions using TinyLLaVA
        
        Args:
            image_path (str): Path to the input image.
            output_path (str): Path to save the masked image.
            
        Returns:
            tuple: Top predictions, SHAP values, masked image, and caption.
        """
        input_tensor, image = self.preprocess_image(image_path)
        top3_labels, top3_scores = self.predict_label(input_tensor)
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
    import json
    
    parser = argparse.ArgumentParser(description='LAVE Pre-trained Model Explainer')
    parser.add_argument('--model_name', type=str, default='densenet121',
                        help='Name of pre-trained model')
    parser.add_argument('--img_path', type=str, required=True,
                        help='Path to the input image')
    parser.add_argument('--sam_checkpoint', type=str, required=True,
                        help='Path to SAM checkpoint')
    parser.add_argument('--tiny_llava_model_path', type=str, default='bczhou/TinyLLaVA-3.1B',
                        help='Path to TinyLLaVA model')
    parser.add_argument('--output_path', type=str, default='masked_image.jpg',
                        help='Output path for masked image')
    
    args = parser.parse_args()
    
    pipeline = PyTorchExplainableWrapper(
        model_name_or_path=args.model_name,
        sam_checkpoint=args.sam_checkpoint,
        tiny_llava_model_path=args.tiny_llava_model_path
    )
    
    pipeline.run_pipeline(args.img_path, args.output_path)
