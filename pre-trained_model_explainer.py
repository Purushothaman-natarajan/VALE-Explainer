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
import urllib, json

class PyTorchExplainableWrapper:
    def __init__(self, model_name_or_path, sam_checkpoint, tiny_llava_model_path, device=None):
        # Load device configuration (optimized for edge)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Use FP16 if available on edge devices
        self.fp16_enabled = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7

        # Load the pre-trained DenseNet model or the specified model
        self.model = self._load_model(model_name_or_path)
        self.model.eval()  # Set model to evaluation mode

        # Load SAM (Segment Anything Model) only when needed
        self.sam_predictor = None
        self.sam_checkpoint = sam_checkpoint

        # Load TinyLLaVA model for captioning, lazy initialization
        self.tiny_llava_tokenizer = None
        self.tiny_llava_model = None
        self.image_processor = None
        self.context_len = None
        self.tiny_llava_model_path = tiny_llava_model_path

    def _load_model(self, model_name_or_path):
        # Load the model (pre-trained DenseNet as default)
        if model_name_or_path == "densenet121":
            from torchvision.models import densenet121
            model = densenet121(pretrained=True).to(self.device)

            # Apply quantization for edge device
            if self.device == "cpu":
                model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

        else:
            # Load a custom model for the edge device
            model = torch.load(model_name_or_path, map_location=self.device)
        return model

    def _load_sam_model(self):
        if self.sam_predictor is None:
            model_type = "vit_h"  # Assuming SAM's vit_h model
            sam = sam_model_registry[model_type](checkpoint=self.sam_checkpoint)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)

    def _load_tiny_llava(self):
        if self.tiny_llava_tokenizer is None or self.tiny_llava_model is None:
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=self.tiny_llava_model_path,
                model_base=None,
                model_name=get_model_name_from_path(self.tiny_llava_model_path)
            )
            self.tiny_llava_tokenizer = tokenizer
            self.tiny_llava_model = model
            self.image_processor = image_processor
            self.context_len = context_len

    def preprocess_image(self, image_path):
        # Adjust image size for edge devices (reduce resolution if necessary)
        image = Image.open(image_path).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize((128, 128)),  # Reduced resolution for edge computing
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(image).unsqueeze(0).to(self.device)
        
        # Convert to half precision if enabled
        if self.fp16_enabled:
            input_tensor = input_tensor.half()
            
        return input_tensor, image

    def predict_label(self, input_tensor):
        with torch.no_grad():
            outputs = self.model(input_tensor)

        LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        with urllib.request.urlopen(LABELS_URL) as url:
            class_labels = json.loads(url.read().decode())

        # Get top predictions
        top3_prob, top3_catid = torch.topk(outputs, 3)
        top3_labels = [class_labels[i] for i in top3_catid[0]]
        top3_scores = top3_prob[0].tolist()

        return top3_labels, top3_scores

    def explain_with_shap(self, input_tensor):
        background = torch.randn((1, 3, 128, 128)).to(self.device)
        if self.fp16_enabled:
            background = background.half()

        explainer = shap.DeepExplainer(self.model, background)
        shap_values = explainer.shap_values(input_tensor)

        return shap_values

    def segment_image(self, shap_values, input_tensor, image):
        self._load_sam_model()  # Lazy load SAM if not done already

        input_image_np = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        self.sam_predictor.set_image(input_image_np)

        # Top SHAP coordinates for SAM
        shap_values_abs = np.abs(shap_values[0])
        top_indices = np.unravel_index(np.argsort(shap_values_abs, axis=None)[::-1], shap_values_abs.shape)
        top_coordinates = list(zip(top_indices[1], top_indices[2]))

        # Use first SHAP point as SAM prompt
        input_point = np.array([top_coordinates[0]])
        input_label = np.array([1])

        # Predict mask using SAM
        masks, scores, logits = self.sam_predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)
        best_mask = masks[np.argmax(scores)]

        masked_image = np.array(image).copy()
        masked_image[best_mask == 0] = 0

        return masked_image

    def caption_image(self, masked_image_path, prompt="Explain the object in the image:"):
        self._load_tiny_llava()  # Lazy load TinyLLaVA model

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
        eval_model(args)

    def run_pipeline(self, image_path):
        input_tensor, image = self.preprocess_image(image_path)
        top3_labels, top3_scores = self.predict_label(input_tensor)
        print(f"Top 3 Predictions: {[(label, score) for label, score in zip(top3_labels, top3_scores)]}")

        shap_values = self.explain_with_shap(input_tensor)
        shap.image_plot([shap_values], [input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()])

        masked_image = self.segment_image(shap_values, input_tensor, image)
        masked_image_path = "masked_image.jpg"
        plt.imshow(masked_image)
        plt.title("Masked Image")
        plt.show()
        plt.imsave(masked_image_path, masked_image)

        self.caption_image(masked_image_path)

# Example usage:
if __name__ == "__main__":
    model_name_or_path = "densenet121"  # Could also be a path to a custom model
    sam_checkpoint = "/path_to_sam_checkpoint/sam_vit_h_4b8939.pth"
    tiny_llava_model_path = "bczhou/TinyLLaVA-3.1B"

    # Initialize the wrapper
    pipeline = PyTorchExplainableWrapper(
        model_name_or_path=model_name_or_path,
        sam_checkpoint=sam_checkpoint,
        tiny_llava_model_path=tiny_llava_model_path
    )

    # Run the pipeline on an image
    image_path = "/path_to_image/"
    pipeline.run_pipeline(image_path)
