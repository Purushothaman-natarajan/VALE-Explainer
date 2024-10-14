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
from data.CustomModel import CustomModel  # Import your custom model

class PyTorchExplainableWrapper:
    def __init__(self, custom_model_path, sam_checkpoint, tiny_llava_model_path, num_classes, device=None):
        # Load the device configuration
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Load the custom trained model
        self.model = self._load_custom_model(custom_model_path, num_classes)
        self.model.eval()  # Set model to evaluation mode

        # Load SAM (Segment Anything Model)
        self.sam_predictor = self._load_sam_model(sam_checkpoint)

        # Load TinyLLaVA model for captioning
        self.tiny_llava_tokenizer, self.tiny_llava_model, self.image_processor, self.context_len = self._load_tiny_llava(
            tiny_llava_model_path
        )

    def _load_custom_model(self, model_path, num_classes):
        # Load the custom-trained model from the given path
        base_model_name = 'resnet50'  # Change as per your custom model's architecture
        model = CustomModel(base_model_name, num_classes=num_classes).to(self.device)
        model.load_state_dict(torch.load(model_path))
        return model

    def _load_sam_model(self, sam_checkpoint):
        model_type = "vit_h"  # Assuming SAM's vit_h model
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        return SamPredictor(sam)

    def _load_tiny_llava(self, model_path):
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path)
        )
        return tokenizer, model, image_processor, context_len

    def preprocess_image(self, image_path):
        # Load and preprocess image for the custom model
        image = Image.open(image_path).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(image).unsqueeze(0).to(self.device)
        return input_tensor, image

    def predict_label(self, input_tensor, class_names):
        # Make prediction with the custom model
        with torch.no_grad():
            outputs = self.model(input_tensor)

        # Get top predictions
        top3_prob, top3_catid = torch.topk(outputs, 3)
        top3_labels = [class_names[i] for i in top3_catid[0]]
        top3_scores = top3_prob[0].tolist()

        return top3_labels, top3_scores

    def explain_with_shap(self, input_tensor):
        # SHAP explanation
        background = torch.randn((1, 3, 224, 224)).to(self.device)
        explainer = shap.DeepExplainer(self.model, background)
        shap_values = explainer.shap_values(input_tensor)

        # Return SHAP values (also for visualization)
        return shap_values

    def segment_image(self, shap_values, input_tensor, image):
        input_image_np = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        self.sam_predictor.set_image(input_image_np)

        # Top SHAP coordinates for SAM
        shap_values_abs = np.abs(shap_values[0])
        top_indices = np.unravel_index(np.argsort(shap_values_abs, axis=None)[::-1], shap_values_abs.shape)
        top_coordinates = list(zip(top_indices[1], top_indices[2]))

        # Use first SHAP point as SAM prompt
        input_point = np.array([top_coordinates[0]])
        input_label = np.array([1])  # Positive label for SAM

        # Predict mask using SAM
        masks, scores, logits = self.sam_predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)

        # Get best mask based on scores
        best_mask = masks[np.argmax(scores)]

        # Apply mask to the original image
        masked_image = np.array(image).copy()
        masked_image[best_mask == 0] = 0  # Set non-mask regions to black

        return masked_image

    def caption_image(self, masked_image_path, prompt="Explain the object in the image:"):
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

    def run_pipeline(self, image_path, class_names):
        # Step 1: Preprocess the image
        input_tensor, image = self.preprocess_image(image_path)

        # Step 2: Predict the label
        top3_labels, top3_scores = self.predict_label(input_tensor, class_names)
        print(f"Top 3 Predictions: {[(label, score) for label, score in zip(top3_labels, top3_scores)]}")

        # Step 3: Explain with SHAP
        shap_values = self.explain_with_shap(input_tensor)
        shap.image_plot([shap_values], [input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()])

        # Step 4: Segment image using SAM
        masked_image = self.segment_image(shap_values, input_tensor, image)

        # Step 5: Save and display masked image
        masked_image_path = "masked_image.jpg"
        plt.imshow(masked_image)
        plt.title("Masked Image")
        plt.show()
        plt.imsave(masked_image_path, masked_image)

        # Step 6: Caption the masked image using TinyLLaVA
        self.caption_image(masked_image_path)


# Example usage:
if __name__ == "__main__":
    custom_model_path = "/path_to_your_custom_model/model.pth"
    sam_checkpoint = "/path_to_sam_checkpoint/sam_vit_h_4b8939.pth"
    tiny_llava_model_path = "bczhou/TinyLLaVA-3.1B"
    train_dir = "/path_to_train_dir"  # Directory containing class labels for the custom model

    # Get class names
    class_names = [d.name for d in os.scandir(train_dir) if d.is_dir()]
    num_classes = len(class_names)

    # Initialize the wrapper
    pipeline = PyTorchExplainableWrapper(
        custom_model_path=custom_model_path,
        sam_checkpoint=sam_checkpoint,
        tiny_llava_model_path=tiny_llava_model_path,
        num_classes=num_classes
    )

    # Run the pipeline on an image
    image_path = "/path_to_image/image.jpg"
    pipeline.run_pipeline(image_path, class_names)
