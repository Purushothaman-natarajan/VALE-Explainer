### Language-Aware Visual Explanations (LAVE) ; a combination of text and visual explanation framework for image classification.

#### Overview:
This repository hosts an innovative Explainer tool designed specifically for the ImageNet dataset. Unlike traditional methods that require training, our approach provides explanations using SHAP (SHapley Additive exPlanations) values. It offers clear visual masks highlighting the objects of interest using SHAP scores coordinates as prompts to the SAM (Segment Anything Model), and textual explanations for the visual masks using VLM Llava.

#### Architecture:

<p align="center">
<img src= "https://github.com/Purushothaman-natarajan/LAVE-Explainer/blob/main/data/Architecture%20SHAP.jpg" width="800" />
</p>

#### Features:
- **SHAP-Based Explanations:** Utilizes SHAP values to explain model predictions without the need for additional training on the ImageNet dataset.
- **Visual Mask Generation:** Generates clear visual masks highlighting objects of interest in images using coordinates with highest SHAP score as input prompt to the SAM.
- **Textual Explanations:** Provides textual explanations for the visual masks using VLM Llava, enhancing interpretability.
- **Language-Aware:** Incorporates linguistic analysis to ensure coherent and informative textual explanations corresponding to the visual masks.

#### How to Use:
1. **Installation:**
   - Clone this repository to your local machine.
   - Ensure all dependencies listed in the `LAVE_(SHAP)_on_ImageNet.ipynb` file are installed accordingly.

2. **Dataset Preparation:**
   - Any pre-trained model compatible with ImageNet can be used with this explainer.

3. **Model Explanation:**
   - Run the ipynb file, providing the path to the image, SAM model, and the pre-trained model.
   - The script will explain the predictions of the classification model using SHAP. Then, the co-ordinates with highest SHAP value is used to create the visual mask and the corresponding textual explanations for the objects in the image.

4. **Interpreting Results:**
   - The output will include:
     - Predicted Label (Class Label).
     - SHAP explanation for the predicted label. 
     - Visual masks highlighting the objects of interest in the image.
     - Textual explanations corresponding to each visual mask.

5. **Customization:**
   - Modify parameters in the explainer script to adjust the level of detail in the explanations or experiment with different models.

#### Dependencies:
- Python
- SHAP
- SAM (Segment Anything Model)
- VLM Llava
- Other dependencies listed in `ipynb file`

#### Credits:
- SHAP library: [SHAP GitHub Repository](https://github.com/slundberg/shap)
- SAM (Segment Anything Model): [SAM GitHub Repository](https://github.com/samteam/sam)
- VLM Llava: [VLM Llava GitHub Repository](https://github.com/VLM-Llava/vlm-llava)

#### Contact:
For any inquiries or feedback, please contact [purushothamanprt@gmail.com / c30945@srmist.edu.in].

#### Acknowledgments:
We would like to acknowledge the developers of SHAP, SAM, and VLM Llava for their invaluable open-source models, as well as our funder, DRDO, India, in the field of explainable AI.
