# LAVE: Language-Aware Visual Explanations

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2408.12808-b31b1b.svg)](https://arxiv.org/abs/2408.12808)

**A multimodal visual and language explanation framework for image classifiers using eXplainable AI and Language Models**

</div>

## Overview

LAVE (Language-Aware Visual Explanation) is a novel explainability framework that provides both **visual** and **textual** explanations for image classification models. It combines:

- **SHAP** (SHapley Additive exPlanations) - For computing feature importance
- **SAM** (Segment Anything Model) - For generating precise visual masks
- **VLM Llava/TinyLLaVA** - For generating natural language explanations

### Key Features

- 🎯 **No Training Required** - Works directly with pre-trained models
- 🖼️ **Visual Explanations** - Highlight important image regions using SHAP + SAM
- 📝 **Textual Explanations** - Generate human-readable descriptions of predictions
- 🔄 **Flexible** - Supports both custom-trained and pre-trained models
- ⚡ **Edge-Optimized** - Includes optimizations for deployment on edge devices

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Input     │────▶│   SHAP       │────▶│     SAM     │
│   Image     │     │   Explainer  │     │  Segmenter  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                                ▼
                                        ┌───────────────┐
                                        │   TinyLLaVA   │
                                        │   Captioner   │
                                        └───────┬───────┘
                                                │
                                                ▼
                                        ┌───────────────┐
                                        │ Combined      │
                                        │ Explanation   │
                                        └───────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster inference)

### Setup

```bash
# Clone the repository
git clone https://github.com/Purushothaman-natarajan/LAVE-Explainer.git
cd LAVE-Explainer

# Create and activate conda environment
conda env create -f environment.yaml
conda activate LAVE-Explainer

# Or install dependencies directly
pip install -r requirements.txt
```

### Additional Model Downloads

You will need to download the following models separately:

1. **SAM Checkpoint**: [sam_vit_h_4b8939.pth](https://github.com/facebookresearch/segment-anything#model-checkpoints)
2. **TinyLLaVA**: Download from [HuggingFace](https://huggingface.co/bczhou/TinyLLaVA-3.1B)

## Quick Start

### Using Pre-trained Models

```bash
python pre-trained_model_explainer.py \
    --model_name densenet121 \
    --img_path path/to/image.jpg \
    --sam_checkpoint path/to/sam_vit_h_4b8939.pth \
    --tiny_llava_model_path bczhou/TinyLLaVA-3.1B
```

### Using Custom-Trained Models

```bash
python custom_model_explainer.py \
    --model_path path/to/your/model.pth \
    --img_path path/to/image.jpg \
    --sam_checkpoint path/to/sam_vit_h_4b8939.pth \
    --tiny_llava_model_path bczhou/TinyLLaVA-3.1B \
    --num_classes 10
```

## Training Your Own Model

### 1. Prepare Your Data

Organize your dataset as follows:

```
dataset/
├── class_1/
│   ├── image1.jpg
│   └── image2.jpg
├── class_2/
│   ├── image1.jpg
│   └── image2.jpg
└── ...
```

### 2. Process the Data

```bash
python data_loader.py \
    --path ./dataset \
    --target_folder ./processed_data \
    --dim 224 \
    --batch_size 32 \
    --num_workers 4 \
    --augment_data
```

### 3. Train the Model

```bash
python train.py \
    --base_model_names "resnet50,vgg16" \
    --shape 224 \
    --data_path ./processed_data \
    --log_dir ./logs \
    --model_dir ./models \
    --epochs 100 \
    --optimizer adam \
    --learning_rate 0.001 \
    --batch_size 32
```

### 4. Test the Model

```bash
python test.py \
    --data_path ./processed_data/test \
    --base_model_name resnet50 \
    --model_path ./models/resnet50_best_model.pth \
    --log_dir ./logs
```

### 5. Make Predictions

```bash
python predict.py \
    --model_path ./models/resnet50_best_model.pth \
    --img_path ./test_image.jpg \
    --train_dir ./processed_data/train \
    --base_model_names resnet50
```

## API Usage

### Using the Explainers in Python

```python
from pre_trained_model_explainer import PyTorchExplainableWrapper

# Initialize the explainer
explainer = PyTorchExplainableWrapper(
    model_name_or_path="densenet121",
    sam_checkpoint="./sam_vit_h_4b8939.pth",
    tiny_llava_model_path="bczhou/TinyLLaVA-3.1B"
)

# Run the full explanation pipeline
explainer.run_pipeline("path/to/image.jpg")
```

### Individual Components

```python
# Get SHAP explanations
shap_values = explainer.explain_with_shap(input_tensor)

# Generate visual mask
masked_image = explainer.segment_image(shap_values, input_tensor, original_image)

# Generate text explanation
caption = explainer.caption_image(masked_image_path)
```

## Supported Models

The training framework supports multiple backbone architectures:

| Model | Description |
|-------|-------------|
| ResNet18/50 | Residual Networks |
| VGG16 | Visual Geometry Group |
| MobileNetV2/V3 | Efficient mobile networks |
| EfficientNet | Efficient scaling networks |
| DenseNet121 | Densely connected networks |
| AlexNet | Classic CNN |
| And many more... |

## Research Paper

If you use this code in your research, please cite our paper:

```bibtex
@article{natarajan2024vale,
  title={VALE: A Multimodal Visual and Language Explanation Framework for Image Classifiers Using eXplainable AI and Language Models},
  author={Natarajan, Purushothaman and Nambiar, Athira},
  journal={arXiv preprint arXiv:2408.12808},
  year={2024}
}
```

## Project Structure

```
VALE-Explainer/
├── assets/                    # Images and diagrams
├── experiments_notebook/      # Jupyter notebooks
├── sonar_dataset/             # Sample dataset
├── data_loader.py            # Data loading and preprocessing
├── train.py                  # Model training script
├── test.py                   # Model testing script
├── predict.py                # Prediction script
├── pre-trained_model_explainer.py   # Pre-trained model explainer
├── custom_model_explainer.py        # Custom model explainer
├── environment.yaml          # Conda environment
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) before submitting PRs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [SHAP](https://github.com/shap/shap) - SHapley Additive exPlanations
- [SAM](https://github.com/facebookresearch/segment-anything) - Segment Anything Model
- [TinyLLaVA](https://github.com/bczhou/TinyLLaVA) - Efficient Large Language Models
- [DRDO, India](https://www.drdo.gov.in/) - Research funding

## Contact

- **Purushothaman Natarajan** - [purushothamanprt@gmail.com](mailto:purushothamanprt@gmail.com)
- **Athira Nambiar** - [c30945@srmist.edu.in](mailto:c30945@srmist.edu.in)

---

<div align="center">

⭐ Star us on GitHub if you find this project useful!

</div>
