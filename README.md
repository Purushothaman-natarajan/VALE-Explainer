### Language-Aware Visual Explanations (LAVE) ; a combination of text and visual explanation framework for image classification.

This repository presents **LAVE**, a unique explainer tool for the ImageNet dataset, combining both **SHAP** (SHapley Additive exPlanations) and **SAM** (Segment Anything Model) for visual masks, and **VLM Llava** for generating textual explanations. It allows users to generate both visual and textual explanations without requiring additional training. 

## Overview

LAVE offers an explainer for models based on the ImageNet dataset, allowing you to interpret model predictions through:

- **SHAP-based explanations** for visual and textual insights.
- **Visual masks** highlighting objects of interest in images, generated by SAM using SHAP coordinates.
- **Textual explanations** generated by VLM Llava for each visual mask.


## Architecture

<p align="center">
<img src= "https://github.com/Purushothaman-natarajan/LAVE-Explainer/blob/main/data/Architecture%20SHAP.jpg" width="800" />
</p>



## Features

1. **SHAP-Based Explanations**: Provides visual and textual explanations based on SHAP values, requiring no additional training on ImageNet.
2. **SAM Integration**: Uses SAM to generate visual masks from SHAP coordinates highlighting important regions.
3. **Textual Explanations**: VLM Llava generates language-aware, coherent textual descriptions for the visual explanations.
4. **Modular Use**: Capable of using both custom-trained models and pre-trained models directly from TorchHub.


## How to Use

### Installation

Clone the repository and install the dependencies listed in `requirements.txt`:

```bash
git clone https://github.com/Purushothaman-natarajan/LAVE-Explainer.git
cd LAVE-Explainer
```

#### Create a Conda Environment

```bash
conda env create -f environment.yaml
conda activate LAVE-Explainer
```

## Explainer Script

You can use either of the following scripts based on your model type:

1. **Pre-trained Models**: Use `pre-trained_model_explainer.py` for playing with pre-trained models available on TorchHub.
   
   ```bash
   python pre-trained_model_explainer.py --model_name <pretrained_model_name> --img_path <path_to_image>
   ```

2. **Custom-Trained Models**: Use `custom_model_explainer.py` for explaining models trained using the codes in this repo.

   ```bash
   python custom_model_explainer.py --model_path <path_to_custom_model> --img_path <path_to_image>
   ```

---

## Model Training

This section covers how to use the provided scripts to train a image classification model using transfer learning for your datasets.

#### 1. `data_loader.py`

This script is used to load, process, split dataset - train, val, tests. and augment data.

**Command Line Usage:**

```sh
python data_loader.py --path <path_to_data> --target_folder <path_to_target_folder> --dim <dimension> --batch_size <batch_size> --num_workers <num_workers> [--augment_data]
```

**Arguments:**

- `--path`: Path to the data.
- `--target_folder`: Path to the target folder where processed data will be saved.
- `--dim`: Dimension for resizing the images.
- `--batch_size`: Batch size for data loading.
- `--num_workers`: Number of workers for data loading.
- `--augment_data` (optional): Flag to enable data augmentation.

**Example:**

```sh
python data_loader.py --path "./data" --target_folder "./processed_data" --dim 224 --batch_size 32 --num_workers 4 --augment_data
```

**Dataset Structure:**

```sh
├── Dataset (Raw)
   ├── class_name_1
   │   └── *.jpg
   ├── class_name_2
   │   └── *.jpg
   ├── class_name_3
   │   └── *.jpg
   └── class_name_4
       └── *.jpg
```

#### 2. `train.py`

This script is used for training and storing the models leveraging transfer learning.

**Command Line Usage:**

```sh
python train.py --base_model_names <model_names> --shape <shape> --data_path <data_path> --log_dir <log_dir> --model_dir <model_dir> --epochs <epochs> --optimizer <optimizer> --learning_rate <learning_rate> --batch_size <batch_size>
```

**Arguments:**

- `--base_model_names`: Comma-separated list of base model names (e.g., 'vgg16,alexnet').
- `--shape`: Image shape (size).
- `--data_path`: Path to the data.
- `--log_dir`: Path to the log directory.
- `--model_dir`: Path to the model directory.
- `--epochs`: Number of training epochs.
- `--optimizer`: Optimizer type ('adam' or 'sgd').
- `--learning_rate`: Learning rate for the optimizer.
- `--batch_size`: Batch size for training.

**Example:**

```sh
python train.py --base_model_names "vgg16,alexnet" --shape 224 --data_path "./data" --log_dir "./logs" --model_dir "./models" --epochs 100 --optimizer "adam" --learning_rate 0.001 --batch_size 32
```

#### 3. `test.py`

This script is used for testing and storing the test logs of the trained models.

**Command Line Usage:**

```sh
python test.py --data_path <data_path> --base_model_name <base_model_name> --model_path <model_path> --models_folder_path <models_folder_path> --log_dir <log_dir>
```

**Arguments:**

- `--data_path`: Path to the test data.
- `--base_model_name`: Name of the base model.
- `--model_path` (optional): Path to the specific model file.
- `--models_folder_path` (optional): Path to the folder containing models.
- `--log_dir`: Path to the log directory.

**Example:**

```sh
python test.py --data_path "./test_data" --base_model_name "vgg16" --model_path "./models/vgg16_model.pth" --log_dir "./logs"
```

#### 4. `predict.py`

This script is used for making predictions on new images.

**Command Line Usage:**

```sh
python predict.py --model_path <model_path> --img_path <img_path> --train_dir <train_dir> --base_model_name <base_model_name>
```

**Arguments:**

- `--model_path`: Path to the model file.
- `--img_path`: Path to the image file.
- `--train_dir`: Path to the training dataset.
- `--base_model_name`: Name of the base model.

**Example:**

```sh
python predict.py --model_path "./models/vgg16_model.pth" --img_path "./images/test_image.jpg" --train_dir "./data/train" --base_model_name "vgg16"
```

-----

### Running Scripts in Jupyter Notebook:

You can also run these scripts programmatically using Python's `subprocess` module specially for Jupyter Notebook users. Here is an example of how to do this for each script:

```python
import subprocess

# Run data_loader.py
subprocess.run([
    "python", "data_loader.py",
    "--path", "./data",
    "--target_folder", "./processed_data",
    "--dim", "224",
    "--batch_size", "32",
    "--num_workers", "4",
    "--augment_data"
])

# Run train.py
subprocess.run([
    "python", "train.py",
    "--base_model_names", "vgg16,alexnet",
    "--shape", "224",
    "--data_path", "./data",
    "--log_dir", "./logs",
    "--model_dir", "./models",
    "--epochs", "100",
    "--optimizer", "adam",
    "--learning_rate", "0.001",
    "--batch_size", "32"
])

# Run test.py
subprocess.run([
    "python", "test.py",
    "--data_path", "./test_data",
    "--base_model_name", "vgg16",
    "--model_path", "./models/vgg16_model.pth",
    "--log_dir", "./logs"
])

# Run predict.py
subprocess.run([
    "python", "predict.py",
    "--model_path", "./models/vgg16_model.pth",
    "--img_path", "./images/test_image.jpg",
    "--train_dir", "./data/train",
    "--base_model_name", "vgg16"
])
```


## Dependencies

- Python
- SHAP
- SAM (Segment Anything Model)
- VLM Llava
- PyTorch (for pre-trained models and custom model training)
- Other dependencies listed in `requirements.txt`

#### Contact:
For any inquiries or feedback, please contact [purushothamanprt@gmail.com / c30945@srmist.edu.in].

#### Acknowledgments:
We would like to acknowledge the developers of SHAP, SAM, and VLM Llava for their invaluable open-source models, as well as our funder, DRDO, India, in the field of explainable AI.

## Citation

If you use this repository, please cite the following paper:

```
@article{natarajan2024vale,
  title={VALE: A Multimodal Visual and Language Explanation Framework for Image Classifiers using eXplainable AI and Language Models},
  author={Natarajan, Purushothaman and Nambiar, Athira},
  journal={arXiv preprint arXiv:2408.12808},
  year={2024}
}
```
------