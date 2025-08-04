# Polygon Colorizer

A deep learning project for colorizing polygon shapes based on text descriptions using Conditioned UNet implementation by diffusion model UNet2DConditionModel and CLIP text encoders.

## Overview

This project implements a polygon colorization system using:
- **Diffusers U-Net**: For generating colored polygon outputs
- **CLIP Text Encoder**: For processing color text descriptions
- **WandB**: For experiment tracking and visualization
- **PyTorch**: For the deep learning framework

The model takes grayscale polygon images and color text descriptions as input, and generates colored polygon outputs.

## Project Structure

```
polygon-colorizer/
├── main.py                 # Main training script
├── README.md              # Readme file
└── dataset/                 # Dataset directory
    ├── training/
    │   ├── inputs/       # Grayscale polygon images
    │   ├── outputs/      # Colored polygon images
    │   └── data.json     # Metadata with input/output mappings
    └── validation/
        ├── inputs/
        ├── outputs/
        └── data.json
```

## Setup Instructions

### 1. Environment Setup

Clone the repo (for running locally):

```bash
git clone https://github.com/sprnjt/polygon-colorizer.git
cd polygon-colorizer
```

Install the required dependencies:

```bash
pip install wandb diffusers transformers accelerate torch torchvision
```

### 2. Weights & Biases (WandB) Setup

For Kaggle Notebooks:
1. Click "Add-ons" → "Secrets" → "Add a new secret"
2. Label: `wandb_api_key`
3. Value: Your actual WandB API key

For Local Development:
- Set your WandB API key as an environment variable or pass it as a command line argument

### 3. Dataset Setup

#### Option A: Using Ayna Dataset in Kaggle
- Download the Ayna dataset from the provided source
- The dataset contains two main folders: `training` and `validation`
- Each folder has:
  - `inputs/`: Grayscale polygon shapes
  - `outputs/`: Completed colored polygons
  - `data.json`: Metadata with fields like `input_polygon`, `colour`, and `output_image`

#### Option B: Local Dataset
- Download the dataset locally
- Organize it according to the structure shown above

## Usage

### Training

Run the main training script:

```bash
python main.py --data_dir /path/to/your/dataset --output_dir ./models
```

#### Command Line Arguments

- `--data_dir`: Path to the root dataset directory (required)
- `--output_dir`: Directory to save the best model (default: `./best_unet_model`)
- `--learning_rate`: Optimizer learning rate (default: `1e-4`)
- `--batch_size`: Batch size for training (default: `8`)
- `--num_epochs`: Total number of training epochs (default: `130`)
- `--image_size`: Image resolution (default: `128`)
- `--device`: Device to use (default: `cuda`)
- `--text_encoder_model`: CLIP model ID (default: `openai/clip-vit-base-patch32`)
- `--wandb_project`: WandB project name (default: `polygon-colorizer-diffusers`)
- `--wandb_entity`: WandB entity/username
- `--wandb_api_key`: Your WandB API key

### Example Training Command

```bash
python main.py \
    --data_dir ./data \
    --output_dir ./models \
    --batch_size 16 \
    --num_epochs 100 \
    --learning_rate 5e-5 \
    --wandb_project "my-polygon-colorizer" \
    --wandb_api_key "your_api_key_here"
```

## Model Architecture

The model consists of:
1. **CLIP Text Encoder**: Processes color text descriptions
2. **U-Net with Cross-Attention**: Generates colored polygon outputs
3. **Conditional Generation**: Uses text embeddings to condition the image generation

## Resources

### Model Weights
Download pre-trained model weights from: [Google Drive](https://drive.google.com/drive/folders/1b2Zd0eoi33r4gD7msTQCx89kmqX3VDUR?usp=drive_link)

Model on Kaggle: [Kaggle](https://www.kaggle.com/models/suparnojitsarkar/p-colorizer-model)

### Kaggle Notebook
For complete training and inference examples, refer to the Kaggle notebook: [Kaggle Notebook](https://www.kaggle.com/code/suparnojitsarkar/p-colorizer-final-version)

[ipynb file](https://drive.google.com/file/d/1eZqsozuu4w7xBry3-YxqxtfyCQETT1vP/view?usp=sharing)

### WandB Run Details
Track training progress and view experiment details: [Wandb Report Link](https://api.wandb.ai/links/suparnojit2026-iisc/2v63a4in)

[PDF](https://drive.google.com/file/d/1sBwtDu-HVfrS5zAvOHKDZmn0NN0c8kU8/view?usp=sharing)

### Complete Report
For detailed project documentation and analysis: [Docs](https://docs.google.com/document/d/1N7OuYE1vUVSWzFI4aR0V3RzDNJeeb1BHAjezBm9dCUE/edit?usp=sharing)

[PDF](https://drive.google.com/file/d/17RFKwSh_to91GXDeGuwwNZg5W9XeqfQJ/view?usp=sharing)

## Dataset Format

The `data.json` file in each split folder contains metadata with the following structure:

```json
[
  {
    "input_polygon": "path/to/input/image.png",
    "colour": "red",
    "output_image": "path/to/output/image.png"
  }
]
```

## Training Process

1. **Data Loading**: Loads polygon images and color text descriptions
2. **Text Encoding**: Uses CLIP tokenizer and encoder to process color names
3. **Image Processing**: Converts images to tensors and normalizes them
4. **Model Training**: Trains the U-Net with cross-attention to the text embeddings
5. **Validation**: Evaluates model performance and logs sample predictions
6. **Model Saving**: Saves the best model based on validation loss

## Monitoring

The training process is monitored through WandB, which tracks:
- Training and validation losses
- Sample predictions with input, ground truth, and generated outputs
- Model hyperparameters and configuration

## Requirements

- Python 3.7+
- PyTorch 1.8+
- CUDA-compatible GPU (recommended)
- WandB account for experiment tracking

## Contact

[LinkedIn](https://www.linkedin.com/in/suparnojit)