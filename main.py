import os
import json
import argparse
import warnings
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Hugging Face Libraries
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

import wandb

warnings.filterwarnings("ignore")

#Dataset Definition

class PolygonDatasetDiffusers(Dataset):
    def __init__(self, data_dir, tokenizer, split='training', image_size=128):
        self.split_dir = os.path.join(data_dir, split)
        self.image_size = image_size
        self.tokenizer = tokenizer

        
        json_path = os.path.join(self.split_dir, 'data.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Error: 'data.json' not found in {self.split_dir}")

        with open(json_path, 'r') as f:
            self.data_map = json.load(f)

        print(f"[{split.upper()}] Found {len(self.data_map)} samples in {self.split_dir}.")

        self.input_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.output_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        item = self.data_map[idx]

        input_img_path = os.path.join(self.split_dir, 'inputs', os.path.basename(item['input_polygon']))
        input_img = Image.open(input_img_path).convert("L")
        input_tensor = self.input_transform(input_img)

        color_name = item['colour']
        text_inputs = self.tokenizer(
            color_name, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.squeeze(0)

        output_img_path = os.path.join(self.split_dir, 'outputs', os.path.basename(item['output_image']))
        output_img = Image.open(output_img_path).convert("RGB")
        output_tensor = self.output_transform(output_img)

        return {
            "polygon_image": input_tensor,
            "text_input_ids": text_input_ids,
            "target_output": output_tensor,
            "color_name": color_name
        }

#Training and Validation Functions
def train_one_epoch(loader, unet, text_encoder, optimizer, loss_fn, device):
    unet.train()
    total_loss = 0.0
    for batch in loader:
        polygon_images = batch['polygon_image'].to(device)
        text_input_ids = batch['text_input_ids'].to(device)
        target_outputs = batch['target_output'].to(device)

        with torch.no_grad():
            encoder_hidden_states = text_encoder(text_input_ids).last_hidden_state

        predictions = unet(sample=polygon_images, timestep=0, encoder_hidden_states=encoder_hidden_states).sample
        loss = loss_fn(predictions, target_outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate_and_log_images(loader, unet, text_encoder, loss_fn, device, epoch):
    unet.eval()
    total_loss = 0.0
    logged_images = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            polygon_images = batch['polygon_image'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            target_outputs = batch['target_output'].to(device)

            encoder_hidden_states = text_encoder(text_input_ids).last_hidden_state

            predictions = unet(sample=polygon_images, timestep=0, encoder_hidden_states=encoder_hidden_states).sample
            loss = loss_fn(predictions, target_outputs)
            total_loss += loss.item()

            if i == 0:
                preds_denorm = (predictions.clamp(-1, 1) + 1) / 2
                targets_denorm = (target_outputs.clamp(-1, 1) + 1) / 2
                polys_denorm = (polygon_images.clamp(-1, 1) + 1) / 2

                for j in range(min(8, predictions.shape[0])): # Log up to 8 images
                    color_name = batch['color_name'][j]
                    # Concatenate (Input, Ground Truth, Prediction)
                    # Convert 1-channel input to 3-channel for concatenation
                    input_3_channel = polys_denorm[j].cpu().repeat(3, 1, 1)
                    comparison_img = torch.cat([input_3_channel, targets_denorm[j].cpu(), preds_denorm[j].cpu()], dim=2)
                    logged_images.append(wandb.Image(
                        comparison_img,
                        caption=f"Epoch {epoch} | Color: {color_name}"
                    ))

    avg_loss = total_loss / len(loader)
    wandb.log({"val_loss": avg_loss, "val_predictions": logged_images})
    return avg_loss

def main(args):

    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)
    else:
        try:
            wandb.login()
        except:
            wandb.login(anonymous="must")
            print("Could not find W&B secret. Proceeding in anonymous mode.")

    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_dir = args.data_dir
    print(f"Loading dataset from: {data_dir}")

    tokenizer = CLIPTokenizer.from_pretrained(args.text_encoder_model)

    train_dataset = PolygonDatasetDiffusers(data_dir, tokenizer, split='training', image_size=args.image_size)
    val_dataset = PolygonDatasetDiffusers(data_dir, tokenizer, split='validation', image_size=args.image_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    #Model Setup
    text_encoder = CLIPTextModel.from_pretrained(args.text_encoder_model).to(device)
    text_encoder.requires_grad_(False) # Freeze the text encoder

    unet = UNet2DConditionModel(
        in_channels=1,
        out_channels=3,
        block_out_channels=(128, 128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        cross_attention_dim=text_encoder.config.hidden_size,
    ).to(device)

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.learning_rate)
    best_val_loss = float('inf')

    os.makedirs(args.output_dir, exist_ok=True)

    #Training Loop
    print("\nStarting training with Diffusers U-Net...")
    for epoch in range(args.num_epochs):
        train_loss = train_one_epoch(train_loader, unet, text_encoder, optimizer, loss_fn, device)
        val_loss = validate_and_log_images(val_loader, unet, text_encoder, loss_fn, device, epoch)

        print(f"Epoch {epoch+1}/{args.num_epochs} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = os.path.join(args.output_dir, "best_unet_model")
            unet.save_pretrained(model_save_path)
            print(f"-> Saved new best model to {model_save_path} with val_loss: {best_val_loss:.4f}")

    wandb.finish()
    print("\nTraining complete.")
    print(f"Best model saved at: {os.path.join(args.output_dir, 'best_unet_model')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Polygon Colorizer using Diffusers U-Net")

    parser.add_argument('--data_dir', type=str, required=True, help='Path to the root dataset directory containing train/ and val/ folders.')
    parser.add_argument('--output_dir', type=str, default="./best_unet_model", help='Directory to save the best model.')

    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Optimizer learning rate.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and validation.')
    parser.add_argument('--num_epochs', type=int, default=130, help='Total number of training epochs.')
    parser.add_argument('--image_size', type=int, default=128, help='Image resolution.')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use for training (cuda or cpu).')

    parser.add_argument('--text_encoder_model', type=str, default="openai/clip-vit-base-patch32", help='Hugging Face model ID for the text encoder.')

    parser.add_argument('--wandb_project', type=str, default="polygon-colorizer-diffusers-final", help='W&B project name.')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (username or team). Defaults to your default entity.')
    parser.add_argument('--wandb_api_key', type=str, default=None, help='Your W&B API key.')

    args = parser.parse_args()
    main(args)