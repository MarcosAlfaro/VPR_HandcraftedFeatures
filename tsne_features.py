import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
from config import PARAMS
from functions.models import load_model
from functions.misc_functions import select_device
from functions.img_process_functions import select_tf
import torchvision.transforms as transforms
import functions.img_process_functions as img_proc


class RepImagesDataset(Dataset):
    """Dataset for loading RepImages from FR_A environment"""
    
    def __init__(self, input_type="RGB", tf=transforms.ToTensor()):
        self.input_type = input_type
        self.tf = tf
        
        # Paths
        self.rgb_dir = f"{PARAMS.cold_path}FR_A/Train/"
        
        if input_type != "RGB":
            self.features_dir = f"{PARAMS.cold_path}FEATURES/{input_type}/FR_A/Train/"
        
        # Get all place folders
        self.places = sorted([d for d in os.listdir(self.rgb_dir) 
                             if os.path.isdir(os.path.join(self.rgb_dir, d))])
        
        # Collect all images with their place labels
        self.img_paths = []
        self.labels = []
        
        for place_idx, place in enumerate(self.places):
            place_path = os.path.join(self.rgb_dir, place)
            images = sorted([f for f in os.listdir(place_path) if f.endswith('.jpeg')])
            
            for img in images:
                self.img_paths.append((place, img))
                self.labels.append(place_idx)
        
        print(f"Loaded {len(self.img_paths)} images from {len(self.places)} places")
        print(f"Places: {self.places}")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        place, img_name = self.img_paths[idx]
        label = self.labels[idx]
        
        if self.input_type == "RGB":
            # Load RGB image
            img_path = os.path.join(self.rgb_dir, place, img_name)
            image = img_proc.load_image(img_path, rgb=True)
            image = img_proc.tf_image(image, tf=self.tf)
        else:
            # Load feature image
            img_path = os.path.join(self.features_dir, place, img_name.replace('.jpeg', '.npy'))
            image = img_proc.load_image(img_path, rgb=False)
            image = img_proc.equalize_image(image) if PARAMS.eq else image
            image = img_proc.inverse_image(image) if PARAMS.inv else image
            image = img_proc.sharpen_image(image) if PARAMS.sh else image
            image = img_proc.apply_colormap(image, PARAMS.color_rep) if PARAMS.color_rep is not None else image
            image = img_proc.tf_image(image, tf=self.tf)
            # Convert single channel to 3 channels for model compatibility
            image = torch.cat((image, image, image), dim=0)
        
        return image, label, place


def extract_embeddings(model, dataloader, device):
    """Extract embeddings from a model for all images in dataloader"""
    model.eval()
    embeddings = []
    labels = []
    places = []
    
    with torch.no_grad():
        for batch_idx, (images, batch_labels, batch_places) in enumerate(dataloader):
            images = images.to(device)
            output = model(images)
            
            embeddings.append(output.cpu().numpy())
            labels.extend(batch_labels.numpy())
            places.extend(batch_places)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    return embeddings, labels, places


def plot_tsne(descriptors, labels, place_names, title, perplexity=30, save_path=None):
    """Apply t-SNE and visualize embeddings"""
    print(f"\nCalculando t-SNE para {title}...")
    print(f"Shape: {descriptors.shape}")
    
    # 1. Execute t-SNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, max_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(descriptors)

    # 2. Create DataFrame
    df = pd.DataFrame()
    df['x'] = tsne_results[:, 0]
    df['y'] = tsne_results[:, 1]
    df['Place ID'] = labels
    df['Place Name'] = [place_names[i] for i in labels]

    # 3. Visualize
    plt.figure(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = sns.color_palette("hsv", len(unique_labels))
    
    sns.scatterplot(
        x="x", y="y",
        hue="Place Name",
        palette=colors,
        data=df,
        legend="full",
        alpha=0.7,
        s=100
    )
    
    plt.title(title, fontsize=21)
    plt.xlabel("t-SNE Dimension 1", fontsize=18)
    plt.ylabel("t-SNE Dimension 2", fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def main():
    """Main function to run t-SNE analysis on EXP03_COLD models"""
    
    # Setup
    device = select_device()
    tf = select_tf(model=PARAMS.model)
    
    # Features to analyze (available in EXP03_COLD)
    features = ["RGB", "GRAYSCALE", "MAGNITUDE", "ANGLE", "HUE"]
    features_spanish = ["RGB", "INTENSIDAD", "GRADIENTE (MAG.)", "GRADIENTE (ANG.)", "TONO"]
    
    # Create output directory for plots
    output_dir = "tsne_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    all_embeddings = {}
    all_labels = None
    place_names = None
    
    for feature in features:
        print(f"\n{'='*60}")
        print(f"Processing feature: {feature}")
        print(f"{'='*60}")
        
        # Load model
        model_path = f"{PARAMS.saved_models_path}EXP03_COLD/{feature}/net.pth"
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
        
        print(f"Loading model from: {model_path}")
        model = load_model(
            model=PARAMS.model,
            backbone=PARAMS.backbone,
            embedding_size=PARAMS.embedding_size,
            state_dict_path=model_path,
            device=device
        )
        model.eval()
        
        # Load dataset
        dataset = RepImagesDataset(input_type=feature, tf=tf)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        
        if place_names is None:
            place_names = dataset.places
        
        # Extract embeddings
        embeddings, labels, places = extract_embeddings(model, dataloader, device)
        
        if all_labels is None:
            all_labels = labels
        
        all_embeddings[feature] = embeddings
        
        # Plot t-SNE for this feature
        perplexity = min(30, len(embeddings) // 3)  # Adjust perplexity based on data size
        plot_tsne(
            embeddings, 
            labels, 
            place_names,
            f"t-SNE: {feature} Features (EXP03_COLD, FR_A RepImages)",
            perplexity=perplexity,
            save_path=f"{output_dir}/tsne_{feature}.png"
        )
        
        print(f"\nFeature {feature}: Mean embedding norm = {np.linalg.norm(embeddings, axis=1).mean():.4f}")
    
    # Compare embeddings from different features
    if len(all_embeddings) > 1:
        print(f"\n{'='*60}")
        print("Creating comparison plots")
        print(f"{'='*60}")
        
        # Plot all features in subplots
        n_features = len(all_embeddings)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (feature, embeddings) in enumerate(all_embeddings.items()):
            if idx >= 6:
                break
            
            perplexity = min(30, len(embeddings) // 3)
            tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=1000, random_state=42)
            tsne_results = tsne.fit_transform(embeddings)
            
            ax = axes[idx]
            unique_labels = np.unique(all_labels)
            colors = sns.color_palette("hsv", len(unique_labels))
            
            for label_idx, label in enumerate(unique_labels):
                mask = all_labels == label
                ax.scatter(
                    tsne_results[mask, 0],
                    tsne_results[mask, 1],
                    c=[colors[label_idx]],
                    label=place_names[label],
                    alpha=0.6,
                    s=50
                )
            
            ax.set_title(f"{features_spanish[idx]}", fontsize=21)
            ax.set_xlabel("t-SNE Dim 1", fontsize=18)
            ax.set_ylabel("t-SNE Dim 2", fontsize=18)
            ax.grid()
            
        
        # Hide unused subplots and place legend in bottom right
        for idx in range(n_features, 6):
            axes[idx].axis('off')
        
        # Place legend in the bottom-right empty subplot
        if n_features < 6:
            handles, labels_list = axes[0].get_legend_handles_labels()
            axes[5].legend(handles, labels_list, fontsize=24, loc='center', frameon=True, 
                          ncol=2, title="Estancias", title_fontsize=26)
        
        plt.suptitle("Comparación de descriptores t-SNE (COLD)", fontsize=28, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/tsne_comparison_all.png", dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {output_dir}/tsne_comparison_all.png")
        plt.show()
    
    print(f"\n{'='*60}")
    print("t-SNE analysis completed!")
    print(f"Plots saved in: {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()