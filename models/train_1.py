import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset, random_split
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from generator import Generator
from discriminator import Discriminator

# Custom dataset for person folders
class PersonDataset(Dataset):
    def __init__(self, root_dir, transform=None, authorized_ids=None):
        self.root_dir = root_dir
        self.transform = transform
        self.authorized_ids = authorized_ids or []
        self.samples = []
        
        # Collect samples
        for person_id in os.listdir(root_dir):
            person_dir = os.path.join(root_dir, person_id)
            if not os.path.isdir(person_dir):
                continue
            for img_name in os.listdir(person_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(person_dir, img_name)
                    label = 1 if person_id in self.authorized_ids else 0
                    self.samples.append((img_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# Enhanced Data Augmentation Transforms
def get_enhanced_transforms():
    return transforms.Compose([
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def save_sample_images(generator, latent_dim, device, epoch, out_dir='samples'):
    os.makedirs(out_dir, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(16, latent_dim, 1, 1, device=device)
        fake_imgs = generator(noise)
        fake_imgs = (fake_imgs + 1) / 2  # Denormalize to [0,1]
        utils.save_image(fake_imgs, f"{out_dir}/epoch_{epoch+1}.png", nrow=4)
    generator.train()

def calculate_accuracy(model, dataloader, device):
    """Calculate accuracy on the full dataset without affecting computational graph"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # Ensure no gradients are computed during evaluation
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    model.train()
    return correct / total if total > 0 else 0

# Early Stopping Implementation
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 100
    authorized_ids = ['person3']  # Update with your authorized person ID
    
    # Enhanced hyperparameters
    batch_size = 32  # Reduced for better stability
    lr = 0.0002  # Standard DCGAN learning rate
    weight_decay = 1e-5  # L2 regularization
    
    # Create enhanced datasets with augmentation
    transform = get_enhanced_transforms()
    full_dataset = PersonDataset(
        root_dir='D:\\Personal\\Myprojects-new\\Mini_project\\Face_recognition\\dataset-1',
        transform=transform,
        authorized_ids=authorized_ids
    )
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize models
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    
    # Optimizers with weight decay
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)
    
    # Learning rate schedulers
    scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(optimizer_g, mode='min', patience=10, factor=0.5, verbose=True)
    scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(optimizer_d, mode='min', patience=10, factor=0.5, verbose=True)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=25, min_delta=0.001)
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Training history
    d_losses = []
    g_losses = []
    train_accuracies = []
    val_accuracies = []
    val_losses = []
    
    num_epochs = 200
    
    print(f"Starting training on {device}")
    print(f"Dataset size: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    for epoch in range(num_epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_val_loss = 0.0
        batches = 0
        
        # Training phase
        generator.train()
        discriminator.train()
        
        for i, (real_imgs, labels) in enumerate(train_loader):
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            batch_size_current = real_imgs.size(0)
            
            # Create labels for GAN training
            real_labels = torch.ones(batch_size_current, device=device)
            fake_labels = torch.zeros(batch_size_current, device=device)
            
            # ========================
            # Train Discriminator - FIXED VERSION
            # ========================
            optimizer_d.zero_grad()
            
            # Real images
            outputs_real = discriminator(real_imgs)
            d_loss_real = criterion(outputs_real, real_labels)
            
            # Fake images - CRITICAL FIX: Use .detach() to prevent backward through generator
            noise = torch.randn(batch_size_current, latent_dim, 1, 1, device=device)
            fake_imgs = generator(noise)
            outputs_fake = discriminator(fake_imgs.detach())  # DETACH HERE - This prevents the error
            d_loss_fake = criterion(outputs_fake, fake_labels)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()  # This won't affect generator's graph due to detach()
            optimizer_d.step()
            
            # ========================
            # Train Generator - FIXED VERSION
            # ========================
            optimizer_g.zero_grad()
            
            # Generate NEW fake images for generator training 
            noise = torch.randn(batch_size_current, latent_dim, 1, 1, device=device)
            fake_imgs_for_gen = generator(noise)  # Fresh computational graph
            outputs = discriminator(fake_imgs_for_gen)  # No detach for generator training
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()  # Clean backward pass through generator
            optimizer_g.step()
            
            # Accumulate losses
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            batches += 1
        
        # Validation phase
        generator.eval()
        discriminator.eval()
        val_batches = 0
        
        with torch.no_grad():  # No gradients needed for validation
            for real_imgs, labels in val_loader:
                real_imgs = real_imgs.to(device)
                labels = labels.to(device)
                
                outputs = discriminator(real_imgs)
                val_loss = criterion(outputs, labels.float())
                epoch_val_loss += val_loss.item()
                val_batches += 1
        
        # Calculate average losses
        avg_d_loss = epoch_d_loss / batches
        avg_g_loss = epoch_g_loss / batches
        avg_val_loss = epoch_val_loss / val_batches if val_batches > 0 else float('inf')
        
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        val_losses.append(avg_val_loss)
        
        # Calculate accuracies on full datasets
        train_acc = calculate_accuracy(discriminator, train_loader, device)
        val_acc = calculate_accuracy(discriminator, val_loader, device)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler_g.step(avg_val_loss)
        scheduler_d.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"D_loss: {avg_d_loss:.4f} G_loss: {avg_g_loss:.4f} Val_loss: {avg_val_loss:.4f} "
              f"Train_Acc: {train_acc:.2%} Val_Acc: {val_acc:.2%}")
        
        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        # Save samples every 20 epochs
        if (epoch + 1) % 20 == 0:
            save_sample_images(generator, latent_dim, device, epoch)
    
    # Save final models
    os.makedirs('saved_models_fixed', exist_ok=True)
    torch.save(generator.state_dict(), 'saved_models_fixed/generator.pth')
    torch.save(discriminator.state_dict(), 'saved_models_fixed/discriminator.pth')
    
    # Enhanced plotting
    plt.figure(figsize=(20, 12))
    
    # Loss plot
    plt.subplot(2, 3, 1)
    plt.plot(d_losses, label='Discriminator Loss', alpha=0.7)
    plt.plot(g_losses, label='Generator Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(2, 3, 2)
    plt.plot(train_accuracies, label='Training Accuracy', alpha=0.7)
    plt.plot(val_accuracies, label='Validation Accuracy', alpha=0.7)
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    
    # Overfitting analysis
    plt.subplot(2, 3, 3)
    generalization_gap = [train_acc - val_acc for train_acc, val_acc in zip(train_accuracies, val_accuracies)]
    plt.plot(generalization_gap, label='Generalization Gap', alpha=0.7)
    plt.title('Overfitting Analysis')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Gap')
    plt.legend()
    plt.grid(True)
    
    # Loss comparison
    plt.subplot(2, 3, 4)
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Discriminator vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Training stability
    plt.subplot(2, 3, 5)
    d_g_ratio = [d/g if g > 0 else 0 for d, g in zip(d_losses, g_losses)]
    plt.plot(d_g_ratio, label='D/G Loss Ratio', alpha=0.7)
    plt.title('Training Balance (D/G Loss Ratio)')
    plt.xlabel('Epochs')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid(True)
    
    # Final metrics summary
    plt.subplot(2, 3, 6)
    final_metrics = ['Train Acc', 'Val Acc', 'D Loss', 'G Loss']
    final_values = [train_accuracies[-1], val_accuracies[-1], d_losses[-1], g_losses[-1]]
    bars = plt.bar(final_metrics, final_values, alpha=0.7)
    plt.title('Final Training Metrics')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, final_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('fixed_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save training data
    with open('fixed_training_data.txt', 'w') as f:
        f.write("Epoch,Train_Acc,Val_Acc,D_Loss,G_Loss,Val_Loss\n")
        for epoch, (train_acc, val_acc, d_loss, g_loss, val_loss) in enumerate(
            zip(train_accuracies, val_accuracies, d_losses, g_losses, val_losses)):
            f.write(f"{epoch+1},{train_acc},{val_acc},{d_loss},{g_loss},{val_loss}\n")
    
    print("Training completed successfully!")
    print(f"Final Training Accuracy: {train_accuracies[-1]:.2%}")
    print(f"Final Validation Accuracy: {val_accuracies[-1]:.2%}")
    print(f"Generalization Gap: {train_accuracies[-1] - val_accuracies[-1]:.2%}")

if __name__ == '__main__':
    main()
