import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet_classic import UNetBaseline
from data.synthetic_dataset import SyntheticMedicalDataset

def count_parameters(model):
    """Modelin toplam eğitilebilir parametre sayısını hesaplar"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train():
    # 1. GPU check (MPS Activate)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Başarılı: GPU (MPS) aktif edildi!")
    else:
        device = torch.device("cpu")
        print("⚠️ Uyarı: GPU bulunamadı, CPU kullanılıyor.")

    # 2. Dataset Load
    train_dataset = SyntheticMedicalDataset(num_samples=400, size=28)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 3. Model
    model = UNetBaseline(in_channels=1, out_channels=1).to(device)
    
    # Parameter count
    total_params = count_parameters(model)
    print(f"📊 Klasik U-Net Toplam Parametre Sayısı (Baseline): {total_params:,}")

    # 4. Loss func and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Training Loop
    epochs = 5
    print("\n🏋️ Model Eğitimi Başlıyor...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, masks in progress_bar:
            # GPU
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix(loss=loss.item())
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"✅ Epoch {epoch+1} Tamamlandı | Ortalama Kayıp (Loss): {epoch_loss:.4f}")

    
    # Save weights
    torch.save(model.state_dict(), "unet_baseline.pth")
    print("💾 Model ağırlıkları 'unet_baseline.pth' olarak başarıyla kaydedildi!")

    print("\n🎉 Eğitim Başarıyla Tamamlandı!")

if __name__ == "__main__":
    train()