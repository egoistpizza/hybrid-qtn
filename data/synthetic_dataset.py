import torch
from torch.utils.data import Dataset
import numpy as np

class SyntheticMedicalDataset(Dataset):
    """Eğitim için yapay hücre/tümör segmentasyon verisi üretir"""
    def __init__(self, num_samples=200, size=28):
        self.num_samples = num_samples
        self.size = size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Boş bir siyah görüntü ve maske oluşturuyoruz (28x28)
        image = np.zeros((self.size, self.size), dtype=np.float32)
        mask = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Rastgele bir merkeze sahip daire çiziyoruz (Hücre/Tümör simülasyonu)
        cx, cy = np.random.randint(5, self.size-5, size=2)
        radius = np.random.randint(3, 8)
        
        for x in range(self.size):
            for y in range(self.size):
                if (x - cx)**2 + (y - cy)**2 < radius**2:
                    image[x, y] = 1.0
                    mask[x, y] = 1.0 # Maskede tümörün olduğu yer tam beyaz (1.0)
                    
        # Görüntüye gerçekçilik katmak için arka plan gürültüsü (noise) ekleyelim
        noise = np.random.normal(0, 0.2, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)
        
        # PyTorch formatına dönüştürme: [Kanal, Yükseklik, Genişlik]
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        
        return image_tensor, mask_tensor