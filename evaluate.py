import torch
import matplotlib.pyplot as plt

from models.unet_classic import UNetBaseline
from data.synthetic_dataset import SyntheticMedicalDataset

def visualize_prediction():
    # 1. Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 2. Get random test data
    dataset = SyntheticMedicalDataset(num_samples=1, size=28)
    image, true_mask = dataset[0] # Tek bir örnek 

    model = UNetBaseline(in_channels=1, out_channels=1).to(device)

    # Load saved weghts
    model.load_state_dict(torch.load("unet_baseline.pth", map_location=device))
    print("💾 Eğitilmiş model ağırlıkları başarıyla yüklendi!")
    
    
    # 3. Model Evaluation mode
    #model.eval()
    
    # add extra batch size [1, 1, 28, 28]
    input_tensor = image.unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        prediction = model(input_tensor)
       
        prediction_probs = torch.sigmoid(prediction)

    # GPU to  CPU and numpy
    img_np = image.squeeze().numpy()
    true_mask_np = true_mask.squeeze().numpy()
    pred_probs_np = prediction_probs.squeeze().cpu().numpy() # Eşiklenmemiş ham olasılıklar

    # visualize with Matplotlib 
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title("1. Orijinal Görüntü")
    axes[0].axis('off')
    
    axes[1].imshow(true_mask_np, cmap='gray')
    axes[1].set_title("2. Gerçek Maske")
    axes[1].axis('off')
    
    axes[2].imshow(pred_probs_np, cmap='gray') 
    axes[2].set_title("3. Modelin Eminlik Haritası")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_prediction()