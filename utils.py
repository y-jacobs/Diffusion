import torch
import matplotlib.pyplot as plt


def visualize_results(model, test_loader, device='cuda'):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            break  # Just visualize the first batch
    # Convert to numpy for visualization
    data = data.cpu().numpy()
    output = output.cpu().numpy()
    # Plot original and reconstructed images
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(data[i][0], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(output[i][0], cmap='gray')
        axes[1, i].axis('off')
    plt.show()