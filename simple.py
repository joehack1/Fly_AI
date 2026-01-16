# simplest_ai.py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

print("⚡ Building your first AI...")

# 1. Load the famous MNIST dataset (handwritten digits)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

print(f"✅ Loaded {len(train_dataset)} training images")
print(f"✅ Loaded {len(test_dataset)} test images")

# 2. Let's look at some data
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i in range(10):
    img, label = train_dataset[i]
    ax = axes[i // 5, i % 5]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f"Label: {label}")
    ax.axis('off')
plt.suptitle("Sample handwritten digits from MNIST")
plt.tight_layout()
plt.show()

# 3. Build the SIMPLEST possible neural network
class SimpleAI(nn.Module):
    def __init__(self):
        super(SimpleAI, self).__init__()
        # Just one layer! Input: 28x28=784 pixels → Output: 10 digits
        self.layer = nn.Linear(28*28, 10)
    
    def forward(self, x):
        # Flatten the image from 28x28 to 784
        x = x.view(-1, 28*28)
        return self.layer(x)

# Create our AI
model = SimpleAI()
print(f"\n🧠 Our AI has {sum(p.numel() for p in model.parameters())} parameters")

# 4. Train it!
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

print("\n🚀 Training started...")
losses = []

for epoch in range(5):  # Just 5 epochs for quick training
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# 5. Test it!
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print(f"\n🎯 Test Accuracy: {accuracy:.2f}%")
print(f"That means it correctly identified {correct} out of {total} digits!")

# 6. Visualize some predictions
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
model.eval()
with torch.no_grad():
    for i in range(10):
        img, true_label = test_dataset[i]
        output = model(img.unsqueeze(0))
        pred_label = torch.argmax(output).item()
        
        ax = axes[i // 5, i % 5]
        ax.imshow(img.squeeze(), cmap='gray')
        color = 'green' if pred_label == true_label else 'red'
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
        ax.axis('off')

plt.suptitle(f"AI Predictions (Overall Accuracy: {accuracy:.1f}%)")
plt.tight_layout()
plt.show()

# 7. Save your AI!
torch.save(model.state_dict(), 'my_first_ai.pth')
print("\n💾 AI model saved as 'my_first_ai.pth'")
print("🎉 Congratulations! You just built and trained your first AI!")