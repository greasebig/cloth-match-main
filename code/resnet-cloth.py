import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import wandb

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
wandb.init(project='your_project_name')

train_dataset = ImageFolder(root='/data/lujunda/baidu/duomo/data/leftdata/clipscore/cloth-matvh/train_test_256/arrange_train', transform=transform)
test_dataset = ImageFolder(root='/data/lujunda/baidu/duomo/data/leftdata/clipscore/cloth-matvh/train_test_256/arrange_train', transform=transform)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


model = torchvision.models.resnet50(pretrained=True)


num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 30
save_every = 10


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 1):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        # 每个batch打印loss
        wandb.log({"Training Loss": loss.item()})
        if i % 5 == 0:  # 每10个batch打印一次
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    if (epoch + 1) % save_every == 0:
        torch.save(model.state_dict(), f'/data/lujunda/baidu/duomo/data/leftdata/clipscore/cloth-matvh/checkpoint/checkpoint_epoch_{epoch+1}.pth')
torch.save(model.state_dict(), '/data/lujunda/baidu/duomo/data/leftdata/clipscore/cloth-matvh/checkpoint/resnet50_weights-all.pth')
# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Accuracy on test set: {accuracy:.4f}")


wandb.finish()