import sqlite3

DATABASE_FILE = "DB_of_images.db"
conn = sqlite3.connect(DATABASE_FILE)
cursor = conn.cursor()
cursor.execute('SELECT * FROM image')
dataset = cursor.fetchall()
conn.close()

# ------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
print(torch.__version__)
from sklearn.model_selection import train_test_split


samples = []
labels = []
for img_obj in dataset:
    labels.append(int(img_obj[3]))
    img_bytes = img_obj[2]
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    img_normalized = transform(img)
    samples.append(img_normalized)

class_names = ['Human', 'Animal', 'Food']

# show some of samples
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(samples[i].permute(1, 2, 0).numpy())
    plt.xlabel(class_names[labels[i]])
# plt.show()

train_images, test_images, train_labels, test_labels = train_test_split(samples, labels, test_size=0.1, random_state=42)

# Convert the data to PyTorch tensors
train_images_tensor = torch.stack(train_images)
train_labels_tensor = torch.tensor(train_labels)
test_images_tensor = torch.stack(test_images)
test_labels_tensor = torch.tensor(test_labels)

# Create a DataLoader for training and validation data
train_dataset = torch.utils.data.TensorDataset(train_images_tensor, train_labels_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(test_images_tensor, test_labels_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# Build the model
num_classes = len(class_names)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(299 * 299 * 3, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create an instance of the model
model = CNNModel()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of training epochs
num_epochs = 30

# Lists to store training and validation metrics
train_accuracy_list = []
val_accuracy_list = []
train_loss_list = []
val_loss_list = []

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs.view(inputs.size(0), -1))

        # Calculate the loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        total_loss += loss.item()

        # Calculate training accuracy
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

    # Calculate average training accuracy and loss
    train_accuracy = correct_train / total_train
    train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()  # Set the model to evaluation mode
    correct_val = 0
    total_val = 0
    total_val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.view(inputs.size(0), -1))
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item()

            # Calculate validation accuracy
            _, predicted_val = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted_val == labels).sum().item()

    # Calculate average validation accuracy and loss
    val_accuracy = correct_val / total_val
    val_loss = total_val_loss / len(test_loader)

    # Append metrics to lists
    train_accuracy_list.append(train_accuracy)
    val_accuracy_list.append(val_accuracy)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)

    # Print metrics for the epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


epochs_range = range(30)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_accuracy_list, label='Training Accuracy')
plt.plot(epochs_range, val_accuracy_list, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss_list, label='Training Loss')
plt.plot(epochs_range, val_loss_list, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
# plt.show()

