---
layout: post
title: "Computer Vision Experiments Using Machine Learning"
date: 2024-09-25
---

# CelebA Project—Computer Vision Using Machine Learning

# A Kaggle Project by John Aziz 

This is a personal data science project. I graduated in July 2024 with a Bachelor's degree in Data Science, and since then I have been building some projects with the intent of improving my skills and being able to secure a job.

I am using a dataset here for machine learning that I obtained from Kaggle—https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

Relevant features of the dataset listed on Kaggle:

* 202,599 number of face images of various celebrities
* 10,177 unique identities, but names of identities are not given
* 40 binary attribute annotations per image

Everything undertaken here is fairly exploratory work, revisiting and refreshing methods that I used during my undergraduate degree, and trying out some new methods that I have not worked with before.

The first task we are undertaking is to build a binary classifier for smiling or not smiling, a suggested task from the authors of the dataset.

We will develop a convolutional neural network (CNN) using PyTorch and a pre-trained model (MobileNetV2). The goal, as you can probably surmise, is to determine whether the subject in an image is smiling or not.

We use the CelebA dataset, with the 'Smiling' attribute as the target. The attribute annotations are loaded and processed, converting labels for binary classification (0 for not smiling, 1 for smiling).

The dataset is split into training, validation, and test sets, and transformed for smaller image dimensions (128x128) to speed up training.

For our model architecture we employ MobileNetV2, a lightweight and efficient model for image classification, fine-tuned for our binary classification task. 

The model is trained over 3 epochs using binary cross-entropy loss and optimized using Adam. We employ accuracy and loss metrics to track performance.

The model is validated on a separate dataset after each epoch and finally tested to evaluate its performance.



```python
# Required Libraries
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm.notebook import tqdm

# Paths
image_dir = '/Users/johnaziz/Downloads/archive-3/img_align_celeba/img_align_celeba/'
attr_path = '/Users/johnaziz/Downloads/archive-3/list_attr_celeba.csv'
partition_path = '/Users/johnaziz/Downloads/archive-3/list_eval_partition.csv'

# Load the attribute data and partition data
print("Loading data...")
attr_df = pd.read_csv(attr_path)
partition_df = pd.read_csv(partition_path)

# Filter for the 'Smiling' attribute
attr_df['Smiling'] = attr_df['Smiling'].replace(-1, 0)  # Convert -1 to 0 for binary classification

# Merge partition info into attr_df to facilitate train/val/test split
print("Preparing datasets...")
attr_df = attr_df.merge(partition_df, on="image_id")

# Define PyTorch Dataset
class CelebADataset(Dataset):
    def __init__(self, img_dir, df, transform=None):
        self.img_dir = img_dir
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        image = Image.open(img_name)

        label = torch.tensor(self.df.iloc[idx, 1], dtype=torch.float32)  # Smiling label

        if self.transform:
            image = self.transform(image)

        return image, label

# Define smaller image transformations (128x128 instead of 224x224)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prepare the train, validation, and test datasets
train_df = attr_df[attr_df['partition'] == 0]  # Training set
val_df = attr_df[attr_df['partition'] == 1]    # Validation set
test_df = attr_df[attr_df['partition'] == 2]   # Test set

train_dataset = CelebADataset(image_dir, train_df[['image_id', 'Smiling']], transform=transform)
val_dataset = CelebADataset(image_dir, val_df[['image_id', 'Smiling']], transform=transform)
test_dataset = CelebADataset(image_dir, test_df[['image_id', 'Smiling']], transform=transform)

# DataLoader for batching (smaller batch size for faster training)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define a faster model (MobileNetV2) with pretrained weights
class SmileClassifier(nn.Module):
    def __init__(self):
        super(SmileClassifier, self).__init__()
        self.model = models.mobilenet_v2(weights='IMAGENET1K_V1')  # Use MobileNetV2
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 1)  # Binary classification output

    def forward(self, x):
        return self.model(x)

# Initialize the model, loss function, and optimizer
model = SmileClassifier()
criterion = nn.BCEWithLogitsLoss()  # Binary classification loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop for 3 epochs (quick training)
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3):
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print("Training...")

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            labels = labels.unsqueeze(1)  # Ensure labels are in shape [batch_size, 1]

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs).round()  # Use sigmoid + round for binary classification predictions
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f'Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        print("Validating...")
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                preds = torch.sigmoid(outputs).round()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        print(f'Validation Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_acc:.4f}')

    print('Training Complete')

# Train the model for 3 epochs
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3)

# Evaluation on the test set
def evaluate_model(model, test_loader):
    print("\nEvaluating the model on the test set...")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            preds = torch.sigmoid(outputs).round()
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_loss /= len(test_loader.dataset)
    test_acc = test_correct / test_total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

# Evaluate the model on the test set
evaluate_model(model, test_loader)


```

    Loading data...
    Preparing datasets...
    
    Epoch 1/3
    Training...



      0%|          | 0/5087 [00:00<?, ?it/s]


    Training Loss: 0.1961, Accuracy: 0.9191
    Validating...
    Validation Loss: 0.1767, Accuracy: 0.9290
    
    Epoch 2/3
    Training...



      0%|          | 0/5087 [00:00<?, ?it/s]


    Training Loss: 0.1762, Accuracy: 0.9273
    Validating...
    Validation Loss: 0.1675, Accuracy: 0.9297
    
    Epoch 3/3
    Training...



      0%|          | 0/5087 [00:00<?, ?it/s]


    Training Loss: 0.1690, Accuracy: 0.9299
    Validating...
    Validation Loss: 0.1615, Accuracy: 0.9311
    Training Complete
    
    Evaluating the model on the test set...
    Test Loss: 0.1720, Test Accuracy: 0.9279


Throughout the training epochs, both the training and validation losses consistently decreased, while accuracy steadily increased. This indicates that the model was learning effectively and generalizing well on the validation set.

With a test accuracy of 92.79%, we have achieved a strong performance in distinguishing between smiling and non-smiling faces. This is vastly better than 50% which would be achieved by randomly guessing. This means that the model is predicting accurately.

The model’s final performance on the test set shows only a slight increase in loss compared to the validation set, meaning that the model maintained good generalization and performance across unseen data.

Next, we move on to building a predictor for all of the attributes in the dataset.

We once again load the CelebA dataset, merge the attribute and partition data, and convert the attribute labels into binary format.

We define a PyTorch Dataset class that loads the images from the specified directory and returns both the image and the corresponding 40 attribute labels for each sample.

The images are resized to 128x128 to reduce computation and memory usage (especially important given I am training on my laptop) and normalisation is applied for better performance with the pre-trained model.

The model is based on MobileNetV2. The final layer of the model is modified to output 40 logits, each representing one of the 40 attributes in the dataset.

We use Binary Cross Entropy with Logits (BCEWithLogitsLoss) as the loss function, which is suitable for multi-label binary classification.

The model is once again trained for 3 epochs using a batch size of 32.

After training, the model is evaluated on the test set to assess its performance in predicting multiple attributes simultaneously.


```python
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# Paths to the data
image_dir = '/Users/johnaziz/Downloads/archive-3/img_align_celeba/img_align_celeba/'
attr_path = '/Users/johnaziz/Downloads/archive-3/list_attr_celeba.csv'
partition_path = '/Users/johnaziz/Downloads/archive-3/list_eval_partition.csv'

# Load attribute and partition data
print("Loading data...")
attr_df = pd.read_csv(attr_path)
partition_df = pd.read_csv(partition_path)

# Merge attribute data with partition data on image_id
attr_df = attr_df.merge(partition_df, on="image_id")

# Convert -1 to 0 for binary classification (0 = attribute absent, 1 = attribute present)
attr_df.iloc[:, 1:-1] = attr_df.iloc[:, 1:-1].replace(-1, 0)

# Convert all columns except 'image_id' and 'partition' to numeric, coercing errors
for column in attr_df.columns[1:-1]:
    attr_df[column] = pd.to_numeric(attr_df[column], errors='coerce')

# Check for any remaining NaN values after the conversion
print("Checking for NaN values in the dataset...")
print(attr_df.isnull().sum())

# Fill any remaining NaN values with 0 (assuming it's a missing label)
attr_df = attr_df.fillna(0)

# Verify the types again after the fix
print("Verifying column data types after conversion...")
print(attr_df.dtypes)

# Split data into train, validation, and test sets based on partition
train_df = attr_df[attr_df['partition'] == 0]
val_df = attr_df[attr_df['partition'] == 1]
test_df = attr_df[attr_df['partition'] == 2]

# Define PyTorch Dataset for multi-label classification
class CelebAMultiLabelDataset(Dataset):
    def __init__(self, img_dir, df, transform=None):
        self.img_dir = img_dir
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        image = Image.open(img_name)

        # Updated slice to exclude 'image_id' and 'partition' columns
        labels = torch.tensor(self.df.iloc[idx, 1:-1].values.astype(float), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels

# Define image transformations (128x128 resolution)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prepare the datasets and DataLoader
train_dataset = CelebAMultiLabelDataset(image_dir, train_df, transform=transform)
val_dataset = CelebAMultiLabelDataset(image_dir, val_df, transform=transform)
test_dataset = CelebAMultiLabelDataset(image_dir, test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model for multi-label classification using MobileNetV2
class MultiLabelAttributeClassifier(nn.Module):
    def __init__(self):
        super(MultiLabelAttributeClassifier, self).__init__()
        self.model = models.mobilenet_v2(weights='IMAGENET1K_V1')  # Load pre-trained weights
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 40)  # 40 attributes to predict

    def forward(self, x):
        return self.model(x)

# Initialize model, loss function, and optimizer
model = MultiLabelAttributeClassifier()
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy loss for each label
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to calculate per-attribute accuracy
def calculate_per_attribute_accuracy(outputs, labels):
    # Apply sigmoid to the outputs to get probabilities
    probs = torch.sigmoid(outputs)
    
    # Convert probabilities to binary predictions (threshold at 0.5)
    preds = (probs > 0.5).float()

    # Calculate per-attribute accuracy
    correct_per_attribute = (preds == labels).float().mean(dim=0)
    
    return correct_per_attribute

# Training loop with per-attribute accuracy calculation
def train_model_with_per_attribute_accuracy(model, train_loader, val_loader, criterion, optimizer, num_epochs=3):
    num_attributes = 40  
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Training phase
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Training Loss: {epoch_loss:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        total_correct_per_attribute = torch.zeros(num_attributes).to(device)  # Track accuracy for each attribute
        total_batches = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                # Calculate accuracy per attribute for this batch
                correct_per_attribute = calculate_per_attribute_accuracy(outputs, labels)
                total_correct_per_attribute += correct_per_attribute
                total_batches += 1

        val_epoch_loss = val_loss / len(val_loader.dataset)
        per_attribute_accuracy = total_correct_per_attribute / total_batches

        print(f'Validation Loss: {val_epoch_loss:.4f}')
        print(f'Per-Attribute Validation Accuracy: {per_attribute_accuracy.cpu().numpy()}')

    print('Training complete')

# Train the model with per-attribute validation accuracy
train_model_with_per_attribute_accuracy(model, train_loader, val_loader, criterion, optimizer, num_epochs=3)

# Evaluation on the test set (if needed)
def evaluate_model(model, test_loader):
    print("\nEvaluating the model on the test set...")
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')

# Evaluate the model on the test set
evaluate_model(model, test_loader)

```

    Loading data...
    Checking for NaN values in the dataset...
    image_id               0
    5_o_Clock_Shadow       0
    Arched_Eyebrows        0
    Attractive             0
    Bags_Under_Eyes        0
    Bald                   0
    Bangs                  0
    Big_Lips               0
    Big_Nose               0
    Black_Hair             0
    Blond_Hair             0
    Blurry                 0
    Brown_Hair             0
    Bushy_Eyebrows         0
    Chubby                 0
    Double_Chin            0
    Eyeglasses             0
    Goatee                 0
    Gray_Hair              0
    Heavy_Makeup           0
    High_Cheekbones        0
    Male                   0
    Mouth_Slightly_Open    0
    Mustache               0
    Narrow_Eyes            0
    No_Beard               0
    Oval_Face              0
    Pale_Skin              0
    Pointy_Nose            0
    Receding_Hairline      0
    Rosy_Cheeks            0
    Sideburns              0
    Smiling                0
    Straight_Hair          0
    Wavy_Hair              0
    Wearing_Earrings       0
    Wearing_Hat            0
    Wearing_Lipstick       0
    Wearing_Necklace       0
    Wearing_Necktie        0
    Young                  0
    partition              0
    dtype: int64
    Verifying column data types after conversion...
    image_id               object
    5_o_Clock_Shadow        int64
    Arched_Eyebrows         int64
    Attractive              int64
    Bags_Under_Eyes         int64
    Bald                    int64
    Bangs                   int64
    Big_Lips                int64
    Big_Nose                int64
    Black_Hair              int64
    Blond_Hair              int64
    Blurry                  int64
    Brown_Hair              int64
    Bushy_Eyebrows          int64
    Chubby                  int64
    Double_Chin             int64
    Eyeglasses              int64
    Goatee                  int64
    Gray_Hair               int64
    Heavy_Makeup            int64
    High_Cheekbones         int64
    Male                    int64
    Mouth_Slightly_Open     int64
    Mustache                int64
    Narrow_Eyes             int64
    No_Beard                int64
    Oval_Face               int64
    Pale_Skin               int64
    Pointy_Nose             int64
    Receding_Hairline       int64
    Rosy_Cheeks             int64
    Sideburns               int64
    Smiling                 int64
    Straight_Hair           int64
    Wavy_Hair               int64
    Wearing_Earrings        int64
    Wearing_Hat             int64
    Wearing_Lipstick        int64
    Wearing_Necklace        int64
    Wearing_Necktie         int64
    Young                   int64
    partition               int64
    dtype: object
    
    Epoch 1/3


    100%|███████████████████████████████████████| 5087/5087 [57:02<00:00,  1.49it/s]


    Training Loss: 0.2272
    Validation Loss: 0.2063
    Per-Attribute Validation Accuracy: [0.9354778  0.8435021  0.8092104  0.8323903  0.9883756  0.9565124
     0.81417364 0.8181081  0.8700815  0.939101   0.9642117  0.8301891
     0.92022085 0.94774705 0.96062946 0.9916465  0.95958203 0.9722725
     0.9110716  0.8751137  0.98152244 0.9376734  0.955547   0.9385792
     0.95827365 0.75376856 0.96738195 0.75965625 0.94039077 0.94457674
     0.9609407  0.92246675 0.8345131  0.8601271  0.9052845  0.9853469
     0.9236335  0.8854073  0.96164525 0.8600953 ]
    
    Epoch 2/3


    100%|███████████████████████████████████████| 5087/5087 [58:05<00:00,  1.46it/s]


    Training Loss: 0.2052
    Validation Loss: 0.1994
    Per-Attribute Validation Accuracy: [0.9381355  0.8535852  0.805207   0.82777    0.98806435 0.9545908
     0.8556167  0.82208353 0.86395156 0.9518734  0.9643533  0.84239507
     0.9223344  0.94940764 0.9621988  0.99541134 0.96592265 0.9768015
     0.91328573 0.8661564  0.9811292  0.9353771  0.9591198  0.9387805
     0.95681435 0.7545234  0.9623498  0.7628582  0.9434604  0.9436803
     0.9683381  0.927348   0.837907   0.8657128  0.9104584  0.9889608
     0.9249922  0.88690764 0.9647149  0.8743775 ]
    
    Epoch 3/3


    100%|███████████████████████████████████████| 5087/5087 [53:30<00:00,  1.58it/s]


    Training Loss: 0.1981
    Validation Loss: 0.1932
    Per-Attribute Validation Accuracy: [0.93954456 0.85644424 0.8130256  0.8421714  0.989684   0.9578114
     0.85269797 0.8270244  0.9111537  0.9530812  0.96284366 0.8554881
     0.927074   0.9556476  0.96481556 0.9933072  0.9660643  0.9781602
     0.9225264  0.87979364 0.983595   0.9391923  0.95906025 0.9133864
     0.9594907  0.76101494 0.9608401  0.75714016 0.9443159  0.94544154
     0.9701497  0.9240174  0.8425236  0.8602873  0.90820324 0.98897016
     0.9297225  0.8905812  0.9644633  0.87331146]
    Training complete
    
    Evaluating the model on the test set...
    Test Loss: 0.2011


However, I realised that this formatting was not displaying the data very informatively, so because I did not want to wait to train the model again as it was taking hours per run, so I used pandas to create a more visually readable table:


```python
import pandas as pd

# Attribute names 
attribute_names = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie",
    "Young"
]

# Validation accuracies for each epoch
epoch_1_accuracy = [0.9354778, 0.8435021, 0.8092104, 0.8323903, 0.9883756, 0.9565124,
                    0.81417364, 0.8181081, 0.8700815, 0.939101, 0.9642117, 0.8301891,
                    0.92022085, 0.94774705, 0.96062946, 0.9916465, 0.95958203, 0.9722725,
                    0.9110716, 0.8751137, 0.98152244, 0.9376734, 0.955547, 0.9385792,
                    0.95827365, 0.75376856, 0.96738195, 0.75965625, 0.94039077, 0.94457674,
                    0.9609407, 0.92246675, 0.8345131, 0.8601271, 0.9052845, 0.9853469,
                    0.9236335, 0.8854073, 0.96164525, 0.8600953]

epoch_2_accuracy = [0.9381355, 0.8535852, 0.805207, 0.82777, 0.98806435, 0.9545908,
                    0.8556167, 0.82208353, 0.86395156, 0.9518734, 0.9643533, 0.84239507,
                    0.9223344, 0.94940764, 0.9621988, 0.99541134, 0.96592265, 0.9768015,
                    0.91328573, 0.8661564, 0.9811292, 0.9353771, 0.9591198, 0.9387805,
                    0.95681435, 0.7545234, 0.9623498, 0.7628582, 0.9434604, 0.9436803,
                    0.9683381, 0.927348, 0.837907, 0.8657128, 0.9104584, 0.9889608,
                    0.9249922, 0.88690764, 0.9647149, 0.8743775]

epoch_3_accuracy = [0.93954456, 0.85644424, 0.8130256, 0.8421714, 0.989684, 0.9578114,
                    0.85269797, 0.8270244, 0.9111537, 0.9530812, 0.96284366, 0.8554881,
                    0.927074, 0.9556476, 0.96481556, 0.9933072, 0.9660643, 0.9781602,
                    0.9225264, 0.87979364, 0.983595, 0.9391923, 0.95906025, 0.9133864,
                    0.9594907, 0.76101494, 0.9608401, 0.75714016, 0.9443159, 0.94544154,
                    0.9701497, 0.9240174, 0.8425236, 0.8602873, 0.90820324, 0.98897016,
                    0.9297225, 0.8905812, 0.9644633, 0.87331146]

# Create DataFrame
df = pd.DataFrame({
    'Attribute': attribute_names,
    'Epoch 1 Accuracy': epoch_1_accuracy,
    'Epoch 2 Accuracy': epoch_2_accuracy,
    'Epoch 3 Accuracy': epoch_3_accuracy
})

# Display the DataFrame
df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Attribute</th>
      <th>Epoch 1 Accuracy</th>
      <th>Epoch 2 Accuracy</th>
      <th>Epoch 3 Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5_o_Clock_Shadow</td>
      <td>0.935478</td>
      <td>0.938136</td>
      <td>0.939545</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Arched_Eyebrows</td>
      <td>0.843502</td>
      <td>0.853585</td>
      <td>0.856444</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Attractive</td>
      <td>0.809210</td>
      <td>0.805207</td>
      <td>0.813026</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bags_Under_Eyes</td>
      <td>0.832390</td>
      <td>0.827770</td>
      <td>0.842171</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bald</td>
      <td>0.988376</td>
      <td>0.988064</td>
      <td>0.989684</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Bangs</td>
      <td>0.956512</td>
      <td>0.954591</td>
      <td>0.957811</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Big_Lips</td>
      <td>0.814174</td>
      <td>0.855617</td>
      <td>0.852698</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Big_Nose</td>
      <td>0.818108</td>
      <td>0.822084</td>
      <td>0.827024</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Black_Hair</td>
      <td>0.870081</td>
      <td>0.863952</td>
      <td>0.911154</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Blond_Hair</td>
      <td>0.939101</td>
      <td>0.951873</td>
      <td>0.953081</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Blurry</td>
      <td>0.964212</td>
      <td>0.964353</td>
      <td>0.962844</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Brown_Hair</td>
      <td>0.830189</td>
      <td>0.842395</td>
      <td>0.855488</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Bushy_Eyebrows</td>
      <td>0.920221</td>
      <td>0.922334</td>
      <td>0.927074</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Chubby</td>
      <td>0.947747</td>
      <td>0.949408</td>
      <td>0.955648</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Double_Chin</td>
      <td>0.960629</td>
      <td>0.962199</td>
      <td>0.964816</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Eyeglasses</td>
      <td>0.991646</td>
      <td>0.995411</td>
      <td>0.993307</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Goatee</td>
      <td>0.959582</td>
      <td>0.965923</td>
      <td>0.966064</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Gray_Hair</td>
      <td>0.972272</td>
      <td>0.976801</td>
      <td>0.978160</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Heavy_Makeup</td>
      <td>0.911072</td>
      <td>0.913286</td>
      <td>0.922526</td>
    </tr>
    <tr>
      <th>19</th>
      <td>High_Cheekbones</td>
      <td>0.875114</td>
      <td>0.866156</td>
      <td>0.879794</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Male</td>
      <td>0.981522</td>
      <td>0.981129</td>
      <td>0.983595</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Mouth_Slightly_Open</td>
      <td>0.937673</td>
      <td>0.935377</td>
      <td>0.939192</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Mustache</td>
      <td>0.955547</td>
      <td>0.959120</td>
      <td>0.959060</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Narrow_Eyes</td>
      <td>0.938579</td>
      <td>0.938781</td>
      <td>0.913386</td>
    </tr>
    <tr>
      <th>24</th>
      <td>No_Beard</td>
      <td>0.958274</td>
      <td>0.956814</td>
      <td>0.959491</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Oval_Face</td>
      <td>0.753769</td>
      <td>0.754523</td>
      <td>0.761015</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Pale_Skin</td>
      <td>0.967382</td>
      <td>0.962350</td>
      <td>0.960840</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Pointy_Nose</td>
      <td>0.759656</td>
      <td>0.762858</td>
      <td>0.757140</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Receding_Hairline</td>
      <td>0.940391</td>
      <td>0.943460</td>
      <td>0.944316</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Rosy_Cheeks</td>
      <td>0.944577</td>
      <td>0.943680</td>
      <td>0.945442</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Sideburns</td>
      <td>0.960941</td>
      <td>0.968338</td>
      <td>0.970150</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Smiling</td>
      <td>0.922467</td>
      <td>0.927348</td>
      <td>0.924017</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Straight_Hair</td>
      <td>0.834513</td>
      <td>0.837907</td>
      <td>0.842524</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Wavy_Hair</td>
      <td>0.860127</td>
      <td>0.865713</td>
      <td>0.860287</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Wearing_Earrings</td>
      <td>0.905285</td>
      <td>0.910458</td>
      <td>0.908203</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Wearing_Hat</td>
      <td>0.985347</td>
      <td>0.988961</td>
      <td>0.988970</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Wearing_Lipstick</td>
      <td>0.923633</td>
      <td>0.924992</td>
      <td>0.929723</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Wearing_Necklace</td>
      <td>0.885407</td>
      <td>0.886908</td>
      <td>0.890581</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Wearing_Necktie</td>
      <td>0.961645</td>
      <td>0.964715</td>
      <td>0.964463</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Young</td>
      <td>0.860095</td>
      <td>0.874378</td>
      <td>0.873311</td>
    </tr>
  </tbody>
</table>
</div>



Certain attributes like glasses (99.33%), male (98.36%), and bald (98.97%) achieved consistently high accuracy across all epochs. These attributes tend to have visually distinct features that the model could make out, making them easier to predict accurately.

Some attributes like Attractive (81.30%), Oval Face (76.10%), and Pointy Nose (75.71%) had lower accuracy. These features would seem to be more subjective (indeed, up to the discretion of those who labelled the dataset) or harder to detect, making them more difficult for the model to predict accurately.

Most attributes showed steady improvement across the epochs, with small increases in accuracy. 

Next, we moved onto using Logistic Regression to model the presence of multiple facial attributes in the CelebA dataset. The goal is to assess the conditional probability of each attribute, predicting whether a given attribute is present based on other features. For each of the 40 attributes, we train a logistic regression classifier and evaluate its performance using accuracy, precision, recall, and F1-score.

Logistic regression is a suitable choice for this multi-label binary classification task, allowing us to understand how each attribute correlates with the input features. 


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Paths to data
image_dir = '/Users/johnaziz/Downloads/archive-3/img_align_celeba/img_align_celeba/'
attr_path = '/Users/johnaziz/Downloads/archive-3/list_attr_celeba.csv'
partition_path = '/Users/johnaziz/Downloads/archive-3/list_eval_partition.csv'

# Load attribute and partition data
print("Loading data...")
attr_df = pd.read_csv(attr_path)
partition_df = pd.read_csv(partition_path)

# Merge attribute data with partition data on image_id
train_df = attr_df.merge(partition_df, on="image_id")

# Convert -1 to 0 for binary classification (0 = attribute absent, 1 = attribute present)
train_df.iloc[:, 1:-1] = train_df.iloc[:, 1:-1].replace(-1, 0)

# Attribute names
attribute_names = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie",
    "Young"
]

# Prepare an empty DataFrame to store the logistic regression results
logreg_results = pd.DataFrame(columns=['Attribute', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

# Iterate over each attribute to train logistic regression models and evaluate them
for i, attribute in enumerate(attribute_names):
    print(f"Processing {i+1}/{len(attribute_names)}: {attribute}")
    
    # Prepare the dataset: use all attributes except the target as features
    X = train_df.drop(columns=['image_id', 'partition', attribute])
    y = train_df[attribute]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"  Data split complete. Training Logistic Regression model for '{attribute}'...")
    
    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000)
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"  Error training model for {attribute}: {e}")
        continue

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)

    print(f"  Completed {attribute}. Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    # Create a temporary DataFrame to store the results for the current attribute
    temp_df = pd.DataFrame({
        'Attribute': [attribute],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-Score': [f1]
    })

    # Concatenate the temp_df to logreg_results using pd.concat
    logreg_results = pd.concat([logreg_results, temp_df], ignore_index=True)

# Now let's move to the conditional probability calculation

# Initialize an empty DataFrame to store conditional probabilities
conditional_probs = pd.DataFrame(index=attribute_names, columns=attribute_names)

# Iterate over each attribute pair and calculate conditional probabilities
for feature_a in attribute_names:
    for feature_b in attribute_names:
        # Calculate P(A | B), where A is feature_a and B is feature_b
        if feature_a != feature_b:
            # Calculate joint probability P(A ∩ B)
            joint_prob = ((train_df[feature_a] == 1) & (train_df[feature_b] == 1)).mean()  
            # Calculate marginal probability P(B)
            prob_b = (train_df[feature_b] == 1).mean()

            # Avoid division by zero
            if prob_b > 0:
                # Calculate conditional probability P(A | B)
                conditional_probs.at[feature_a, feature_b] = joint_prob / prob_b  
            else:
                conditional_probs.at[feature_a, feature_b] = None  # Undefined if P(B) == 0
        else:
            # P(A | A) = 1
            conditional_probs.at[feature_a, feature_b] = 1.0  

# Display the logistic regression results
print("\nLogistic Regression Results:")
print(logreg_results)

# Display the conditional probabilities table
print("\nConditional Probability Table:")
print(conditional_probs)

```

    Loading data...
    Processing 1/40: 5_o_Clock_Shadow
      Data split complete. Training Logistic Regression model for '5_o_Clock_Shadow'...
      Completed 5_o_Clock_Shadow. Accuracy: 0.9260, Precision: 0.7235, Recall: 0.5325, F1-Score: 0.6135
    Processing 2/40: Arched_Eyebrows
      Data split complete. Training Logistic Regression model for 'Arched_Eyebrows'...
      Completed Arched_Eyebrows. Accuracy: 0.8024, Precision: 0.6660, Recall: 0.5221, F1-Score: 0.5853
    Processing 3/40: Attractive
      Data split complete. Training Logistic Regression model for 'Attractive'...
      Completed Attractive. Accuracy: 0.7862, Precision: 0.7783, Recall: 0.8132, F1-Score: 0.7954
    Processing 4/40: Bags_Under_Eyes
      Data split complete. Training Logistic Regression model for 'Bags_Under_Eyes'...
      Completed Bags_Under_Eyes. Accuracy: 0.8327, Precision: 0.6577, Recall: 0.3975, F1-Score: 0.4955
    Processing 5/40: Bald
      Data split complete. Training Logistic Regression model for 'Bald'...
      Completed Bald. Accuracy: 0.9781, Precision: 0.5485, Recall: 0.1243, F1-Score: 0.2027
    Processing 6/40: Bangs
      Data split complete. Training Logistic Regression model for 'Bangs'...
      Completed Bangs. Accuracy: 0.8471, Precision: 0.6180, Recall: 0.0264, F1-Score: 0.0506
    Processing 7/40: Big_Lips
      Data split complete. Training Logistic Regression model for 'Big_Lips'...
      Completed Big_Lips. Accuracy: 0.7740, Precision: 0.5880, Recall: 0.2170, F1-Score: 0.3170
    Processing 8/40: Big_Nose
      Data split complete. Training Logistic Regression model for 'Big_Nose'...
      Completed Big_Nose. Accuracy: 0.8328, Precision: 0.7050, Recall: 0.5019, F1-Score: 0.5864
    Processing 9/40: Black_Hair
      Data split complete. Training Logistic Regression model for 'Black_Hair'...
      Completed Black_Hair. Accuracy: 0.8007, Precision: 0.6251, Recall: 0.4077, F1-Score: 0.4935
    Processing 10/40: Blond_Hair
      Data split complete. Training Logistic Regression model for 'Blond_Hair'...
      Completed Blond_Hair. Accuracy: 0.8674, Precision: 0.5866, Recall: 0.3654, F1-Score: 0.4503
    Processing 11/40: Blurry
      Data split complete. Training Logistic Regression model for 'Blurry'...
      Completed Blurry. Accuracy: 0.9524, Precision: 0.6818, Recall: 0.0230, F1-Score: 0.0446
    Processing 12/40: Brown_Hair
      Data split complete. Training Logistic Regression model for 'Brown_Hair'...
      Completed Brown_Hair. Accuracy: 0.8048, Precision: 0.5628, Recall: 0.2938, F1-Score: 0.3861
    Processing 13/40: Bushy_Eyebrows
      Data split complete. Training Logistic Regression model for 'Bushy_Eyebrows'...
      Completed Bushy_Eyebrows. Accuracy: 0.8719, Precision: 0.6365, Recall: 0.2353, F1-Score: 0.3436
    Processing 14/40: Chubby
      Data split complete. Training Logistic Regression model for 'Chubby'...
      Completed Chubby. Accuracy: 0.9558, Precision: 0.6873, Recall: 0.4234, F1-Score: 0.5240
    Processing 15/40: Double_Chin
      Data split complete. Training Logistic Regression model for 'Double_Chin'...
      Completed Double_Chin. Accuracy: 0.9642, Precision: 0.6840, Recall: 0.4364, F1-Score: 0.5329
    Processing 16/40: Eyeglasses
      Data split complete. Training Logistic Regression model for 'Eyeglasses'...
      Completed Eyeglasses. Accuracy: 0.9373, Precision: 0.5446, Recall: 0.1155, F1-Score: 0.1906
    Processing 17/40: Goatee
      Data split complete. Training Logistic Regression model for 'Goatee'...
      Completed Goatee. Accuracy: 0.9518, Precision: 0.6408, Recall: 0.5241, F1-Score: 0.5766
    Processing 18/40: Gray_Hair
      Data split complete. Training Logistic Regression model for 'Gray_Hair'...
      Completed Gray_Hair. Accuracy: 0.9626, Precision: 0.6080, Recall: 0.3263, F1-Score: 0.4247
    Processing 19/40: Heavy_Makeup
      Data split complete. Training Logistic Regression model for 'Heavy_Makeup'...
      Completed Heavy_Makeup. Accuracy: 0.9063, Precision: 0.8400, Recall: 0.9361, F1-Score: 0.8855
    Processing 20/40: High_Cheekbones
      Data split complete. Training Logistic Regression model for 'High_Cheekbones'...
      Completed High_Cheekbones. Accuracy: 0.8508, Precision: 0.8311, Recall: 0.8430, F1-Score: 0.8370
    Processing 21/40: Male
      Data split complete. Training Logistic Regression model for 'Male'...
      Completed Male. Accuracy: 0.9323, Precision: 0.8955, Recall: 0.9479, F1-Score: 0.9209
    Processing 22/40: Mouth_Slightly_Open
      Data split complete. Training Logistic Regression model for 'Mouth_Slightly_Open'...
      Completed Mouth_Slightly_Open. Accuracy: 0.7633, Precision: 0.7601, Recall: 0.7526, F1-Score: 0.7563
    Processing 23/40: Mustache
      Data split complete. Training Logistic Regression model for 'Mustache'...
      Completed Mustache. Accuracy: 0.9639, Precision: 0.6205, Recall: 0.3195, F1-Score: 0.4218
    Processing 24/40: Narrow_Eyes
      Data split complete. Training Logistic Regression model for 'Narrow_Eyes'...
      Completed Narrow_Eyes. Accuracy: 0.8848, Precision: 0.5538, Recall: 0.0077, F1-Score: 0.0152
    Processing 25/40: No_Beard
      Data split complete. Training Logistic Regression model for 'No_Beard'...
      Completed No_Beard. Accuracy: 0.9452, Precision: 0.9655, Recall: 0.9690, F1-Score: 0.9673
    Processing 26/40: Oval_Face
      Data split complete. Training Logistic Regression model for 'Oval_Face'...
      Completed Oval_Face. Accuracy: 0.7535, Precision: 0.6342, Recall: 0.3393, F1-Score: 0.4421
    Processing 27/40: Pale_Skin
      Data split complete. Training Logistic Regression model for 'Pale_Skin'...
      Completed Pale_Skin. Accuracy: 0.9581, Precision: 1.0000, Recall: 0.0000, F1-Score: 0.0000
    Processing 28/40: Pointy_Nose
      Data split complete. Training Logistic Regression model for 'Pointy_Nose'...
      Completed Pointy_Nose. Accuracy: 0.7344, Precision: 0.5779, Recall: 0.1794, F1-Score: 0.2738
    Processing 29/40: Receding_Hairline
      Data split complete. Training Logistic Regression model for 'Receding_Hairline'...
      Completed Receding_Hairline. Accuracy: 0.9230, Precision: 0.5646, Recall: 0.1318, F1-Score: 0.2137
    Processing 30/40: Rosy_Cheeks
      Data split complete. Training Logistic Regression model for 'Rosy_Cheeks'...
      Completed Rosy_Cheeks. Accuracy: 0.9355, Precision: 0.5728, Recall: 0.1337, F1-Score: 0.2168
    Processing 31/40: Sideburns
      Data split complete. Training Logistic Regression model for 'Sideburns'...
      Completed Sideburns. Accuracy: 0.9555, Precision: 0.6561, Recall: 0.4359, F1-Score: 0.5238
    Processing 32/40: Smiling
      Data split complete. Training Logistic Regression model for 'Smiling'...
      Completed Smiling. Accuracy: 0.8526, Precision: 0.8566, Recall: 0.8346, F1-Score: 0.8454
    Processing 33/40: Straight_Hair
      Data split complete. Training Logistic Regression model for 'Straight_Hair'...
      Completed Straight_Hair. Accuracy: 0.7948, Precision: 0.5342, Recall: 0.0946, F1-Score: 0.1607
    Processing 34/40: Wavy_Hair
      Data split complete. Training Logistic Regression model for 'Wavy_Hair'...
      Completed Wavy_Hair. Accuracy: 0.7663, Precision: 0.6356, Recall: 0.6317, F1-Score: 0.6336
    Processing 35/40: Wearing_Earrings
      Data split complete. Training Logistic Regression model for 'Wearing_Earrings'...
      Completed Wearing_Earrings. Accuracy: 0.8295, Precision: 0.6239, Recall: 0.2767, F1-Score: 0.3834
    Processing 36/40: Wearing_Hat
      Data split complete. Training Logistic Regression model for 'Wearing_Hat'...
      Completed Wearing_Hat. Accuracy: 0.9522, Precision: 0.6090, Recall: 0.0812, F1-Score: 0.1432
    Processing 37/40: Wearing_Lipstick
      Data split complete. Training Logistic Regression model for 'Wearing_Lipstick'...
      Completed Wearing_Lipstick. Accuracy: 0.9295, Precision: 0.9224, Recall: 0.9290, F1-Score: 0.9257
    Processing 38/40: Wearing_Necklace
      Data split complete. Training Logistic Regression model for 'Wearing_Necklace'...
      Completed Wearing_Necklace. Accuracy: 0.8798, Precision: 0.5626, Recall: 0.0681, F1-Score: 0.1215
    Processing 39/40: Wearing_Necktie
      Data split complete. Training Logistic Regression model for 'Wearing_Necktie'...
      Completed Wearing_Necktie. Accuracy: 0.9301, Precision: 0.5950, Recall: 0.1294, F1-Score: 0.2125
    Processing 40/40: Young
      Data split complete. Training Logistic Regression model for 'Young'...
      Completed Young. Accuracy: 0.8496, Precision: 0.8612, Recall: 0.9599, F1-Score: 0.9079
    
    Logistic Regression Results:
                  Attribute  Accuracy  Precision    Recall  F1-Score
    0      5_o_Clock_Shadow  0.925962   0.723488  0.532543  0.613502
    1       Arched_Eyebrows  0.802443   0.666038  0.522085  0.585341
    2            Attractive  0.786229   0.778317  0.813206  0.795379
    3       Bags_Under_Eyes  0.832651   0.657713  0.397470  0.495499
    4                  Bald  0.978060   0.548544  0.124312  0.202691
    5                 Bangs  0.847137   0.617978  0.026370  0.050582
    6              Big_Lips  0.774013   0.587991  0.216992  0.316999
    7              Big_Nose  0.832848   0.704993  0.501934  0.586382
    8            Black_Hair  0.800740   0.625139  0.407711  0.493539
    9            Blond_Hair  0.867399   0.586620  0.365433  0.450332
    10               Blurry  0.952394   0.681818  0.023041  0.044577
    11           Brown_Hair  0.804788   0.562797  0.293798  0.386060
    12       Bushy_Eyebrows  0.871866   0.636534  0.235325  0.343616
    13               Chubby  0.955750   0.687326  0.423423  0.524024
    14          Double_Chin  0.964215   0.684036  0.436412  0.532861
    15           Eyeglasses  0.937340   0.544627  0.115533  0.190628
    16               Goatee  0.951826   0.640791  0.524054  0.576573
    17            Gray_Hair  0.962562   0.608035  0.326340  0.424725
    18         Heavy_Makeup  0.906269   0.840046  0.936121  0.885485
    19      High_Cheekbones  0.850790   0.831138  0.842963  0.837009
    20                 Male  0.932330   0.895451  0.947872  0.920916
    21  Mouth_Slightly_Open  0.763327   0.760125  0.752591  0.756339
    22             Mustache  0.963944   0.620489  0.319544  0.421844
    23          Narrow_Eyes  0.884822   0.553846  0.007702  0.015193
    24             No_Beard  0.945188   0.965535  0.969041  0.967285
    25            Oval_Face  0.753504   0.634236  0.339278  0.442074
    26            Pale_Skin  0.958095   1.000000  0.000000  0.000000
    27          Pointy_Nose  0.734403   0.577898  0.179415  0.273819
    28    Receding_Hairline  0.923001   0.564581  0.131800  0.213710
    29          Rosy_Cheeks  0.935464   0.572785  0.133727  0.216831
    30            Sideburns  0.955479   0.656085  0.435852  0.523759
    31              Smiling  0.852616   0.856551  0.834577  0.845421
    32        Straight_Hair  0.794842   0.534228  0.094593  0.160727
    33            Wavy_Hair  0.766264   0.635623  0.631652  0.633631
    34     Wearing_Earrings  0.829492   0.623875  0.276733  0.383400
    35          Wearing_Hat  0.952172   0.609023  0.081162  0.143236
    36     Wearing_Lipstick  0.929492   0.922419  0.929015  0.925705
    37     Wearing_Necklace  0.879763   0.562604  0.068122  0.121529
    38      Wearing_Necktie  0.930133   0.595016  0.129360  0.212517
    39                Young  0.849556   0.861239  0.959867  0.907882
    
    Conditional Probability Table:
                        5_o_Clock_Shadow Arched_Eyebrows Attractive  \
    5_o_Clock_Shadow                 1.0        0.028545   0.092004   
    Arched_Eyebrows             0.068573             1.0   0.375102   
    Attractive                  0.424276        0.720059        1.0   
    Bags_Under_Eyes             0.396785        0.143113    0.13436   
    Bald                        0.024782         0.00538   0.001377   
    Bangs                       0.061068        0.135071    0.17246   
    Big_Lips                    0.187955        0.413237    0.26688   
    Big_Nose                    0.416681        0.175541   0.120039   
    Black_Hair                  0.360632        0.238547   0.240973   
    Blond_Hair                  0.014745        0.222721   0.201593   
    Blurry                      0.030156        0.023239   0.012106   
    Brown_Hair                  0.191153        0.217138   0.257211   
    Bushy_Eyebrows              0.357435        0.131725   0.156569   
    Chubby                      0.050808        0.023054   0.003756   
    Double_Chin                 0.048366        0.019375   0.003689   
    Eyeglasses                  0.070217        0.004234   0.011528   
    Goatee                      0.161441        0.017434   0.028084   
    Gray_Hair                   0.017099        0.009004   0.002494   
    Heavy_Makeup                0.000444        0.741727   0.613543   
    High_Cheekbones             0.228948        0.583953   0.527443   
    Male                        0.999112        0.083435   0.227086   
    Mouth_Slightly_Open         0.389456        0.538843   0.493889   
    Mustache                    0.093533        0.012997   0.014263   
    Narrow_Eyes                 0.125333        0.127935   0.092148   
    No_Beard                    0.281711        0.959641   0.906504   
    Oval_Face                   0.175298        0.274099   0.369449   
    Pale_Skin                   0.019853        0.058976   0.059962   
    Pointy_Nose                 0.246669        0.390996   0.377134   
    Receding_Hairline           0.062711         0.07225   0.032562   
    Rosy_Cheeks                 0.002576        0.157867   0.105323   
    Sideburns                   0.225884        0.012128   0.033939   
    Smiling                     0.386303        0.559771    0.55413   
    Straight_Hair               0.268431        0.171732   0.224861   
    Wavy_Hair                   0.155711        0.474746   0.417343   
    Wearing_Earrings            0.009682        0.380126   0.236399   
    Wearing_Hat                 0.069106        0.012886   0.019425   
    Wearing_Lipstick            0.000977          0.8533   0.706201   
    Wearing_Necklace            0.015012        0.242781   0.144983   
    Wearing_Necktie             0.144608        0.015567   0.033101   
    Young                       0.791215        0.875411   0.931871   
    
                        Bags_Under_Eyes      Bald     Bangs  Big_Lips  Big_Nose  \
    5_o_Clock_Shadow           0.215558  0.122718  0.044775  0.086748  0.197449   
    Arched_Eyebrows            0.186773  0.063998  0.237911  0.458174  0.199827   
    Attractive                 0.336607  0.031449  0.583119  0.568023  0.262312   
    Bags_Under_Eyes                 1.0  0.513965   0.14963  0.200287  0.468095   
    Bald                       0.056387       1.0       0.0  0.021503   0.07065   
    Bangs                      0.110867       0.0       1.0  0.173783  0.106617   
    Big_Lips                   0.235753  0.230702  0.276075       1.0   0.30013   
    Big_Nose                    0.53665  0.738289  0.164968  0.292323       1.0   
    Black_Hair                 0.241495  0.012976   0.20538  0.290151  0.301751   
    Blond_Hair                 0.073155   0.00022   0.23104  0.162919  0.046426   
    Blurry                     0.036698  0.037387  0.046175  0.036425  0.035925   
    Brown_Hair                 0.168171   0.00066  0.271549  0.194179  0.108006   
    Bushy_Eyebrows             0.215895  0.095667  0.082223  0.155068  0.231227   
    Chubby                     0.129807  0.401364  0.011202  0.060797  0.189831   
    Double_Chin                0.128312  0.343083  0.011983  0.043108   0.16043   
    Eyeglasses                 0.044492  0.243237  0.030675  0.042985  0.127326   
    Goatee                     0.108406  0.247196  0.013644  0.070985  0.147845   
    Gray_Hair                  0.109806  0.242138  0.012732  0.010557  0.112846   
    Heavy_Makeup               0.105221   0.00044  0.525677  0.513662  0.141889   
    High_Cheekbones            0.526589  0.447548  0.517438  0.494968  0.507892   
    Male                       0.709453  0.996261  0.226318  0.270145  0.745665   
    Mouth_Slightly_Open        0.541331  0.481856  0.494643  0.526883  0.537061   
    Mustache                   0.084809   0.14647  0.009704  0.052701  0.116487   
    Narrow_Eyes                0.182406  0.141192   0.12498  0.179522  0.155611   
    No_Beard                    0.72767  0.551572  0.951252  0.849954  0.664723   
    Oval_Face                  0.163273  0.316252  0.285877   0.19424  0.197849   
    Pale_Skin                  0.029798  0.011876  0.063174  0.057108  0.024202   
    Pointy_Nose                0.176808  0.113042  0.289068  0.319299  0.149129   
    Receding_Hairline          0.142306  0.330548   0.00013  0.089864  0.178866   
    Rosy_Cheeks                0.019809  0.004618  0.101469  0.099498  0.040955   
    Sideburns                  0.102229   0.14581  0.016445  0.040217  0.111436   
    Smiling                    0.593181  0.513086  0.544108  0.493512  0.573154   
    Straight_Hair               0.22779  0.015615  0.227653  0.181613  0.187242   
    Wavy_Hair                  0.202673    0.0011   0.39477  0.419084  0.208498   
    Wearing_Earrings           0.114076   0.03079  0.242828  0.276868  0.147887   
    Wearing_Hat                0.046036  0.005058  0.007848  0.043394  0.073891   
    Wearing_Lipstick           0.192805  0.001759  0.665408  0.645485  0.198565   
    Wearing_Necklace            0.08867  0.013635  0.211892  0.210372   0.10165   
    Wearing_Necktie            0.173575  0.375192  0.015631  0.041468   0.16889   
    Young                      0.583193  0.231581  0.791136  0.853664  0.558612   
    
                        Black_Hair Blond_Hair  ... Sideburns   Smiling  \
    5_o_Clock_Shadow      0.167519   0.011073  ...  0.444231  0.089056   
    Arched_Eyebrows       0.266195   0.401794  ...  0.057298  0.310006   
    Attractive            0.516195   0.698129  ...    0.3078  0.589102   
    Bags_Under_Eyes        0.20649   0.101124  ...  0.370076  0.251718   
    Bald                  0.001217   0.000033  ...  0.057909  0.023887   
    Bangs                 0.130116   0.236634  ...  0.044109  0.171078   
    Big_Lips              0.292024   0.265084  ...  0.171369  0.246506   
    Big_Nose                0.2958   0.073575  ...  0.462486   0.27884   
    Black_Hair                 1.0     0.0001  ...  0.314613  0.238131   
    Blond_Hair            0.000062        1.0  ...  0.008647  0.181194   
    Blurry                0.036021   0.045359  ...  0.028736  0.038712   
    Brown_Hair            0.023436    0.04069  ...  0.147873  0.214991   
    Bushy_Eyebrows        0.302814   0.016376  ...  0.325443  0.141109   
    Chubby                0.061623   0.007371  ...    0.1677  0.066265   
    Double_Chin           0.036248   0.007838  ...  0.070749  0.068579   
    Eyeglasses            0.058013   0.017243  ...  0.112499  0.054285   
    Goatee                0.088278   0.003102  ...  0.571054  0.043862   
    Gray_Hair             0.000206   0.017076  ...   0.04734  0.043975   
    Heavy_Makeup          0.344157   0.675149  ...  0.000175  0.476088   
    High_Cheekbones       0.462576   0.599373  ...  0.180802   0.80782   
    Male                   0.51898   0.058333  ...  0.999039  0.346046   
    Mouth_Slightly_Open   0.462783   0.567488  ...  0.330335  0.761255   
    Mustache              0.064264     0.0005  ...  0.315748  0.027634   
    Narrow_Eyes           0.108888   0.114065  ...  0.114945  0.141273   
    No_Beard               0.76962   0.986959  ...   0.01118  0.878334   
    Oval_Face             0.309787   0.338192  ...  0.193467  0.380356   
    Pale_Skin             0.028965    0.07124  ...  0.010219  0.028105   
    Pointy_Nose           0.240221   0.401461  ...  0.186217   0.29728   
    Receding_Hairline     0.079778   0.033986  ...  0.102018  0.087192   
    Rosy_Cheeks           0.048131   0.149885  ...  0.002096  0.122485   
    Sideburns             0.074311   0.003302  ...       1.0  0.037228   
    Smiling               0.479823   0.590234  ...  0.317582       1.0   
    Straight_Hair         0.289879   0.213721  ...  0.175474  0.210773   
    Wavy_Hair               0.2487   0.463229  ...  0.181763  0.355824   
    Wearing_Earrings      0.190564   0.278158  ...  0.009346  0.258004   
    Wearing_Hat           0.008789   0.005503  ...  0.107695  0.034218   
    Wearing_Lipstick      0.412919   0.810159  ...  0.000262  0.566423   
    Wearing_Necklace      0.098057     0.2361  ...  0.014848  0.153263   
    Wearing_Necktie        0.08345   0.006437  ...  0.136781  0.072111   
    Young                  0.86419   0.826235  ...  0.619879  0.759217   
    
                        Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat  \
    5_o_Clock_Shadow         0.143148  0.054152         0.005695    0.158484   
    Arched_Eyebrows          0.220004  0.396624         0.537177    0.070992   
    Attractive               0.552982  0.669313          0.64129    0.205439   
    Bags_Under_Eyes          0.223604  0.129742         0.123524    0.194337   
    Bald                     0.001682  0.000077         0.003658    0.002343   
    Bangs                    0.165577  0.187245         0.194822    0.024547   
    Big_Lips                 0.209843  0.315782         0.352884    0.215624   
    Big_Nose                  0.21072  0.153018         0.183588    0.357608   
    Black_Hair               0.332789  0.186195         0.241326     0.04339   
    Blond_Hair               0.151769  0.214522         0.217891    0.016806   
    Blurry                   0.035171  0.044637         0.024532    0.066816   
    Brown_Hair                0.19253  0.295827         0.206343    0.029028   
    Bushy_Eyebrows           0.192554  0.110759         0.087026     0.11153   
    Chubby                   0.042513  0.024574          0.03062    0.118456   
    Double_Chin              0.035455  0.021114         0.023357    0.075983   
    Eyeglasses               0.056558  0.030875         0.023435    0.142697   
    Goatee                   0.042584  0.025655          0.01126    0.156244   
    Gray_Hair                0.037871  0.016944         0.019229    0.009676   
    Heavy_Makeup             0.322604   0.61669         0.743259    0.083316   
    High_Cheekbones           0.43681  0.539031          0.69686    0.264005   
    Male                     0.484842  0.183677         0.035244    0.699837   
    Mouth_Slightly_Open      0.469495  0.512603         0.618273    0.491139   
    Mustache                 0.031382  0.017098         0.008726     0.11153   
    Narrow_Eyes              0.117451  0.125541         0.120467    0.097983   
    No_Beard                 0.850883  0.921784          0.97808    0.638317   
    Oval_Face                0.287836  0.309218         0.354609    0.191892   
    Pale_Skin                0.049192  0.049626         0.034957    0.028315   
    Pointy_Nose              0.262754   0.36649         0.376999    0.124771   
    Receding_Hairline        0.049713  0.034243          0.08666    0.001732   
    Rosy_Cheeks              0.051963  0.112953         0.176272    0.009371   
    Sideburns                0.047582  0.032142         0.002795    0.125586   
    Smiling                  0.487566  0.536776          0.65835    0.340395   
    Straight_Hair                 1.0  0.017886         0.144555    0.020676   
    Wavy_Hair                0.027426       1.0         0.433065    0.068446   
    Wearing_Earrings         0.131045  0.256024              1.0    0.099613   
    Wearing_Hat              0.004808  0.010379         0.025551         1.0   
    Wearing_Lipstick         0.413836  0.734184         0.851656    0.118863   
    Wearing_Necklace         0.102719  0.184959         0.254311    0.065798   
    Wearing_Necktie          0.112548  0.018766         0.003344    0.039214   
    Young                    0.816967  0.828478         0.799718    0.706559   
    
                        Wearing_Lipstick Wearing_Necklace Wearing_Necktie  \
    5_o_Clock_Shadow             0.00023         0.013567        0.221015   
    Arched_Eyebrows             0.482213         0.527114        0.057154   
    Attractive                  0.766097         0.604263        0.233302   
    Bags_Under_Eyes             0.083487         0.147513        0.488325   
    Bald                        0.000084         0.002489        0.115802   
    Bangs                       0.213488         0.261189        0.032582   
    Big_Lips                    0.328998         0.411954         0.13732   
    Big_Nose                    0.098574         0.193875        0.544733   
    Black_Hair                   0.20911         0.190784        0.274572   
    Blond_Hair                  0.253785         0.284149        0.013101   
    Blurry                      0.021313         0.048328        0.039166   
    Brown_Hair                  0.248143         0.200618        0.100326   
    Bushy_Eyebrows              0.078796         0.075824        0.221423   
    Chubby                      0.009936         0.027215        0.220337   
    Double_Chin                 0.008807         0.024646        0.216332   
    Eyeglasses                   0.01097         0.035122        0.180695   
    Goatee                      0.000136         0.012323        0.121844   
    Gray_Hair                   0.007836         0.019869        0.216603   
    Heavy_Makeup                0.799457         0.651748        0.001018   
    High_Cheekbones             0.604628         0.619155        0.372115   
    Male                        0.005464         0.060491        0.997624   
    Mouth_Slightly_Open         0.538536         0.589893        0.440334   
    Mustache                    0.000063         0.010878        0.114852   
    Narrow_Eyes                 0.106671         0.141573        0.126731   
    No_Beard                    0.999122         0.973347        0.679609   
    Oval_Face                   0.359139         0.215149        0.199633   
    Pale_Skin                   0.056867          0.04315        0.021993   
    Pointy_Nose                 0.398224         0.354915        0.183342   
    Receding_Hairline           0.044141         0.053346          0.2295   
    Rosy_Cheeks                 0.135465         0.157187        0.006584   
    Sideburns                   0.000031         0.006824        0.106299   
    Smiling                     0.577987         0.600851        0.478075   
    Straight_Hair               0.182552         0.174086        0.322563   
    Wavy_Hair                    0.49662         0.480673        0.082474   
    Wearing_Earrings            0.340574          0.39072        0.008689   
    Wearing_Hat                 0.012192          0.02593        0.026134   
    Wearing_Lipstick                 1.0         0.826878        0.001969   
    Wearing_Necklace            0.215222              1.0        0.000543   
    Wearing_Necktie             0.000303         0.000321             1.0   
    Young                       0.884981         0.790029        0.396416   
    
                            Young  
    5_o_Clock_Shadow     0.113664  
    Arched_Eyebrows      0.302111  
    Attractive           0.617345  
    Bags_Under_Eyes      0.154217  
    Bald                 0.006718  
    Bangs                0.155008  
    Big_Lips             0.265711  
    Big_Nose             0.169351  
    Black_Hair           0.267262  
    Blond_Hair           0.158058  
    Blurry               0.043016  
    Brown_Hair            0.22764  
    Bushy_Eyebrows       0.158262  
    Chubby                0.02043  
    Double_Chin          0.011331  
    Eyeglasses           0.035117  
    Goatee                0.04879  
    Gray_Hair            0.002424  
    Heavy_Makeup         0.451076  
    High_Cheekbones      0.450547  
    Male                 0.341005  
    Mouth_Slightly_Open  0.479577  
    Mustache             0.026746  
    Narrow_Eyes          0.109396  
    No_Beard             0.858761  
    Oval_Face            0.311643  
    Pale_Skin            0.047718  
    Pointy_Nose          0.299495  
    Receding_Hairline    0.051699  
    Rosy_Cheeks          0.071446  
    Sideburns            0.045281  
    Smiling              0.473107  
    Straight_Hair         0.22008  
    Wavy_Hair             0.34223  
    Wearing_Earrings     0.195299  
    Wearing_Hat           0.04426  
    Wearing_Lipstick     0.540444  
    Wearing_Necklace     0.125576  
    Wearing_Necktie      0.037261  
    Young                     1.0  
    
    [40 rows x 40 columns]


This is an interesting probability table to dig through and see the associations between various different attributes in the dataset.

For example, Smiling and Mouth Slightly Open had a relatively high conditional probability, meaning these attributes often occur together.

Conversely, attributes like Bald and Young had very low conditional probabilities, reflecting the mutual exclusivity of these features. Similarly, Heavy Makeup and Wearing Lipstick suggest a strong association between these features.

In this next phase of the project, we applied clustering techniques to understand patterns in the dataset using MiniBatchKMeans. We also used PCA (Principal Component Analysis) to reduce the dimensions of the data for visualisation.

We loaded the attribute and partition data, ensuring that attributes like '-1' (indicating absence of a feature) were converted to '0'.
The two datasets were merged based on image_id, and unnecessary columns like image_id and partition were removed, leaving only the binary attributes.

We sampled 10% of the data to make the clustering task more computationally efficient. We scaled the attributes using StandardScaler to ensure that all features contributed equally to the clustering process.

We employed MiniBatchKMeans (a faster version of KMeans for large datasets) to cluster the sampled dataset into 5 clusters.

The model identified patterns in the facial attributes by grouping them into clusters based on their similarity.

To visualize the high-dimensional data, we applied PCA to reduce the dimensions of the dataset to 2 components.

This allowed us to visualize the clusters formed by MiniBatchKMeans on a 2D plane, helping us observe how well the attributes were 


```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Define the paths
attr_path = '/Users/johnaziz/Downloads/archive-3/list_attr_celeba.csv'
partition_path = '/Users/johnaziz/Downloads/archive-3/list_eval_partition.csv'

# Load the attribute data
print("Loading attribute data...")
attributes_df = pd.read_csv(attr_path)
attributes_df.columns = attributes_df.columns.str.strip()  # Remove any leading/trailing whitespace from column names

# Convert -1 to 0 for binary classification (0 = attribute absent, 1 = attribute present)
attributes_df.iloc[:, 1:] = attributes_df.iloc[:, 1:].replace(-1, 0)

# Load the partition data
print("Loading partition data...")
partition_df = pd.read_csv(partition_path)
partition_df.columns = partition_df.columns.str.strip()  # Remove any leading/trailing whitespace from column names

# Merge the two datasets on the image_id
print("Merging attributes and partition data...")
train_df = pd.merge(attributes_df, partition_df, on="image_id")

# Drop unnecessary columns like 'image_id' and 'partition'
attributes_only = train_df.drop(columns=['image_id', 'partition'])

# Sample a subset of the data (e.g., 10% of the dataset) for testing
attributes_sample = attributes_only.sample(frac=0.1, random_state=42)

# Scale the attributes
scaler = StandardScaler()
attributes_scaled = scaler.fit_transform(attributes_sample)

# Verbose print to monitor progress
print("Starting MiniBatchKMeans clustering on a 10% sample...")

# Track the time taken for MiniBatchKMeans
start_time = time.time()

# MiniBatchKMeans Clustering
minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=1000)
minibatch_kmeans.fit(attributes_scaled)

# Add the cluster labels to the DataFrame
attributes_sample = pd.DataFrame(attributes_scaled, columns=attributes_only.columns)
attributes_sample['Cluster'] = minibatch_kmeans.labels_

# Print completion time for MiniBatchKMeans
end_time = time.time()
print(f"MiniBatchKMeans clustering completed in {end_time - start_time:.2f} seconds.")

# Show the size of each cluster
print("Cluster sizes:")
print(attributes_sample['Cluster'].value_counts())

# Verbose print for PCA
print("Starting PCA for dimensionality reduction...")

# Track the time taken for PCA
start_time = time.time()

# Visualize the clusters using PCA 
pca = PCA(n_components=2)
pca_result = pca.fit_transform(attributes_scaled)

# Print completion time for PCA
end_time = time.time()
print(f"PCA completed in {end_time - start_time:.2f} seconds.")

# Plot the PCA result with cluster labels
plt.figure(figsize=(10,7))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=attributes_sample['Cluster'], palette='viridis', s=50)
plt.title('MiniBatchKMeans Clusters of CelebA Attributes (Sampled Data)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

print("Plotting complete.")


```

    Loading attribute data...
    Loading partition data...
    Merging attributes and partition data...
    Starting MiniBatchKMeans clustering on a 10% sample...


    /Users/johnaziz/anaconda3/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:1930: FutureWarning: The default value of `n_init` will change from 3 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      super()._check_params_vs_input(X, default_n_init=3)


    MiniBatchKMeans clustering completed in 0.11 seconds.
    Cluster sizes:
    Cluster
    3    6578
    1    5723
    4    4735
    2    1791
    0    1433
    Name: count, dtype: int64
    Starting PCA for dimensionality reduction...
    PCA completed in 0.07 seconds.



    
![png](https://raw.githubusercontent.com/johnaziz23/blog/main/_posts/Computer%20Vision%20Faces_files/figure-html/cell-6-output-4.png)
    


    Plotting complete.


The MiniBatchKMeans identified 5 distinct clusters, and we plotted the clusters using a scatterplot, with colours representing different clusters. This gave us a visual insight into how the attributes were grouped.

Next, we sought to etend the clustering analysis of the attributes by adding additional steps for visualizing, validating, and exploring the clusters.

We calculate the average attributes for each cluster, to generate insights into the feature distribution across clusters.

We produced an elbow plot to help assess whether 5 clusters is the optimal number.

We produced t-SNE plot shows the clusters in a reduced 2D space, providing a visual validation of cluster separation.

Next, we produced a system for displaying sample images from each cluster to help give a more visual and real-world sense of the groupings formed during clustering.

Finally, we produced an inter-cluster Distance Heatmap: to highlight how different or similar the clusters are from one another based on their centroids.


```python
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
import time
from PIL import Image

# Paths
attr_path = '/Users/johnaziz/Downloads/archive-3/list_attr_celeba.csv'
partition_path = '/Users/johnaziz/Downloads/archive-3/list_eval_partition.csv'
image_dir = '/Users/johnaziz/Downloads/archive-3/img_align_celeba/img_align_celeba/'  

# Load the attribute and partition data
print("Loading attribute data...")
attributes_df = pd.read_csv(attr_path)
attributes_df.columns = attributes_df.columns.str.strip()

print("Loading partition data...")
partition_df = pd.read_csv(partition_path)
partition_df.columns = partition_df.columns.str.strip()

# Convert -1 to 0 for binary classification
attributes_df.iloc[:, 1:] = attributes_df.iloc[:, 1:].replace(-1, 0)

# Merge attributes and partition data
print("Merging attributes and partition data...")
train_df = pd.merge(attributes_df, partition_df, on="image_id")

# Drop unnecessary columns
attributes_only = train_df.drop(columns=['image_id', 'partition'])

# Sample a subset of the data (e.g., 10% of the dataset) for testing
attributes_sample = attributes_only.sample(frac=0.1, random_state=42)

# Scale the attributes
scaler = StandardScaler()
attributes_scaled = scaler.fit_transform(attributes_sample)

# Start MiniBatchKMeans clustering
print("Starting MiniBatchKMeans clustering...")
minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=1000)
minibatch_kmeans.fit(attributes_scaled)

# Add the cluster labels to both attributes_sample and train_df (matching rows)
attributes_sample = pd.DataFrame(attributes_scaled, columns=attributes_only.columns)
attributes_sample['Cluster'] = minibatch_kmeans.labels_

# Now, add the 'Cluster' column back to the original train_df, matching the sampled rows
train_df.loc[attributes_sample.index, 'Cluster'] = minibatch_kmeans.labels_

# Show the size of each cluster
print("Cluster sizes:")
print(attributes_sample['Cluster'].value_counts())

### 1. Analyze Cluster Characteristics
print("\nMean attribute values for each cluster:")
cluster_means = attributes_sample.groupby('Cluster').mean()
print(cluster_means)

# Plot heatmap of the cluster means
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_means.T, cmap="coolwarm", annot=True)
plt.title('Heatmap of Attribute Means per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Attribute')
plt.show()

### 2. Elbow Method to Determine Optimal Number of Clusters
inertias = []
k_values = range(1, 11)

for k in k_values:
    minibatch_kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000)
    minibatch_kmeans.fit(attributes_scaled)
    inertias.append(minibatch_kmeans.inertia_)

# Plot the elbow method
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertias, '-o')
plt.title('Elbow Method to Determine Optimal Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.xticks(k_values)
plt.show()

### 3. t-SNE for Visualization
print("Starting t-SNE for dimensionality reduction...")
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(attributes_scaled)

# Plot the t-SNE result with cluster labels
plt.figure(figsize=(10,7))
sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=attributes_sample['Cluster'], palette='viridis', s=50)
plt.title('t-SNE Clusters of CelebA Attributes (Sampled Data)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

### 4. Display Sample Images from Each Cluster
def display_cluster_images(cluster_num, num_images=5):
    cluster_data = train_df[train_df['Cluster'] == cluster_num]
    image_ids = cluster_data['image_id'].sample(n=num_images, random_state=42).tolist()
    
    plt.figure(figsize=(15, 5))
    for i, image_id in enumerate(image_ids):
        
        if not image_id.endswith('.jpg'):
            img_path = os.path.join(image_dir, f"{image_id}.jpg")
        else:
            img_path = os.path.join(image_dir, image_id)  
        img = Image.open(img_path)
        plt.subplot(1, num_images, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Cluster {cluster_num}')
    plt.show()

# Display 5 sample images from each cluster
for cluster in attributes_sample['Cluster'].unique():
    display_cluster_images(cluster, num_images=5)

### 5. Inter-cluster Distance Heatmap
centroids = minibatch_kmeans.cluster_centers_
distances = cdist(centroids, centroids)

# Plot heatmap of the inter-cluster distances
plt.figure(figsize=(8, 6))
sns.heatmap(distances, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Inter-cluster Distance Heatmap')
plt.xlabel('Cluster')
plt.ylabel('Cluster')
plt.show()

```

    Loading attribute data...
    Loading partition data...
    Merging attributes and partition data...
    Starting MiniBatchKMeans clustering...


    /Users/johnaziz/anaconda3/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:1930: FutureWarning: The default value of `n_init` will change from 3 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      super()._check_params_vs_input(X, default_n_init=3)


    Cluster sizes:
    Cluster
    3    6578
    1    5723
    4    4735
    2    1791
    0    1433
    Name: count, dtype: int64
    
    Mean attribute values for each cluster:
             5_o_Clock_Shadow  Arched_Eyebrows  Attractive  Bags_Under_Eyes  \
    Cluster                                                                   
    0               -0.096651        -0.501052   -0.987884         0.868152   
    1               -0.353430         0.518107    0.513912        -0.211347   
    2                0.852991        -0.433202   -0.509239         0.341458   
    3                0.351109        -0.496026   -0.386724         0.209468   
    4               -0.353985         0.378375    0.407695        -0.427446   
    
                 Bald     Bangs  Big_Lips  Big_Nose  Black_Hair  Blond_Hair  ...  \
    Cluster                                                                  ...   
    0        1.199428 -0.352821 -0.269672  1.181759   -0.364762   -0.358408  ...   
    1       -0.149169  0.214955  0.137953 -0.228385   -0.076991    0.410953  ...   
    2        0.252597 -0.308701  0.061158  0.654210    0.280353   -0.402915  ...   
    3       -0.092912 -0.111544 -0.257996  0.101412    0.150809   -0.314270  ...   
    4       -0.149169  0.118696  0.250158 -0.469945   -0.112104    0.200761  ...   
    
             Sideburns   Smiling  Straight_Hair  Wavy_Hair  Wearing_Earrings  \
    Cluster                                                                    
    0        -0.136335  0.161372      -0.039087  -0.481216         -0.354227   
    1        -0.240641  0.946678      -0.070434   0.383172          0.649148   
    2         2.375972 -0.335077      -0.095092  -0.363045         -0.409659   
    3        -0.234626 -0.170862       0.139904  -0.365973         -0.401716   
    4        -0.240641 -0.828939      -0.061430   0.328251          0.035632   
    
             Wearing_Hat  Wearing_Lipstick  Wearing_Necklace  Wearing_Necktie  \
    Cluster                                                                     
    0          -0.059743         -0.875502         -0.279066         1.340242   
    1          -0.193574          0.964261          0.404673        -0.283932   
    2           0.386841         -0.942611         -0.306015         0.213793   
    3           0.163961         -0.926164         -0.297424         0.101707   
    4          -0.122055          0.742693          0.124284        -0.284595   
    
                Young  
    Cluster            
    0       -1.666307  
    1        0.181071  
    2       -0.322223  
    3        0.014282  
    4        0.387476  
    
    [5 rows x 40 columns]



    
![png](https://raw.githubusercontent.com/johnaziz23/blog/main/_posts/Computer%20Vision%20Faces_files/figure-html/cell-7-output-4.png)
    


    /Users/johnaziz/anaconda3/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:1930: FutureWarning: The default value of `n_init` will change from 3 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      super()._check_params_vs_input(X, default_n_init=3)
    /Users/johnaziz/anaconda3/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:1930: FutureWarning: The default value of `n_init` will change from 3 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      super()._check_params_vs_input(X, default_n_init=3)
    /Users/johnaziz/anaconda3/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:1930: FutureWarning: The default value of `n_init` will change from 3 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      super()._check_params_vs_input(X, default_n_init=3)
    /Users/johnaziz/anaconda3/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:1930: FutureWarning: The default value of `n_init` will change from 3 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      super()._check_params_vs_input(X, default_n_init=3)
    /Users/johnaziz/anaconda3/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:1930: FutureWarning: The default value of `n_init` will change from 3 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      super()._check_params_vs_input(X, default_n_init=3)
    /Users/johnaziz/anaconda3/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:1930: FutureWarning: The default value of `n_init` will change from 3 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      super()._check_params_vs_input(X, default_n_init=3)
    /Users/johnaziz/anaconda3/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:1930: FutureWarning: The default value of `n_init` will change from 3 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      super()._check_params_vs_input(X, default_n_init=3)
    /Users/johnaziz/anaconda3/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:1930: FutureWarning: The default value of `n_init` will change from 3 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      super()._check_params_vs_input(X, default_n_init=3)
    /Users/johnaziz/anaconda3/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:1930: FutureWarning: The default value of `n_init` will change from 3 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      super()._check_params_vs_input(X, default_n_init=3)
    /Users/johnaziz/anaconda3/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:1930: FutureWarning: The default value of `n_init` will change from 3 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      super()._check_params_vs_input(X, default_n_init=3)



    
![png](https://raw.githubusercontent.com/johnaziz23/blog/main/_posts/Computer%20Vision%20Faces_files/figure-html/cell-7-output-6.png)

    


    Starting t-SNE for dimensionality reduction...



    
![png](https://raw.githubusercontent.com/johnaziz23/blog/main/_posts/Computer%20Vision%20Faces_files/figure-html/cell-7-output-8.png)

![png](https://raw.githubusercontent.com/johnaziz23/blog/main/_posts/Computer%20Vision%20Faces_files/figure-html/cell-7-output-9.png)

![png](https://raw.githubusercontent.com/johnaziz23/blog/main/_posts/Computer%20Vision%20Faces_files/figure-html/cell-7-output-10.png)

![png](https://raw.githubusercontent.com/johnaziz23/blog/main/_posts/Computer%20Vision%20Faces_files/figure-html/cell-7-output-11.png)

![png](https://raw.githubusercontent.com/johnaziz23/blog/main/_posts/Computer%20Vision%20Faces_files/figure-html/cell-7-output-12.png)

![png](https://raw.githubusercontent.com/johnaziz23/blog/main/_posts/Computer%20Vision%20Faces_files/figure-html/cell-7-output-13.png)
    


1. Cluster Sizes:

The dataset was clustered into five distinct groups using MiniBatchKMeans. Here are the sizes of each cluster:

* Cluster 3: 6578 samples

* Cluster 1: 5723 samples

* Cluster 4: 4735 samples

* Cluster 2: 1791 samples

* Cluster 0: 1433 samples

2. Cluster Attribute Analysis:

From the attribute analysis, we can infer general trends:

Cluster 0: Likely older, bald individuals with prominent facial features (e.g., big noses).

Cluster 1: Younger individuals with positive attributes like smiling and wearing lipstick, potentially more fashionable.

Cluster 2: Individuals with masculine features (sideburns, 5 o'clock shadow) but less likely to smile.

Cluster 3: A mix of features without strong positive or negative traits, potentially a catch-all cluster.

Cluster 4: Attractive individuals with more fashionable traits (smiling, lipstick, earrings).

The model helps in identifying distinct groupings based on facial attributes. 

The Elbow Method graph shows how inertia (sum of squared distances within clusters) decreases as the number of clusters increases. The key observations are:

The inertia drops significantly between 1 and 5 clusters.

After 5 clusters, the rate of decrease seems to flatten out somewhat, indicating diminishing returns in reducing the sum of squared distances.

From this graph, 5 clusters does appear to be a good choice, as this is where the "elbow" occurs—further increases in the number of clusters do not significantly reduce the inertia. 

The t-SNE plot gives a visual confirmation that the clustering has produced distinct groupings of attributes, though some inter-cluster mixing might still occur, particularly between Clusters 1 and 2.

Next, we moved on to training a generative adverserial network to try and generate new faces from the faces already in the dataset. This is an interesting approach that I explored a few years ago on the Artbreeder website.

The GAN consists of two main components: a Generator, which creates fake images from random noise, and a Discriminator, which attempts to differentiate between real and fake images. Over the course of training, the GAN learns to generate increasingly realistic images of faces by continuously improving both components.



```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
from torch.cuda.amp import GradScaler, autocast  # AMP for mixed precision training
import os
import psutil

# Set this to detect anomalies when debugging
# torch.autograd.set_detect_anomaly(False)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CelebA Dataset path
dataset_path = '/Users/johnaziz/Downloads/archive-3/img_align_celeba'

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Load CelebA dataset
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  

# Generator network (smaller version for faster training)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 128, 4, 1, 0, bias=False),  
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh()  # Output in the range [-1, 1]
        )

    def forward(self, input):
        return self.main(input)

# Discriminator network (smaller version for faster training)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # Output a probability score between 0 and 1
        )

    def forward(self, input):
        return self.main(input)

# Initialize the networks
netG = Generator().to(device)
netD = Discriminator().to(device)

# Loss function
criterion = nn.BCELoss()

# Optimizers
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Mixed Precision Training Scaler
scaler = GradScaler()

# Training loop
num_epochs = 10 
noise_dim = 100

print("Starting Training...")

for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}] started...")
    for i, data in enumerate(dataloader, 0):
        
        ############################
        # (1) Update Discriminator
        ############################
        netD.zero_grad()

        real_images = data[0].to(device)
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, device=device)

        # Mixed precision for real images
        with autocast():
            output_real = netD(real_images).view(-1)
            lossD_real = criterion(output_real, real_labels)
        
        # Backprop for real images
        scaler.scale(lossD_real).backward()

        # Generate fake images
        noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
        fake_images = netG(noise)
        fake_labels = torch.zeros(batch_size, device=device)

        # Mixed precision for fake images
        with autocast():
            output_fake = netD(fake_images.detach()).view(-1)
            lossD_fake = criterion(output_fake, fake_labels)
        
        # Backprop for fake images
        scaler.scale(lossD_fake).backward()

        # Update discriminator
        scaler.step(optimizerD)
        scaler.update()

        ############################
        # (2) Update Generator
        ############################
        netG.zero_grad()

        real_labels.fill_(1)  # Generator tries to make the discriminator think the fake images are real
        
        with autocast():
            output = netD(fake_images).view(-1)
            lossG = criterion(output, real_labels)
        
        # Backprop for generator
        scaler.scale(lossG).backward()

        # Update generator
        scaler.step(optimizerG)
        scaler.update()

        # Print loss values every 100 steps
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], "
                  f"Loss D: {(lossD_real + lossD_fake).item():.4f}, Loss G: {lossG.item():.4f}")

    # Save generated images occasionally to monitor progress
    if epoch % 5 == 0:
        with torch.no_grad():
            fake_images = netG(noise).detach().cpu()
            vutils.save_image(fake_images, f"fake_images_epoch_{epoch}.png", normalize=True)

# Save the generator model at the end of training
model_save_path = '/Users/johnaziz/Downloads/generator.pth'
torch.save(netG.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

![png](https://raw.githubusercontent.com/johnaziz23/blog/main/_posts/Computer%20Vision%20Faces_files/figure-html/creenshot 2024-09-25 at 19.38.26.png)

Over the 10 epochs, both the Generator and Discriminator improved significantly. Early in training, the Discriminator could easily distinguish fake from real images (indicated by high Generator loss), but as training progressed, the Generator’s images became more realistic, and the Discriminator found it more challenging to differentiate between the two (indicated by lower Discriminator loss).

Loss oscillations in later epochs indicated the model still had room for improvement, particularly in stabilizing the training. GANs are known for being challenging to train due to their adversarial nature, and the balance between the Generator and Discriminator is crucial for achieving optimal results.

Next, we tested out the generation, including generating some faces:


```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import torch.nn.functional as F

# Function to denormalize the images back to the [0, 1] range
def denormalize(img_tensor):
    img_tensor = img_tensor * 0.5 + 0.5  # Reverse normalization (mean=0.5, std=0.5)
    return img_tensor

# Function to upscale images
def upscale_images(images, upscale_factor=4):
    # Rescale images using bicubic interpolation
    upscaled_images = F.interpolate(images, scale_factor=upscale_factor, mode='bicubic', align_corners=True)
    return upscaled_images

# Function to generate and visualize fake faces with upscaling
def generate_faces(generator, num_images=16, noise_dim=100, device="cpu", upscale_factor=4):
    # Set generator to evaluation mode
    generator.eval()
    
    # Generate random noise
    noise = torch.randn(num_images, noise_dim, 1, 1, device=device)
    
    # Generate fake images from noise
    with torch.no_grad():
        fake_images = generator(noise).detach().cpu()
    
    # Denormalize images to [0, 1] range
    fake_images = denormalize(fake_images)

    # Upscale the images to a higher resolution
    upscaled_images = upscale_images(fake_images, upscale_factor=upscale_factor)
    
    # Plot the upscaled generated images in a grid
    grid = make_grid(upscaled_images, nrow=int(np.sqrt(num_images)), normalize=True)
    plt.figure(figsize=(10, 10))  # Larger figure for better resolution
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))  # Convert (C, H, W) to (H, W, C)
    plt.axis('off')
    plt.show()

# Load the trained generator model
model_path = '/Users/johnaziz/Downloads/generator.pth'
netG = Generator().to(torch.device('cpu'))
netG.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
print(f"Generator model loaded from {model_path}")

# Generate 16 faces with 4x upscaling (e.g., 32x32 -> 128x128)
generate_faces(netG, num_images=16, noise_dim=100, device=torch.device('cpu'), upscale_factor=4)'''
    
![png](https://raw.githubusercontent.com/johnaziz23/blog/main/_posts/Computer%20Vision%20Faces_files/figure-html/cell-9-output-2.png)

The model has converged, and it is producing faces, although the quality is more like a horror movie. However as a proof of concept and a basis to iterate upon, the model is working.
