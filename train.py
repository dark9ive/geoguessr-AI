import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from transformers import ViTForImageClassification, ViTFeatureExtractor
from const import PIC_FOLDER
from tqdm import tqdm
import argparse
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_confusion_matrix(matrix, labels, epoch=None):
    os.makedirs("matrix", exist_ok=True)
    
    filepath = f"matrix/evaluate.txt"

    if epoch is not None:
        filepath = f"matrix/epoch_{epoch+1}.txt"
    with open(filepath, "w") as f:
        f.write("\t" + "\t".join(labels) + "\n")
        for i, row in enumerate(matrix):
            row_percentage = [round(value * 100) for value in row]  # Convert to percentages and round to integer
            f.write(labels[i] + "\t" + "\t".join(f"{value}" for value in row_percentage) + "\n")



def train_model(model, train_loader, valid_loader, criterion, optimizer, scaler, num_epochs, num_classes, class_labels, start_epoch=0):
    for epoch in range(start_epoch, start_epoch+num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs+start_epoch}", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs).logits
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs+start_epoch}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f'checkpoint/epoch_{epoch+1}.pth')
        
        #   Eval
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        confusion_matrix = np.zeros((num_classes, num_classes))

        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader, desc=f"Validation", unit="batch"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                with autocast():
                    outputs = model(inputs).logits
                    loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)
        
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1


        val_loss = val_running_loss / len(valid_loader.dataset)
        val_acc = val_running_corrects.double() / len(valid_loader.dataset)
        
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
        save_confusion_matrix(confusion_matrix, class_labels, epoch)


def evaluate_model(model, test_loader, num_classes, class_labels):
    model.eval()
    test_running_corrects = 0
    
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).logits
            _, preds = torch.max(outputs, 1)
            test_running_corrects += torch.sum(preds == labels.data)
            
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    test_acc = test_running_corrects.double() / len(test_loader.dataset)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
    save_confusion_matrix(confusion_matrix, class_labels)



def main(lr, dr, start_epoch, num_epochs, batch_size, seed):

    img_height, img_width = 224, 224

    
    #   Data preprocessing
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224-in21k')
    train_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])

    
    #   Load data
    full_dataset = datasets.ImageFolder(root=PIC_FOLDER)

    class_to_idx = full_dataset.class_to_idx
    class_labels = list(class_to_idx.keys())
    num_classes = len(class_labels)

    print(class_labels)

    
    #   Define split size
    train_size = int(0.7 * len(full_dataset))
    valid_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - valid_size


    #   Split dataset
    torch.manual_seed(seed)
    train_dataset, valid_dataset, test_dataset = random_split(full_dataset, [train_size, valid_size, test_size])
    train_dataset.dataset.transform = train_transform
    valid_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

    
    #   Load pretrained model
    model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224-in21k', num_labels=num_classes)
    model.classifier = nn.Sequential(
        nn.Dropout(dr),
        model.classifier,
        nn.Dropout(dr)
    )
    model = model.to(device)


    #   Define loss_fn and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    #optimizer = optim.AdamW(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scaler = GradScaler()

    #   Load weight
    if start_epoch != 0:
        model.load_state_dict(torch.load(f'checkpoint/epoch_{start_epoch}.pth'))
    
    #   Train
    train_model(model, train_loader, valid_loader, criterion, optimizer, scaler, num_epochs, num_classes, class_labels, start_epoch)

    #   Test
    evaluate_model(model, test_loader, num_classes, class_labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate a ViT model for image classification.")

    parser.add_argument("--num_epochs", "-e", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--start_epoch", "-se", type=int, default=0, help="Number of epoch to start.")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--seed", "-s", type=int, default=114514, help="Random seed of train/test split.")
    parser.add_argument("--dropout_rate", "-dr", type=float, default=0.3, help="Dropout rate for training.")

    args = parser.parse_args()
    main(lr=args.learning_rate, dr=args.dropout_rate, start_epoch=args.start_epoch, num_epochs=args.num_epochs, batch_size=args.batch_size, seed=args.seed)
