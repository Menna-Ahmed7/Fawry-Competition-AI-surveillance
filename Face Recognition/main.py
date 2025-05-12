import os
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
# MTCNN: for detection, InceptionResnetV1: Recognition
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import torch.nn.functional as F

# -------------------------
# 1. Create a custom dataset
# -------------------------
class FaceDataset(Dataset):
    def __init__(self, csv_file, mtcnn, transform=None):
        """
        csv_file: Path to CSV containing columns 'gt' and 'image_path'
        mtcnn: An instance of MTCNN to detect and crop faces.
        transform: Optional transform to be applied on the face tensor.
        """
        self.df = pd.read_csv(csv_file)
        self.mtcnn = mtcnn
        self.transform = transform
        # Sorted list of unique persons and mapping to indices
        self.persons = sorted(self.df['gt'].unique())
        self.person2idx = {p: idx for idx, p in enumerate(self.persons)}
    
    def __len__(self):
        return len(self.df)
    
    # for single image
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        person = row['gt']
        label = self.person2idx[person]
        
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a dummy tensor and label in case of error
            return torch.zeros(3, 160, 160), label
        
        # Use MTCNN to detect and crop the face
        face = self.mtcnn(img)
        if face is None:
            # If no face is *detected*, return a tensor of zeros.
            face = torch.zeros(3, 160, 160)
        # 
        if self.transform:
            face = self.transform(face)
        return face, label

def main():
    # -------------------------
    # 2. Prepare training data and dataloader
    # -------------------------
    mtcnn_train = MTCNN(image_size=160, margin=0, keep_all=False, post_process=True)
    # Getting data ready as : each face(image of face only*(cropped)*), label: 0,1,..
    train_dataset = FaceDataset('D:/ComputerEngineering/Fawry/Fawry-Competition-final/surveillance-for-retail-stores/face_identification/face_identification/trainset.csv', mtcnn_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    
    num_classes = len(train_dataset.persons)
    print(f"Number of classes: {num_classes}")
    
    # -------------------------
    # 3. Set up the model for fine-tuning
    # -------------------------
    model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=num_classes)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Testing
    model.load_state_dict(torch.load('D:/ComputerEngineering/Fawry/Fawry-Competition-final/Face Recognition/finetuned_facenet.pth', map_location=device))

    
    # -------------------------
    # 4. Training loop
    # -------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # epochs = 10  # Adjust the number of epochs as needed
    # for epoch in range(epochs):
    #     model.train()
    #     running_loss = 0.0
    #     for faces, labels in train_loader:
    #         faces, labels = faces.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #         outputs = model(faces)  # Forward pass
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item() * faces.size(0)
    #     epoch_loss = running_loss / len(train_loader.dataset)
    #     print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    # # Save the fine-tuned model (optional)
    # torch.save(model.state_dict(), 'finetuned_facenet.pth')
    # print("Model saved as finetuned_facenet.pth")
    
    # # -------------------------
    # 5. Evaluation Phase: Predict on test set
    # -------------------------
    
    test_df = pd.read_csv('D:/ComputerEngineering/Fawry/Fawry-Competition-final/surveillance-for-retail-stores/face_identification/face_identification/eval_set.csv')
    mtcnn_eval = MTCNN(image_size=160, margin=0, keep_all=False, post_process=True)
    model.eval()
    
    results = []
    confidence_threshold = 0.8  # Tuning threshold based on your validation set
    
    # Inverse mapping from class index to person label
    idx2person = {idx: person for person, idx in train_dataset.person2idx.items()}
    
    with torch.no_grad():
        for _, row in test_df.iterrows():
            image_path = 'D:/ComputerEngineering/Fawry/Fawry-Competition-final/surveillance-for-retail-stores/face_identification/face_identification/test/' + row['image_path']
            try:
                img = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                pred = "doesn't_exist"
                results.append({'gt': pred, 'image': image_path})
                continue
            face = mtcnn_eval(img)
            if face is None:
                pred = "doesn't_exist"
            else:
                face = face.unsqueeze(0).to(device)
                logits = model(face)
                probs = F.softmax(logits, dim=1)
                # pred_idx: person number
                max_prob, pred_idx = torch.max(probs, dim=1)
                if max_prob.item() < confidence_threshold:
                    pred = "doesn't_exist"
                else:
                    pred = idx2person[pred_idx.item()]
            results.append({'gt': pred, 'image': image_path})
    
    submission_df = pd.DataFrame(results)
    submission_df.to_csv('submission.csv', index=False)
    print("Submission file saved as submission.csv")

if __name__ == '__main__':
    main()
