import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

# ---------------------
# Load the dataset
# ---------------------
def load_dataset():
    q_H_init_list = np.load('data/2000_q_H_init_list.dat', allow_pickle=True)
    labeled_pcd_init_list = np.load('data/2000_labeled_pcd_init_list.dat', allow_pickle=True)
    q_H_goal_list = np.load('data/2000_q_H_goal_list.dat', allow_pickle=True)
    labeled_pcd_goal_list = np.load('data/2000_labeled_pcd_goal_list.dat', allow_pickle=True)
    return q_H_init_list, labeled_pcd_init_list, q_H_goal_list, labeled_pcd_goal_list

# ---------------------
# Preprocess the dataset (Normalization)
# ---------------------
def normalize_point_cloud(pcd):
    """
    Normalize a point cloud to fit within a unit sphere.
    pcd shape: (N, 4) with columns [x, y, z, label].
    """
    points = pcd[:, :3]
    centroid = np.mean(points, axis=0)
    points -= centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    points /= (max_dist + 1e-9)
    return np.hstack((points, pcd[:, 3:]))

def preprocess_point_clouds(labeled_pcd_list):
    """
    Normalize each point cloud in the dataset.
    labeled_pcd_list shape: (num_samples, N, 4)
    """
    return np.array([normalize_point_cloud(pcd) for pcd in labeled_pcd_list])

def normalize_joint_angles(q_H_init_list, q_H_goal_list):
    q_H_min = np.array([-3.141368118925281, -0.248997453133789, -2.6643015908664056, 0.0])
    q_H_max = np.array([ 3.1415394736319917,  1.2392816988875348, -1.3229245882839409, 2.541304])
    q_H_init_list_norm = (q_H_init_list - q_H_min) / (q_H_max - q_H_min)
    q_H_goal_list_norm = (q_H_goal_list - q_H_min) / (q_H_max - q_H_min)
    return q_H_init_list_norm, q_H_goal_list_norm, q_H_min, q_H_max

# ---------------------
# Define the Dataset
# ---------------------
class ArmConfigDataset(Dataset):
    def __init__(self, q_H_init_list, labeled_pcd_init_list, q_H_goal_list):
        """
        This dataset will only use q_H_init and labeled_pcd_init to predict q_H_goal.
        We assume q_H_goal_list is already normalized if q_H_init_list is normalized.
        """
        self.q_H_init_list = q_H_init_list
        self.labeled_pcd_init_list = labeled_pcd_init_list
        self.q_H_goal_list = q_H_goal_list

    def __len__(self):
        return len(self.q_H_init_list)

    def __getitem__(self, idx):
        q_H_init = torch.tensor(self.q_H_init_list[idx], dtype=torch.float32)   # (4,)
        pcd_init = self.labeled_pcd_init_list[idx]  # (N,4)
        q_H_goal = torch.tensor(self.q_H_goal_list[idx], dtype=torch.float32)    # (4,)

        # pcd_init should be transposed for PointNet: (features, N)
        pcd_init_t = torch.tensor(pcd_init.T, dtype=torch.float32)  # shape (4, N)

        return q_H_init, pcd_init_t, q_H_goal

# ---------------------
# Define the Encoder and Predictor
# ---------------------
class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(PointNetEncoder, self).__init__()
        self.mlp1 = nn.Conv1d(4, 64, 1)
        self.mlp2 = nn.Conv1d(64, 128, 1)
        self.mlp3 = nn.Conv1d(128, latent_dim, 1)
        self.relu = nn.ReLU()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # x: (batch_size, 4, N)
        x = self.relu(self.mlp1(x))
        x = self.relu(self.mlp2(x))
        x = self.relu(self.mlp3(x))
        x = self.global_max_pool(x)  # (batch_size, latent_dim, 1)
        x = torch.flatten(x, start_dim=1)  # (batch_size, latent_dim)
        return x

class ArmConfigPredictor(nn.Module):
    def __init__(self, latent_dim=512, input_dim=4, output_dim=4):
        super(ArmConfigPredictor, self).__init__()
        # Map q_H_init (4D) -> 128D
        self.fc_q_init = nn.Linear(input_dim, 128)

        # Concatenate 128 (q_H_init) + 512 (latent vector) = 640
        self.fc1 = nn.Linear(640, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()

    def forward(self, q_H_init, latent_vector):
        # Map q_H_init from 4D to 128D
        q_init_emb = self.relu(self.fc_q_init(q_H_init))  # (batch_size, 128)

        # Concatenate q_init_emb (128D) with latent_vector (512D)
        x = torch.cat([q_init_emb, latent_vector], dim=1)  # (batch_size, 640)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CombinedModel(nn.Module):
    def __init__(self, encoder, predictor):
        super(CombinedModel, self).__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(self, q_H_init, pcd_init):
        # Encode the point cloud into a latent vector
        latent_vector = self.encoder(pcd_init)
        # Predict q_H_goal
        q_H_goal_pred = self.predictor(q_H_init, latent_vector)
        return q_H_goal_pred

# ---------------------
# Train the Combined Model
# ---------------------
def train_model(dataset, model, device, batch_size=32, epochs=50, learning_rate=1e-3):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for q_H_init, pcd_init, q_H_goal in dataloader:
            q_H_init = q_H_init.to(device)
            pcd_init = pcd_init.to(device)
            q_H_goal = q_H_goal.to(device)

            optimizer.zero_grad()
            predicted_q_H_goal = model(q_H_init, pcd_init)
            loss = criterion(predicted_q_H_goal, q_H_goal)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.6f}")

    print("Training complete.")
    return model


if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    q_H_init_list, labeled_pcd_init_list, q_H_goal_list, labeled_pcd_goal_list = load_dataset()

    # Preprocess point clouds
    labeled_pcd_init_list = preprocess_point_clouds(labeled_pcd_init_list)

    # Normalize joint angles
    q_H_init_list_norm, q_H_goal_list_norm, q_H_min, q_H_max = normalize_joint_angles(q_H_init_list, q_H_goal_list)

    # Initialize dataset
    dataset = ArmConfigDataset(q_H_init_list_norm, labeled_pcd_init_list, q_H_goal_list_norm)

    # Initialize encoder and predictor
    encoder = PointNetEncoder(latent_dim=512)
    predictor = ArmConfigPredictor(latent_dim=512, input_dim=4, output_dim=4)
    model = CombinedModel(encoder, predictor)

    # Train the model end-to-end
    model = train_model(dataset, model, device, batch_size=32, epochs=100, learning_rate=1e-3)

    # Save the trained model
    Path("models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "models/arm_config_predictor.pth")
    print("Model saved successfully.")
