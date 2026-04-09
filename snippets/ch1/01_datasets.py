import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

delta_timestamps = {
    # 0.2, and 0.1 seconds *before* each frame
    "observation.images.wrist_camera": [-0.2, -0.1, 0.0]
}

# Optionally, use StreamingLeRobotDataset to avoid downloading the dataset
dataset = LeRobotDataset(
    "lerobot/svla_so101_pickplace",
    delta_timestamps=delta_timestamps
)

# Streams frames from the Hugging Face Hub without loading into memory
streaming_dataset = StreamingLeRobotDataset(
    "lerobot/svla_so101_pickplace",
    delta_timestamps=delta_timestamps
)

# Get the 100th frame in the dataset by 
sample = dataset[100]
print(sample)
# {
# 'observation.state': tensor([...]), 
# 'action': tensor([...]), 
# 'observation.images.wrist_camera': tensor([3, C, H, W]), for delta timesteps
# ...
# }

batch_size=16
# wrap the dataset in a DataLoader to use process it batches for training purposes
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size
)

# Iterate over the DataLoader in a training loop
num_epochs = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

for epoch in range(num_epochs):
    for batch in data_loader:
        # Move data to the appropriate device (e.g., GPU)
        observations = batch["observation.state"].to(device)
        actions = batch["action"].to(device)
        images = batch["observation.images.wrist_camera"].to(device)

        # Next, you can do amazing_model.forward(batch)
        ...