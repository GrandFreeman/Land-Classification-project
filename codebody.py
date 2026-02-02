
import warnings
warnings.filterwarnings('ignore')

import os
from tqdm import tqdm
import tensorflow as tf

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import  random_split
import torch.nn.functional as F

#===============def_frames import=========================

from def_frames import set_seed
from def_frames import CNN_ViT_Hybrid
from def_frames import evaluate
from def_frames import model_metrics
from def_frames import print_metrics

#=========================================================

from variables_set import data_dir 
from variables_set import dataset_url 
from preparation import pytorch_state_dict_url 
from preparation import pytorch_state_dict_name 
from preparation import pytorch_state_dict_path


print("Imported libraries")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ["CUDA_VISIBLE_DEVICES"] = ""


gpu_list = tf.config.list_physical_devices('GPU')
device = "gpu" if gpu_list != [] else "cpu"
print(f"TensorFlow {tf.__version__}  |  GPUs found: {tf.config.list_physical_devices('GPU')}")

SEED = 7331
set_seed(SEED)
print(f"Global seed set to {SEED} - Processes are now deterministic.")

if not os.path.exists(pytorch_state_dict_path):
    print("Unable to find the PyTorch model at give path. Please check...")
else:
    print(f"Found the pre-trained PyTorch model:\n{pytorch_state_dict_name} --at------> {pytorch_state_dict_path}")

#============== variables input ================
from variables_set import dataset_path

from variables_set import img_w, img_h
from variables_set import batch_size
from variables_set import num_classes
from variables_set import agri_class_labels

from variables_set import depth
from variables_set import attn_heads
from variables_set import embed_dim
#===============================================

train_transform = transforms.Compose([
    transforms.Resize((img_w, img_h)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
full_dataset = datasets.ImageFolder(dataset_path, transform=train_transform)
test_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

pytorch_model = CNN_ViT_Hybrid(num_classes=num_classes,
                      heads=attn_heads,
                      depth=depth,
                      embed_dim=embed_dim).to(device)

if device=="cpu":
    map_location=torch.device("cpu")
else:
    map_location=torch.device("cuda")

pytorch_model.load_state_dict(torch.load(pytorch_state_dict_path, map_location=map_location), strict=False)
print("Loaded model state dict, now getting predictions")

all_preds_pytorch = []
all_labels_pytorch = []
all_probs_pytorch = []

pytorch_model.eval()
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Step")):
#    for images, labels in test_loader:
        images = images.to(device)
        outputs = pytorch_model(images)
        preds = torch.argmax(outputs, dim=1)
        probs = F.softmax(outputs, dim=1)[:, 1]  # probability for class 1
        all_probs_pytorch.extend(probs.cpu())
        all_preds_pytorch.extend(preds.cpu().numpy().flatten())
        all_labels_pytorch.extend(labels.numpy())

evaluate(pytorch_model, test_loader, nn.CrossEntropyLoss(), device)

print_metrics(all_labels_pytorch, all_preds_pytorch, all_probs_pytorch, agri_class_labels, 
    'Pytorch CNN-Vit Hybrid Model'
)
