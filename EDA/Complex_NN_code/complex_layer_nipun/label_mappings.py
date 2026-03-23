"""
Label Mappings for Hierarchical SAR Classification

Classes (0-indexed):
    0: AG (Agriculture)     → Group 0 (Vegetation)
    1: FR (Forest)          → Group 0 (Vegetation)
    2: HD (High Density)    → Group 1 (Urban)
    3: HR (High-Rise)       → Group 1 (Urban)
    4: LD (Low Density)     → Group 1 (Urban)
    5: IR (Industrial)      → Group 1 (Urban)
    6: WR (Water)           → Group 2 (Water)

Groups:
    0: Vegetation (AG, FR)       → 2 classes
    1: Urban (HD, HR, LD, IR)    → 4 classes
    2: Water (WR)                → 1 class
"""

import torch


# Class names (0-indexed, matching dataset labels after -1)
CLASS_NAMES = [
    "AG",  # 0: Agriculture
    "FR",  # 1: Forest
    "HD",  # 2: High Density
    "HR",  # 3: High-Rise
    "LD",  # 4: Low Density
    "IR",  # 5: Industrial
    "WR",  # 6: Water
]

GROUP_NAMES = [
    "Vegetation",  # 0
    "Urban",       # 1
    "Water",       # 2
]

NUM_CLASSES = 7
NUM_GROUPS = 3


# Mapping: class_id → group_id
CLASS_TO_GROUP = torch.tensor([
    0,  # AG → Vegetation
    0,  # FR → Vegetation
    1,  # HD → Urban
    1,  # HR → Urban
    1,  # LD → Urban
    1,  # IR → Urban
    2,  # WR → Water
], dtype=torch.long)


# Mapping: class_id → subclass_id within group
# Vegetation: AG=0, FR=1
# Urban: HD=0, HR=1, LD=2, IR=3
# Water: WR=0
CLASS_TO_SUBCLASS = torch.tensor([
    0,  # AG → 0 in Vegetation
    1,  # FR → 1 in Vegetation
    0,  # HD → 0 in Urban
    1,  # HR → 1 in Urban
    2,  # LD → 2 in Urban
    3,  # IR → 3 in Urban
    0,  # WR → 0 in Water
], dtype=torch.long)


# Group sizes (number of classes per group)
GROUP_SIZES = [2, 4, 1]  # Vegetation, Urban, Water


# Classes belonging to each group
GROUP_TO_CLASSES = [
    [0, 1],        # Vegetation: AG, FR
    [2, 3, 4, 5],  # Urban: HD, HR, LD, IR
    [6],           # Water: WR
]


def get_group_labels(class_labels):
    """
    Convert class labels to group labels.
    
    Args:
        class_labels: Tensor of shape (B,) with values in [0, 6]
        
    Returns:
        group_labels: Tensor of shape (B,) with values in [0, 2]
    """
    device = class_labels.device
    return CLASS_TO_GROUP.to(device)[class_labels]


def get_subclass_labels(class_labels):
    """
    Convert class labels to subclass labels within their group.
    
    Args:
        class_labels: Tensor of shape (B,) with values in [0, 6]
        
    Returns:
        subclass_labels: Tensor of shape (B,) with subclass indices
    """
    device = class_labels.device
    return CLASS_TO_SUBCLASS.to(device)[class_labels]


def get_full_class_from_group_subclass(group_labels, subclass_labels):
    """
    Reconstruct full class labels from group and subclass labels.
    
    Args:
        group_labels: Tensor of shape (B,) with values in [0, 2]
        subclass_labels: Tensor of shape (B,) with subclass indices
        
    Returns:
        class_labels: Tensor of shape (B,) with values in [0, 6]
    """
    device = group_labels.device
    batch_size = group_labels.shape[0]
    
    class_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for g in range(NUM_GROUPS):
        mask = group_labels == g
        classes_in_group = torch.tensor(GROUP_TO_CLASSES[g], device=device)
        class_labels[mask] = classes_in_group[subclass_labels[mask]]
    
    return class_labels
