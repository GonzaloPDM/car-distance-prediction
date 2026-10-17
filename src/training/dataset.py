import torch
import numpy as np

from torchvision.datasets.cityscapes import Cityscapes
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


# -------------------------------- DATASET TRANSFORMATION -------------------------------- #
# 
# The following code is the implementation of the transformation from cityscape format to maskrcnn format, so that images and target
# can be correctly read
#

class CityscapesInstance(Cityscapes):
    def __getitem__(self, index: int):
        # Tuple image, target from Cityscape without processing
        image, target = super().__getitem__(index)
        
        # Add image_id to target dict
        target["image_id"] = index
            
        return image, target


# We are going to map the Cityscapes IDs to the desired ones
CITYSCAPES_ID_TO_MODEL_ID = {}
VEHICLE_INSTANCE_IDS = [26, 27, 28, 29, 30, 32, 33]
MODEL_ID_TO_NAME = {0: "background"}

current_model_id = 1

for c in Cityscapes.classes:
    if c.id in VEHICLE_INSTANCE_IDS:
        CITYSCAPES_ID_TO_MODEL_ID[c.id] = current_model_id
        MODEL_ID_TO_NAME[current_model_id] = c.name
        current_model_id += 1

# "car":        1, (0, 0, 142)
# "truck":      2, (0, 0, 70)
# "bus":        3, (0, 60, 100)
# "caravan":    4, (0, 0, 90)
# "trailer":    5, (0, 0, 110)
# "motorcycle": 6, (0, 0, 230)
# "bicycle":    7, (119, 11, 32)

# Now we implement a function that returns an empty target
def _get_empty_target(image):
    target = {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "masks": torch.zeros((0, *F.get_size(image)), dtype=torch.uint8),
        "image_id": torch.tensor([0]),
        "area": torch.zeros((0,), dtype=torch.float32),
        "iscrowd":torch.zeros((0,), dtype=torch.int64)
    }
    return image, target


def transform_cityscapes_to_maskrcnn(image, target_pil):
    # TRANSFORM IMAGE
    # From PIL to tensor (C, H, W)
    image = F.to_image(image)
    image = F.to_dtype(image, torch.float, scale=True)
    
    # TRANSFORM TARGET
    mask_np = np.array(target_pil)  
    obj_ids = np.unique(mask_np)

    masks_list = []
    labels_list = []
    
    for obj_id in obj_ids:
        if obj_id == 0: continue 

        class_id_cityscapes = obj_id // 1000
        
        if class_id_cityscapes in CITYSCAPES_ID_TO_MODEL_ID:
            label_model = CITYSCAPES_ID_TO_MODEL_ID[class_id_cityscapes]
            mask_instance = (mask_np == obj_id).astype(np.uint8)
            masks_list.append(torch.from_numpy(mask_instance))
            labels_list.append(torch.tensor(label_model, dtype=torch.int64))

    # If none of the considered vehicles appears in the image
    if not masks_list:
        return _get_empty_target(image)

    # Convert to torch format
    masks = torch.stack(masks_list, dim=0)      # (N, H, W)
    labels = torch.stack(labels_list, dim=0)    # (N,)
    boxes = masks_to_boxes(masks)

    # Filter incorrect data
    keep = (boxes[:,3] > boxes[:,1]) & (boxes[:,2] > boxes[:,0])
    
    if not keep.any():
        return _get_empty_target(image)
    boxes = boxes[keep]
    masks = masks[keep]
    labels = labels[keep]

    # Calculate area and iscrowd
    area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
    iscrowd = torch.zeros((len(boxes)), dtype=torch.int64)

    # Create proper target dictionary
    target = {}
    target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(image))
    target["labels"] = labels
    target["masks"] = tv_tensors.Mask(masks)
    target["area"] = area
    target["iscrowd"] = iscrowd
    
    return image, target
