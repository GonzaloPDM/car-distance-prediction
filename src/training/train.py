import os
import json
import torch

import src.training.vendor.utils as utils
from src.training.vendor.engine import train_one_epoch, evaluate
from src.detection.model import get_model_instance_segmentation
from src.training.dataset import CityscapesInstance, transform_cityscapes_to_maskrcnn, CITYSCAPES_ID_TO_MODEL_ID


# -------------------------------- FINE-TUNNING -------------------------------- #
#
# Fine tunning of the maskrcnn model, using cityscape dataset. It can be resumed from the last checkpoint.
#
def train(data_root, model_dir, checkpoints_dir, num_epochs=15, batch_size=3):
    """
    Args:
        data_root: Root folder of the Cityscapes dataset
        model_dir: Folder where the best model is saved (maskrcnn_cityscapes.pth)
        checkpoints_dir: Folder where the last epoch checkpoint and progress file are saved
        num_epochs: Total number of epochs
        batch_size: Training batch size
    """
    PATH_PROGRESS_FILE = os.path.join(checkpoints_dir, "last_progress.json")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # train on the accelerator (or on the CPU if an accelerator is not available)
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')

    # A class for every considered kind of vehicle is needed, as well as background (7+1)
    num_classes = len(CITYSCAPES_ID_TO_MODEL_ID) + 1

    # Load dataset class
    dataset_train = CityscapesInstance(
        root=data_root, 
        split = 'train',
        mode = 'fine',
        target_type = 'instance',
        transforms = transform_cityscapes_to_maskrcnn
    )
    dataset_val = CityscapesInstance(
        root=data_root, 
        split = 'val',
        mode='fine',
        target_type='instance',
        transforms = transform_cityscapes_to_maskrcnn
    )


    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=utils.collate_fn
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        collate_fn=utils.collate_fn
    )

    # Get the modified pre-trained model
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device for acceleration
    model.to(device)

    # Establish the modifiable parameteres for training
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[12, 16],
        gamma=0.1
    )


    # Initialize variables
    best_map = 0.0
    best_model_path = os.path.join(model_dir, "maskrcnn_cityscapes.pth")
    checkpoint_model_path = os.path.join(checkpoints_dir, "maskrcnn_cityscapes_checkpoint.pth")
    start_epoch = 0

    # If previous epochs were completed, load variables from checkpoint
    if os.path.exists(PATH_PROGRESS_FILE):
        print("Starting from last checkpoint...")

        with open(PATH_PROGRESS_FILE, 'r') as f:
            progress_data = json.load(f)

            start_epoch = progress_data.get('last_completed_epoch', -1) + 1
            best_map = progress_data.get('best_map_achieved', 0.0)

            checkpoint = torch.load(checkpoint_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Chekpoint model loaded")

    # Calculate and evaluate model in every epoch
    for epoch in range(start_epoch, num_epochs):

        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        lr_scheduler.step()

        # evaluate on the test dataset
        coco_evaluator = evaluate(model, data_loader_val, device=device)

        # Obtain metric Average Precision (mAP)
        current_map = coco_evaluator.coco_eval['segm'].stats[0] 

        # Save state dict of checkpoint model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'map': current_map,
        }, checkpoint_model_path)

        # Save in best model if is better than the already best one
        if current_map > best_map:
            print(f"Upgrade in mAP of {best_map:.4f} to {current_map:.4f}. Saving the model...")
            best_map = current_map

            # Save state dict of the model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map': best_map,
            }, best_model_path)
        else:
            print(f"mAP of {current_map:.4f} is not better than {best_map:.4f}. Do not saving this model.")

        progress_data = {
            'last_completed_epoch': epoch,
            'best_map_achieved': best_map,
        }

        with open(PATH_PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f, indent=4)

        print(f"Progress of epoch {epoch} saved. Best mAP: {best_map:.4f}")
