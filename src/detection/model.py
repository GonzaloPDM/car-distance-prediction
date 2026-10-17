import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights


# 7 considered vehicles + background
NUM_CLASSES = 8


def get_model_instance_segmentation(
        num_classes : int,
        freeze_backbone : bool =False
    ):
    """
        This method return a pre-trained model on COCO, with its output layer modified, to only return 'num-classes'
        different classes instead of the 21 original classes.

    Args:
        num_classes (int): number of classes to segmentation
        freeze_backbone (bool, optional): If true, then the backbone weights wont be modified in the fine-tunning procces, thus enhancing GPU memory and time. Defaults to False.

    Returns:
        model: the modified pre-trained model
    """
    # load an instance segmentation model pre-trained on COCO
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)

    # Modify box classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Modify mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    # Freeze backbone weights if do not want to re-calculate them
    if freeze_backbone:
        for name, param in model.named_parameters():
            # Freeze all parameters apart from the ones at the headers (roi_heads)
            if 'roi_heads' not in name:
                param.requires_grad = False

    return model


def load_model(model_path, device, num_classes=NUM_CLASSES):
    """
        Build the model architecture, load the fine-tuned weights and set it ready for inference.

    Args:
        model_path: Path to the .pth file saved by the training process
        device: The device where the model will be loaded
        num_classes: Number of classes of the model (background included)

    Returns:
        model: The model in evaluation mode on the given device
    """
    model = get_model_instance_segmentation(num_classes)

    # weights_only=False is needed because the checkpoint stores more than tensors
    data_model = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(data_model['model_state_dict'])

    model.to(device)
    model.eval()
    return model
