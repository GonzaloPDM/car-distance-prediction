import torch


def get_predictions(image_tensor, model, img_h, img_w, score_threshold=0.5, max_height=0):
    """
        This method calculates the predictions of the car segmentation model of a given image,
        and returns the predictions with a confidence level above a given threshold
    Args:
        image: The image to be processed
        model: The model that computes the predictions
        img_h, img_w: The dimensions of the image.
        score_threshold: The confidence threshold
        max_height: The row from which the model evaluates the image. 0 means no cropping (slower),
                    1 means cropping whole image (no image processed). Default 0 to no cropping.

    Returns:
        pred_masks: The segmentation masks
        pred_boxes: The bounding boxes
        pred_labels: The labels for each predicted class
    """

    # Check that max_height is not to low (minimum computes 40% of the image)
    if 0 < max_height < 0.6*img_h:
        input_tensor = image_tensor[:, max_height:, :]
    else:
        input_tensor = image_tensor

    # Turn off gradient mode during the prediction execution
    with torch.no_grad():
        predictions = model([input_tensor])

    prediction = predictions[0]

    # Only keep predictions if the model has a certain level of confidence about them (over 50% score)
    keep_idx = prediction['scores'] > score_threshold

    # If nothing detected, we return the original image
    if not keep_idx.any():
        empty_box = torch.empty((0, 4), device=image_tensor.device)
        empty_mask = torch.empty((0, 1, image_tensor.shape[1], image_tensor.shape[2]), device=image_tensor.device)
        empty_label = torch.empty((0), dtype=torch.int64, device=image_tensor.device)
        return empty_mask, empty_box, empty_label

    pred_boxes = prediction['boxes'][keep_idx]
    pred_masks = prediction['masks'][keep_idx]
    pred_labels = prediction['labels'][keep_idx]

    # Adjust coordinates if image cropped
    if 0 < max_height < 0.6*img_h:
        pred_boxes[:, [1, 3]] += max_height
        full_masks = torch.zeros((len(pred_masks), 1, img_h, img_w), device=image_tensor.device)
        full_masks[:, :, max_height:, :] = pred_masks
        return full_masks, pred_boxes, pred_labels

    return pred_masks, pred_boxes, pred_labels


class VehicleDetector:
    """
        Runs the model over a sequence of frames. It only predicts every 'frames_per_prediction' frames
        (reusing the last predictions in between) and crops the sky: the model only evaluates the image
        from a little above the highest vehicle found in the previous prediction.
    """
    RESET_MAX_HEIGHT = 25   # Each 25 frames, model predicts the whole image

    def __init__(self, model, device, img_h, img_w, score_threshold=0.5, frames_per_prediction=1):
        """
        Args:
            model: The segmentation model that predicts masks and bouding boxes
            device: The device where the computations will be done
            img_h, img_w: The dimensions of the frames
            score_threshold: The asurance of the model for shwoing the instance predictions 
            frames_per_prediction: The amount of frames that passes beetwen each prediction. Default is 1 (every frame)
        """
        if frames_per_prediction <= 0:
            raise ValueError("Frames per predictions must be > 0")

        self.model = model
        self.device = device
        self.img_h = img_h
        self.img_w = img_w
        self.score_threshold = score_threshold
        self.frames_per_prediction = frames_per_prediction

        self.prev_max_height = 0     # First process the whole image
        self.frame_count = 0
        self.last_predictions = None

    def __call__(self, frame_tensor):
        """ Return (masks, boxes, labels) for the given frame in Tensor format """
        # Let between model predictions as many frames as frames_per_prediction
        if self.frame_count % self.frames_per_prediction == 0:

            # Each certain amount of frames, processes the whole image
            if self.frame_count % self.RESET_MAX_HEIGHT == 0:
                self.prev_max_height = 0

            masks, boxes, labels = get_predictions(frame_tensor, self.model, 
                                                   self.img_h, self.img_w, 
                                                   self.score_threshold,
                                                   self.prev_max_height
                                                   )

            # Update the maximum height reached by any bounding box (10% margin above it)
            if len(boxes) > 0:
                self.prev_max_height = int(torch.min(boxes[:, 1]).item()) - int(self.img_h*0.1)

            self.last_predictions = (masks, boxes, labels)

        self.frame_count += 1
        return self.last_predictions
