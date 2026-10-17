import cv2
import numpy as np
from torchvision.transforms import functional as F_torch

from src.lanes.filters import filter_lane_pixels
from src.lanes.perspective import get_homography, warp
from src.lanes.fitting import calculate_road_lines, LaneTracker
from src.lanes.mask import get_road_mask
from src.detection.inference import get_predictions, VehicleDetector
from src.detection.draw import draw_predictions
from src.distance.draw import draw_predictions_in_lanes
from src.distance.estimation import FOCAL_LENGTH


class LaneDetector:
    """ Calculates the mask of the own lane: filter -> bird eye view -> line fitting -> mask """

    def __init__(self, img_h, img_w, mode='sobel', src_pts=None, dst_pts=None, tracker=None):
        """
        Args:
            img_h, img_w: The dimensions of the frames
            mode: The lane pixels filter, 'sobel' or 'canny'
            src_pts: The points of a rectangle align with the road lines in the image from the perspective of the car.
            dst_pts: The points of the src_pts rectanggle, but from a "bird eye" perspective.
            tracker: Optional LaneTracker. If None, a full sliding windows search is done in every frame.
        """
        self.img_h = img_h
        self.img_w = img_w
        self.mode = mode
        self.tracker = tracker
        self.M, self.Minv = get_homography(img_h, img_w, src_pts, dst_pts)

    def __call__(self, image, debug_image=False):
        """
        Returns:
            lane_mask: RGB image with the lane drawn in green, or None if the lane was not found
            debug: The bird eye debug image (None if debug_image is False)
        """
        binary = filter_lane_pixels(image, self.mode)
        no_perspective = warp(binary, self.M, self.img_h, self.img_w)

        if self.tracker is not None:
            left_fit, right_fit, debug = self.tracker.update(no_perspective, debug_image=debug_image)
        else:
            left_fit, right_fit, debug = calculate_road_lines(no_perspective, debug_image=debug_image)

        lane_mask = None
        if left_fit is not None and right_fit is not None:
            lane_mask = get_road_mask(no_perspective, left_fit, right_fit, self.Minv, self.img_h, self.img_w)

        return lane_mask, debug


def draw_lane(image, lane_mask):
    """ Blend the lane mask over the image """
    return cv2.addWeighted(image, 1, lane_mask, 0.3, 0)


# -------------------------------- SINGLE IMAGE -------------------------------- #

def segment_vehicles(image, model, device='cpu', score_threshold=0.5):
    """ Draw the vehicle segmentation predictions over an image """
    # Convert the image to Tensor format
    image_tensor = F_torch.to_tensor(image).to(device)
    _, img_h, img_w = image_tensor.shape

    pred_masks, pred_boxes, pred_labels = get_predictions(image_tensor, model, img_h, img_w, score_threshold)
    return draw_predictions(image_tensor, pred_masks, pred_boxes, pred_labels)


def predict_distance(image, model, device='cpu', score_threshold=0.5, show_lane=False, mode='sobel',
                     src_pts=None, dst_pts=None, focal_lenght=FOCAL_LENGTH):
    """ Draw the vehicles over an image, with the distance to the ones in the own lane """
    img_h, img_w = image.shape[:2]

    lane_mask, _ = LaneDetector(img_h, img_w, mode, src_pts, dst_pts)(image)
    if lane_mask is None:
        lane_mask = np.zeros_like(image)

    image_tensor = F_torch.to_tensor(image).to(device)
    masks, boxes, labels = get_predictions(image_tensor, model, img_h, img_w, score_threshold)

    if show_lane:
        image_tensor = F_torch.to_tensor(draw_lane(image, lane_mask)).to(device)

    return draw_predictions_in_lanes(image_tensor, masks, boxes, labels, lane_mask, focal_lenght)


# -------------------------------- VIDEO PROCESSORS -------------------------------- #
#
# Each function returns a build_processor(img_h, img_w) for src.video.process_video
#

def lane_processor(mode='sobel', src_pts=None, dst_pts=None, reset_every=25, debug=False):
    """ Draws the own lane. If debug, also returns the bird eye debug image (black if the lane was not found) """
    def build(img_h, img_w):
        detector = LaneDetector(img_h, img_w, mode, src_pts, dst_pts,
                                tracker=LaneTracker(margin=50, reset_every=reset_every))

        def process(frame):
            lane_mask, debug_img = detector(frame, debug_image=debug)
            frame_processed = draw_lane(frame, lane_mask) if lane_mask is not None else frame

            if not debug:
                return frame_processed
            return frame_processed, (debug_img if debug_img is not None else np.zeros_like(frame))
        return process
    return build


def segmentation_processor(model, device='cpu', score_threshold=0.5, frames_per_prediction=1):
    """ Draws the vehicle segmentation predictions """
    def build(img_h, img_w):
        detector = VehicleDetector(model, device, img_h, img_w, score_threshold, frames_per_prediction)

        def process(frame):
            frame_tensor = F_torch.to_tensor(frame).to(device)
            masks, boxes, labels = detector(frame_tensor)
            return draw_predictions(frame_tensor, masks, boxes, labels)
        return process
    return build


def distance_processor(model, device='cpu', score_threshold=0.5, frames_per_prediction=1,
                       src_pts=None, dst_pts=None, focal_lenght=FOCAL_LENGTH, show_lane=False, mode='sobel'):
    """ Draws the vehicles and the distance to the ones in the own lane """
    def build(img_h, img_w):
        lanes = LaneDetector(img_h, img_w, mode, src_pts, dst_pts)
        detector = VehicleDetector(model, device, img_h, img_w, score_threshold, frames_per_prediction)

        def process(frame):
            # Lane mask
            lane_mask, _ = lanes(frame)
            if lane_mask is None:
                lane_mask = np.zeros_like(frame)

            # Vehicle segmentation masks
            frame_tensor = F_torch.to_tensor(frame).to(device)
            masks, boxes, labels = detector(frame_tensor)

            # Resulting frame
            if show_lane:
                frame_tensor = F_torch.to_tensor(draw_lane(frame, lane_mask)).to(device)

            return draw_predictions_in_lanes(frame_tensor, masks, boxes, labels, lane_mask, focal_lenght)
        return process
    return build
