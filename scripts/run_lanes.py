import os
import sys
import argparse

import cv2
import numpy as np

from src.lanes.filters import filter_lane_pixels, LANE_FILTER_MODES
from src.lanes.perspective import get_homography, warp
from src.lanes.fitting import LaneTracker
from src.lanes.mask import get_road_mask


def process_lines_video(video_path, output_path, mode='sobel',
                        src_pts=None,
                        dst_pts=None,
                        check_path=None,
                        reset_every=25
                        ):
    """
        This functions processes a video by segmenting the own line of the vehicle.

    Args:
        video_path: The path to the video file
        output_path: The path where the resulting video is written
        mode: The lane pixels filter, 'sobel' or 'canny'
        src_pts: The points of a rectangle align with the road lines in the image from the perspective of the car.
        dst_pts: The points of the src_pts rectanggle, but from a "bird eye" perspective.
        check_path: If set, then also a debug video will be processed and display to that path
        reset_every: Every this amount of frames the lanes history is dropped. None means never.

    Returns:
        true if video is correctly processed, False if an error ocurred.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return False

    # Create Writer with same properties that the original video
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (img_w, img_h))

    debug = False
    if check_path is not None:
        debug = True
        out_check = cv2.VideoWriter(check_path, fourcc, fps, (img_w, img_h))

    print(f"Processing video... Resolution: {img_w}x{img_h}, FPS: {fps}")

    # Calculate homography
    M, Minv = get_homography(img_h, img_w, src_pts, dst_pts)

    tracker = LaneTracker(margin=50, reset_every=reset_every)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # Video ended

        frame_processed = frame

        # Filter potential line pixels and remove perspective
        filter_frame = filter_lane_pixels(frame, mode)
        no_perspective_frame = warp(filter_frame, M, img_h, img_w)

        # Calculate road lines
        left_fit, right_fit, no_perspective_lines = tracker.update(no_perspective_frame, debug_image=debug)

        if left_fit is not None and right_fit is not None:
            road_mask = get_road_mask(no_perspective_frame, left_fit, right_fit, Minv, img_h, img_w)

            # Draw the mask on the original image
            frame_processed = cv2.addWeighted(frame, 1, road_mask, 0.3, 0)

        out.write(frame_processed)

        if check_path is not None:
            if no_perspective_lines is not None:
                out_check.write(no_perspective_lines)
            else:
                out_check.write(np.zeros_like(frame))

        # Show the frame processing in real time
        cv2.imshow("Processed video", frame_processed)

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"{frame_count} frames processed...")

        # Exit if press 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if frame_count == 0:
        print("Error, video has not been processed")

    # Free resources
    cap.release()
    out.release()
    if check_path is not None:
        out_check.release()
    cv2.destroyAllWindows()
    return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Own lane segmentation")
    parser.add_argument('--input', '-i', type=str, required=True, help="Path to the input video file")
    parser.add_argument('--output', '-o', type=str, required=True, help="Path to save the result")
    parser.add_argument('--mode', type=str, default='sobel', choices=LANE_FILTER_MODES,
                        help="Lane detection method: 'sobel' (default) or 'canny'")
    parser.add_argument('--check', type=str, default=None,
                        help="If set, also saves the bird eye debug video to this path")
    parser.add_argument('--srcpts', type=float, nargs=8,
                        help="The four points of a trapezoid drawn over a straight lane: x1 y1 x2 y2 x3 y3 x4 y4")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    src_pts = np.float32(np.array(args.srcpts).reshape(4, 2)) if args.srcpts else None

    result = process_lines_video(args.input, args.output, mode=args.mode, src_pts=src_pts, check_path=args.check)

    if result:
        print(f"Video saved in: {args.output}")
    else:
        print("An error ocurred while openning the video.")
