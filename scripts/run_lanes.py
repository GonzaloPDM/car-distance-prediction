import os
import sys
import argparse

import numpy as np

from src.lanes.filters import LANE_FILTER_MODES
from src.pipeline import lane_processor
from src.video import process_video


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
    parser.add_argument('--show', action='store_true', help="If set, the video is displayed while processing")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    src_pts = np.float32(np.array(args.srcpts).reshape(4, 2)) if args.srcpts else None

    processor = lane_processor(mode=args.mode, src_pts=src_pts, debug=args.check is not None)
    result = process_video(args.input, args.output, processor, check_path=args.check, show=args.show)

    if result:
        print(f"Video saved in: {args.output}")
    else:
        print("An error ocurred while openning the video.")
