import cv2


def process_video(video_path, output_path, build_processor, check_path=None, show=False):
    """
        Generic video loop: reads every frame, processes it and writes the result with the same
        resolution and FPS as the original video.

    Args:
        video_path: The path to the video file
        output_path: The path where the resulting video is written
        build_processor: Function (img_h, img_w) -> processor. The processor is a callable that receives a
                         BGR frame and returns the processed frame, or (processed_frame, check_frame) when
                         check_path is set. It can keep state between frames.
        check_path: If set, the second image returned by the processor is written to this path (debug video)
        show: If True, then the video is displayed while computing. Press 'q' to stop.

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
    out_check = cv2.VideoWriter(check_path, fourcc, fps, (img_w, img_h)) if check_path is not None else None

    print(f"Processing video... Resolution: {img_w}x{img_h}, FPS: {fps}")

    process_frame = build_processor(img_h, img_w)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # Video ended

        result = process_frame(frame)

        if out_check is not None:
            frame_processed, check_frame = result
            out_check.write(check_frame)
        else:
            frame_processed = result

        out.write(frame_processed)

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"{frame_count} frames processed...")

        # Show the frame processing in real time, exit if press 'q'
        if show:
            cv2.imshow("Processed video", frame_processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Free resources
    cap.release()
    out.release()
    if out_check is not None:
        out_check.release()
    if show:
        cv2.destroyAllWindows()

    if frame_count == 0:
        print("Error, no frames have been processed")
        return False
    return True
