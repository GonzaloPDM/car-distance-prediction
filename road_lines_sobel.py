import os
import cv2
import numpy as np


def road_line_filter_image(image, saturation_thresh=(170, 255), sobelx_thresh=(20, 100), debug_image=False):
    """ Filter the road lines in the image using color and gradient thresholds.
    
        Args:
            image: Input image to be filtered
            saturation_thresh: Saturation threshold
            sobelx_thresh: Sobel x threshold
        Returns: 
            combined_binary: Binary image after applying the filters
    """
    # Transform to HLS color space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(float)
    h_channel = hls[:,:,0] # Hue channel
    l_channel = hls[:,:,1] # Lightness channel
    s_channel = hls[:,:,2] # Saturation channel

    # Filter white pixels (Hihg luminosity)
    white_mask = np.zeros_like(l_channel)
    white_mask[(l_channel > 220) & (l_channel <= 255)] = 1

    # Filter yellow pixels (High saturation + Hue: 15 - 35)
    yellow_mask = np.zeros_like(s_channel)
    yellow_mask[((h_channel >= 15) & (h_channel <= 35)) & 
                ((s_channel > saturation_thresh[0]) & (s_channel <= saturation_thresh[1]))] = 1
    
    # Sobel X: vertical edges
    l_blur = cv2.GaussianBlur(l_channel, (5,5), 1)
    sobel_x = cv2.Sobel(l_blur, cv2.CV_64F, 1, 0)
    abs_sobel_x = np.absolute(sobel_x)
    scaled_sobel = np.uint8(255*abs_sobel_x/np.max(abs_sobel_x))
    
    sobelx_mask = np.zeros_like(scaled_sobel)
    sobelx_mask[(scaled_sobel >= sobelx_thresh[0]) & (scaled_sobel <= sobelx_thresh[1])] = 1
    
    # Create final mask
    color_binary = None
    if debug_image:
        color_binary = np.dstack((np.zeros_like(sobelx_mask), sobelx_mask, (yellow_mask==1) | (white_mask==1))) * 255
    
    combined_binary = np.zeros_like(sobelx_mask)
    combined_binary[(yellow_mask == 1) | (white_mask == 1) | (sobelx_mask == 1)] = 1
    
    return combined_binary, color_binary



def remove_road_perspective(image, 
                            img_h, img_w,
                            src_pts, dst_pts):
    """
        This method calculate the homography of the road, given the points of a rectangle aling with straight road lines
    Args:
        img: The road image from the camera perspective
        src_pts: The points of a rectangle align with the road lines in the image from the perspective of the car.
        dst_pts: The points of the src_pts rectanggle, but from a "bird eye" perspective. 

    Returns:
        no_perspective_image: The image with removed perspective, like seen from above
        M: The homography matrix calculated for removing perspective
    """
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    no_perspective_image = cv2.warpPerspective(image, M, (img_w, img_h), flags=cv2.INTER_LINEAR)

    return no_perspective_image, M



def calculate_road_lines(binary_warped, debug_image=False):
    """ Calculate the road lines in a edges image with non perspective transformation applied, using sliding windows and polynomial fitting. 
        Then, draw the windows and the lines in an output image.
    
        Args:
            binary_warped: The binary image with the road lines in white
            debug_image: If True, then the debug image is processed and return as out_img
        Returns:
            left_fit: The polynomial coefficients for the left road line
            right_fit: The polynomial coefficients for the right road line
            out_img: The output image with the detected lines and windows drawn
    """  
    img_h, img_w = binary_warped.shape[:2]

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(img_h*0.8):int(img_h*0.98),int(img_w*0.35):int(img_w*0.65)], axis=0)

    # The peak of the left and right halves of the histogram will be the starting point for the left and right lines
    mid_point = histogram.shape[0]//2
    left_peak = np.argmax(histogram[:mid_point]) + int(img_w*0.35)
    right_peak = np.argmax(histogram[mid_point:]) + mid_point + int(img_w*0.35)

    if left_peak == 0 or right_peak == 0:
        return None, None, None

    # Set parameters for sliding windows
    nwindows = 9
    window_height = int(img_h // nwindows)
    margin = img_w // 20
    minpix = 50
    left_current = left_peak 
    right_current = right_peak
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    out_img = None
    if debug_image:
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Slide windows from bottom to top, finding line pixels
    for window in range(nwindows):
        # Define new window
        win_y_low = img_h - window * window_height
        win_y_high = img_h - (window + 1) * window_height
        win_xleft_low = left_current - margin
        win_xleft_high = left_current + margin
        win_xright_low = right_current - margin
        win_xright_high = right_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = (binary_warped[win_y_high:win_y_low, win_xleft_low:win_xleft_high]).nonzero()
        good_right_inds = (binary_warped[win_y_high:win_y_low, win_xright_low:win_xright_high]).nonzero()

        # Adjust pixels to image coordinates
        good_left_inds = (good_left_inds[0] + win_y_high, good_left_inds[1] + win_xleft_low)
        good_right_inds = (good_right_inds[0] + win_y_high, good_right_inds[1] + win_xright_low)

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        new_left_current = left_current
        new_right_current = right_current

        # If found > minpix pixels, recenter next window on their mean position (ponderated to center of the image)
        if len(good_left_inds[0]) > minpix:
            new_left_current = int(np.mean(good_left_inds[1]))
        if len(good_right_inds[0]) > minpix:
            new_right_current = int(np.mean(good_right_inds[1]))

        # Check wether boxes colides
        left_box_right_edge = new_left_current + margin
        right_box_left_edge = new_right_current - margin
        buffer = 10 
        
        # If no colision, we let the means, if not, do not renew them
        if right_box_left_edge > (left_box_right_edge + buffer):
            
            left_current = new_left_current
            right_current = new_right_current

        if debug_image:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,
                          (left_current - margin, win_y_high), 
                          (left_current + margin, win_y_low),
                          (0,255,0), 2)
            cv2.rectangle(out_img,
                          (right_current - margin, win_y_high), 
                          (right_current + margin, win_y_low),
                          (0,255,0), 2)

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds, axis=1)
    right_lane_inds = np.concatenate(right_lane_inds, axis=1)

    if (len(left_lane_inds[0]) == 0) | (len(right_lane_inds[0]) == 0):
        return None, None, None

    if debug_image:
        # Draw lane pixels
        out_img[left_lane_inds[0], left_lane_inds[1]] = [255, 0, 0]
        out_img[right_lane_inds[0], right_lane_inds[1]] = [0, 0, 255]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(left_lane_inds[0], left_lane_inds[1], 1)
    right_fit = np.polyfit(right_lane_inds[0], right_lane_inds[1], 1)

    return left_fit, right_fit, out_img


def get_road_mask(no_perspective_image, left_fit, right_fit, Minv, h_img, img_w):
    """
        Create the mask of the detected road area
    """
    
    # Calculate left and right lanes
    ploty = np.linspace(0, h_img-1, h_img)
    left_fitx = np.polyval(left_fit, ploty)
    right_fitx = np.polyval(right_fit, ploty)
    
    # Create empty RGB image to draw the mask
    warp_zero = np.zeros_like(no_perspective_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Set format to points for cv2.fillpoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the mask on the empty image, this will be the mask
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0)) # Verde

    # Apply inverse homography to the road mask
    road_mask = cv2.warpPerspective(color_warp, Minv, (img_w, h_img)) 

    return road_mask



def search_around_poly(binary_warped, left_fit, right_fit, margin=50, debug_image=False):
    """
    Busca líneas de carril basándose en los polinomios del frame anterior (ROI Tracking).
    Esto evita usar histogramas y ventanas deslizantes nuevamente.
    """
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Define the search area around the previous polynomials
    left_center_x = np.polyval(left_fit, nonzeroy)
    right_center_x = np.polyval(right_fit, nonzeroy)

    # Define the search area around the previous polynomials
    left_lane_inds = ((nonzerox > (left_center_x - margin)) & 
                      (nonzerox < (left_center_x + margin)))
    
    right_lane_inds = ((nonzerox > (right_center_x - margin)) & 
                       (nonzerox < (right_center_x + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # If we don't find enough pixels, we assume detection failed
    min_inds = 50
    if len(leftx) < min_inds or len(rightx) < min_inds:
        return None, None, None

    # Fit new polynomials
    new_left_fit = np.polyfit(lefty, leftx, 1)
    new_right_fit = np.polyfit(righty, rightx, 1)
    
    # Generate x and y values for plotting
    out_img = None
    if debug_image:
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        out_img = out_img.astype(np.uint8)

        # Draw green area where research have been done for each line
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        
        # Calculate the trajectory of the PREVIOUS polynomial (where we are searching)
        left_fitx = np.polyval(left_fit, ploty)
        right_fitx = np.polyval(right_fit, ploty)

        # Create an empty image to draw the transparency
        window_img = np.zeros_like(out_img)

        # Generate polygon points for the left margin
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        
        # Generate polygon points for the right margin
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the tunnels in green on the empty image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        
        # Blend with the original image (Transparency)
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        out_img[lefty, leftx] = [255, 0, 0]   # Red (or Blue in BGR)
        out_img[righty, rightx] = [0, 0, 255] # Blue (or Red in BGR)

    return new_left_fit, new_right_fit, out_img


def process_lines_video(video_path, output_path,  
                        src_pts=None, 
                        dst_pts=None, 
                        check_path=None
                        ):
    """
        This functions processes a video by segmenting the own line of the vehicle.

    Args:
        video_path: The path to the video file
        output_path: The path where the resulting video is written
        src_pts: The points of a rectangle align with the road lines in the image from the perspective of the car.
        dst_pts: The points of the src_pts rectanggle, but from a "bird eye" perspective.
        check_path: If set, then also a debug video will be processed and display to that path
        
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

    if src_pts is None:
        p1_top_left = (int(img_w * 0.45), int(img_h * 0.63)) 
        p2_top_right = (int(img_w * 0.52), int(img_h * 0.63)) 
        p3_bot_right = (int(img_w * 0.84), int(img_h * 0.99)) 
        p4_bot_left  = (int(img_w * 0.28), int(img_h * 0.99)) 

        src_pts = np.float32([p1_top_left, p2_top_right, p3_bot_right, p4_bot_left])

    if dst_pts is None:
        dst_pts = np.float32([(int(img_w*0.4), 0), 
                    (int(img_w*0.6), 0), 
                    (int(img_w*0.6), img_h), 
                    (int(img_w*0.4), img_h)
                ])

    print(f"Processing video... Resolution: {img_w}x{img_h}, FPS: {fps}")
    
    # Calculate homography
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = np.linalg.inv(M)

    prev_left_fit = None
    prev_right_fit = None
    
    frame_count = 0

    RESET_LINES_FRAMES = 25     # Reset lanes history each 25 frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # Video ended

        frame_processed = frame

        # Filter potential line pixels
        filter_frame, _ = road_line_filter_image(frame, sobelx_thresh=(20,100))

        # Remove perspective
        no_perspective_frame = cv2.warpPerspective(filter_frame, M, (img_w, img_h), flags=cv2.INTER_LINEAR)

        # Calculate road lines
        left_fit, right_fit = None, None
        no_perspective_lines = None

        # A) Try Optimized Search (if we have history)
        if prev_left_fit is not None and prev_right_fit is not None:
            left_fit, right_fit, no_perspective_lines = search_around_poly(
                                                            no_perspective_frame, 
                                                            prev_left_fit, 
                                                            prev_right_fit, 
                                                            debug_image=debug,
                                                            margin=50
                                                        )
        
        # B) Fallback: Full Sliding Windows Search (if optimization failed)
        if left_fit is None and right_fit is None:
            left_fit, right_fit, no_perspective_lines = calculate_road_lines(
                no_perspective_frame, debug_image=debug
            )
            if left_fit is None and prev_left_fit is not None:
                 left_fit, right_fit = prev_left_fit, prev_right_fit


        if left_fit is not None and right_fit is not None:
            # Save for next frame
            if frame_count % RESET_LINES_FRAMES == 0:
                prev_left_fit = None
                prev_right_fit = None
            else:
                prev_left_fit = left_fit
                prev_right_fit = right_fit

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

    # Liberar recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return True
    

if __name__ == '__main__':

    CWD = os.getcwd()
    PATH_CITYSCAPE_DATA = os.path.join(CWD, 'datasets/cityscape_data')
    PATH_IMAGES = os.path.join(CWD, "display_elements/road_segmentation/images")
    PATH_VIDEOS = os.path.join(CWD, "display_elements/road_segmentation/videos")
    PATH_RESULT_IMAGES = os.path.join(CWD, "results/road_segmentation/images")
    PATH_RESULT_VIDEOS = os.path.join(CWD, "results/road_segmentation/videos")

    try:
        os.mkdir(PATH_RESULT_IMAGES)
        os.mkdir(PATH_RESULT_VIDEOS)
    except FileExistsError:
        pass
    
    # Get the video capture
    video_name = "video1_recortado"
    video_path = os.path.join(PATH_VIDEOS, f"{video_name}.mp4")
    output_path = os.path.join(PATH_RESULT_VIDEOS,f"{video_name}_result_filter.mp4")
    check_path = os.path.join(PATH_RESULT_VIDEOS, f"{video_name}_wraped_filter.mp4")

    result = process_lines_video(video_path, output_path, check_path)

    if result:
        print(f"Video saved in: {output_path}")
    else:
        print("An error ocurred while openning the video.")


