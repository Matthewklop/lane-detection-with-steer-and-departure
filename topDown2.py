import cv2
import numpy as np
import os
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors

# Defining variables to hold meter-to-pixel conversion
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 720  # meters per pixel in x dimension (lane width / image width)

CWD_PATH = os.getcwd()

def readVideo():
    """Function to read the video input"""
    inpImage = cv2.VideoCapture(os.path.join(CWD_PATH, 'drive.mp4'))
    return inpImage

def processImage(inpImage):
    """Applies various filters to prepare the image for lane detection"""
    hls = cv2.cvtColor(inpImage, cv2.COLOR_BGR2HLS)
    lower_white = np.array([0, 160, 10])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(inpImage, lower_white, upper_white)
    hls_result = cv2.bitwise_and(inpImage, inpImage, mask=mask)

    gray = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresh, (3, 3), 0)
    canny = cv2.Canny(blur, 40, 60)

    return canny  # We only need the Canny edge image for display

def perspectiveWarp(inpImage):
    """Applies a perspective transformation to the image"""
    img_size = (inpImage.shape[1], inpImage.shape[0])

    # Define source and destination points for perspective transform
    src = np.float32([[590, 440], [690, 440], [200, 640], [1000, 640]])
    dst = np.float32([[200, 0], [1200, 0], [200, 710], [1200, 710]])

    # Perspective matrix
    matrix = cv2.getPerspectiveTransform(src, dst)
    minv = cv2.getPerspectiveTransform(dst, src)

    # Warp the image to bird's eye view
    birdseye = cv2.warpPerspective(inpImage, matrix, img_size)

    return birdseye, minv

def draw_lane_lines(original_image, warped_image, Minv):
    """Draw lane lines on the original image with correct format for points"""
    
    # Define points for the lane area
    pts_left = np.array([[200, 710], [200, 0]], dtype=np.int32)  # Left points
    pts_right = np.array([[1200, 0], [1200, 710]], dtype=np.int32)  # Right points

    # Stack points together to form a polygon
    pts = np.vstack((pts_left, pts_right))

    # Initialize a blank image for drawing
    color_warp = np.zeros_like(original_image).astype(np.uint8)

    # Ensure points are in the correct format for cv2.fillPoly (should be int32)
    cv2.fillPoly(color_warp, [pts], (0, 255, 0))

    # Warp the blank image back to the original perspective
    newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)

    return result

def main():
    """Main function to run lane detection"""
    video = readVideo()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Original frame
        original_frame = frame

        # Apply perspective warp
        birdView, minv = perspectiveWarp(frame)

        # Process the image (for canny edge detection)
        canny_edge = processImage(birdView)

        # Final image with lane overlays
        final_img = draw_lane_lines(original_frame, canny_edge, minv)

        # Show the required steps in separate windows
        cv2.imshow("Original", original_frame)  # Show the original frame
        cv2.imshow("Canny Edge", canny_edge)    # Show the Canny edge detection
        cv2.imshow("Bird's Eye View", birdView) # Show the bird's eye view (warped image)
        cv2.imshow("Final", final_img)          # Show the final lane overlay image

        # Wait for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
