import cv2
import numpy as np

# Initialize variables for drawing
ix, iy, drawing = -1, -1, False
bx, by = -1, -1
img = cv2.imread(r"e:\DEGREE\YEAR 4\BERR4723_Digital Image Processing\Assignment\apple-banana-cherry.jpg")

if img is None:
    print("Error: Image not found or unable to load.")
else:
    # Resize the image (optional step)
    new_width = 600  # Set desired width
    new_height = 400  # Set desired height
    img = cv2.resize(img, (new_width, new_height))

    # Convert to BGRA format (with an alpha channel)
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # Fully opaque
    img_bgra = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))  # Merge BGR channels with alpha


# Function to draw the bounding box
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img, bx, by
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Image", img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bx, by = x, y  # Save the final rectangle's bottom-right corner
        cv2.rectangle(img, (ix, iy), (bx, by), (0, 255, 0), 2)
        cv2.imshow("Image", img)

# Display the image and set up mouse callback for selection
cv2.imshow("Image", img)
cv2.setMouseCallback("Image", draw_rectangle)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Ensure selection is valid
if ix != -1 and iy != -1:
    # Step 1: Adjust the coordinates to remove the border (optional border thickness can be subtracted)
    border_thickness = 2  # Adjust the border thickness if needed
    selected_region = img[iy + border_thickness:by - border_thickness, ix + border_thickness:bx - border_thickness]

    # Step 2: Convert only the selected region to grayscale
    gray_region = cv2.cvtColor(selected_region, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Region", gray_region)  # Show grayscale image
    cv2.waitKey(0)

    # Step 3: Apply Gaussian blur to reduce noise 
    blurred_region = cv2.GaussianBlur(gray_region, (5, 5), 0)

    # Step 3.1: Apply histogram equalization to make the object darker (enhance contrast)
    equalized_region = cv2.equalizeHist(blurred_region)
    cv2.imshow("Equalized Region", equalized_region)  # Show equalized image
    cv2.waitKey(0)

    # Step 4: Perform thresholding to create a binary image (only white and black)
    _, binary_region = cv2.threshold(equalized_region, 190, 210, cv2.THRESH_BINARY)
    cv2.imshow("Binary Region", binary_region)  # Show binary image
    cv2.waitKey(0)

    # Step 5: Invert the binary image (white to black and black to white)
    inverted_region = cv2.bitwise_not(binary_region)
    cv2.imshow("Inverted Region", inverted_region)  # Show inverted image
    cv2.waitKey(0)

    # Step 6: Create an alpha mask where white is 0 (transparent) and black is 255 (opaque)
    alpha_mask = inverted_region  # This mask now ensures the object is removed (transparent)
    alpha_mask = alpha_mask.astype(np.uint8)

    # Step 7: Apply the alpha mask to the color channels (B, G, R) of the selected region
    for c in range(3):  # Iterate over color channels (0: Blue, 1: Green, 2: Red)
        selected_region[:, :, c] = selected_region[:, :, c] * (alpha_mask / 255.0)

    # Step 8: Create a new image (BGRA) for saving the selected region
    selected_region_bgra = cv2.merge((selected_region[:, :, 0], selected_region[:, :, 1], selected_region[:, :, 2], alpha_mask))

    # Step 9: Compress and save the selected region with transparency as a PNG file
    png_compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 5]  # Level 5 compression (out of 0-9)
    cv2.imwrite("Cropped_PNG_Image.png", selected_region_bgra, png_compression_params)

    # Display the saved PNG image with transparency
    cv2.imshow("Saved PNG Image", selected_region_bgra)
    cv2.waitKey(0)
    cv2.destroyAllWindows()