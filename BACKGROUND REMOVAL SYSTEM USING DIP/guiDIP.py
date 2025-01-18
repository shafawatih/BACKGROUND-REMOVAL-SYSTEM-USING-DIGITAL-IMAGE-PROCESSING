import cv2
import numpy as np
from tkinter import Tk, Button, filedialog, Canvas, messagebox
from PIL import Image, ImageTk

# Initialize global variables
ix, iy, drawing, bx, by = -1, -1, False, -1, -1
img = None
img_copy = None
processed_region = None  # Variable to store the processed region

# Function to load an image
def load_image():
    global img, img_copy
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return
    img = cv2.imread(file_path)
    if img is None:
        messagebox.showerror("Error", "Failed to load image.")
        return
    img = cv2.resize(img, (600, 400))  # Resize for better visualization
    img_copy = img.copy()
    display_image(img)

# Function to display an image on the canvas
def display_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tk = ImageTk.PhotoImage(Image.fromarray(image_rgb))
    canvas.img_tk = img_tk  # Store a reference to avoid garbage collection
    canvas.create_image(0, 0, anchor="nw", image=img_tk)

# Mouse event handler for drawing rectangle
def on_mouse_down(event):
    global ix, iy, drawing
    drawing = True
    ix, iy = event.x, event.y

def on_mouse_move(event):
    global ix, iy, drawing, bx, by
    if drawing:
        bx, by = event.x, event.y
        canvas.delete("rect")  # Clear the previous rectangle
        canvas.create_rectangle(ix, iy, bx, by, outline="green", width=2, tags="rect")

def on_mouse_up(event):
    global ix, iy, bx, by, img_copy
    drawing = False
    bx, by = event.x, event.y
    canvas.create_rectangle(ix, iy, bx, by, outline="green", width=2)  # Draw final rectangle

    # Update the OpenCV image with the drawn rectangle
    cv2.rectangle(img_copy, (ix, iy), (bx, by), (0, 255, 0), 2)
    display_image(img_copy)

# Function to process the selected region
def process_region():
    global ix, iy, bx, by, img, processed_region
    if ix == -1 or iy == -1 or bx == -1 or by == -1:
        messagebox.showerror("Error", "No region selected.")
        return
    selected_region = img[iy:by, ix:bx]

    # Step 1: Convert to grayscale
    gray_region = cv2.cvtColor(selected_region, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Gaussian blur
    blurred_region = cv2.GaussianBlur(gray_region, (5, 5), 0)

    # Step 3: Histogram equalization
    equalized_region = cv2.equalizeHist(blurred_region)

    # Step 4: Thresholding
    _, binary_region = cv2.threshold(equalized_region, 190, 210, cv2.THRESH_BINARY)

    # Step 5: Invert binary image
    inverted_region = cv2.bitwise_not(binary_region)

    # Step 6: Create an alpha mask
    alpha_mask = inverted_region.astype(np.uint8)

    # Step 7: Apply alpha mask to color channels
    selected_region[:, :, 0] = selected_region[:, :, 0] * (alpha_mask / 255.0)
    selected_region[:, :, 1] = selected_region[:, :, 1] * (alpha_mask / 255.0)
    selected_region[:, :, 2] = selected_region[:, :, 2] * (alpha_mask / 255.0)

    # Step 8: Merge into BGRA
    processed_region = cv2.merge((selected_region[:, :, 0], selected_region[:, :, 1], selected_region[:, :, 2], alpha_mask))

    # Display the processed region
    cv2.imshow("Processed Region", processed_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    messagebox.showinfo("Success", "Processed region ready to be saved.")

# Function to save the processed image as PNG
def save_image():
    global processed_region
    if processed_region is None:
        messagebox.showerror("Error", "No processed region to save.")
        return
    save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
    if save_path:
        # Save the processed region
        cv2.imwrite(save_path, processed_region)
        messagebox.showinfo("Success", f"Image saved as {save_path}")

# Initialize the Tkinter GUI
root = Tk()
root.title("Image Processing GUI")

# Create and configure canvas
canvas = Canvas(root, width=600, height=400, bg="white")
canvas.pack()

# Bind mouse events to the canvas
canvas.bind("<ButtonPress-1>", on_mouse_down)
canvas.bind("<B1-Motion>", on_mouse_move)
canvas.bind("<ButtonRelease-1>", on_mouse_up)

# Create buttons
btn_load = Button(root, text="Load Image", command=load_image)
btn_load.pack(side="left", padx=10, pady=10)

btn_process = Button(root, text="Process Region", command=process_region)
btn_process.pack(side="left", padx=10, pady=10)

btn_save = Button(root, text="Save Image", command=save_image)
btn_save.pack(side="left", padx=10, pady=10)

# Run the main event loop
root.mainloop()
