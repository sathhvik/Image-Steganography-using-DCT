import cv2  # OpenCV for image processing
import numpy as np  # NumPy for handling arrays
from tkinter import Tk, filedialog, Button, Label, Text, END, messagebox, Scale, HORIZONTAL  # Tkinter components for GUI
from tkinter.ttk import Progressbar
from scipy.fft import dct, idct  # Import DCT and IDCT from scipy.fft
from PIL import Image, ImageTk  # Pillow for image display in Tkinter
import matplotlib.pyplot as plt  # For displaying histograms

# Functions for DCT and IDCT
def dct2(block):
    """Apply 2D Discrete Cosine Transform to an 8x8 block."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    """Apply 2D Inverse Discrete Cosine Transform to an 8x8 block."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# Steganography Functions
def is_image_compatible(image):
    """Check if image dimensions are compatible (multiples of 8)."""
    rows, cols, _ = image.shape
    return rows % 8 == 0 and cols % 8 == 0

def pad_image(image):
    """Pad image dimensions to the nearest multiple of 8."""
    rows, cols, _ = image.shape
    new_rows = (rows + 7) // 8 * 8
    new_cols = (cols + 7) // 8 * 8
    padded_image = np.zeros((new_rows, new_cols, 3), dtype=image.dtype)
    padded_image[:rows, :cols, :] = image
    return padded_image

def embed_message_dct(image, message, quality_factor):
    """Embed a binary message into an image using DCT."""
    message_bits = ''.join(f'{ord(c):08b}' for c in message)  # Convert message to binary
    message_bits += '00000000'  # Null character to mark end of message
    message_length = len(message_bits)  # Total bits to embed
    idx = 0  # Bit index for embedding
    rows, cols, _ = image.shape

    # Process each 8x8 block in the image
    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            if idx >= message_length:
                break
            block = image[i:i+8, j:j+8, 0]  # Access 8x8 block in the Y channel
            if block.shape[0] != 8 or block.shape[1] != 8:
                continue
            dct_block = dct2(block)  # Apply DCT to block
            dct_block[4, 4] = (dct_block[4, 4] // 2 * 2) + int(message_bits[idx])  # Embed bit in DCT coefficient
            dct_block[4, 4] = dct_block[4, 4] * quality_factor  # Adjust with quality factor
            idx += 1
            image[i:i+8, j:j+8, 0] = idct2(dct_block)  # Apply IDCT to modified block

    return image

def extract_message_dct(image, quality_factor):
    """Extract binary message from an image using DCT."""
    message_bits = ""
    rows, cols, _ = image.shape

    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            block = image[i:i+8, j:j+8, 0]
            if block.shape[0] != 8 or block.shape[1] != 8:
                continue
            dct_block = dct2(block) // quality_factor  # Scale down by quality factor
            message_bits += str(int(dct_block[4, 4]) & 1)  # Retrieve embedded bit

    message = ""
    for i in range(0, len(message_bits), 8):
        byte = message_bits[i:i+8]
        if byte == "00000000":  # End of message
            break
        message += chr(int(byte, 2))  # Convert byte to character
    return message

def can_embed_message(image, message):
    """Check if message can fit in the image based on available space."""
    rows, cols, _ = image.shape
    num_blocks = (rows // 8) * (cols // 8)  # Total 8x8 blocks
    max_bits = num_blocks * 1  # 1 bit per block
    message_bits = len(message) * 8 + 8  # Include null character
    return message_bits <= max_bits, max_bits // 8  # Return if embeddable and max chars

def show_histogram():
    """Display the histogram of the loaded image."""
    if img is None:
        messagebox.showerror("Error", "Load an image first")
        return

    y_channel, cr_channel, cb_channel = cv2.split(img)

    # Plot histograms for Y, Cr, and Cb channels
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.hist(y_channel.ravel(), bins=256, color='gray')
    plt.title('Y (Luminance) Histogram')

    plt.subplot(1, 3, 2)
    plt.hist(cr_channel.ravel(), bins=256, color='red')
    plt.title('Cr (Chrominance Red) Histogram')

    plt.subplot(1, 3, 3)
    plt.hist(cb_channel.ravel(), bins=256, color='blue')
    plt.title('Cb (Chrominance Blue) Histogram')

    plt.tight_layout()
    plt.show()

# Tkinter Functions
def load_image():
    """Load an image, check compatibility, and apply padding if needed."""
    global img_path, img, img_display
    img_path = filedialog.askopenfilename()
    if not img_path:
        return
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2YCrCb)  # Convert image to YCrCb color space

    # Check if the image dimensions are compatible
    if is_image_compatible(img):
        messagebox.showinfo("Compatibility Check", "Image is compatible for DCT Steganography.")
    else:
        response = messagebox.askyesno("Incompatible Image", "Image dimensions are not multiples of 8. Do you want to pad the image?")
        if response:
            img = pad_image(img)
            messagebox.showinfo("Padding Applied", "Image has been padded to compatible dimensions.")
        else:
            messagebox.showwarning("Process Aborted", "Please select a compatible image.")
            return

    display_image(img_path)
    update_info()

def display_image(img_path):
    """Display image in the Tkinter window."""
    img_display = Image.open(img_path)
    img_display = img_display.resize((250, 250))
    img_tk = ImageTk.PhotoImage(img_display)
    label_img.config(image=img_tk)
    label_img.image = img_tk

def update_info():
    """Update image information such as dimensions and max characters."""
    if img is None:
        label_info.config(text="No image loaded")
        return
    rows, cols, _ = img.shape
    _, max_chars = can_embed_message(img, "")
    label_info.config(text=f"Image dimensions: {rows}x{cols}\nMax characters: {max_chars}")

def encode_message():
    """Encode a message into the loaded image and save it."""
    global img
    message = text_message.get("1.0", END).strip()
    quality_factor = scale_quality.get()
    if img is None or not message:
        messagebox.showerror("Error", "Image and message required")
        return

    can_embed, max_chars = can_embed_message(img, message)
    if not can_embed:
        messagebox.showerror("Error", f"Message is too long for this image. Max allowed: {max_chars} characters.")
        return
    
    img_encoded = embed_message_dct(img.copy(), message, quality_factor)  # Embed message
    save_path = filedialog.asksaveasfilename(defaultextension=".png", title="Save Stego Image", 
                                             filetypes=[("PNG files", ".png"), ("JPEG files", ".jpg")])
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(img_encoded, cv2.COLOR_YCrCb2BGR))  # Save the encoded image
        messagebox.showinfo("Success", f"Message encoded and saved as '{save_path}'")
        display_image(save_path)
        
    # Clear the message box after encoding
    clear_message()

def decode_message():
    """Decode the hidden message from the image."""
    global img
    quality_factor = scale_quality.get()
    if img is None:
        messagebox.showerror("Error", "Load an image first")
        return

    decoded_msg = extract_message_dct(img, quality_factor)  # Decode the message
    clear_message()
    text_message.insert(END, f"Decoded message is: {decoded_msg}")  # Display decoded message

def clear_message():
    """Clear the message text box."""
    text_message.delete("1.0", END)

# GUI Setup
root = Tk()
root.title("DCT Steganography")
root.geometry("500x500")
root.configure(bg="#EAEAEA")

Button(root, text="Load Image", command=load_image, bg="#4CAF50", fg="white").pack(pady=5)
label_img = Label(root)
label_img.pack()

label_info = Label(root, text="Image dimensions: N/A\nMax characters: N/A", bg="#EAEAEA")
label_info.pack()

Label(root, text="Enter message:", bg="#EAEAEA").pack()
text_message = Text(root, height=4, wrap="word")
text_message.pack()

Button(root, text="Encode", command=encode_message, bg="#2196F3", fg="white").pack(pady=5)
Button(root, text="Decode", command=decode_message, bg="#FF5722", fg="white").pack(pady=5)
Button(root, text="Clear Message", command=clear_message, bg="#9E9E9E", fg="white").pack(pady=5)
Button(root, text="Show Histogram", command=show_histogram, bg="#8BC34A", fg="white").pack(pady=5)  # New histogram button

Label(root, text="Quality factor:", bg="#EAEAEA").pack()
scale_quality = Scale(root, from_=1, to=10, orient=HORIZONTAL)
scale_quality.set(5)
scale_quality.pack()

root.mainloop()
