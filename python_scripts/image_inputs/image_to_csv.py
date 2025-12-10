from PIL import Image
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd

# --- Function 1: Convert Image to CSV (from previous turn) ---
def image_to_pixel_csv(image_path, output_csv_path):
    """
    Converts a black and white image into a CSV file.
    
    (1 for black, 0 for white)
    """
    try:
        # Load the Image
        img = Image.open(image_path)
        img_bw = img.convert('L') # Convert to Grayscale (0-255)
        width, height = img_bw.size
        pixels = img_bw.load()

        print(f"✅ Image '{os.path.basename(image_path)}' loaded. Dimensions: {width}x{height}")
        
        # Write to CSV
        with open(output_csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['x', 'y', 'pixel_value'])

            # Iterate Pixels & Determine Pixel Value
            for y in range(height):
                for x in range(width):
                    grayscale_value = pixels[x, y]
                    
                    # Map: Black (0) -> 1, White (255) -> 0
                    pixel_output = 1 if grayscale_value < 128 else 0

                    csv_writer.writerow([x, y, pixel_output])

        print(f"🎉 Success! Data written to '{output_csv_path}'")

    except FileNotFoundError:
        print(f"❌ Error: Image file not found at '{image_path}'")
    except Exception as e:
        print(f"❌ An error occurred: {e}. Check if the image format is supported.")


# --- Function 2: Draw CSV Data with Matplotlib (The Extension) ---
def draw_pixel_csv(input_csv_path, image_width, image_height):
    """
    Loads pixel data from a CSV and visualizes it using Matplotlib.
    
    Args:
        input_csv_path (str): The file path to the input CSV.
        image_width (int): The width of the original image (used for plot size).
        image_height (int): The height of the original image (used for plot size).
    """
    try:
        print(f"\n🎨 Starting Matplotlib visualization for '{input_csv_path}'...")
        
        # Load the data from the CSV file
        df = pd.read_csv(input_csv_path)

        # Separate the columns
        x_coords = df['x']
        y_coords = df['y']
        pixel_values = df['pixel_value'] # 1 for black, 0 for white

        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(image_width/100, image_height/100)) # Adjust size based on pixel count

        # Scatter plot the data
        # c=pixel_values: uses the pixel_value column for coloring
        # cmap='gray_r': reverse grayscale. '0' (white) plots as white, '1' (black) plots as black.
        # s=1: makes each point tiny, simulating a pixel
        ax.scatter(
            x_coords, 
            y_coords, 
            c=pixel_values, 
            cmap='gray_r', 
            s=1,
            marker='s' # Use square markers for a more "pixel-like" look
        )

        # Set limits and aspect ratio for accurate image representation
        ax.set_xlim(0, image_width)
        ax.set_ylim(image_height, 0) # Invert Y-axis so (0,0) is top-left, like an image
        ax.set_aspect('equal') # Ensure pixels are square

        # Remove axes ticks and labels for a cleaner "image" look
        ax.axis('off')

        plt.title('Reconstructed Image from CSV Data')
        plt.show()
        print("🖼️ Displayed plot successfully!")

    except FileNotFoundError:
        print(f"❌ Error: CSV file not found at '{input_csv_path}'")
    except Exception as e:
        print(f"❌ An error occurred during drawing: {e}")


# --- Configuration and Execution ---
IMAGE_FILE = 'Untitled.png'      # **CHANGE THIS to your image file name (e.g., lion_bw.png)**
OUTPUT_CSV_FILE = 'pixel_data_10240.csv'

# NOTE: We need the original image dimensions for the Matplotlib plotting function
# to properly size and orient the plot. We'll get them before conversion.
try:
    img = Image.open(IMAGE_FILE)
    IMG_WIDTH, IMG_HEIGHT = img.size
except FileNotFoundError:
    print(f"\n⚠️ Cannot find image file: '{IMAGE_FILE}'. Please fix the file name.")
    exit() # Stop execution if image isn't found

# 1. Run the CSV conversion
image_to_pixel_csv(IMAGE_FILE, OUTPUT_CSV_FILE)

# 2. Run the Matplotlib drawing extension
draw_pixel_csv(OUTPUT_CSV_FILE, IMG_WIDTH, IMG_HEIGHT)