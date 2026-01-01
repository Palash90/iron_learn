from PIL import Image
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd

def image_to_pixel_csv(image_path, output_csv_path):
    """
    Converts a black and white image into a CSV file.
    
    (1 for black, 0 for white)
    """
    try:
        img = Image.open(image_path)
        img_bw = img.convert('L')
        width, height = img_bw.size
        pixels = img_bw.load()

        print(f"✅ Image '{os.path.basename(image_path)}' loaded. Dimensions: {width}x{height}")
        
        with open(output_csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['x', 'y', 'pixel_value'])

            for y in range(height):
                for x in range(width):
                    grayscale_value = pixels[x, y]
                    
                    pixel_output = 1 if grayscale_value < 128 else 0

                    csv_writer.writerow([x, y, pixel_output])

        print(f"🎉 Success! Data written to '{output_csv_path}'")

    except FileNotFoundError:
        print(f"❌ Error: Image file not found at '{image_path}'")
    except Exception as e:
        print(f"❌ An error occurred: {e}. Check if the image format is supported.")


def draw_pixel_csv(input_csv_path, image_width, image_height):
    try:
        print(f"\n🎨 Starting Matplotlib visualization for '{input_csv_path}'...")
        
        df = pd.read_csv(input_csv_path)

        x_coords = df['x']
        y_coords = df['y']
        pixel_values = df['pixel_value']

        fig, ax = plt.subplots(figsize=(image_width/100, image_height/100))

        ax.scatter(
            x_coords, 
            y_coords, 
            c=pixel_values, 
            cmap='gray_r', 
            s=1,
            marker='s'
        )

        ax.set_xlim(0, image_width)
        ax.set_ylim(image_height, 0)
        ax.set_aspect('equal')

        ax.axis('off')

        plt.title('Reconstructed Image from CSV Data')
        plt.show()
        print("🖼️ Displayed plot successfully!")

    except FileNotFoundError:
        print(f"❌ Error: CSV file not found at '{input_csv_path}'")
    except Exception as e:
        print(f"❌ An error occurred during drawing: {e}")


IMAGE_FILE = 'Untitled.png'      
OUTPUT_CSV_FILE = 'pixel_data_10240.csv'

try:
    img = Image.open(IMAGE_FILE)
    IMG_WIDTH, IMG_HEIGHT = img.size
except FileNotFoundError:
    print(f"\n⚠️ Cannot find image file: '{IMAGE_FILE}'. Please fix the file name.")
    exit()
    
image_to_pixel_csv(IMAGE_FILE, OUTPUT_CSV_FILE)

draw_pixel_csv(OUTPUT_CSV_FILE, IMG_WIDTH, IMG_HEIGHT)