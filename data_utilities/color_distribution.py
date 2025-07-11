"""
Script that gives an overview of the RGB and HSV distribution in the TMA images.
"""

# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
Image.MAX_IMAGE_PIXELS = None

# %% LOAD PATHS
img_path = "/local/data3/chrsp39/DPD-AI/TMA"

slide_ids = os.listdir(img_path)

# Initialize cumulative histograms for each channel (256 bins for 0-255 values)
cumulative_r_hist = np.zeros(256)
cumulative_g_hist = np.zeros(256)
cumulative_b_hist = np.zeros(256)

# Initialize HSV histograms
# H: 0-179 (OpenCV convention), S: 0-255, V: 0-255
cumulative_h_hist = np.zeros(180)  # Hue: 0-179
cumulative_s_hist = np.zeros(256)  # Saturation: 0-255
cumulative_v_hist = np.zeros(256)  # Value: 0-255

print(f"Processing {len(slide_ids)} images...")

for i, slide_id in enumerate(slide_ids):
    slide_name = os.path.splitext(slide_id)[0]
    slide_id_path = os.path.join(img_path, slide_id)
    
    print(f"Processing {i+1}/{len(slide_ids)}: {slide_name}")

    # Open image with PIL
    image = Image.open(slide_id_path)
    # Convert to RGB if necessary
    if image.mode in ('RGBA', 'P', 'L'):
        image = image.convert('RGB')
    
    # Get RGB distribution
    r, g, b = image.split()
    
    r_hist = np.array(r.histogram())
    g_hist = np.array(g.histogram())
    b_hist = np.array(b.histogram())
    
    # Add to cumulative histograms
    cumulative_r_hist += r_hist
    cumulative_g_hist += g_hist
    cumulative_b_hist += b_hist
    
    # Convert to HSV for HSV analysis
    # Convert PIL image to numpy array
    image_array = np.array(image)
    # Convert RGB to HSV using cv2
    hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    
    # Split HSV channels
    h, s, v = cv2.split(hsv_image)
    
    # Calculate histograms for HSV
    h_hist, _ = np.histogram(h.flatten(), bins=180, range=(0, 180))
    s_hist, _ = np.histogram(s.flatten(), bins=256, range=(0, 256))
    v_hist, _ = np.histogram(v.flatten(), bins=256, range=(0, 256))
    
    # Add to cumulative HSV histograms
    cumulative_h_hist += h_hist
    cumulative_s_hist += s_hist
    cumulative_v_hist += v_hist

print("Processing complete!")

# %% PLOT RGB DISTRIBUTION
# Create the plot
plt.figure(figsize=(16, 12))

# RGB Plots
# Create x-axis (pixel intensity values 0-255)
x_rgb = np.arange(256)

# Plot histograms for each RGB channel
plt.subplot(3, 3, 1)
plt.plot(x_rgb, cumulative_r_hist, 'r-', linewidth=2, label='Red')
plt.title('Red Channel Distribution')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 2)
plt.plot(x_rgb, cumulative_g_hist, 'g-', linewidth=2, label='Green')
plt.title('Green Channel Distribution')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 3)
plt.plot(x_rgb, cumulative_b_hist, 'b-', linewidth=2, label='Blue')
plt.title('Blue Channel Distribution')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Combined RGB plot
plt.subplot(3, 3, 4)
plt.plot(x_rgb, cumulative_r_hist, 'r-', linewidth=2, label='Red', alpha=0.7)
plt.plot(x_rgb, cumulative_g_hist, 'g-', linewidth=2, label='Green', alpha=0.7)
plt.plot(x_rgb, cumulative_b_hist, 'b-', linewidth=2, label='Blue', alpha=0.7)
plt.title('Combined RGB Distribution')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

# HSV Plots
# Create x-axis for Hue (0-179) and Saturation/Value (0-255)
x_h = np.arange(180)
x_sv = np.arange(256)

plt.subplot(3, 3, 5)
plt.plot(x_h, cumulative_h_hist, color='orange', linewidth=2, label='Hue')
plt.title('Hue Distribution')
plt.xlabel('Hue Value (0-179)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 6)
plt.plot(x_sv, cumulative_s_hist, color='purple', linewidth=2, label='Saturation')
plt.title('Saturation Distribution')
plt.xlabel('Saturation Value (0-255)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 7)
plt.plot(x_sv, cumulative_v_hist, color='brown', linewidth=2, label='Value')
plt.title('Value Distribution')
plt.xlabel('Value/Brightness (0-255)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Combined HSV plot
plt.subplot(3, 3, 8)
# Normalize hue to 0-255 range for visualization
normalized_h_hist = cumulative_h_hist * (255/179)
x_h_norm = np.arange(180) * (255/179)
plt.plot(x_h_norm, normalized_h_hist, color='orange', linewidth=2, label='Hue (normalized)', alpha=0.7)
plt.plot(x_sv, cumulative_s_hist, color='purple', linewidth=2, label='Saturation', alpha=0.7)
plt.plot(x_sv, cumulative_v_hist, color='brown', linewidth=2, label='Value', alpha=0.7)
plt.title('Combined HSV Distribution')
plt.xlabel('Value (0-255)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('color_distribution_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# %% PRINT STATISTICS
print("\n=== RGB & HSV Distribution Statistics ===")
print(f"Total pixels processed: {int(cumulative_r_hist.sum()):,}")

# RGB Statistics
x_rgb = np.arange(256)
print(f"\n--- RGB Channel Statistics ---")
print(f"Red Channel:")
print(f"  Mean intensity: {np.average(x_rgb, weights=cumulative_r_hist):.2f}")
print(f"  Std deviation: {np.sqrt(np.average((x_rgb - np.average(x_rgb, weights=cumulative_r_hist))**2, weights=cumulative_r_hist)):.2f}")
print(f"  Peak intensity: {x_rgb[np.argmax(cumulative_r_hist)]}")

print(f"\nGreen Channel:")
print(f"  Mean intensity: {np.average(x_rgb, weights=cumulative_g_hist):.2f}")
print(f"  Std deviation: {np.sqrt(np.average((x_rgb - np.average(x_rgb, weights=cumulative_g_hist))**2, weights=cumulative_g_hist)):.2f}")
print(f"  Peak intensity: {x_rgb[np.argmax(cumulative_g_hist)]}")

print(f"\nBlue Channel:")
print(f"  Mean intensity: {np.average(x_rgb, weights=cumulative_b_hist):.2f}")
print(f"  Std deviation: {np.sqrt(np.average((x_rgb - np.average(x_rgb, weights=cumulative_b_hist))**2, weights=cumulative_b_hist)):.2f}")
print(f"  Peak intensity: {x_rgb[np.argmax(cumulative_b_hist)]}")

# save them in a text file
with open('rgb_statistics.txt', 'w') as f:
    f.write(f"Total pixels processed: {int(cumulative_r_hist.sum()):,}\n")
    f.write(f"Red Channel:\n")
    f.write(f"  Mean intensity: {np.average(x_rgb, weights=cumulative_r_hist):.2f}\n")
    f.write(f"  Std deviation: {np.sqrt(np.average((x_rgb - np.average(x_rgb, weights=cumulative_r_hist))**2, weights=cumulative_r_hist)):.2f}\n")
    f.write(f"  Peak intensity: {x_rgb[np.argmax(cumulative_r_hist)]}\n\n")
    
    f.write(f"Green Channel:\n")
    f.write(f"  Mean intensity: {np.average(x_rgb, weights=cumulative_g_hist):.2f}\n")
    f.write(f"  Std deviation: {np.sqrt(np.average((x_rgb - np.average(x_rgb, weights=cumulative_g_hist))**2, weights=cumulative_g_hist)):.2f}\n")
    f.write(f"  Peak intensity: {x_rgb[np.argmax(cumulative_g_hist)]}\n\n")
    
    f.write(f"Blue Channel:\n")
    f.write(f"  Mean intensity: {np.average(x_rgb, weights=cumulative_b_hist):.2f}\n")
    f.write(f"  Std deviation: {np.sqrt(np.average((x_rgb - np.average(x_rgb, weights=cumulative_b_hist))**2, weights=cumulative_b_hist)):.2f}\n")
    f.write(f"  Peak intensity: {x_rgb[np.argmax(cumulative_b_hist)]}\n")

# HSV Statistics
x_h = np.arange(180)
x_sv = np.arange(256)
print(f"\n--- HSV Channel Statistics ---")
print(f"Hue Channel (0-179):")
print(f"  Mean hue: {np.average(x_h, weights=cumulative_h_hist):.2f}")
print(f"  Std deviation: {np.sqrt(np.average((x_h - np.average(x_h, weights=cumulative_h_hist))**2, weights=cumulative_h_hist)):.2f}")
print(f"  Peak hue: {x_h[np.argmax(cumulative_h_hist)]}")

print(f"\nSaturation Channel (0-255):")
print(f"  Mean saturation: {np.average(x_sv, weights=cumulative_s_hist):.2f}")
print(f"  Std deviation: {np.sqrt(np.average((x_sv - np.average(x_sv, weights=cumulative_s_hist))**2, weights=cumulative_s_hist)):.2f}")
print(f"  Peak saturation: {x_sv[np.argmax(cumulative_s_hist)]}")

print(f"\nValue/Brightness Channel (0-255):")
print(f"  Mean value: {np.average(x_sv, weights=cumulative_v_hist):.2f}")
print(f"  Std deviation: {np.sqrt(np.average((x_sv - np.average(x_sv, weights=cumulative_v_hist))**2, weights=cumulative_v_hist)):.2f}")
print(f"  Peak value: {x_sv[np.argmax(cumulative_v_hist)]}")

# Additional HSV insights
print(f"\n--- HSV Insights ---")
# Calculate percentages for saturation ranges
low_sat = np.sum(cumulative_s_hist[0:64]) / np.sum(cumulative_s_hist) * 100
med_sat = np.sum(cumulative_s_hist[64:192]) / np.sum(cumulative_s_hist) * 100
high_sat = np.sum(cumulative_s_hist[192:256]) / np.sum(cumulative_s_hist) * 100

print(f"Saturation distribution:")
print(f"  Low saturation (0-63): {low_sat:.1f}%")
print(f"  Medium saturation (64-191): {med_sat:.1f}%")
print(f"  High saturation (192-255): {high_sat:.1f}%")

# Calculate percentages for value/brightness ranges
low_val = np.sum(cumulative_v_hist[0:64]) / np.sum(cumulative_v_hist) * 100
med_val = np.sum(cumulative_v_hist[64:192]) / np.sum(cumulative_v_hist) * 100
high_val = np.sum(cumulative_v_hist[192:256]) / np.sum(cumulative_v_hist) * 100

print(f"\nBrightness distribution:")
print(f"  Dark (0-63): {low_val:.1f}%")
print(f"  Medium brightness (64-191): {med_val:.1f}%")
print(f"  Bright (192-255): {high_val:.1f}%")

# save them in a text file
with open('hsv_statistics.txt', 'w') as f:
    f.write(f"Hue Channel (0-179):\n")
    f.write(f"  Mean hue: {np.average(x_h, weights=cumulative_h_hist):.2f}\n")
    f.write(f"  Std deviation: {np.sqrt(np.average((x_h - np.average(x_h, weights=cumulative_h_hist))**2, weights=cumulative_h_hist)):.2f}\n")
    f.write(f"  Peak hue: {x_h[np.argmax(cumulative_h_hist)]}\n\n")
    
    f.write(f"Saturation Channel (0-255):\n")
    f.write(f"  Mean saturation: {np.average(x_sv, weights=cumulative_s_hist):.2f}\n")
    f.write(f"  Std deviation: {np.sqrt(np.average((x_sv - np.average(x_sv, weights=cumulative_s_hist))**2, weights=cumulative_s_hist)):.2f}\n")
    f.write(f"  Peak saturation: {x_sv[np.argmax(cumulative_s_hist)]}\n\n")
    
    f.write(f"Value/Brightness Channel (0-255):\n")
    f.write(f"  Mean value: {np.average(x_sv, weights=cumulative_v_hist):.2f}\n")
    f.write(f"  Std deviation: {np.sqrt(np.average((x_sv - np.average(x_sv, weights=cumulative_v_hist))**2, weights=cumulative_v_hist)):.2f}\n")
    f.write(f"  Peak value: {x_sv[np.argmax(cumulative_v_hist)]}\n\n")
    
    f.write(f"Saturation distribution:\n")
    f.write(f"  Low saturation (0-63): {low_sat:.1f}%\n")
    f.write(f"  Medium saturation (64-191): {med_sat:.1f}%\n")
    f.write(f"  High saturation (192-255): {high_sat:.1f}%\n\n")
    
    f.write(f"Brightness distribution:\n")
    f.write(f"  Dark (0-63): {low_val:.1f}%\n")
    f.write(f"  Medium brightness (64-191): {med_val:.1f}%\n")
    f.write(f"  Bright (192-255): {high_val:.1f}%\n")

# %%