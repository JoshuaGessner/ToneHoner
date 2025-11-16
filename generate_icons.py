"""
ToneHoner Icon Generator
Creates icons for client and server applications
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_gradient_background(size, color1, color2):
    """Create a gradient background"""
    image = Image.new('RGB', size, color1)
    draw = ImageDraw.Draw(image)
    
    for y in range(size[1]):
        ratio = y / size[1]
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        draw.line([(0, y), (size[0], y)], fill=(r, g, b))
    
    return image

def create_waveform_icon(size, gradient_colors, waveform_color, label_text=""):
    """Create an icon with audio waveform design"""
    img = create_gradient_background(size, gradient_colors[0], gradient_colors[1])
    draw = ImageDraw.Draw(img)
    
    # Draw stylized waveform
    center_y = size[1] // 2
    wave_width = int(size[0] * 0.7)
    start_x = (size[0] - wave_width) // 2
    
    # Multiple waveform bars with varying heights
    bar_count = 12
    bar_width = wave_width // (bar_count * 2)
    spacing = bar_width
    
    for i in range(bar_count):
        x = start_x + i * (bar_width + spacing)
        # Create varying heights for visual interest
        height_factor = abs(i - bar_count // 2) / (bar_count // 2)
        max_height = int(size[1] * 0.4)
        bar_height = int(max_height * (1 - height_factor * 0.5))
        
        # Draw bar with rounded corners effect
        y1 = center_y - bar_height // 2
        y2 = center_y + bar_height // 2
        draw.rectangle([x, y1, x + bar_width, y2], fill=waveform_color)
    
    # Add label if provided
    if label_text:
        try:
            # Try to use a nice font, fall back to default if not available
            font_size = size[0] // 10
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Get text bounding box
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            text_x = (size[0] - text_width) // 2
            text_y = size[1] - text_height - size[1] // 8
            
            # Draw text with shadow for better visibility
            shadow_offset = 2
            draw.text((text_x + shadow_offset, text_y + shadow_offset), label_text, 
                     fill=(0, 0, 0, 128), font=font)
            draw.text((text_x, text_y), label_text, fill=(255, 255, 255), font=font)
        except:
            pass  # Skip label if font rendering fails
    
    return img

def create_client_icon():
    """Create client icon with blue/cyan theme"""
    # Blue gradient for client (microphone/input focus)
    gradient_colors = [(30, 60, 114), (42, 157, 244)]  # Dark blue to bright blue
    waveform_color = (100, 200, 255)  # Light cyan
    
    # Create multiple sizes for ICO format
    sizes = [256, 128, 64, 48, 32, 16]
    images = []
    
    for size in sizes:
        img = create_waveform_icon(
            (size, size), 
            gradient_colors, 
            waveform_color,
            label_text="C" if size >= 64 else ""
        )
        images.append(img)
    
    # Save as ICO with multiple sizes
    images[0].save('client_icon.ico', format='ICO', sizes=[(s, s) for s in sizes])
    # Also save PNG for other uses
    images[0].save('client_icon.png', format='PNG')
    print("✓ Created client_icon.ico and client_icon.png")

def create_server_icon():
    """Create server icon with green/teal theme"""
    # Green gradient for server (processing/enhancement focus)
    gradient_colors = [(34, 87, 122), (87, 187, 138)]  # Dark teal to green
    waveform_color = (144, 238, 144)  # Light green
    
    # Create multiple sizes for ICO format
    sizes = [256, 128, 64, 48, 32, 16]
    images = []
    
    for size in sizes:
        img = create_waveform_icon(
            (size, size), 
            gradient_colors, 
            waveform_color,
            label_text="S" if size >= 64 else ""
        )
        images.append(img)
    
    # Save as ICO with multiple sizes
    images[0].save('server_icon.ico', format='ICO', sizes=[(s, s) for s in sizes])
    # Also save PNG for other uses
    images[0].save('server_icon.png', format='PNG')
    print("✓ Created server_icon.ico and server_icon.png")

def create_installer_icon():
    """Create installer icon with purple/orange theme"""
    # Purple/orange gradient for installer (setup theme)
    gradient_colors = [(88, 24, 69), (199, 125, 72)]  # Purple to orange
    waveform_color = (255, 195, 113)  # Light orange
    
    # Create multiple sizes for ICO format
    sizes = [256, 128, 64, 48, 32, 16]
    images = []
    
    for size in sizes:
        img = create_waveform_icon(
            (size, size), 
            gradient_colors, 
            waveform_color,
            label_text="T" if size >= 64 else ""
        )
        images.append(img)
    
    # Save as ICO with multiple sizes
    images[0].save('installer_icon.ico', format='ICO', sizes=[(s, s) for s in sizes])
    # Also save PNG for other uses
    images[0].save('installer_icon.png', format='PNG')
    print("✓ Created installer_icon.ico and installer_icon.png")

if __name__ == "__main__":
    print("ToneHoner Icon Generator")
    print("=" * 50)
    
    # Check if Pillow is installed
    try:
        import PIL
        print(f"Using Pillow version {PIL.__version__}")
    except ImportError:
        print("ERROR: Pillow is not installed.")
        print("Install it with: pip install Pillow")
        exit(1)
    
    print("\nGenerating icons...")
    create_client_icon()
    create_server_icon()
    create_installer_icon()
    
    print("\n" + "=" * 50)
    print("Icon generation complete!")
    print("\nGenerated files:")
    print("  - client_icon.ico / client_icon.png (Blue theme)")
    print("  - server_icon.ico / server_icon.png (Green theme)")
    print("  - installer_icon.ico / installer_icon.png (Purple/Orange theme)")
    print("\nNext steps:")
    print("  1. Update PyInstaller spec files to use the icon files")
    print("  2. Update Inno Setup scripts to use the icon files")
    print("  3. Rebuild executables and installers")
