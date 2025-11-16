"""
ToneHoner Icon Generator
Creates icons for client and server applications
"""
from PIL import Image, ImageDraw, ImageFont, ImageChops
import os

def create_gradient_background(size, color1, color2, mode='RGB'):
    """Create a vertical gradient background image.
    mode: 'RGB' or 'RGBA'
    """
    image = Image.new(mode, size, color1 + ((255,) if mode == 'RGBA' and len(color1) == 3 else ()))
    draw = ImageDraw.Draw(image)
    
    for y in range(size[1]):
        ratio = y / size[1]
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        a = 255
        draw.line([(0, y), (size[0], y)], fill=(r, g, b, a) if image.mode == 'RGBA' else (r, g, b))
    
    return image

def create_waveform_icon(size, gradient_colors, waveform_color, label_text="", *, round_mask=False):
    """Create an icon with audio waveform design. If round_mask=True, output has transparent
    background with circular content area and subtle outline for sleek look.
    """
    if round_mask:
        # Create RGBA gradient
        img = create_gradient_background(size, gradient_colors[0], gradient_colors[1], mode='RGBA')
        # Circular mask with anti-aliasing
        w, h = size
        mask = Image.new('L', (w*4, h*4), 0)
        mdraw = ImageDraw.Draw(mask)
        margin = int(min(w, h) * 0.08) * 4
        mdraw.ellipse([margin, margin, w*4 - margin, h*4 - margin], fill=255)
        mask = mask.resize((w, h), Image.LANCZOS)
        # Apply mask to alpha
        alpha = img.split()[3]
        alpha = ImageChops.multiply(alpha, mask)
        img.putalpha(alpha)
    else:
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
            shadow_fill = (0, 0, 0, 180) if img.mode == 'RGBA' else (0, 0, 0)
            text_fill = (255, 255, 255, 255) if img.mode == 'RGBA' else (255, 255, 255)
            draw.text((text_x + shadow_offset, text_y + shadow_offset), label_text, 
                     fill=shadow_fill, font=font)
            draw.text((text_x, text_y), label_text, fill=text_fill, font=font)
        except:
            pass  # Skip label if font rendering fails
    
    # Subtle circular outline when round
    if img.mode == 'RGBA':
        w, h = img.size
        outline = Image.new('RGBA', img.size, (0,0,0,0))
        odraw = ImageDraw.Draw(outline)
        radius_margin = int(min(w, h) * 0.06)
        odraw.ellipse([radius_margin, radius_margin, w - radius_margin, h - radius_margin], outline=(255,255,255,90), width=max(1, w//64))
        img = Image.alpha_composite(img, outline)

    return img

def _draw_wave_accent(draw: ImageDraw.ImageDraw, size, color):
    """Draw a subtle wave accent across the banner for branding."""
    w, h = size
    mid = h // 2
    # 6 bars with varying heights
    num = 12
    margin = int(w * 0.06)
    span = w - margin * 2
    bar_w = max(2, span // (num * 2))
    spacing = bar_w
    for i in range(num):
        x = margin + i * (bar_w + spacing)
        t = abs(i - num // 2) / (num / 2)
        bh = int(h * (0.35 - 0.18 * t))
        y1 = mid - bh // 2
        y2 = mid + bh // 2
        draw.rectangle([x, y1, x + bar_w, y2], fill=color)

def create_wizard_banner(size, gradient_colors, accent_color, title_text: str, subtitle_text: str = ""):
    """Create a banner-style image for Inno Setup Wizard (modern style).
    Typical sizes: 352x120 (large), 55x55 (small).
    """
    img = create_gradient_background(size, gradient_colors[0], gradient_colors[1])
    draw = ImageDraw.Draw(img)

    # Accent waveform
    try:
        _draw_wave_accent(draw, size, accent_color)
    except Exception:
        pass

    # Title/subtitle text
    try:
        w, h = size
        # Choose font sizes relative to banner height
        title_fs = max(12, h // 4)
        subtitle_fs = max(8, h // 6)
        try:
            title_font = ImageFont.truetype("arial.ttf", title_fs)
            subtitle_font = ImageFont.truetype("arial.ttf", subtitle_fs)
        except Exception:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()

        # Title
        if title_text:
            tb = draw.textbbox((0, 0), title_text, font=title_font)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
            tx = min(w - tw - 10, 12)
            ty = 10
            # shadow
            draw.text((tx+1, ty+1), title_text, font=title_font, fill=(0,0,0,160))
            draw.text((tx, ty), title_text, font=title_font, fill=(255,255,255))

        # Subtitle
        if subtitle_text:
            sb = draw.textbbox((0, 0), subtitle_text, font=subtitle_font)
            sw = sb[2] - sb[0]
            sh = sb[3] - sb[1]
            sx = min(w - sw - 10, 12)
            sy = ty + th + 6
            draw.text((sx+1, sy+1), subtitle_text, font=subtitle_font, fill=(0,0,0,160))
            draw.text((sx, sy), subtitle_text, font=subtitle_font, fill=(230,230,230))
    except Exception:
        pass

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
            label_text="C" if size >= 64 else "",
            round_mask=True
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
            label_text="S" if size >= 64 else "",
            round_mask=True
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

def create_installer_wizard_images():
    """Create Installer wizard images for Inno Setup modern style.
    - Large banner: 352x120 BMP
    - Small logo:   55x55 BMP
    """
    gradient_colors = [(88, 24, 69), (199, 125, 72)]  # Purple to orange
    accent_color = (255, 195, 113)

    # Large banner
    large = create_wizard_banner((352, 120), gradient_colors, accent_color, "ToneHoner", "Audio Enhancement")
    large.save('installer_wizard_large.bmp', format='BMP')

    # Small square
    small = create_wizard_banner((55, 55), gradient_colors, accent_color, "TH", "")
    small.save('installer_wizard_small.bmp', format='BMP')

    print("✓ Created installer_wizard_large.bmp and installer_wizard_small.bmp")

def create_client_wizard_images():
    """Create client-specific wizard images (blue theme)."""
    gradient_colors = [(30, 60, 114), (42, 157, 244)]
    accent_color = (100, 200, 255)
    large = create_wizard_banner((352, 120), gradient_colors, accent_color, "ToneHoner Client", "Real-time Enhancement")
    large.save('installer_wizard_client_large.bmp', format='BMP')
    small = create_wizard_banner((55, 55), gradient_colors, accent_color, "C", "")
    small.save('installer_wizard_client_small.bmp', format='BMP')
    print("✓ Created client wizard BMPs")

def create_server_wizard_images():
    """Create server-specific wizard images (green/teal theme)."""
    gradient_colors = [(34, 87, 122), (87, 187, 138)]
    accent_color = (144, 238, 144)
    large = create_wizard_banner((352, 120), gradient_colors, accent_color, "ToneHoner Server", "DeepFilterNet Engine")
    large.save('installer_wizard_server_large.bmp', format='BMP')
    small = create_wizard_banner((55, 55), gradient_colors, accent_color, "S", "")
    small.save('installer_wizard_server_small.bmp', format='BMP')
    print("✓ Created server wizard BMPs")

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
    create_installer_wizard_images()
    create_client_wizard_images()
    create_server_wizard_images()
    
    print("\n" + "=" * 50)
    print("Icon generation complete!")
    print("\nGenerated files:")
    print("  - client_icon.ico / client_icon.png (Blue theme)")
    print("  - server_icon.ico / server_icon.png (Green theme)")
    print("  - installer_icon.ico / installer_icon.png (Purple/Orange theme)")
    print("  - installer_wizard_large.bmp (352x120) / installer_wizard_small.bmp (55x55)")
    print("  - installer_wizard_client_large.bmp / installer_wizard_client_small.bmp")
    print("  - installer_wizard_server_large.bmp / installer_wizard_server_small.bmp")
    print("\nNext steps:")
    print("  1. Update PyInstaller spec files to use the icon files")
    print("  2. Update Inno Setup scripts to use the icon files")
    print("  3. Rebuild executables and installers")
