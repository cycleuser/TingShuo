from PIL import Image, ImageDraw, ImageFont
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

def create_mock_gui_screenshot():
    width, height = 720, 920
    img = Image.new('RGB', (width, height), color=(240, 240, 240))
    d = ImageDraw.Draw(img)
    
    # Title Bar
    d.rectangle([0, 0, width, 30], fill=(200, 200, 200))
    d.text((10, 5), "TingShuo 听说", fill=(0, 0, 0))
    
    # Input/Output Section
    d.text((20, 50), "Input Directory:", fill=(0, 0, 0))
    d.rectangle([150, 45, 600, 75], fill=(255, 255, 255), outline=(0, 0, 0))
    d.rectangle([610, 45, 700, 75], fill=(220, 220, 220), outline=(0, 0, 0))
    d.text((630, 50), "Browse", fill=(0, 0, 0))
    
    d.text((20, 90), "Output Directory:", fill=(0, 0, 0))
    d.rectangle([150, 85, 600, 115], fill=(255, 255, 255), outline=(0, 0, 0))
    d.rectangle([610, 85, 700, 115], fill=(220, 220, 220), outline=(0, 0, 0))
    d.text((630, 90), "Browse", fill=(0, 0, 0))
    
    # Engine Settings
    d.rectangle([10, 140, 710, 240], outline=(150, 150, 150))
    d.text((20, 130), "Engine Settings", fill=(100, 100, 100))
    
    d.text((30, 160), "Engine:", fill=(0, 0, 0))
    d.rectangle([100, 155, 250, 185], fill=(255, 255, 255), outline=(0, 0, 0))
    d.text((110, 160), "faster-whisper", fill=(0, 0, 0))
    
    d.text((270, 160), "Model:", fill=(0, 0, 0))
    d.rectangle([330, 155, 450, 185], fill=(255, 255, 255), outline=(0, 0, 0))
    d.text((340, 160), "base", fill=(0, 0, 0))
    
    d.rectangle([470, 155, 550, 185], fill=(220, 220, 220), outline=(0, 0, 0))
    d.text((480, 160), "Download", fill=(0, 0, 0))
    
    # Other sections...
    d.rectangle([10, 260, 710, 320], outline=(150, 150, 150))
    d.text((20, 250), "Output Format", fill=(100, 100, 100))
    d.text((30, 280), "(o) SRT   ( ) LRC   ( ) Markdown", fill=(0, 0, 0))
    
    d.rectangle([10, 340, 710, 420], outline=(150, 150, 150))
    d.text((20, 330), "Features", fill=(100, 100, 100))
    d.text((30, 360), "[ ] Auto Correct", fill=(0, 0, 0))
    d.text((200, 360), "[ ] Summarize", fill=(0, 0, 0))
    d.text((400, 360), "[ ] Translate", fill=(0, 0, 0))
    
    # Log Area
    d.rectangle([10, 450, 710, 800], fill=(255, 255, 255), outline=(0, 0, 0))
    d.text((20, 460), "[INFO] Ready...", fill=(0, 0, 0))
    d.text((20, 480), "[INFO] Loaded faster-whisper model 'base'", fill=(0, 0, 0))
    
    # Start/Stop Buttons
    d.rectangle([10, 820, 100, 860], fill=(100, 200, 100), outline=(0, 0, 0))
    d.text((30, 830), "Start", fill=(0, 0, 0))
    
    d.rectangle([120, 820, 210, 860], fill=(200, 100, 100), outline=(0, 0, 0))
    d.text((140, 830), "Stop", fill=(0, 0, 0))
    
    img.save(os.path.join(IMAGES_DIR, "gui_screenshot.png"))
    print(f"Generated {os.path.join(IMAGES_DIR, 'gui_screenshot.png')}")

def create_mock_web_screenshot():
    width, height = 1200, 800
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    
    # Browser Bar
    d.rectangle([0, 0, width, 40], fill=(230, 230, 230))
    d.text((20, 10), "http://localhost:7860 - TingShuo Web", fill=(0, 0, 0))
    
    # Gradio Header
    d.text((50, 60), "TingShuo (听说) Web Interface", fill=(0, 0, 0), font=None) # Default font, maybe small but OK
    
    # Left Column (Inputs)
    d.rectangle([50, 100, 550, 750], outline=(200, 200, 200))
    d.text((70, 120), "Upload Audio/Video", fill=(0, 0, 0))
    d.rectangle([70, 150, 530, 250], fill=(245, 245, 245), outline=(200, 200, 200))
    d.text((250, 190), "Drop Files Here", fill=(150, 150, 150))
    
    d.text((70, 280), "Engine Settings", fill=(0, 0, 0))
    d.rectangle([70, 310, 530, 350], outline=(200, 200, 200)) # Dropdown
    d.text((80, 320), "faster-whisper", fill=(0, 0, 0))
    
    d.text((70, 380), "Output Format", fill=(0, 0, 0))
    d.text((80, 410), "(o) SRT   ( ) LRC   ( ) Markdown", fill=(0, 0, 0))
    
    d.rectangle([70, 700, 530, 740], fill=(255, 140, 0), outline=(0, 0, 0))
    d.text((250, 710), "Start Processing", fill=(255, 255, 255))
    
    # Right Column (Outputs)
    d.rectangle([600, 100, 1100, 750], outline=(200, 200, 200))
    d.text((620, 120), "Logs", fill=(0, 0, 0))
    d.rectangle([620, 150, 1080, 400], fill=(245, 245, 245), outline=(200, 200, 200))
    d.text((630, 160), "Processing started...", fill=(0, 0, 0))
    
    d.text((620, 430), "Generated Files", fill=(0, 0, 0))
    d.rectangle([620, 460, 1080, 600], fill=(245, 245, 245), outline=(200, 200, 200))
    d.text((630, 480), "Download output.srt", fill=(0, 0, 255))
    
    img.save(os.path.join(IMAGES_DIR, "web_screenshot.png"))
    print(f"Generated {os.path.join(IMAGES_DIR, 'web_screenshot.png')}")

if __name__ == "__main__":
    create_mock_gui_screenshot()
    create_mock_web_screenshot()
