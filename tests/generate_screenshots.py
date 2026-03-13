import sys
import os
import subprocess
import time
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

# Ensure we are in the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

console = Console(record=True, width=120)

def run_command(command, filename, title=None, height=None):
    """Run a command, capture output, and save as SVG."""
    console.clear()
    
    cmd_str = f"$ {command}"
    console.print(Text(cmd_str, style="bold green"))
    
    start_time = time.time()
    
    try:
        # Run the command
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=PROJECT_ROOT
        )
        
        stdout = result.stdout
        stderr = result.stderr
        
        if stdout:
            console.print(stdout)
        if stderr:
            console.print(Text(stderr, style="red"))
            
    except Exception as e:
        console.print(f"Error running command: {e}", style="bold red")

    # Save SVG
    svg_path = os.path.join(IMAGES_DIR, f"{filename}.svg")
    console.save_svg(svg_path, title=title or command)
    print(f"Generated {svg_path}")

def main():
    # 1. Help Command
    run_command(f"{sys.executable} tingshuo.py --help", "help_output", "TingShuo Help")

    # 2. Version Command
    run_command(f"{sys.executable} tingshuo.py --version", "version_output", "TingShuo Version")

    # 3. Create a dummy audio file if not exists
    if not os.path.exists("test_audio.wav"):
        import wave
        import struct
        sample_rate = 44100
        duration = 2 # seconds
        n_frames = int(sample_rate * duration)
        with wave.open("test_audio.wav", 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            data = struct.pack('<' + ('h'*n_frames), *([0]*n_frames))
            wav_file.writeframes(data)

    # 4. Transcription (Basic)
    # We use a very small model (tiny) and enable download if needed
    run_command(
        f"{sys.executable} tingshuo.py -i . -o test_output -f srt --no-recursive -e faster-whisper -m tiny --download", 
        "transcription_basic", 
        "Basic Transcription"
    )

    # 5. Model Download (Explicit)
    # We already downloaded tiny above, but let's show the command
    run_command(
        f"{sys.executable} tingshuo.py --download -e faster-whisper -m tiny", 
        "download_model", 
        "Model Download"
    )

    # 6. List Ollama Models
    run_command(
        f"{sys.executable} tingshuo.py --list-ollama-models", 
        "list_ollama", 
        "List Ollama Models"
    )

    # 7. Translation (Mock/Simulate)
    # We use NLLB for translation (default). It might need to download a model.
    # We skip actual execution if it takes too long, but let's try.
    # NLLB model 'facebook/nllb-200-distilled-600M' is ~600MB. Might be too big.
    # Let's skip translation execution screenshot if it requires huge download.
    # Instead, we show the command.
    
    # 8. Polishing with LLM (using available model)
    # We use 'huihui_ai/qwen3.5-abliterated:0.8B' as discovered
    ollama_model = "huihui_ai/qwen3.5-abliterated:0.8B"
    run_command(
        f"{sys.executable} tingshuo.py -i . -o test_output_polish -f md --polish-llm --ollama-model {ollama_model} --no-recursive",
        "polish_llm",
        "Subtitle Polishing with LLM"
    )

if __name__ == "__main__":
    main()
