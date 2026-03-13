import gradio as gr
import os
import sys
import threading
from queue import Queue
import logging

# Import tingshuo functionality
import tingshuo
from tingshuo import JobConfig, PolishConfig, TranslationConfig, SummarizeConfig, process_batch, SUPPORTED_ENGINES, ENGINE_MODELS, ENGINE_DEFAULT_MODEL, LANGUAGE_CODES

# Setup logging to capture output
log_queue = Queue()
class QueueHandler(logging.Handler):
    def emit(self, record):
        log_queue.put(self.format(record))

logger = logging.getLogger("tingshuo")
logger.addHandler(QueueHandler())
logger.setLevel(logging.INFO)

def process_media(
    input_files, 
    engine, 
    model, 
    language, 
    output_format, 
    polish_method, 
    ollama_url, 
    ollama_model,
    translate_enabled,
    target_langs,
    summarize_enabled
):
    if not input_files:
        return "Please upload files.", ""

    # Create a temporary directory for input if needed, or just use the file paths
    # Gradio provides temp paths for uploaded files.
    # But tingshuo expects an input directory or we need to modify it to accept file list.
    # tingshuo.process_batch takes a config which has input_dir.
    # We might need to adapt tingshuo to accept a list of files or put files in a dir.
    
    # Let's create a temp input dir
    import tempfile
    import shutil
    
    temp_input_dir = tempfile.mkdtemp(prefix="tingshuo_in_")
    temp_output_dir = tempfile.mkdtemp(prefix="tingshuo_out_")
    
    try:
        for f in input_files:
            shutil.copy(f.name, os.path.join(temp_input_dir, os.path.basename(f.name)))
            
        # Build Config
        polish = PolishConfig()
        if polish_method == "LLM (Ollama)":
            polish.method = "llm"
            polish.ollama_url = ollama_url
            polish.ollama_model = ollama_model
        elif polish_method == "NLP (nltk)":
            polish.method = "nlp"
            
        translation = TranslationConfig()
        if translate_enabled:
            translation.enabled = True
            translation.target_languages = target_langs.split(",") if target_langs else []
            
        summarize = SummarizeConfig()
        if summarize_enabled:
            summarize.enabled = True
            
        config = JobConfig(
            input_dir=temp_input_dir,
            output_dir=temp_output_dir,
            format=output_format.lower(),
            engine_name=engine,
            model_name=model,
            language=None if language == "auto" else language,
            polish=polish,
            translation=translation,
            summarize=summarize,
            recursive=False
        )
        
        # Run processing
        logs = []
        def progress_cb(curr, total, name):
            logs.append(f"Processing {curr+1}/{total}: {name}")
            
        success, total = process_batch(config, progress_cb=progress_cb)
        
        # Collect outputs
        output_files = []
        for root, dirs, files in os.walk(temp_output_dir):
            for file in files:
                output_files.append(os.path.join(root, file))
                
        log_str = "\n".join(logs)
        if success == total:
            log_str += "\n\nAll tasks completed successfully."
        else:
            log_str += f"\n\nCompleted {success}/{total} tasks."
            
        return log_str, output_files
        
    except Exception as e:
        return f"Error: {str(e)}", []
    finally:
        # Cleanup input dir (we keep output dir for download)
        shutil.rmtree(temp_input_dir, ignore_errors=True)

with gr.Blocks(title="TingShuo Web") as demo:
    gr.Markdown("# TingShuo (听说) Web Interface")
    gr.Markdown("Generate SRT/LRC subtitles and Markdown transcripts from audio/video files.")
    
    with gr.Row():
        with gr.Column():
            input_files = gr.File(label="Upload Audio/Video", file_count="multiple")
            
            with gr.Group():
                gr.Markdown("### STT Engine Settings")
                engine = gr.Dropdown(choices=list(SUPPORTED_ENGINES), value="faster-whisper", label="Engine")
                model = gr.Dropdown(choices=ENGINE_MODELS["faster-whisper"], value="base", label="Model")
                language = gr.Dropdown(choices=LANGUAGE_CODES, value="auto", label="Language")
                
            with gr.Group():
                gr.Markdown("### Output Settings")
                output_format = gr.Radio(choices=["srt", "lrc", "md"], value="srt", label="Format")
                
            with gr.Group():
                gr.Markdown("### Polishing & Summarization")
                polish_method = gr.Radio(choices=["None", "LLM (Ollama)", "NLP (nltk)"], value="None", label="Polishing Method")
                ollama_url = gr.Textbox(value="http://localhost:11434", label="Ollama URL")
                ollama_model = gr.Textbox(value="qwen2.5", label="Ollama Model")
                summarize_enabled = gr.Checkbox(label="Enable Summarization")

            with gr.Group():
                gr.Markdown("### Translation")
                translate_enabled = gr.Checkbox(label="Enable Translation")
                target_langs = gr.Textbox(placeholder="zh,en,ja", label="Target Languages (comma separated)")
                
            submit_btn = gr.Button("Start Processing", variant="primary")
            
        with gr.Column():
            logs = gr.Textbox(label="Logs", lines=10)
            outputs = gr.File(label="Generated Files")
            
    submit_btn.click(
        process_media,
        inputs=[
            input_files, engine, model, language, output_format, 
            polish_method, ollama_url, ollama_model, 
            translate_enabled, target_langs, summarize_enabled
        ],
        outputs=[logs, outputs]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
