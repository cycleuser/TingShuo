#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# TingShuo (听说) - Subtitle Generator for Audio/Video Files
#
# Copyright (C) 2024 TingShuo Team <wedonotuse@outlook.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
TingShuo (听说) - Generate SRT/LRC subtitles and Markdown transcripts from audio/video files.

Supports multiple speech-to-text engines (faster-whisper, Vosk, OpenAI Whisper,
whisper.cpp) with optional LLM polishing (Ollama / OpenAI-compatible API) and
NLP sentence segmentation (nltk). Features include auto-correction of typos and
verbal mistakes, Markdown transcript generation for speeches/lectures, and content
summarization with multimodal video analysis. Provides both CLI and GUI interfaces.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# §2  Version & Metadata
# ═══════════════════════════════════════════════════════════════════════════════

__version__ = "0.1.6"
__author__ = "TingShuo Team"
__license__ = "GPL-3.0-or-later"

# ═══════════════════════════════════════════════════════════════════════════════
# §3  Imports
# ═══════════════════════════════════════════════════════════════════════════════

import os
import sys
import re
import json
import wave
import shutil
import logging
import argparse
import tempfile
import threading
import subprocess
import zipfile
import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, Empty
from typing import List, Optional, Dict, Callable, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logger = logging.getLogger("tingshuo")

# ═══════════════════════════════════════════════════════════════════════════════
# §4  Constants
# ═══════════════════════════════════════════════════════════════════════════════

AUDIO_EXTENSIONS = frozenset({
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a", ".opus",
})

VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".ts", ".m4v", ".mpg", ".mpeg",
})

ALL_MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

SUPPORTED_ENGINES = ("faster-whisper", "vosk", "whisper", "whisper-cpp")

DEFAULT_ENGINE = "faster-whisper"
DEFAULT_FORMAT = "srt"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5"

FFMPEG_CMD = os.environ.get("TINGSHUO_FFMPEG", "ffmpeg")

ENGINE_MODELS: Dict[str, List[str]] = {
    "faster-whisper": ["tiny", "base", "small", "medium", "large-v2", "large-v3"],
    "vosk": [
        "vosk-model-small-en-us-0.15",
        "vosk-model-small-cn-0.22",
        "vosk-model-small-ja-0.22",
        "vosk-model-small-ko-0.22",
        "vosk-model-small-de-0.15",
        "vosk-model-small-fr-0.22",
        "vosk-model-small-es-0.42",
        "vosk-model-small-it-0.22",
        "vosk-model-small-pt-0.3",
        "vosk-model-small-ru-0.22",
    ],
    "whisper": ["tiny", "base", "small", "medium", "large"],
    "whisper-cpp": ["tiny", "base", "small", "medium", "large"],
}

ENGINE_DEFAULT_MODEL: Dict[str, str] = {
    "faster-whisper": "base",
    "vosk": "",
    "whisper": "base",
    "whisper-cpp": "base",
}

# NLTK language mapping: ISO 639-1 -> nltk tokenizer language
NLTK_LANG_MAP: Dict[str, str] = {
    "en": "english", "de": "german", "fr": "french", "es": "spanish",
    "it": "italian", "pt": "portuguese", "nl": "dutch", "pl": "polish",
    "ru": "russian", "cs": "czech", "et": "estonian", "fi": "finnish",
    "el": "greek", "no": "norwegian", "sl": "slovene", "sv": "swedish",
    "tr": "turkish", "da": "danish",
}

# CJK sentence-ending punctuation for languages without nltk tokenizer support
CJK_SENT_END = re.compile(r"([。！？．!?\.\?!]+)")

# Secondary clause boundaries for elastic splitting
CJK_CLAUSE_SPLIT = re.compile(r"(?<=[，、；：,;:])")
LATIN_CLAUSE_SPLIT = re.compile(r"(?<=[,;:\-\u2013\u2014])\s+")

# Elastic length thresholds (characters)
ELASTIC_CJK_MIN = 10
ELASTIC_CJK_TARGET = 30
ELASTIC_CJK_MAX = 50
ELASTIC_LATIN_MIN = 15
ELASTIC_LATIN_TARGET = 50
ELASTIC_LATIN_MAX = 80

# Common languages for dropdown selection
COMMON_LANGUAGES: List[Tuple[str, str]] = [
    ("auto", "auto (detect)"),
    ("zh", "zh - Chinese"),
    ("en", "en - English"),
    ("ja", "ja - Japanese"),
    ("ko", "ko - Korean"),
    ("fr", "fr - French"),
    ("de", "de - German"),
    ("es", "es - Spanish"),
    ("it", "it - Italian"),
    ("pt", "pt - Portuguese"),
    ("ru", "ru - Russian"),
    ("ar", "ar - Arabic"),
    ("hi", "hi - Hindi"),
    ("nl", "nl - Dutch"),
    ("pl", "pl - Polish"),
    ("sv", "sv - Swedish"),
    ("tr", "tr - Turkish"),
    ("th", "th - Thai"),
    ("vi", "vi - Vietnamese"),
    ("uk", "uk - Ukrainian"),
]
LANGUAGE_CODES: List[str] = [code for code, _ in COMMON_LANGUAGES]

# HuggingFace repo IDs for faster-whisper model download
FASTER_WHISPER_REPO_MAP: Dict[str, str] = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
}

# Vosk model download URLs
VOSK_MODEL_URLS: Dict[str, str] = {
    "vosk-model-small-en-us-0.15": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    "vosk-model-small-cn-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip",
    "vosk-model-small-ja-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-ja-0.22.zip",
    "vosk-model-small-ko-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-ko-0.22.zip",
    "vosk-model-small-de-0.15": "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip",
    "vosk-model-small-fr-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip",
    "vosk-model-small-es-0.42": "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip",
    "vosk-model-small-it-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-it-0.22.zip",
    "vosk-model-small-pt-0.3": "https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip",
    "vosk-model-small-ru-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip",
}

LLM_POLISH_PROMPT = """\
You are a subtitle polishing assistant. You receive a JSON array of subtitle \
segments, each with "s" (start time in seconds), "e" (end time in seconds), \
and "t" (text). Your task is to merge fragmented segments into complete, \
natural sentences and correct transcription errors.

Rules:
1. Merge fragments that belong to the same sentence into one segment.
2. For a merged segment, use the "s" of the first fragment and "e" of the last.
3. Actively correct transcription errors: fix homophones (同音词/近音词), \
   wrong characters (错别字), misheard words, and verbal mistakes (口误). \
   Use surrounding context to determine the intended word. For example, \
   "危机分" should be "微积分" in a math context; "集合论" not "鸡和论".
4. Preserve the original meaning while improving transcription accuracy. \
   Do not add new content or change the speaker's intent.
5. Target segment length: each output segment should be approximately \
   15-50 characters for CJK text, or 8-25 words for non-CJK text. \
   If a merged sentence exceeds this range, split it at natural clause \
   boundaries (commas, semicolons, conjunctions) into multiple segments, \
   assigning appropriate time ranges to each.
6. Do not produce very short segments (fewer than 10 characters or 5 words) \
   unless the utterance is naturally complete at that length.
7. Return a JSON array in the exact same format: [{"s": ..., "e": ..., "t": ...}, ...]
8. Return ONLY the JSON array, no other text.
"""

LLM_AUTOCORRECT_PROMPT = """\
You are a transcription error-correction assistant. You receive a JSON array of \
subtitle segments, each with "s" (start time in seconds), "e" (end time in \
seconds), and "t" (text). Your task is to fix errors in the text while keeping \
everything else unchanged.

Rules:
1. Fix typos, wrong characters (错别字), and obvious verbal mistakes (口误).
2. Fix homophones and misheard words based on context.
3. Remove filler words (um, uh, 嗯, 那个, etc.) only when they add no meaning.
4. Do NOT merge, split, or reorder segments. Keep the EXACT same number of segments.
5. Preserve "s" and "e" values EXACTLY as given.
6. Preserve the original meaning. Do not rephrase, summarize, or add content.
7. Return a JSON array in the exact same format: [{"s": ..., "e": ..., "t": ...}, ...]
8. Return ONLY the JSON array, no other text.
"""

LLM_TRANSCRIPT_PROMPT = """\
You are a transcript formatting assistant. You receive raw transcription text \
from a speech, lecture, or presentation. Your task is to organize it into a \
clean, readable Markdown document.

Rules:
1. Organize the text into logical paragraphs based on topic flow.
2. Add Markdown section headers (## or ###) where major topic shifts occur.
3. Remove filler words, false starts, and verbal hesitations.
4. Fix obvious errors and improve readability while preserving ALL original meaning.
5. Do NOT add content that was not in the original speech.
6. Do NOT add a title or metadata header — start directly with the content.
7. Return ONLY the formatted Markdown text, no explanations.
"""

LLM_SUMMARIZE_PROMPT = """\
You are a content summarization assistant. You receive a transcript of an audio \
or video recording. Your task is to produce a structured Markdown summary.

Rules:
1. Start with a brief overview (2-3 sentences).
2. List the main topics discussed as a bulleted list.
3. Provide key points and details for each topic.
4. End with conclusions or action items if applicable.
5. Use Markdown formatting (headers, bullets, bold for emphasis).
6. Be concise but comprehensive — capture all important information.
7. Return ONLY the Markdown summary, no explanations.
"""

LLM_SUMMARIZE_MULTIMODAL_PROMPT = """\
You are a multimodal content analysis assistant. You receive a transcript of a \
video recording along with keyframe images extracted from the video. Your task \
is to produce a comprehensive Markdown summary that integrates both spoken \
content and visual information.

Rules:
1. Start with a brief overview of the video content (2-3 sentences).
2. Describe the visual elements: slides, diagrams, charts, demonstrations, etc.
3. List the main topics discussed as a bulleted list.
4. For each topic, integrate both what was said and what was shown visually.
5. Note important visual information that complements or extends the spoken content.
6. End with conclusions or key takeaways.
7. Use Markdown formatting (headers, bullets, bold for emphasis).
8. Return ONLY the Markdown summary, no explanations.
"""

# ─── NLLB Language Codes (ISO 639-1 → NLLB) ─────────────────────────────────

NLLB_LANG_MAP: Dict[str, str] = {
    "en": "eng_Latn", "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn", "it": "ita_Latn",
    "pt": "por_Latn", "ru": "rus_Cyrl", "ar": "arb_Arab", "hi": "hin_Deva",
    "nl": "nld_Latn", "pl": "pol_Latn", "sv": "swe_Latn", "tr": "tur_Latn",
    "th": "tha_Thai", "vi": "vie_Latn", "uk": "ukr_Cyrl",
}

NLLB_MODELS: List[str] = [
    "facebook/nllb-200-distilled-600M",
    "facebook/nllb-200-distilled-1.3B",
    "facebook/nllb-200-3.3B",
]
DEFAULT_NLLB_MODEL = "facebook/nllb-200-distilled-600M"

# ─── User Configuration Paths ────────────────────────────────────────────────

CONFIG_DIR = Path.home() / ".config" / "tingshuo"
CONFIG_FILE = CONFIG_DIR / "settings.json"

# ─── LLM Translation Prompt ─────────────────────────────────────────────────

LLM_TRANSLATE_PROMPT = """\
You are a subtitle translation assistant. You receive a JSON array of subtitle \
segments, each with "i" (index) and "t" (text). Translate ALL the "t" values \
from {src_lang} to {tgt_lang}.

Rules:
1. Translate every segment accurately, preserving meaning and nuance.
2. Keep the "i" index unchanged.
3. Return a JSON array in the exact same format: [{{"i": ..., "t": ...}}, ...]
4. Return ONLY the JSON array, no other text.
"""

# ─── UI Language Names ───────────────────────────────────────────────────────

UI_LANG_NAMES: Dict[str, str] = {
    "en": "English", "zh": "中文", "ja": "日本語", "ko": "한국어",
    "fr": "Français", "de": "Deutsch", "es": "Español", "it": "Italiano",
    "pt": "Português", "ru": "Русский",
}

# ─── I18N UI Strings ─────────────────────────────────────────────────────────

UI_STRINGS: Dict[str, Dict[str, str]] = {
    "en": {
        "app_title": "TingShuo", "input_output": "Input / Output",
        "input_dir": "Input Directory:", "output_dir": "Output Directory:",
        "browse": "Browse", "save_same_dir": "Save to same directory as source files",
        "engine_settings": "Engine Settings", "engine": "Engine:", "model": "Model:",
        "download": "Download", "download_all": "All (Engine)", "download_everything": "All Models", "language": "Language:",
        "lang_hint": "(select or type code)", "use_hf_mirror": "Use HF Mirror:",
        "output_format": "Output Format", "subtitle_polishing": "Subtitle Polishing",
        "none_opt": "None", "llm_opt": "LLM", "nlp_opt": "NLP (nltk)",
        "llm_settings": "LLM Settings", "ollama": "Ollama",
        "openai_api": "OpenAI-compatible API", "ollama_url": "Ollama URL:",
        "ollama_model": "Ollama Model:", "refresh": "Refresh",
        "api_url": "API URL:", "api_key": "API Key:", "api_model": "API Model:",
        "translation": "Translation", "enable_trans": "Enable Translation",
        "target_langs": "Target Languages:", "trans_backend": "Backend:",
        "nllb": "NLLB", "nllb_model": "NLLB Model:",
        "start": "Start", "stop": "Stop", "progress": "Progress",
        "ready": "Ready", "done": "Done", "log": "Log",
        "settings": "Settings", "about": "About",
        "file_menu": "File", "help_menu": "Help",
        "ui_language": "Interface Language:", "version": "Version",
        "author": "Author", "license": "License",
        "close": "Close", "apply": "Apply", "save": "Save",
        "starting": "Starting...", "stopping": "Stopping...",
        "select_input": "Please select an input directory.",
        "input_not_exist": "Input directory does not exist:",
        "select_output": "Please select an output directory or enable same directory option.",
        "select_model_first": "Please select a model first.",
        "downloading": "Downloading...", "download_complete": "Download complete:",
        "download_failed": "Download failed:", "error": "Error", "warning": "Warning",
        "about_text": "TingShuo - Subtitle Generator for Audio/Video Files",
        "no_models": "No downloadable models for this engine.",
        "api_required": "API URL and API Key are required for OpenAI-compatible API.",
        "select_target": "Select at least one target language for translation.",
        "translating": "Translating...", "trans_complete": "Translation complete",
        "trans_settings": "Translation Settings",
        "restart_note": "Restart the application for language changes to take effect.",
        "auto_correct": "Auto-Correct", "auto_correcting": "Auto-correcting...",
        "auto_correct_complete": "Auto-correction complete",
        "summarize": "Content Summary", "summarizing": "Summarizing...",
        "summarize_complete": "Summary generated",
        "transcript_opt": "MD (Transcript)",
        "multimodal_note": "Video summarization uses visual analysis (requires vision-capable API)",
        "keyframe_interval": "Keyframe Interval (s):",
    },
    "zh": {
        "app_title": "听说", "input_output": "输入 / 输出",
        "input_dir": "输入目录：", "output_dir": "输出目录：",
        "browse": "浏览", "save_same_dir": "保存到与源文件相同的目录",
        "engine_settings": "引擎设置", "engine": "引擎：", "model": "模型：",
        "download": "下载", "download_all": "引擎全部", "download_everything": "所有模型", "language": "语言：",
        "lang_hint": "（选择或输入语言代码）", "use_hf_mirror": "使用 HF 镜像：",
        "output_format": "输出格式", "subtitle_polishing": "字幕优化",
        "none_opt": "无", "llm_opt": "LLM", "nlp_opt": "NLP (nltk)",
        "llm_settings": "LLM 设置", "ollama": "Ollama",
        "openai_api": "OpenAI 兼容 API", "ollama_url": "Ollama 地址：",
        "ollama_model": "Ollama 模型：", "refresh": "刷新",
        "api_url": "API 地址：", "api_key": "API 密钥：", "api_model": "API 模型：",
        "translation": "翻译", "enable_trans": "启用翻译",
        "target_langs": "目标语言：", "trans_backend": "翻译后端：",
        "nllb": "NLLB", "nllb_model": "NLLB 模型：",
        "start": "开始", "stop": "停止", "progress": "进度",
        "ready": "就绪", "done": "完成", "log": "日志",
        "settings": "设置", "about": "关于",
        "file_menu": "文件", "help_menu": "帮助",
        "ui_language": "界面语言：", "version": "版本",
        "author": "作者", "license": "许可证",
        "close": "关闭", "apply": "应用", "save": "保存",
        "starting": "启动中...", "stopping": "停止中...",
        "select_input": "请选择输入目录。",
        "input_not_exist": "输入目录不存在：",
        "select_output": "请选择输出目录或启用同目录选项。",
        "select_model_first": "请先选择一个模型。",
        "downloading": "下载中...", "download_complete": "下载完成：",
        "download_failed": "下载失败：", "error": "错误", "warning": "警告",
        "about_text": "听说 - 音视频字幕生成工具",
        "no_models": "此引擎没有可下载的模型。",
        "api_required": "使用 OpenAI 兼容 API 需要填写 API 地址和密钥。",
        "select_target": "请至少选择一种翻译目标语言。",
        "translating": "翻译中...", "trans_complete": "翻译完成",
        "trans_settings": "翻译设置",
        "restart_note": "语言更改需要重启应用程序才能生效。",
        "auto_correct": "自动纠错", "auto_correcting": "自动纠错中...",
        "auto_correct_complete": "自动纠错完成",
        "summarize": "内容总结", "summarizing": "总结中...",
        "summarize_complete": "总结生成完成",
        "transcript_opt": "MD（讲稿）",
        "multimodal_note": "视频总结使用视觉分析（需要支持视觉的API）",
        "keyframe_interval": "关键帧间隔（秒）：",
    },
    "ja": {
        "app_title": "TingShuo", "input_output": "入出力",
        "input_dir": "入力ディレクトリ：", "output_dir": "出力ディレクトリ：",
        "browse": "参照", "save_same_dir": "ソースファイルと同じディレクトリに保存",
        "engine_settings": "エンジン設定", "engine": "エンジン：", "model": "モデル：",
        "download": "ダウンロード", "download_all": "エンジン全部", "download_everything": "全モデル", "language": "言語：",
        "lang_hint": "（選択または入力）", "use_hf_mirror": "HFミラー使用：",
        "output_format": "出力形式", "subtitle_polishing": "字幕最適化",
        "none_opt": "なし", "llm_opt": "LLM", "nlp_opt": "NLP (nltk)",
        "llm_settings": "LLM 設定", "ollama": "Ollama",
        "openai_api": "OpenAI互換API", "ollama_url": "Ollama URL：",
        "ollama_model": "Ollamaモデル：", "refresh": "更新",
        "api_url": "API URL：", "api_key": "APIキー：", "api_model": "APIモデル：",
        "translation": "翻訳", "enable_trans": "翻訳を有効化",
        "target_langs": "翻訳先言語：", "trans_backend": "バックエンド：",
        "nllb": "NLLB", "nllb_model": "NLLBモデル：",
        "start": "開始", "stop": "停止", "progress": "進捗",
        "ready": "準備完了", "done": "完了", "log": "ログ",
        "settings": "設定", "about": "バージョン情報",
        "file_menu": "ファイル", "help_menu": "ヘルプ",
        "ui_language": "表示言語：", "version": "バージョン",
        "author": "作者", "license": "ライセンス",
        "close": "閉じる", "apply": "適用", "save": "保存",
        "starting": "開始中...", "stopping": "停止中...",
        "select_input": "入力ディレクトリを選択してください。",
        "input_not_exist": "入力ディレクトリが存在しません：",
        "select_output": "出力ディレクトリを選択するか、同一ディレクトリオプションを有効にしてください。",
        "select_model_first": "まずモデルを選択してください。",
        "downloading": "ダウンロード中...", "download_complete": "ダウンロード完了：",
        "download_failed": "ダウンロード失敗：", "error": "エラー", "warning": "警告",
        "about_text": "TingShuo - 音声・動画字幕生成ツール",
        "no_models": "このエンジンにはダウンロード可能なモデルがありません。",
        "api_required": "OpenAI互換APIにはAPI URLとAPIキーが必要です。",
        "select_target": "翻訳先言語を少なくとも1つ選択してください。",
        "translating": "翻訳中...", "trans_complete": "翻訳完了",
        "trans_settings": "翻訳設定",
        "restart_note": "言語の変更はアプリの再起動後に反映されます。",
        "auto_correct": "自動校正", "auto_correcting": "自動校正中...",
        "auto_correct_complete": "自動校正完了",
        "summarize": "コンテンツ要約", "summarizing": "要約中...",
        "summarize_complete": "要約生成完了",
        "transcript_opt": "MD（原稿）",
        "multimodal_note": "動画要約はビジュアル分析を使用（ビジョン対応APIが必要）",
        "keyframe_interval": "キーフレーム間隔（秒）：",
    },
    "ko": {
        "app_title": "TingShuo", "input_output": "입출력",
        "input_dir": "입력 디렉토리:", "output_dir": "출력 디렉토리:",
        "browse": "찾아보기", "save_same_dir": "소스 파일과 같은 디렉토리에 저장",
        "engine_settings": "엔진 설정", "engine": "엔진:", "model": "모델:",
        "download": "다운로드", "download_all": "엔진 전부", "download_everything": "전체 모델", "language": "언어:",
        "lang_hint": "(선택 또는 입력)", "use_hf_mirror": "HF 미러 사용:",
        "output_format": "출력 형식", "subtitle_polishing": "자막 최적화",
        "none_opt": "없음", "llm_opt": "LLM", "nlp_opt": "NLP (nltk)",
        "llm_settings": "LLM 설정", "ollama": "Ollama",
        "openai_api": "OpenAI 호환 API", "ollama_url": "Ollama URL:",
        "ollama_model": "Ollama 모델:", "refresh": "새로고침",
        "api_url": "API URL:", "api_key": "API 키:", "api_model": "API 모델:",
        "translation": "번역", "enable_trans": "번역 활성화",
        "target_langs": "대상 언어:", "trans_backend": "백엔드:",
        "nllb": "NLLB", "nllb_model": "NLLB 모델:",
        "start": "시작", "stop": "중지", "progress": "진행",
        "ready": "준비", "done": "완료", "log": "로그",
        "settings": "설정", "about": "정보",
        "file_menu": "파일", "help_menu": "도움말",
        "ui_language": "인터페이스 언어:", "version": "버전",
        "author": "저자", "license": "라이선스",
        "close": "닫기", "apply": "적용", "save": "저장",
        "starting": "시작 중...", "stopping": "중지 중...",
        "select_input": "입력 디렉토리를 선택해 주세요.",
        "input_not_exist": "입력 디렉토리가 존재하지 않습니다:",
        "select_output": "출력 디렉토리를 선택하거나 같은 디렉토리 옵션을 활성화해 주세요.",
        "select_model_first": "먼저 모델을 선택해 주세요.",
        "downloading": "다운로드 중...", "download_complete": "다운로드 완료:",
        "download_failed": "다운로드 실패:", "error": "오류", "warning": "경고",
        "about_text": "TingShuo - 오디오/비디오 자막 생성기",
        "no_models": "이 엔진에 다운로드 가능한 모델이 없습니다.",
        "api_required": "OpenAI 호환 API에는 API URL과 API 키가 필요합니다.",
        "select_target": "번역 대상 언어를 하나 이상 선택해 주세요.",
        "translating": "번역 중...", "trans_complete": "번역 완료",
        "trans_settings": "번역 설정",
        "restart_note": "언어 변경은 앱을 다시 시작해야 적용됩니다.",
        "auto_correct": "자동 교정", "auto_correcting": "자동 교정 중...",
        "auto_correct_complete": "자동 교정 완료",
        "summarize": "콘텐츠 요약", "summarizing": "요약 중...",
        "summarize_complete": "요약 생성 완료",
        "transcript_opt": "MD (원고)",
        "multimodal_note": "비디오 요약은 시각 분석 사용 (비전 API 필요)",
        "keyframe_interval": "키프레임 간격 (초):",
    },
    "fr": {
        "app_title": "TingShuo", "input_output": "Entrée / Sortie",
        "input_dir": "Répertoire d'entrée :", "output_dir": "Répertoire de sortie :",
        "browse": "Parcourir", "save_same_dir": "Enregistrer dans le même répertoire",
        "engine_settings": "Paramètres moteur", "engine": "Moteur :", "model": "Modèle :",
        "download": "Télécharger", "download_all": "Tout (moteur)", "download_everything": "Tous les modèles", "language": "Langue :",
        "lang_hint": "(sélectionner ou saisir)", "use_hf_mirror": "Miroir HF :",
        "output_format": "Format de sortie", "subtitle_polishing": "Polissage des sous-titres",
        "none_opt": "Aucun", "llm_opt": "LLM", "nlp_opt": "NLP (nltk)",
        "llm_settings": "Paramètres LLM", "ollama": "Ollama",
        "openai_api": "API compatible OpenAI", "ollama_url": "URL Ollama :",
        "ollama_model": "Modèle Ollama :", "refresh": "Actualiser",
        "api_url": "URL API :", "api_key": "Clé API :", "api_model": "Modèle API :",
        "translation": "Traduction", "enable_trans": "Activer la traduction",
        "target_langs": "Langues cibles :", "trans_backend": "Backend :",
        "nllb": "NLLB", "nllb_model": "Modèle NLLB :",
        "start": "Démarrer", "stop": "Arrêter", "progress": "Progression",
        "ready": "Prêt", "done": "Terminé", "log": "Journal",
        "settings": "Paramètres", "about": "À propos",
        "file_menu": "Fichier", "help_menu": "Aide",
        "ui_language": "Langue d'interface :", "version": "Version",
        "author": "Auteur", "license": "Licence",
        "close": "Fermer", "apply": "Appliquer", "save": "Enregistrer",
        "starting": "Démarrage...", "stopping": "Arrêt en cours...",
        "select_input": "Veuillez sélectionner un répertoire d'entrée.",
        "input_not_exist": "Le répertoire d'entrée n'existe pas :",
        "select_output": "Veuillez sélectionner un répertoire de sortie.",
        "select_model_first": "Veuillez d'abord sélectionner un modèle.",
        "downloading": "Téléchargement...", "download_complete": "Téléchargement terminé :",
        "download_failed": "Échec du téléchargement :", "error": "Erreur", "warning": "Avertissement",
        "about_text": "TingShuo - Générateur de sous-titres pour audio/vidéo",
        "no_models": "Aucun modèle téléchargeable pour ce moteur.",
        "api_required": "L'URL et la clé API sont requises.",
        "select_target": "Sélectionnez au moins une langue cible.",
        "translating": "Traduction en cours...", "trans_complete": "Traduction terminée",
        "trans_settings": "Paramètres de traduction",
        "restart_note": "Redémarrez l'application pour appliquer le changement de langue.",
        "auto_correct": "Auto-correction", "auto_correcting": "Auto-correction en cours...",
        "auto_correct_complete": "Auto-correction terminée",
        "summarize": "Résumé du contenu", "summarizing": "Résumé en cours...",
        "summarize_complete": "Résumé généré",
        "transcript_opt": "MD (Transcription)",
        "multimodal_note": "Le résumé vidéo utilise l'analyse visuelle (API vision requise)",
        "keyframe_interval": "Intervalle d'images clés (s) :",
    },
    "de": {
        "app_title": "TingShuo", "input_output": "Eingabe / Ausgabe",
        "input_dir": "Eingabeverzeichnis:", "output_dir": "Ausgabeverzeichnis:",
        "browse": "Durchsuchen", "save_same_dir": "Im selben Verzeichnis speichern",
        "engine_settings": "Engine-Einstellungen", "engine": "Engine:", "model": "Modell:",
        "download": "Herunterladen", "download_all": "Alle (Engine)", "download_everything": "Alle Modelle", "language": "Sprache:",
        "lang_hint": "(auswählen oder eingeben)", "use_hf_mirror": "HF-Mirror:",
        "output_format": "Ausgabeformat", "subtitle_polishing": "Untertitel-Optimierung",
        "none_opt": "Keine", "llm_opt": "LLM", "nlp_opt": "NLP (nltk)",
        "llm_settings": "LLM-Einstellungen", "ollama": "Ollama",
        "openai_api": "OpenAI-kompatible API", "ollama_url": "Ollama-URL:",
        "ollama_model": "Ollama-Modell:", "refresh": "Aktualisieren",
        "api_url": "API-URL:", "api_key": "API-Schlüssel:", "api_model": "API-Modell:",
        "translation": "Übersetzung", "enable_trans": "Übersetzung aktivieren",
        "target_langs": "Zielsprachen:", "trans_backend": "Backend:",
        "nllb": "NLLB", "nllb_model": "NLLB-Modell:",
        "start": "Starten", "stop": "Stoppen", "progress": "Fortschritt",
        "ready": "Bereit", "done": "Fertig", "log": "Protokoll",
        "settings": "Einstellungen", "about": "Über",
        "file_menu": "Datei", "help_menu": "Hilfe",
        "ui_language": "Oberflächensprache:", "version": "Version",
        "author": "Autor", "license": "Lizenz",
        "close": "Schließen", "apply": "Anwenden", "save": "Speichern",
        "starting": "Starten...", "stopping": "Stoppen...",
        "select_input": "Bitte wählen Sie ein Eingabeverzeichnis.",
        "input_not_exist": "Eingabeverzeichnis existiert nicht:",
        "select_output": "Bitte wählen Sie ein Ausgabeverzeichnis.",
        "select_model_first": "Bitte wählen Sie zuerst ein Modell.",
        "downloading": "Herunterladen...", "download_complete": "Download abgeschlossen:",
        "download_failed": "Download fehlgeschlagen:", "error": "Fehler", "warning": "Warnung",
        "about_text": "TingShuo - Untertitelgenerator für Audio-/Videodateien",
        "no_models": "Keine herunterladbaren Modelle für diese Engine.",
        "api_required": "API-URL und API-Schlüssel sind erforderlich.",
        "select_target": "Bitte wählen Sie mindestens eine Zielsprache.",
        "translating": "Übersetzen...", "trans_complete": "Übersetzung abgeschlossen",
        "trans_settings": "Übersetzungseinstellungen",
        "restart_note": "Starten Sie die Anwendung neu, um Sprachänderungen zu übernehmen.",
        "auto_correct": "Autokorrektur", "auto_correcting": "Autokorrektur läuft...",
        "auto_correct_complete": "Autokorrektur abgeschlossen",
        "summarize": "Inhaltszusammenfassung", "summarizing": "Zusammenfassung wird erstellt...",
        "summarize_complete": "Zusammenfassung erstellt",
        "transcript_opt": "MD (Transkript)",
        "multimodal_note": "Videozusammenfassung nutzt visuelle Analyse (Vision-API erforderlich)",
        "keyframe_interval": "Keyframe-Intervall (s):",
    },
    "es": {
        "app_title": "TingShuo", "input_output": "Entrada / Salida",
        "input_dir": "Directorio de entrada:", "output_dir": "Directorio de salida:",
        "browse": "Examinar", "save_same_dir": "Guardar en el mismo directorio",
        "engine_settings": "Configuración del motor", "engine": "Motor:", "model": "Modelo:",
        "download": "Descargar", "download_all": "Todo (motor)", "download_everything": "Todos modelos", "language": "Idioma:",
        "lang_hint": "(seleccionar o escribir)", "use_hf_mirror": "Espejo HF:",
        "output_format": "Formato de salida", "subtitle_polishing": "Optimización de subtítulos",
        "none_opt": "Ninguno", "llm_opt": "LLM", "nlp_opt": "NLP (nltk)",
        "llm_settings": "Configuración LLM", "ollama": "Ollama",
        "openai_api": "API compatible con OpenAI", "ollama_url": "URL de Ollama:",
        "ollama_model": "Modelo Ollama:", "refresh": "Actualizar",
        "api_url": "URL de API:", "api_key": "Clave de API:", "api_model": "Modelo de API:",
        "translation": "Traducción", "enable_trans": "Activar traducción",
        "target_langs": "Idiomas destino:", "trans_backend": "Backend:",
        "nllb": "NLLB", "nllb_model": "Modelo NLLB:",
        "start": "Iniciar", "stop": "Detener", "progress": "Progreso",
        "ready": "Listo", "done": "Completado", "log": "Registro",
        "settings": "Configuración", "about": "Acerca de",
        "file_menu": "Archivo", "help_menu": "Ayuda",
        "ui_language": "Idioma de interfaz:", "version": "Versión",
        "author": "Autor", "license": "Licencia",
        "close": "Cerrar", "apply": "Aplicar", "save": "Guardar",
        "starting": "Iniciando...", "stopping": "Deteniendo...",
        "select_input": "Seleccione un directorio de entrada.",
        "input_not_exist": "El directorio de entrada no existe:",
        "select_output": "Seleccione un directorio de salida.",
        "select_model_first": "Seleccione primero un modelo.",
        "downloading": "Descargando...", "download_complete": "Descarga completa:",
        "download_failed": "Descarga fallida:", "error": "Error", "warning": "Advertencia",
        "about_text": "TingShuo - Generador de subtítulos para audio/video",
        "no_models": "No hay modelos descargables para este motor.",
        "api_required": "Se requieren URL y clave de API.",
        "select_target": "Seleccione al menos un idioma destino.",
        "translating": "Traduciendo...", "trans_complete": "Traducción completa",
        "trans_settings": "Configuración de traducción",
        "restart_note": "Reinicie la aplicación para aplicar el cambio de idioma.",
        "auto_correct": "Autocorrección", "auto_correcting": "Autocorrigiendo...",
        "auto_correct_complete": "Autocorrección completada",
        "summarize": "Resumen de contenido", "summarizing": "Resumiendo...",
        "summarize_complete": "Resumen generado",
        "transcript_opt": "MD (Transcripción)",
        "multimodal_note": "El resumen de video usa análisis visual (requiere API de visión)",
        "keyframe_interval": "Intervalo de fotogramas clave (s):",
    },
    "it": {
        "app_title": "TingShuo", "input_output": "Input / Output",
        "input_dir": "Directory di input:", "output_dir": "Directory di output:",
        "browse": "Sfoglia", "save_same_dir": "Salva nella stessa directory",
        "engine_settings": "Impostazioni motore", "engine": "Motore:", "model": "Modello:",
        "download": "Scarica", "download_all": "Tutto (motore)", "download_everything": "Tutti modelli", "language": "Lingua:",
        "lang_hint": "(seleziona o digita)", "use_hf_mirror": "Mirror HF:",
        "output_format": "Formato di output", "subtitle_polishing": "Ottimizzazione sottotitoli",
        "none_opt": "Nessuno", "llm_opt": "LLM", "nlp_opt": "NLP (nltk)",
        "llm_settings": "Impostazioni LLM", "ollama": "Ollama",
        "openai_api": "API compatibile OpenAI", "ollama_url": "URL Ollama:",
        "ollama_model": "Modello Ollama:", "refresh": "Aggiorna",
        "api_url": "URL API:", "api_key": "Chiave API:", "api_model": "Modello API:",
        "translation": "Traduzione", "enable_trans": "Attiva traduzione",
        "target_langs": "Lingue di destinazione:", "trans_backend": "Backend:",
        "nllb": "NLLB", "nllb_model": "Modello NLLB:",
        "start": "Avvia", "stop": "Ferma", "progress": "Progresso",
        "ready": "Pronto", "done": "Completato", "log": "Registro",
        "settings": "Impostazioni", "about": "Informazioni",
        "file_menu": "File", "help_menu": "Aiuto",
        "ui_language": "Lingua interfaccia:", "version": "Versione",
        "author": "Autore", "license": "Licenza",
        "close": "Chiudi", "apply": "Applica", "save": "Salva",
        "starting": "Avvio...", "stopping": "Arresto...",
        "select_input": "Selezionare una directory di input.",
        "input_not_exist": "La directory di input non esiste:",
        "select_output": "Selezionare una directory di output.",
        "select_model_first": "Selezionare prima un modello.",
        "downloading": "Download in corso...", "download_complete": "Download completato:",
        "download_failed": "Download fallito:", "error": "Errore", "warning": "Avviso",
        "about_text": "TingShuo - Generatore di sottotitoli per audio/video",
        "no_models": "Nessun modello scaricabile per questo motore.",
        "api_required": "URL e chiave API sono necessari.",
        "select_target": "Selezionare almeno una lingua di destinazione.",
        "translating": "Traduzione in corso...", "trans_complete": "Traduzione completata",
        "trans_settings": "Impostazioni traduzione",
        "restart_note": "Riavviare l'applicazione per applicare il cambio di lingua.",
        "auto_correct": "Autocorrezione", "auto_correcting": "Autocorrezione in corso...",
        "auto_correct_complete": "Autocorrezione completata",
        "summarize": "Riepilogo contenuto", "summarizing": "Riepilogo in corso...",
        "summarize_complete": "Riepilogo generato",
        "transcript_opt": "MD (Trascrizione)",
        "multimodal_note": "Il riepilogo video usa l'analisi visiva (richiede API vision)",
        "keyframe_interval": "Intervallo fotogrammi chiave (s):",
    },
    "pt": {
        "app_title": "TingShuo", "input_output": "Entrada / Saída",
        "input_dir": "Diretório de entrada:", "output_dir": "Diretório de saída:",
        "browse": "Procurar", "save_same_dir": "Salvar no mesmo diretório",
        "engine_settings": "Configurações do mecanismo", "engine": "Mecanismo:", "model": "Modelo:",
        "download": "Baixar", "download_all": "Tudo (motor)", "download_everything": "Todos modelos", "language": "Idioma:",
        "lang_hint": "(selecionar ou digitar)", "use_hf_mirror": "Espelho HF:",
        "output_format": "Formato de saída", "subtitle_polishing": "Otimização de legendas",
        "none_opt": "Nenhum", "llm_opt": "LLM", "nlp_opt": "NLP (nltk)",
        "llm_settings": "Configurações LLM", "ollama": "Ollama",
        "openai_api": "API compatível com OpenAI", "ollama_url": "URL do Ollama:",
        "ollama_model": "Modelo Ollama:", "refresh": "Atualizar",
        "api_url": "URL da API:", "api_key": "Chave da API:", "api_model": "Modelo da API:",
        "translation": "Tradução", "enable_trans": "Ativar tradução",
        "target_langs": "Idiomas alvo:", "trans_backend": "Backend:",
        "nllb": "NLLB", "nllb_model": "Modelo NLLB:",
        "start": "Iniciar", "stop": "Parar", "progress": "Progresso",
        "ready": "Pronto", "done": "Concluído", "log": "Registro",
        "settings": "Configurações", "about": "Sobre",
        "file_menu": "Arquivo", "help_menu": "Ajuda",
        "ui_language": "Idioma da interface:", "version": "Versão",
        "author": "Autor", "license": "Licença",
        "close": "Fechar", "apply": "Aplicar", "save": "Salvar",
        "starting": "Iniciando...", "stopping": "Parando...",
        "select_input": "Selecione um diretório de entrada.",
        "input_not_exist": "O diretório de entrada não existe:",
        "select_output": "Selecione um diretório de saída.",
        "select_model_first": "Selecione primeiro um modelo.",
        "downloading": "Baixando...", "download_complete": "Download concluído:",
        "download_failed": "Download falhou:", "error": "Erro", "warning": "Aviso",
        "about_text": "TingShuo - Gerador de legendas para áudio/vídeo",
        "no_models": "Nenhum modelo disponível para download.",
        "api_required": "URL e chave da API são necessárias.",
        "select_target": "Selecione pelo menos um idioma alvo.",
        "translating": "Traduzindo...", "trans_complete": "Tradução concluída",
        "trans_settings": "Configurações de tradução",
        "restart_note": "Reinicie o aplicativo para aplicar a alteração de idioma.",
        "auto_correct": "Autocorreção", "auto_correcting": "Autocorrigindo...",
        "auto_correct_complete": "Autocorreção concluída",
        "summarize": "Resumo de conteúdo", "summarizing": "Resumindo...",
        "summarize_complete": "Resumo gerado",
        "transcript_opt": "MD (Transcrição)",
        "multimodal_note": "O resumo de vídeo usa análise visual (requer API de visão)",
        "keyframe_interval": "Intervalo de quadros-chave (s):",
    },
    "ru": {
        "app_title": "TingShuo", "input_output": "Вход / Выход",
        "input_dir": "Входная директория:", "output_dir": "Выходная директория:",
        "browse": "Обзор", "save_same_dir": "Сохранять в ту же директорию",
        "engine_settings": "Настройки движка", "engine": "Движок:", "model": "Модель:",
        "download": "Скачать", "download_all": "Все (движок)", "download_everything": "Все модели", "language": "Язык:",
        "lang_hint": "(выберите или введите)", "use_hf_mirror": "Зеркало HF:",
        "output_format": "Формат вывода", "subtitle_polishing": "Обработка субтитров",
        "none_opt": "Нет", "llm_opt": "LLM", "nlp_opt": "NLP (nltk)",
        "llm_settings": "Настройки LLM", "ollama": "Ollama",
        "openai_api": "API совместимый с OpenAI", "ollama_url": "URL Ollama:",
        "ollama_model": "Модель Ollama:", "refresh": "Обновить",
        "api_url": "URL API:", "api_key": "Ключ API:", "api_model": "Модель API:",
        "translation": "Перевод", "enable_trans": "Включить перевод",
        "target_langs": "Целевые языки:", "trans_backend": "Бэкенд:",
        "nllb": "NLLB", "nllb_model": "Модель NLLB:",
        "start": "Старт", "stop": "Стоп", "progress": "Прогресс",
        "ready": "Готово", "done": "Завершено", "log": "Журнал",
        "settings": "Настройки", "about": "О программе",
        "file_menu": "Файл", "help_menu": "Справка",
        "ui_language": "Язык интерфейса:", "version": "Версия",
        "author": "Автор", "license": "Лицензия",
        "close": "Закрыть", "apply": "Применить", "save": "Сохранить",
        "starting": "Запуск...", "stopping": "Остановка...",
        "select_input": "Выберите входную директорию.",
        "input_not_exist": "Входная директория не существует:",
        "select_output": "Выберите выходную директорию.",
        "select_model_first": "Сначала выберите модель.",
        "downloading": "Загрузка...", "download_complete": "Загрузка завершена:",
        "download_failed": "Загрузка не удалась:", "error": "Ошибка", "warning": "Предупреждение",
        "about_text": "TingShuo - Генератор субтитров для аудио/видео файлов",
        "no_models": "Нет загружаемых моделей для этого движка.",
        "api_required": "URL и ключ API необходимы.",
        "select_target": "Выберите хотя бы один целевой язык.",
        "translating": "Перевод...", "trans_complete": "Перевод завершён",
        "trans_settings": "Настройки перевода",
        "restart_note": "Перезапустите приложение для применения изменений языка.",
        "auto_correct": "Автокоррекция", "auto_correcting": "Автокоррекция...",
        "auto_correct_complete": "Автокоррекция завершена",
        "summarize": "Резюме содержания", "summarizing": "Создание резюме...",
        "summarize_complete": "Резюме создано",
        "transcript_opt": "MD (Стенограмма)",
        "multimodal_note": "Резюме видео использует визуальный анализ (требуется API с поддержкой изображений)",
        "keyframe_interval": "Интервал ключевых кадров (с):",
    },
}

# ─── I18N Helper ─────────────────────────────────────────────────────────────

_current_ui_lang = "en"


def tr(key: str) -> str:
    """Get translated UI string for the current language."""
    lang_dict = UI_STRINGS.get(_current_ui_lang, UI_STRINGS["en"])
    return lang_dict.get(key, UI_STRINGS["en"].get(key, key))


def set_ui_language(lang: str) -> None:
    """Set the current UI language."""
    global _current_ui_lang
    _current_ui_lang = lang if lang in UI_STRINGS else "en"


# ═══════════════════════════════════════════════════════════════════════════════
# §5  Data Structures
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    segments: List[Segment]
    language: Optional[str] = None
    duration: Optional[float] = None


@dataclass
class PolishConfig:
    method: str = "none"  # "none", "llm", "nlp"
    ollama_url: str = DEFAULT_OLLAMA_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    api_url: str = ""
    api_key: str = ""
    api_model: str = ""


@dataclass
class TranslationConfig:
    enabled: bool = False
    method: str = "nllb"  # "nllb" or "llm"
    target_languages: List[str] = field(default_factory=list)
    nllb_model: str = DEFAULT_NLLB_MODEL
    # LLM settings (reuses PolishConfig-style fields)
    ollama_url: str = DEFAULT_OLLAMA_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    api_url: str = ""
    api_key: str = ""
    api_model: str = ""


@dataclass
class SummarizeConfig:
    enabled: bool = False
    keyframe_interval: int = 60  # seconds between extracted keyframes


@dataclass
class JobConfig:
    input_dir: str = ""
    output_dir: str = ""
    format: str = DEFAULT_FORMAT
    engine_name: str = DEFAULT_ENGINE
    model_name: str = ""
    language: Optional[str] = None
    hf_mirror: str = ""
    polish: PolishConfig = field(default_factory=PolishConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    recursive: bool = True
    auto_correct: bool = False
    summarize: SummarizeConfig = field(default_factory=SummarizeConfig)


# ═══════════════════════════════════════════════════════════════════════════════
# §6  Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)


def scan_media_files(directory: str, recursive: bool = True) -> List[Path]:
    root = Path(directory)
    if not root.is_dir():
        logger.error("Input directory does not exist: %s", directory)
        return []
    pattern = "**/*" if recursive else "*"
    files = sorted(
        p for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in ALL_MEDIA_EXTENSIONS
    )
    logger.info("Found %d media file(s) in %s", len(files), directory)
    return files


def check_ffmpeg() -> bool:
    try:
        subprocess.run(
            [FFMPEG_CMD, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_audio(media_path: Path, temp_dir: str) -> Path:
    out_name = media_path.stem + ".wav"
    out_path = Path(temp_dir) / out_name
    cmd = [
        FFMPEG_CMD, "-y", "-i", str(media_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {media_path.name}:\n{result.stderr[:500]}"
        )
    return out_path


def extract_keyframes(
    media_path: Path, temp_dir: str, interval: int = 60, max_frames: int = 10,
) -> List[Path]:
    """Extract keyframes from a video file at the given interval.

    Returns a list of JPEG file paths. Returns empty list for audio files.
    """
    if media_path.suffix.lower() not in VIDEO_EXTENSIONS:
        return []

    frames_dir = Path(temp_dir) / "keyframes"
    os.makedirs(frames_dir, exist_ok=True)

    cmd = [
        FFMPEG_CMD, "-y", "-i", str(media_path),
        "-vf", f"fps=1/{interval},scale='min(1024,iw)':-1",
        "-q:v", "2",
        str(frames_dir / "frame_%04d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("Keyframe extraction failed for %s: %s", media_path.name, result.stderr[:300])
        return []

    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        return []

    # Cap at max_frames by selecting evenly spaced subset
    if len(frames) > max_frames:
        step = len(frames) / max_frames
        frames = [frames[int(i * step)] for i in range(max_frames)]

    logger.info("Extracted %d keyframe(s) from %s", len(frames), media_path.name)
    return frames


def encode_image_base64(image_path: Path) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def setup_hf_mirror(url: str) -> None:
    if url:
        os.environ["HF_ENDPOINT"] = url
        logger.info("HuggingFace mirror set to: %s", url)


def format_time_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_time_lrc(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    return f"[{m:02d}:{s:02d}.{cs:02d}]"


# ─── Ollama Model Query ──────────────────────────────────────────────────────


def query_ollama_models(ollama_url: str) -> List[str]:
    """Query Ollama server for installed models. Returns sorted list of model names."""
    url = ollama_url.rstrip("/") + "/api/tags"
    req = Request(url, method="GET")
    try:
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = data.get("models", [])
        return sorted(m.get("name", "") for m in models if m.get("name"))
    except (URLError, HTTPError, TimeoutError, json.JSONDecodeError) as e:
        logger.warning("Failed to query Ollama models at %s: %s", ollama_url, e)
        return []


# ─── Model Download Functions ────────────────────────────────────────────────


def _download_faster_whisper_model(
    model_name: str,
    hf_mirror: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> None:
    if hf_mirror:
        setup_hf_mirror(hf_mirror)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise EngineNotAvailableError(
            "faster-whisper (huggingface_hub) is not installed. "
            "Install with: pip install faster-whisper"
        )
    repo_id = FASTER_WHISPER_REPO_MAP.get(model_name)
    if not repo_id:
        raise ValueError(
            f"Unknown faster-whisper model: '{model_name}'. "
            f"Available: {', '.join(FASTER_WHISPER_REPO_MAP.keys())}"
        )
    if progress_cb:
        progress_cb(f"Downloading faster-whisper model: {model_name} ...")
    snapshot_download(repo_id)
    if progress_cb:
        progress_cb(f"Download complete: faster-whisper/{model_name}")


def _download_whisper_model(
    model_name: str,
    hf_mirror: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> None:
    try:
        import whisper
    except ImportError:
        raise EngineNotAvailableError(
            "openai-whisper is not installed. "
            "Install with: pip install openai-whisper"
        )
    if progress_cb:
        progress_cb(f"Downloading OpenAI Whisper model: {model_name} ...")
    # Use internal _download if available to avoid loading into GPU memory
    if hasattr(whisper, "_MODELS") and hasattr(whisper, "_download"):
        if model_name in whisper._MODELS:
            download_root = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
            os.makedirs(download_root, exist_ok=True)
            whisper._download(whisper._MODELS[model_name], download_root, False)
        else:
            whisper.load_model(model_name)
    else:
        whisper.load_model(model_name)
    if progress_cb:
        progress_cb(f"Download complete: whisper/{model_name}")


def _download_vosk_model(
    model_name: str,
    hf_mirror: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> None:
    url = VOSK_MODEL_URLS.get(model_name)
    if not url:
        raise ValueError(
            f"Unknown Vosk model: '{model_name}'. "
            f"Available: {', '.join(VOSK_MODEL_URLS.keys())}"
        )
    cache_dir = Path.home() / ".cache" / "vosk"
    model_dir = cache_dir / model_name
    if model_dir.is_dir():
        if progress_cb:
            progress_cb(f"Vosk model already cached: {model_name}")
        return

    if progress_cb:
        progress_cb(f"Downloading Vosk model: {model_name} ...")

    os.makedirs(cache_dir, exist_ok=True)
    tmp_zip = cache_dir / f"{model_name}.zip"
    try:
        req = Request(url)
        with urlopen(req, timeout=600) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(tmp_zip, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)  # 1 MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb and total_size > 0:
                        pct = downloaded * 100 // total_size
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        progress_cb(
                            f"Downloading {model_name}: "
                            f"{mb_done:.1f} MB / {mb_total:.1f} MB ({pct}%)"
                        )

        if progress_cb:
            progress_cb(f"Extracting {model_name} ...")
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(cache_dir)
        if progress_cb:
            progress_cb(f"Download complete: vosk/{model_name}")
    except (URLError, HTTPError, TimeoutError) as e:
        raise RuntimeError(f"Failed to download Vosk model '{model_name}': {e}")
    finally:
        if tmp_zip.exists():
            tmp_zip.unlink()


def _download_whisper_cpp_model(
    model_name: str,
    hf_mirror: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> None:
    try:
        from pywhispercpp.model import Model as CppModel
    except ImportError:
        raise EngineNotAvailableError(
            "pywhispercpp is not installed. "
            "Install with: pip install pywhispercpp"
        )
    if progress_cb:
        progress_cb(f"Downloading whisper.cpp model: {model_name} ...")
    CppModel(model_name)
    if progress_cb:
        progress_cb(f"Download complete: whisper-cpp/{model_name}")


_DOWNLOAD_MAP: Dict[str, Callable] = {
    "faster-whisper": _download_faster_whisper_model,
    "vosk": _download_vosk_model,
    "whisper": _download_whisper_model,
    "whisper-cpp": _download_whisper_cpp_model,
}


def download_model(
    engine_name: str,
    model_name: str,
    hf_mirror: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> None:
    """Download a single model for the given engine."""
    fn = _DOWNLOAD_MAP.get(engine_name)
    if fn is None:
        raise ValueError(
            f"Unknown engine: '{engine_name}'. "
            f"Supported: {', '.join(SUPPORTED_ENGINES)}"
        )
    models = ENGINE_MODELS.get(engine_name, [])
    if models and model_name not in models:
        raise ValueError(
            f"Unknown model '{model_name}' for engine '{engine_name}'. "
            f"Available: {', '.join(models)}"
        )
    fn(model_name, hf_mirror=hf_mirror, progress_cb=progress_cb)


def download_all_models(
    engine_name: str,
    hf_mirror: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[int, int]:
    """Download all known models for the given engine. Returns (success, total)."""
    models = ENGINE_MODELS.get(engine_name, [])
    if not models:
        if progress_cb:
            progress_cb(f"No downloadable models defined for engine: {engine_name}")
        return 0, 0

    success = 0
    for model_name in models:
        try:
            download_model(engine_name, model_name, hf_mirror=hf_mirror, progress_cb=progress_cb)
            success += 1
        except Exception as e:
            logger.error("Failed to download %s/%s: %s", engine_name, model_name, e)
            if progress_cb:
                progress_cb(f"FAILED: {engine_name}/{model_name}: {e}")
    return success, len(models)


def download_everything(
    hf_mirror: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[int, int]:
    """Download all known models for all engines. Returns (success, total)."""
    total_success = 0
    total_count = 0
    for engine_name in SUPPORTED_ENGINES:
        models = ENGINE_MODELS.get(engine_name, [])
        total_count += len(models)
        for model_name in models:
            try:
                download_model(engine_name, model_name, hf_mirror=hf_mirror, progress_cb=progress_cb)
                total_success += 1
            except Exception as e:
                logger.error("Failed to download %s/%s: %s", engine_name, model_name, e)
                if progress_cb:
                    progress_cb(f"FAILED: {engine_name}/{model_name}: {e}")
    return total_success, total_count


# ─── Settings Persistence ────────────────────────────────────────────────────


def load_settings() -> Dict:
    """Load user settings from config file."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load settings: %s", e)
    return {}


def save_settings(settings: Dict) -> None:
    """Save user settings to config file."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as e:
        logger.warning("Failed to save settings: %s", e)


# ─── Translation Functions ───────────────────────────────────────────────────


def translate_with_nllb(
    texts: List[str],
    src_lang: str,
    tgt_lang: str,
    model_name: str = DEFAULT_NLLB_MODEL,
    hf_mirror: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """Translate a list of texts using Helsinki-NLP / NLLB model."""
    if hf_mirror:
        setup_hf_mirror(hf_mirror)

    src_nllb = NLLB_LANG_MAP.get(src_lang)
    tgt_nllb = NLLB_LANG_MAP.get(tgt_lang)
    if not src_nllb or not tgt_nllb:
        logger.error(
            "NLLB does not support language pair: %s -> %s", src_lang, tgt_lang
        )
        return texts

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError:
        raise EngineNotAvailableError(
            "transformers is not installed. Install with: pip install transformers sentencepiece"
        )

    if progress_cb:
        progress_cb(f"Loading NLLB model: {model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer.src_lang = src_nllb
    tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_nllb)

    translated: List[str] = []
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model.generate(
            **inputs, forced_bos_token_id=tgt_token_id, max_new_tokens=512,
        )
        for out in outputs:
            translated.append(tokenizer.decode(out, skip_special_tokens=True))
        if progress_cb:
            done = min(i + batch_size, len(texts))
            progress_cb(f"NLLB translating: {done}/{len(texts)} segments")

    return translated


def translate_with_llm(
    texts: List[str],
    src_lang: str,
    tgt_lang: str,
    config: TranslationConfig,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """Translate a list of texts using LLM (Ollama or OpenAI-compatible API)."""
    prompt = LLM_TRANSLATE_PROMPT.format(src_lang=src_lang, tgt_lang=tgt_lang)
    translated: List[str] = list(texts)  # fallback copy
    batch_size = 30

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = json.dumps(
            [{"i": j, "t": t} for j, t in enumerate(batch)],
            ensure_ascii=False,
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": payload},
        ]

        # Reuse the existing calling infrastructure
        polish_cfg = PolishConfig(
            ollama_url=config.ollama_url,
            ollama_model=config.ollama_model,
            api_url=config.api_url,
            api_key=config.api_key,
            api_model=config.api_model,
        )

        if config.api_url and config.api_key:
            response = _call_openai_api(messages, polish_cfg)
        else:
            response = _call_ollama(messages, polish_cfg)

        if not response:
            logger.warning("LLM returned empty response for translation batch %d", i)
            continue

        try:
            cleaned = _extract_json_array(response)
            items = json.loads(cleaned)
            for item in items:
                idx = item.get("i", -1)
                txt = item.get("t", "")
                if 0 <= idx < len(batch) and txt:
                    translated[i + idx] = txt
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to parse LLM translation response: %s", e)

        if progress_cb:
            done = min(i + batch_size, len(texts))
            progress_cb(f"LLM translating: {done}/{len(texts)} segments")

    return translated


def translate_segments(
    segments: List[Segment],
    src_lang: str,
    tgt_lang: str,
    config: TranslationConfig,
    hf_mirror: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[Segment]:
    """Translate all segments from src_lang to tgt_lang. Returns new segments."""
    if src_lang == tgt_lang:
        return segments

    texts = [seg.text for seg in segments]

    if config.method == "nllb":
        translated_texts = translate_with_nllb(
            texts, src_lang, tgt_lang,
            model_name=config.nllb_model,
            hf_mirror=hf_mirror,
            progress_cb=progress_cb,
        )
    else:
        translated_texts = translate_with_llm(
            texts, src_lang, tgt_lang, config,
            progress_cb=progress_cb,
        )

    return [
        Segment(start=seg.start, end=seg.end, text=txt)
        for seg, txt in zip(segments, translated_texts)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# §7  STT Engine Abstraction + Implementations
# ═══════════════════════════════════════════════════════════════════════════════


class EngineNotAvailableError(Exception):
    pass


class STTEngine(ABC):
    name: str = ""

    @abstractmethod
    def transcribe(
        self, audio_path: Path, language: Optional[str] = None
    ) -> TranscriptionResult:
        ...

    @classmethod
    def check_available(cls) -> bool:
        return False

    @classmethod
    def get_models(cls) -> List[str]:
        return ENGINE_MODELS.get(cls.name, [])


# ─── faster-whisper ──────────────────────────────────────────────────────────


class FasterWhisperEngine(STTEngine):
    name = "faster-whisper"

    def __init__(self, model_name: str = "base"):
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise EngineNotAvailableError(
                "faster-whisper is not installed. "
                "Install with: pip install faster-whisper"
            )
        self.model_name = model_name
        logger.info("Loading faster-whisper model '%s' ...", model_name)
        self._model = WhisperModel(model_name, device="auto", compute_type="auto")

    def transcribe(
        self, audio_path: Path, language: Optional[str] = None
    ) -> TranscriptionResult:
        logger.info("Transcribing with faster-whisper: %s", audio_path.name)
        kwargs = {"beam_size": 5}
        if language:
            kwargs["language"] = language
        segments_gen, info = self._model.transcribe(str(audio_path), **kwargs)
        segments = [
            Segment(start=seg.start, end=seg.end, text=seg.text.strip())
            for seg in segments_gen
            if seg.text.strip()
        ]
        return TranscriptionResult(
            segments=segments,
            language=getattr(info, "language", language),
            duration=getattr(info, "duration", None),
        )

    @classmethod
    def check_available(cls) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False


# ─── Vosk ────────────────────────────────────────────────────────────────────


class VoskEngine(STTEngine):
    name = "vosk"

    def __init__(self, model_name: str = ""):
        try:
            from vosk import Model as VoskModel, SetLogLevel
        except ImportError:
            raise EngineNotAvailableError(
                "vosk is not installed. Install with: pip install vosk"
            )
        SetLogLevel(-1)
        if model_name and os.path.isdir(model_name):
            logger.info("Loading Vosk model from path: %s", model_name)
            self._model = VoskModel(model_path=model_name)
        elif model_name:
            logger.info("Loading Vosk model: %s", model_name)
            self._model = VoskModel(model_name=model_name)
        else:
            logger.info("Loading default Vosk model (English) ...")
            self._model = VoskModel(lang="en")

    def transcribe(
        self, audio_path: Path, language: Optional[str] = None
    ) -> TranscriptionResult:
        from vosk import KaldiRecognizer

        logger.info("Transcribing with Vosk: %s", audio_path.name)
        wf = wave.open(str(audio_path), "rb")
        sample_rate = wf.getframerate()
        rec = KaldiRecognizer(self._model, sample_rate)
        rec.SetWords(True)

        words: List[Dict] = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                part = json.loads(rec.Result())
                if "result" in part:
                    words.extend(part["result"])
        final = json.loads(rec.FinalResult())
        if "result" in final:
            words.extend(final["result"])
        wf.close()

        segments = self._group_words(words)
        return TranscriptionResult(segments=segments, language=language)

    @staticmethod
    def _group_words(words: List[Dict], pause_threshold: float = 0.7) -> List[Segment]:
        if not words:
            return []
        segments: List[Segment] = []
        current_start = words[0].get("start", 0.0)
        current_end = words[0].get("end", 0.0)
        current_text = [words[0].get("word", "")]

        for i in range(1, len(words)):
            w = words[i]
            w_start = w.get("start", 0.0)
            w_end = w.get("end", 0.0)
            gap = w_start - current_end
            if gap > pause_threshold:
                text = " ".join(current_text).strip()
                if text:
                    segments.append(Segment(current_start, current_end, text))
                current_start = w_start
                current_text = []
            current_end = w_end
            current_text.append(w.get("word", ""))

        text = " ".join(current_text).strip()
        if text:
            segments.append(Segment(current_start, current_end, text))
        return segments

    @classmethod
    def check_available(cls) -> bool:
        try:
            import vosk  # noqa: F401
            return True
        except ImportError:
            return False


# ─── OpenAI Whisper ──────────────────────────────────────────────────────────


class WhisperEngine(STTEngine):
    name = "whisper"

    def __init__(self, model_name: str = "base"):
        try:
            import whisper
        except ImportError:
            raise EngineNotAvailableError(
                "openai-whisper is not installed. "
                "Install with: pip install openai-whisper"
            )
        self.model_name = model_name
        logger.info("Loading OpenAI Whisper model '%s' ...", model_name)
        self._model = whisper.load_model(model_name)

    def transcribe(
        self, audio_path: Path, language: Optional[str] = None
    ) -> TranscriptionResult:
        logger.info("Transcribing with OpenAI Whisper: %s", audio_path.name)
        kwargs = {}
        if language:
            kwargs["language"] = language
        result = self._model.transcribe(str(audio_path), **kwargs)
        segments = [
            Segment(start=seg["start"], end=seg["end"], text=seg["text"].strip())
            for seg in result.get("segments", [])
            if seg.get("text", "").strip()
        ]
        return TranscriptionResult(
            segments=segments,
            language=result.get("language", language),
            duration=result.get("duration"),
        )

    @classmethod
    def check_available(cls) -> bool:
        try:
            import whisper  # noqa: F401
            return True
        except ImportError:
            return False


# ─── whisper.cpp (pywhispercpp) ──────────────────────────────────────────────


class WhisperCppEngine(STTEngine):
    name = "whisper-cpp"

    def __init__(self, model_name: str = "base"):
        try:
            from pywhispercpp.model import Model as CppModel
        except ImportError:
            raise EngineNotAvailableError(
                "pywhispercpp is not installed. "
                "Install with: pip install pywhispercpp"
            )
        self.model_name = model_name
        logger.info("Loading whisper.cpp model '%s' ...", model_name)
        self._model = CppModel(model_name)

    def transcribe(
        self, audio_path: Path, language: Optional[str] = None
    ) -> TranscriptionResult:
        logger.info("Transcribing with whisper.cpp: %s", audio_path.name)
        result = self._model.transcribe(str(audio_path))
        segments = []
        for seg in result:
            t0 = getattr(seg, "t0", 0) / 100.0
            t1 = getattr(seg, "t1", 0) / 100.0
            text = getattr(seg, "text", "").strip()
            if text:
                segments.append(Segment(start=t0, end=t1, text=text))
        return TranscriptionResult(segments=segments, language=language)

    @classmethod
    def check_available(cls) -> bool:
        try:
            from pywhispercpp.model import Model  # noqa: F401
            return True
        except ImportError:
            return False


# ─── Engine Factory ──────────────────────────────────────────────────────────

_ENGINE_MAP: Dict[str, type] = {
    "faster-whisper": FasterWhisperEngine,
    "vosk": VoskEngine,
    "whisper": WhisperEngine,
    "whisper-cpp": WhisperCppEngine,
}


def create_engine(engine_name: str, model_name: Optional[str] = None) -> STTEngine:
    cls = _ENGINE_MAP.get(engine_name)
    if cls is None:
        raise ValueError(
            f"Unknown engine: '{engine_name}'. "
            f"Supported: {', '.join(SUPPORTED_ENGINES)}"
        )
    model = model_name or ENGINE_DEFAULT_MODEL.get(engine_name, "base")
    if engine_name == "vosk":
        return cls(model_name=model_name or "")
    return cls(model_name=model)


# ═══════════════════════════════════════════════════════════════════════════════
# §8  Subtitle Formatters (SRT + LRC + MD)
# ═══════════════════════════════════════════════════════════════════════════════


def generate_srt(segments: List[Segment]) -> str:
    lines: List[str] = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{format_time_srt(seg.start)} --> {format_time_srt(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def generate_lrc(segments: List[Segment], title: str = "") -> str:
    lines: List[str] = []
    if title:
        lines.append(f"[ti:{title}]")
    lines.append(f"[re:TingShuo v{__version__}]")
    lines.append("")
    for seg in segments:
        lines.append(f"{format_time_lrc(seg.start)}{seg.text}")
    return "\n".join(lines)


def generate_transcript(
    segments: List[Segment], config: PolishConfig,
) -> str:
    """Generate a Markdown transcript from segments using LLM.

    Falls back to simple paragraph grouping if LLM is unavailable.
    """
    full_text = " ".join(seg.text for seg in segments)
    if not full_text.strip():
        return ""

    # Try LLM-based transcript generation
    messages = [
        {"role": "system", "content": LLM_TRANSCRIPT_PROMPT},
        {"role": "user", "content": full_text},
    ]

    response = ""
    if config.api_url and config.api_key:
        response = _call_openai_api(messages, config)
    elif config.ollama_model:
        response = _call_ollama(messages, config)

    if response and response.strip():
        logger.info("Markdown transcript generated via LLM")
        return response.strip()

    # Fallback: simple paragraph grouping (every ~10 segments)
    logger.info("LLM unavailable for transcript, using simple paragraph grouping")
    paragraphs: List[str] = []
    group_size = 10
    for i in range(0, len(segments), group_size):
        group = segments[i:i + group_size]
        para = " ".join(seg.text for seg in group).strip()
        if para:
            paragraphs.append(para)
    return "\n\n".join(paragraphs)


def write_subtitle(
    segments: List[Segment], output_path: Path, fmt: str,
    title: str = "", polish_config: Optional[PolishConfig] = None,
) -> None:
    os.makedirs(output_path.parent, exist_ok=True)
    if fmt == "lrc":
        content = generate_lrc(segments, title=title)
    elif fmt == "md":
        config = polish_config or PolishConfig()
        content = generate_transcript(segments, config)
    else:
        content = generate_srt(segments)
    output_path.write_text(content, encoding="utf-8-sig")
    logger.info("Saved: %s", output_path)


# ═══════════════════════════════════════════════════════════════════════════════
# §9  Polishing (LLM + NLP)
# ═══════════════════════════════════════════════════════════════════════════════


def _segments_to_json(segments: List[Segment]) -> str:
    return json.dumps(
        [{"s": round(seg.start, 3), "e": round(seg.end, 3), "t": seg.text}
         for seg in segments],
        ensure_ascii=False,
    )


def _json_to_segments(text: str) -> List[Segment]:
    data = json.loads(text)
    return [Segment(start=d["s"], end=d["e"], text=d["t"]) for d in data]


def _call_ollama(messages: List[Dict], config: PolishConfig) -> str:
    url = config.ollama_url.rstrip("/") + "/api/chat"
    body = json.dumps({
        "model": config.ollama_model,
        "messages": messages,
        "stream": False,
    }).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result.get("message", {}).get("content", "")
    except (URLError, HTTPError, TimeoutError) as e:
        logger.warning("Ollama API call failed: %s", e)
        return ""


def _call_openai_api(messages: List[Dict], config: PolishConfig) -> str:
    base = config.api_url.rstrip("/")
    if not base.endswith("/v1"):
        url = base + "/v1/chat/completions"
    else:
        url = base + "/chat/completions"
    body = json.dumps({
        "model": config.api_model,
        "messages": messages,
        "temperature": 0.3,
    }).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }
    req = Request(url, data=body, headers=headers)
    try:
        with urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        choices = result.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""
    except (URLError, HTTPError, TimeoutError) as e:
        logger.warning("OpenAI-compatible API call failed: %s", e)
        return ""


def _extract_json_array(text: str) -> str:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def polish_with_llm(
    segments: List[Segment], config: PolishConfig
) -> List[Segment]:
    if not segments:
        return segments

    logger.info("Polishing subtitles with LLM ...")
    batch_size = 30
    polished: List[Segment] = []

    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        seg_json = _segments_to_json(batch)
        messages = [
            {"role": "system", "content": LLM_POLISH_PROMPT},
            {"role": "user", "content": seg_json},
        ]

        if config.api_url and config.api_key:
            response = _call_openai_api(messages, config)
        else:
            response = _call_ollama(messages, config)

        if not response:
            logger.warning("LLM returned empty response for batch %d, keeping originals", i)
            polished.extend(batch)
            continue

        try:
            cleaned = _extract_json_array(response)
            new_segments = _json_to_segments(cleaned)
            polished.extend(new_segments)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to parse LLM response: %s. Keeping originals.", e)
            polished.extend(batch)

    logger.info("LLM polishing complete: %d -> %d segments", len(segments), len(polished))
    return polished


def auto_correct_with_llm(
    segments: List[Segment], config: PolishConfig
) -> List[Segment]:
    """Use LLM to fix typos, wrong characters, and verbal mistakes in segments."""
    if not segments:
        return segments

    logger.info("Auto-correcting transcription with LLM ...")
    batch_size = 30
    corrected: List[Segment] = []

    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        seg_json = _segments_to_json(batch)
        messages = [
            {"role": "system", "content": LLM_AUTOCORRECT_PROMPT},
            {"role": "user", "content": seg_json},
        ]

        if config.api_url and config.api_key:
            response = _call_openai_api(messages, config)
        else:
            response = _call_ollama(messages, config)

        if not response:
            logger.warning("LLM returned empty response for auto-correct batch %d, keeping originals", i)
            corrected.extend(batch)
            continue

        try:
            cleaned = _extract_json_array(response)
            new_segments = _json_to_segments(cleaned)
            # Validate segment count matches — auto-correct must not merge/split
            if len(new_segments) != len(batch):
                logger.warning(
                    "Auto-correct returned %d segments (expected %d) for batch %d. Keeping originals.",
                    len(new_segments), len(batch), i,
                )
                corrected.extend(batch)
            else:
                corrected.extend(new_segments)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to parse auto-correct response: %s. Keeping originals.", e)
            corrected.extend(batch)

    logger.info("Auto-correction complete: %d segments processed", len(corrected))
    return corrected


def polish_with_nlp(
    segments: List[Segment], language: Optional[str] = None
) -> List[Segment]:
    if not segments:
        return segments

    logger.info("Polishing subtitles with NLP sentence segmentation ...")

    lang_code = (language or "en").split("-")[0].split("_")[0].lower()

    # Check if this is a CJK language without nltk tokenizer support
    if lang_code in ("zh", "ja", "ko"):
        return _polish_cjk(segments)

    try:
        import nltk
    except ImportError:
        logger.warning(
            "nltk is not installed. Install with: pip install nltk. "
            "Skipping NLP polishing."
        )
        return segments

    try:
        nltk_lang = NLTK_LANG_MAP.get(lang_code, "english")
        nltk.data.find(f"tokenizers/punkt_tab/{nltk_lang}")
    except LookupError:
        logger.info("Downloading nltk punkt_tab data ...")
        nltk.download("punkt_tab", quiet=True)

    nltk_lang = NLTK_LANG_MAP.get(lang_code, "english")

    # Build a combined text with segment boundary markers
    full_text = " ".join(seg.text for seg in segments)
    sentences = nltk.sent_tokenize(full_text, language=nltk_lang)

    # Elastic post-processing: split overly long sentences, merge very short ones
    refined: List[str] = []
    for sent in sentences:
        if len(sent) > ELASTIC_LATIN_MAX:
            # Split at clause boundaries and accumulate to target length
            parts = LATIN_CLAUSE_SPLIT.split(sent)
            buf = ""
            for part in parts:
                if buf and len(buf) + len(part) > ELASTIC_LATIN_TARGET:
                    refined.append(buf.strip())
                    buf = part
                else:
                    buf = buf + part if buf else part
            if buf.strip():
                refined.append(buf.strip())
        elif len(sent) < ELASTIC_LATIN_MIN and refined and len(refined[-1]) < ELASTIC_LATIN_TARGET:
            # Merge very short sentence with previous short one
            refined[-1] = refined[-1] + " " + sent
        else:
            refined.append(sent)

    return _map_sentences_to_segments(refined, segments)


def _split_long_segment(
    text: str, start: float, end: float, is_cjk: bool = False,
) -> List[Segment]:
    """Split an overly long segment at clause boundaries with proportional timestamps."""
    text = text.strip()
    if not text:
        return []

    max_len = ELASTIC_CJK_MAX if is_cjk else ELASTIC_LATIN_MAX
    target_len = ELASTIC_CJK_TARGET if is_cjk else ELASTIC_LATIN_TARGET

    # Already at or below target length - no splitting needed
    if len(text) <= target_len:
        return [Segment(start=start, end=end, text=text)]

    # Split at secondary clause boundaries
    splitter = CJK_CLAUSE_SPLIT if is_cjk else LATIN_CLAUSE_SPLIT
    parts = splitter.split(text)

    # If regex produced no splits and text is within max, accept it
    if len(parts) <= 1:
        if len(text) <= max_len:
            return [Segment(start=start, end=end, text=text)]
        return _hard_split_segment(text, start, end, target_len, is_cjk)

    # Accumulate chunks to target length
    total_chars = len(text)
    duration = end - start
    result: List[Segment] = []
    buf = ""
    buf_char_offset = 0  # character offset of buffer start within original text

    for part in parts:
        if buf and len(buf) + len(part) > target_len:
            # Emit current buffer
            seg_start = start + (buf_char_offset / total_chars) * duration
            seg_end = start + ((buf_char_offset + len(buf)) / total_chars) * duration
            result.append(Segment(start=round(seg_start, 3), end=round(seg_end, 3), text=buf.strip()))
            buf_char_offset += len(buf)
            buf = part
        else:
            buf = buf + part if buf else part

    if buf.strip():
        seg_start = start + (buf_char_offset / total_chars) * duration
        result.append(Segment(start=round(seg_start, 3), end=round(end, 3), text=buf.strip()))

    # Recursively split any still-oversized segments
    final: List[Segment] = []
    for seg in result:
        if len(seg.text) > max_len:
            final.extend(_hard_split_segment(seg.text, seg.start, seg.end, target_len, is_cjk))
        else:
            final.append(seg)

    return final if final else [Segment(start=start, end=end, text=text)]


def _hard_split_segment(
    text: str, start: float, end: float, target_len: int, is_cjk: bool,
) -> List[Segment]:
    """Hard-split text at target length on word/character boundaries."""
    total_chars = len(text)
    duration = end - start
    result: List[Segment] = []
    pos = 0

    while pos < total_chars:
        chunk_end = min(pos + target_len, total_chars)
        if chunk_end < total_chars and not is_cjk:
            # For Latin, try to break at a space
            space_pos = text.rfind(" ", pos, chunk_end + 10)
            if space_pos > pos:
                chunk_end = space_pos + 1
        chunk = text[pos:chunk_end].strip()
        if chunk:
            seg_start = start + (pos / total_chars) * duration
            seg_end = start + (chunk_end / total_chars) * duration if chunk_end < total_chars else end
            result.append(Segment(start=round(seg_start, 3), end=round(seg_end, 3), text=chunk))
        pos = chunk_end

    return result if result else [Segment(start=start, end=end, text=text)]


def _merge_short_segments(
    segments: List[Segment], is_cjk: bool = False,
) -> List[Segment]:
    """Merge consecutive short segments that are below the minimum length threshold."""
    if not segments:
        return segments

    min_len = ELASTIC_CJK_MIN if is_cjk else ELASTIC_LATIN_MIN
    target_len = ELASTIC_CJK_TARGET if is_cjk else ELASTIC_LATIN_TARGET
    max_len = ELASTIC_CJK_MAX if is_cjk else ELASTIC_LATIN_MAX
    joiner = "" if is_cjk else " "

    result: List[Segment] = []
    buf_texts: List[str] = []
    buf_start: Optional[float] = None
    buf_end: float = 0.0

    for seg in segments:
        combined_len = len(joiner.join(buf_texts + [seg.text]))

        if buf_start is None:
            # Start a new buffer
            buf_texts = [seg.text]
            buf_start = seg.start
            buf_end = seg.end
        elif len(joiner.join(buf_texts)) < min_len and combined_len <= target_len:
            # Current buffer is too short and merging won't exceed target
            buf_texts.append(seg.text)
            buf_end = seg.end
        else:
            # Emit current buffer, start new one
            result.append(Segment(
                start=buf_start, end=buf_end,
                text=joiner.join(buf_texts).strip(),
            ))
            buf_texts = [seg.text]
            buf_start = seg.start
            buf_end = seg.end

    # Flush remaining buffer
    if buf_texts and buf_start is not None:
        merged = joiner.join(buf_texts).strip()
        if merged:
            # Try to merge with last result segment if both are short
            if result and len(merged) < min_len and len(result[-1].text) + len(merged) <= max_len:
                prev = result.pop()
                merged = joiner.join([prev.text, merged]).strip()
                result.append(Segment(start=prev.start, end=buf_end, text=merged))
            else:
                result.append(Segment(start=buf_start, end=buf_end, text=merged))

    return result


def _polish_cjk(segments: List[Segment]) -> List[Segment]:
    """Polish CJK subtitles: split on sentence boundaries with elastic length control."""
    # Pass 1: Split on sentence-ending punctuation
    raw_result: List[Segment] = []
    buf_text: List[str] = []
    buf_start: Optional[float] = None
    buf_end: float = 0.0

    for seg in segments:
        if buf_start is None:
            buf_start = seg.start
        buf_text.append(seg.text)
        buf_end = seg.end

        combined = "".join(buf_text)
        if CJK_SENT_END.search(combined):
            raw_result.append(Segment(start=buf_start, end=buf_end, text=combined.strip()))
            buf_text = []
            buf_start = None

    # Handle remainder without sentence-ending punctuation
    if buf_text and buf_start is not None:
        remainder = "".join(buf_text).strip()
        if remainder:
            # Split unpunctuated remainder at clause boundaries instead of one mega-segment
            raw_result.extend(_split_long_segment(remainder, buf_start, buf_end, is_cjk=True))

    # Pass 2: Elastic refinement - split oversized, merge undersized
    refined: List[Segment] = []
    for seg in raw_result:
        if len(seg.text) > ELASTIC_CJK_MAX:
            refined.extend(_split_long_segment(seg.text, seg.start, seg.end, is_cjk=True))
        else:
            refined.append(seg)

    result = _merge_short_segments(refined, is_cjk=True)

    logger.info("CJK NLP polishing: %d -> %d segments", len(segments), len(result))
    return result


def _map_sentences_to_segments(
    sentences: List[str], segments: List[Segment]
) -> List[Segment]:
    result: List[Segment] = []
    seg_idx = 0
    seg_count = len(segments)

    for sentence in sentences:
        if seg_idx >= seg_count:
            break

        sent_start = segments[seg_idx].start
        sent_end = segments[seg_idx].end
        remaining = sentence.strip()

        while seg_idx < seg_count:
            seg_text = segments[seg_idx].text.strip()
            sent_end = segments[seg_idx].end

            if seg_text in remaining:
                remaining = remaining[remaining.index(seg_text) + len(seg_text):].strip()
                seg_idx += 1
                if not remaining:
                    break
            else:
                seg_idx += 1
                break

        result.append(Segment(start=sent_start, end=sent_end, text=sentence.strip()))

    # Append any remaining segments that weren't matched
    while seg_idx < seg_count:
        result.append(segments[seg_idx])
        seg_idx += 1

    # Elastic guardrail: split oversized and merge undersized segments
    guarded: List[Segment] = []
    for seg in result:
        if len(seg.text) > ELASTIC_LATIN_MAX:
            guarded.extend(_split_long_segment(seg.text, seg.start, seg.end, is_cjk=False))
        else:
            guarded.append(seg)
    result = _merge_short_segments(guarded, is_cjk=False)

    logger.info("NLP polishing: %d -> %d segments", len(segments), len(result))
    return result


# ─── Multimodal & Summarization ──────────────────────────────────────────────


def _call_ollama_multimodal(
    text: str, images_b64: List[str], config: PolishConfig,
) -> str:
    """Call Ollama API with multimodal content (text + images)."""
    url = config.ollama_url.rstrip("/") + "/api/chat"
    body = json.dumps({
        "model": config.ollama_model,
        "messages": [
            {"role": "system", "content": LLM_SUMMARIZE_MULTIMODAL_PROMPT},
            {"role": "user", "content": text, "images": images_b64},
        ],
        "stream": False,
    }).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result.get("message", {}).get("content", "")
    except (URLError, HTTPError, TimeoutError) as e:
        logger.warning("Ollama multimodal API call failed: %s", e)
        return ""


def _call_openai_api_multimodal(
    text: str, images_b64: List[str], config: PolishConfig,
) -> str:
    """Call OpenAI-compatible API with vision content (text + images)."""
    base = config.api_url.rstrip("/")
    if not base.endswith("/v1"):
        url = base + "/v1/chat/completions"
    else:
        url = base + "/chat/completions"

    # Build multimodal user content
    user_content: List[Dict] = [{"type": "text", "text": text}]
    for img_b64 in images_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })

    body = json.dumps({
        "model": config.api_model,
        "messages": [
            {"role": "system", "content": LLM_SUMMARIZE_MULTIMODAL_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
    }).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }
    req = Request(url, data=body, headers=headers)
    try:
        with urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        choices = result.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""
    except (URLError, HTTPError, TimeoutError) as e:
        logger.warning("OpenAI multimodal API call failed: %s", e)
        return ""


def summarize_content(
    segments: List[Segment],
    media_path: Path,
    config: JobConfig,
    temp_dir: str,
    progress_cb: Optional[Callable] = None,
) -> str:
    """Generate a content summary. Uses multimodal analysis for video files."""
    full_text = " ".join(seg.text for seg in segments)
    if not full_text.strip():
        return ""

    polish_cfg = config.polish
    is_video = media_path.suffix.lower() in VIDEO_EXTENSIONS

    # Try multimodal summarization for video files
    if is_video:
        logger.info("Extracting keyframes for multimodal summarization ...")
        frames = extract_keyframes(
            media_path, temp_dir,
            interval=config.summarize.keyframe_interval,
        )
        if frames:
            images_b64 = [encode_image_base64(f) for f in frames]
            logger.info("Attempting multimodal summary with %d keyframe(s) ...", len(images_b64))

            # Try OpenAI-compatible API first (more likely to support vision)
            if polish_cfg.api_url and polish_cfg.api_key:
                response = _call_openai_api_multimodal(full_text, images_b64, polish_cfg)
                if response and response.strip():
                    logger.info("Multimodal summary generated via OpenAI-compatible API")
                    return response.strip()

            # Try Ollama multimodal
            if polish_cfg.ollama_model:
                response = _call_ollama_multimodal(full_text, images_b64, polish_cfg)
                if response and response.strip():
                    logger.info("Multimodal summary generated via Ollama")
                    return response.strip()

            logger.info("Multimodal summary failed, falling back to text-only summary")

    # Text-only summary
    messages = [
        {"role": "system", "content": LLM_SUMMARIZE_PROMPT},
        {"role": "user", "content": full_text},
    ]

    response = ""
    if polish_cfg.api_url and polish_cfg.api_key:
        response = _call_openai_api(messages, polish_cfg)
    elif polish_cfg.ollama_model:
        response = _call_ollama(messages, polish_cfg)

    if response and response.strip():
        logger.info("Text-only summary generated via LLM")
        return response.strip()

    # Last resort: simple stats-based summary
    logger.info("LLM unavailable for summary, generating basic stats summary")
    word_count = len(full_text.split())
    duration = segments[-1].end if segments else 0
    dur_min = int(duration // 60)
    dur_sec = int(duration % 60)
    first_text = segments[0].text if segments else ""
    last_text = segments[-1].text if segments else ""
    return (
        f"## Summary\n\n"
        f"- **Duration**: {dur_min}m {dur_sec}s\n"
        f"- **Word count**: {word_count}\n"
        f"- **Segments**: {len(segments)}\n\n"
        f"### Opening\n\n{first_text}\n\n"
        f"### Closing\n\n{last_text}\n"
    )


def write_summary(summary: str, output_path: Path) -> None:
    """Write a summary Markdown file."""
    os.makedirs(output_path.parent, exist_ok=True)
    output_path.write_text(summary, encoding="utf-8")
    logger.info("Summary saved: %s", output_path)


# ═══════════════════════════════════════════════════════════════════════════════
# §10  Core Processing Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def process_file(
    media_path: Path,
    config: JobConfig,
    engine: STTEngine,
    progress_cb: Optional[Callable] = None,
) -> Optional[Path]:
    logger.info("Processing: %s", media_path.name)
    temp_dir = tempfile.mkdtemp(prefix="tingshuo_")
    try:
        # Determine output path
        ext_map = {"lrc": ".lrc", "md": ".md", "srt": ".srt"}
        ext = ext_map.get(config.format, ".srt")
        if config.output_dir:
            out_dir = Path(config.output_dir)
        else:
            out_dir = media_path.parent
        output_path = out_dir / (media_path.stem + ext)

        # Extract audio
        wav_path = extract_audio(media_path, temp_dir)

        # Transcribe
        result = engine.transcribe(wav_path, language=config.language)
        if not result.segments:
            logger.warning("No speech detected in: %s", media_path.name)
            return None

        # Auto-correct (before polishing)
        segments = result.segments
        if config.auto_correct:
            segments = auto_correct_with_llm(segments, config.polish)

        # Polish
        polished = segments
        if config.polish.method == "llm":
            polished = polish_with_llm(segments, config.polish)
        elif config.polish.method == "nlp":
            polished = polish_with_nlp(segments, config.language)

        # Write output (SRT / LRC / MD)
        write_subtitle(
            polished, output_path, config.format,
            title=media_path.stem, polish_config=config.polish,
        )

        # Translation: generate translated subtitle files
        if config.translation.enabled and config.translation.target_languages:
            detected_lang = (result.language or config.language or "en").split("-")[0].split("_")[0].lower()
            logger.info("Detected language: %s", detected_lang)
            for tgt_lang in config.translation.target_languages:
                tgt = tgt_lang.strip().lower()
                if tgt == detected_lang:
                    logger.info("Skipping translation to %s (same as source)", tgt)
                    continue
                logger.info("Translating subtitles: %s -> %s", detected_lang, tgt)
                try:
                    translated = translate_segments(
                        polished, detected_lang, tgt, config.translation,
                        hf_mirror=config.hf_mirror,
                    )
                    trans_path = out_dir / f"{media_path.stem}.{tgt}{ext}"
                    write_subtitle(
                        translated, trans_path, config.format,
                        title=media_path.stem, polish_config=config.polish,
                    )
                    logger.info("Translated subtitle saved: %s", trans_path)
                except Exception as te:
                    logger.error("Translation to %s failed: %s", tgt, te)

        # Summarization (must happen before temp_dir cleanup)
        if config.summarize.enabled:
            logger.info("Generating content summary for %s ...", media_path.name)
            summary = summarize_content(
                polished, media_path, config, temp_dir,
            )
            if summary:
                summary_path = out_dir / f"{media_path.stem}.summary.md"
                write_summary(summary, summary_path)

        return output_path

    except Exception as e:
        logger.error("Failed to process %s: %s", media_path.name, e)
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def process_batch(
    config: JobConfig,
    progress_cb: Optional[Callable] = None,
    stop_event: Optional[threading.Event] = None,
) -> Tuple[int, int]:
    if not check_ffmpeg():
        logger.error(
            "ffmpeg not found. Please install ffmpeg and ensure it is on your PATH."
        )
        return 0, 0

    files = scan_media_files(config.input_dir, config.recursive)
    if not files:
        logger.warning("No media files found in: %s", config.input_dir)
        return 0, 0

    if config.hf_mirror:
        setup_hf_mirror(config.hf_mirror)

    # Create engine once for the whole batch
    try:
        engine = create_engine(config.engine_name, config.model_name or None)
    except (EngineNotAvailableError, ValueError) as e:
        logger.error("Engine error: %s", e)
        return 0, 0

    total = len(files)
    success = 0
    for i, fpath in enumerate(files):
        if stop_event and stop_event.is_set():
            logger.info("Processing stopped by user.")
            break

        if progress_cb:
            progress_cb(i, total, fpath.name)

        result = process_file(fpath, config, engine)
        if result:
            success += 1

    if progress_cb:
        progress_cb(total, total, "Done")

    logger.info("Completed: %d/%d files processed successfully.", success, total)
    return success, total


# ═══════════════════════════════════════════════════════════════════════════════
# §11  CLI Interface
# ═══════════════════════════════════════════════════════════════════════════════


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tingshuo",
        description="TingShuo (听说) - Generate SRT/LRC/MD subtitles and transcripts from audio/video files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  tingshuo -i ./videos -e faster-whisper -f srt\n"
            "  tingshuo -i ./audio -e vosk -f lrc -o ./subtitles\n"
            "  tingshuo -i ./lectures -f md --polish-llm --ollama-model qwen2.5\n"
            "  tingshuo -i ./media --auto-correct --polish-llm\n"
            "  tingshuo -i ./media --summarize --ollama-model qwen2.5\n"
            "  tingshuo -i ./media --polish-nlp -l zh\n"
            "  tingshuo --download -e faster-whisper -m large-v3\n"
            "  tingshuo --download-all -e faster-whisper\n"
            "  tingshuo --list-ollama-models\n"
            "  tingshuo --gui\n"
        ),
    )

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--gui", action="store_true", help="Launch the graphical interface"
    )

    # Input/Output
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "-i", "--input", metavar="DIR",
        help="Input directory containing audio/video files (required for CLI mode)",
    )
    io_group.add_argument(
        "-o", "--output", metavar="DIR", default="",
        help="Output directory for subtitles (default: same as source files)",
    )
    io_group.add_argument(
        "-f", "--format", choices=["srt", "lrc", "md"], default="srt",
        help="Output format: srt, lrc, or md (Markdown transcript) (default: srt)",
    )
    io_group.add_argument(
        "--no-recursive", action="store_true",
        help="Do not scan subdirectories recursively",
    )

    # Engine
    eng_group = parser.add_argument_group("STT Engine")
    eng_group.add_argument(
        "-e", "--engine", choices=SUPPORTED_ENGINES, default=DEFAULT_ENGINE,
        help=f"Speech-to-text engine (default: {DEFAULT_ENGINE})",
    )
    eng_group.add_argument(
        "-m", "--model", metavar="NAME", default="",
        help="Model name or path (default: engine-specific, usually 'base')",
    )
    eng_group.add_argument(
        "-l", "--language", metavar="CODE", default="auto",
        help="Language code, e.g. zh, en, ja. Use 'auto' for auto-detection (default: auto)",
    )

    # HF Mirror
    hf_group = parser.add_argument_group("HuggingFace Mirror")
    hf_group.add_argument(
        "--hf-mirror", metavar="URL", default="",
        help="HuggingFace mirror URL, e.g. https://hf-mirror.com (for China mainland)",
    )

    # Model Management
    mgmt_group = parser.add_argument_group("Model Management")
    mgmt_group.add_argument(
        "--download", action="store_true",
        help="Download the model specified by -e and -m, then exit",
    )
    mgmt_group.add_argument(
        "--download-all", action="store_true",
        help="Download all known models for the engine specified by -e, then exit",
    )
    mgmt_group.add_argument(
        "--list-ollama-models", action="store_true",
        help="List installed Ollama models from the server (uses --ollama-url), then exit",
    )

    # Auto-Correction
    correct_group = parser.add_argument_group("Auto-Correction")
    correct_group.add_argument(
        "--auto-correct", action="store_true",
        help="Auto-correct typos, wrong characters, and verbal mistakes using LLM",
    )

    # Polishing
    pol_group = parser.add_argument_group("Subtitle Polishing")
    pol_excl = pol_group.add_mutually_exclusive_group()
    pol_excl.add_argument(
        "--polish-llm", action="store_true",
        help="Polish subtitles using LLM (Ollama or OpenAI-compatible API)",
    )
    pol_excl.add_argument(
        "--polish-nlp", action="store_true",
        help="Polish subtitles using NLP sentence segmentation (nltk)",
    )

    # LLM settings
    llm_group = parser.add_argument_group("LLM Settings (used with --polish-llm, --auto-correct, --summarize)")
    llm_group.add_argument(
        "--ollama-url", metavar="URL", default=DEFAULT_OLLAMA_URL,
        help=f"Ollama API URL (default: {DEFAULT_OLLAMA_URL})",
    )
    llm_group.add_argument(
        "--ollama-model", metavar="NAME", default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model name (default: {DEFAULT_OLLAMA_MODEL})",
    )
    llm_group.add_argument(
        "--api-url", metavar="URL", default="",
        help="OpenAI-compatible API base URL",
    )
    llm_group.add_argument(
        "--api-key", metavar="KEY", default="",
        help="API key for OpenAI-compatible service",
    )
    llm_group.add_argument(
        "--api-model", metavar="NAME", default="",
        help="Model name for OpenAI-compatible API",
    )

    # Misc
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose/debug logging"
    )

    # Translation
    trans_group = parser.add_argument_group("Translation")
    trans_group.add_argument(
        "--translate", action="store_true",
        help="Enable subtitle translation to target language(s)",
    )
    trans_group.add_argument(
        "--target-lang", metavar="CODES", default="",
        help="Comma-separated target language codes for translation, e.g. zh,en,ja",
    )
    trans_group.add_argument(
        "--trans-backend", choices=["nllb", "llm"], default="nllb",
        help="Translation backend: nllb (Helsinki-NLP/NLLB) or llm (default: nllb)",
    )
    trans_group.add_argument(
        "--nllb-model", metavar="NAME", default=DEFAULT_NLLB_MODEL,
        help=f"NLLB model name (default: {DEFAULT_NLLB_MODEL})",
    )

    # Summarization
    sum_group = parser.add_argument_group("Summarization")
    sum_group.add_argument(
        "--summarize", action="store_true",
        help="Generate a content summary (.summary.md) alongside the output",
    )
    sum_group.add_argument(
        "--keyframe-interval", metavar="SECONDS", type=int, default=60,
        help="Seconds between keyframe extractions for video summarization (default: 60)",
    )

    return parser


def run_cli(args: argparse.Namespace) -> None:
    setup_logging(args.verbose)

    if args.gui:
        launch_gui()
        return

    # ── Early-exit: list Ollama models ──
    if args.list_ollama_models:
        models = query_ollama_models(args.ollama_url)
        if models:
            print(f"Installed Ollama models at {args.ollama_url}:")
            for m in models:
                print(f"  {m}")
        else:
            logger.error(
                "Could not fetch models from %s. "
                "Ensure Ollama is running.", args.ollama_url
            )
            sys.exit(1)
        return

    # ── Early-exit: download model(s) ──
    if args.download:
        model = args.model
        if not model:
            default = ENGINE_DEFAULT_MODEL.get(args.engine, "base")
            model = default
            logger.info("No model specified, using default: %s", model)

        def dl_progress(msg: str) -> None:
            print(msg)

        try:
            download_model(args.engine, model, hf_mirror=args.hf_mirror, progress_cb=dl_progress)
        except (EngineNotAvailableError, ValueError, RuntimeError) as e:
            logger.error("Download failed: %s", e)
            sys.exit(1)
        return

    if args.download_all:
        def dl_progress(msg: str) -> None:
            print(msg)

        try:
            success, total = download_all_models(
                args.engine, hf_mirror=args.hf_mirror, progress_cb=dl_progress,
            )
            print(f"\nDownloaded {success}/{total} models for {args.engine}.")
            if success < total:
                sys.exit(1)
        except (EngineNotAvailableError, ValueError) as e:
            logger.error("Download failed: %s", e)
            sys.exit(1)
        return

    # ── Normal transcription mode ──
    if not args.input:
        logger.error("Input directory is required. Use -i/--input or --gui.")
        sys.exit(1)

    if not os.path.isdir(args.input):
        logger.error("Input directory does not exist: %s", args.input)
        sys.exit(1)

    # Map "auto" language to None (auto-detect)
    language = args.language
    if language and language.lower() == "auto":
        language = None

    polish = PolishConfig()
    if args.polish_llm:
        polish.method = "llm"
        polish.ollama_url = args.ollama_url
        polish.ollama_model = args.ollama_model
        polish.api_url = args.api_url
        polish.api_key = args.api_key
        polish.api_model = args.api_model
    elif args.polish_nlp:
        polish.method = "nlp"

    # If auto-correct or summarize or md format is used, ensure LLM config is populated
    if args.auto_correct or args.summarize or args.format == "md":
        polish.ollama_url = args.ollama_url
        polish.ollama_model = args.ollama_model
        polish.api_url = args.api_url
        polish.api_key = args.api_key
        polish.api_model = args.api_model

    translation = TranslationConfig()
    if args.translate:
        translation.enabled = True
        translation.method = args.trans_backend
        translation.nllb_model = args.nllb_model
        translation.ollama_url = args.ollama_url
        translation.ollama_model = args.ollama_model
        translation.api_url = args.api_url
        translation.api_key = args.api_key
        translation.api_model = args.api_model
        if args.target_lang:
            translation.target_languages = [
                t.strip() for t in args.target_lang.split(",") if t.strip()
            ]
        if not translation.target_languages:
            logger.error("--translate requires --target-lang with at least one language code.")
            sys.exit(1)

    summarize_cfg = SummarizeConfig()
    if args.summarize:
        summarize_cfg.enabled = True
        summarize_cfg.keyframe_interval = args.keyframe_interval

    config = JobConfig(
        input_dir=args.input,
        output_dir=args.output,
        format=args.format,
        engine_name=args.engine,
        model_name=args.model,
        language=language,
        hf_mirror=args.hf_mirror,
        polish=polish,
        translation=translation,
        recursive=not args.no_recursive,
        auto_correct=args.auto_correct,
        summarize=summarize_cfg,
    )

    def cli_progress(current: int, total: int, filename: str) -> None:
        if current < total:
            print(f"[{current + 1}/{total}] Processing: {filename}")
        else:
            print(f"\nAll done. {filename}")

    success, total = process_batch(config, progress_cb=cli_progress)
    if total == 0:
        sys.exit(1)
    if success < total:
        logger.warning("%d file(s) failed.", total - success)


# ═══════════════════════════════════════════════════════════════════════════════
# §12  GUI Interface (tkinter)
# ═══════════════════════════════════════════════════════════════════════════════


def launch_gui() -> None:
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, scrolledtext, messagebox
    except ImportError:
        logger.error(
            "tkinter is not available. Install it or use CLI mode.\n"
            "On Debian/Ubuntu: sudo apt install python3-tk"
        )
        sys.exit(1)

    # Load saved settings and apply UI language
    _saved = load_settings()
    set_ui_language(_saved.get("ui_language", "en"))

    class QueueLogHandler(logging.Handler):
        def __init__(self, log_queue: Queue):
            super().__init__()
            self.log_queue = log_queue

        def emit(self, record: logging.LogRecord):
            self.log_queue.put(("log", self.format(record)))

    class TingShuoGUI:
        def __init__(self, root: tk.Tk):
            self.root = root
            self.root.title("TingShuo 听说")
            self.root.geometry("720x920")
            self.root.minsize(680, 800)

            self.stop_event = threading.Event()
            self.msg_queue: Queue = Queue()
            self.worker_thread: Optional[threading.Thread] = None
            self._settings = dict(_saved)

            self._build_menu()
            self._build_ui()
            self._setup_log_handler()
            self._poll_queue()

        # ── Menu Bar ──

        def _build_menu(self) -> None:
            menubar = tk.Menu(self.root)
            self.root.config(menu=menubar)

            help_menu = tk.Menu(menubar, tearoff=0)
            help_menu.add_command(label=tr("settings"), command=self._show_settings)
            help_menu.add_separator()
            help_menu.add_command(label=tr("about"), command=self._show_about)
            menubar.add_cascade(label=tr("help_menu"), menu=help_menu)

        def _show_about(self) -> None:
            dialog = tk.Toplevel(self.root)
            dialog.title(tr("about"))
            dialog.geometry("420x280")
            dialog.resizable(False, False)
            dialog.transient(self.root)
            dialog.grab_set()

            frame = ttk.Frame(dialog, padding=20)
            frame.pack(fill=tk.BOTH, expand=True)

            ttk.Label(frame, text="TingShuo 听说", font=("", 16, "bold")).pack(pady=(0, 8))
            ttk.Label(frame, text=tr("about_text")).pack()
            ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
            ttk.Label(frame, text=f"{tr('version')}: {__version__}").pack()
            ttk.Label(frame, text=f"{tr('author')}: {__author__}").pack()
            ttk.Label(frame, text=f"{tr('license')}: {__license__}").pack()
            ttk.Label(
                frame, text="https://github.com/cycleuser/TingShuo",
                foreground="blue", cursor="hand2",
            ).pack(pady=(8, 0))
            ttk.Button(frame, text=tr("close"), command=dialog.destroy).pack(pady=(12, 0))

        def _show_settings(self) -> None:
            dialog = tk.Toplevel(self.root)
            dialog.title(tr("settings"))
            dialog.geometry("420x200")
            dialog.resizable(False, False)
            dialog.transient(self.root)
            dialog.grab_set()

            frame = ttk.Frame(dialog, padding=20)
            frame.pack(fill=tk.BOTH, expand=True)

            ttk.Label(frame, text=tr("ui_language")).grid(row=0, column=0, sticky=tk.W, pady=4)
            lang_items = [f"{c} - {n}" for c, n in UI_LANG_NAMES.items()]
            cur_display = f"{_current_ui_lang} - {UI_LANG_NAMES.get(_current_ui_lang, 'English')}"
            lang_var = tk.StringVar(value=cur_display)
            lang_combo = ttk.Combobox(
                frame, textvariable=lang_var, values=lang_items,
                state="readonly", width=25,
            )
            lang_combo.grid(row=0, column=1, sticky=tk.W, padx=8, pady=4)

            ttk.Label(frame, text=tr("restart_note"), foreground="gray").grid(
                row=1, column=0, columnspan=2, sticky=tk.W, pady=8,
            )

            def _save_lang() -> None:
                sel = lang_var.get()
                code = sel.split(" - ")[0] if " - " in sel else "en"
                self._settings["ui_language"] = code
                save_settings(self._settings)
                set_ui_language(code)
                messagebox.showinfo(tr("settings"), tr("restart_note"))
                dialog.destroy()

            btn_frame = ttk.Frame(frame)
            btn_frame.grid(row=2, column=0, columnspan=2, pady=(8, 0))
            ttk.Button(btn_frame, text=tr("save"), command=_save_lang).pack(side=tk.LEFT, padx=4)
            ttk.Button(btn_frame, text=tr("close"), command=dialog.destroy).pack(side=tk.LEFT, padx=4)

        # ── Main UI ──

        def _build_ui(self) -> None:
            pad = {"padx": 8, "pady": 4}
            main = ttk.Frame(self.root, padding=8)
            main.pack(fill=tk.BOTH, expand=True)

            # ── Input / Output ──
            io_frame = ttk.LabelFrame(main, text=tr("input_output"), padding=6)
            io_frame.pack(fill=tk.X, **pad)

            ttk.Label(io_frame, text=tr("input_dir")).grid(row=0, column=0, sticky=tk.W)
            self.input_var = tk.StringVar()
            ttk.Entry(io_frame, textvariable=self.input_var, width=50).grid(
                row=0, column=1, sticky=tk.EW, padx=4,
            )
            ttk.Button(io_frame, text=tr("browse"), command=self._browse_input).grid(row=0, column=2)

            ttk.Label(io_frame, text=tr("output_dir")).grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
            self.output_var = tk.StringVar()
            self.output_entry = ttk.Entry(io_frame, textvariable=self.output_var, width=50)
            self.output_entry.grid(row=1, column=1, sticky=tk.EW, padx=4, pady=(4, 0))
            ttk.Button(io_frame, text=tr("browse"), command=self._browse_output).grid(
                row=1, column=2, pady=(4, 0),
            )

            self.same_dir_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                io_frame, text=tr("save_same_dir"),
                variable=self.same_dir_var, command=self._toggle_output_dir,
            ).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(4, 0))
            self._toggle_output_dir()
            io_frame.columnconfigure(1, weight=1)

            # ── Engine Settings ──
            eng_frame = ttk.LabelFrame(main, text=tr("engine_settings"), padding=6)
            eng_frame.pack(fill=tk.X, **pad)

            ttk.Label(eng_frame, text=tr("engine")).grid(row=0, column=0, sticky=tk.W)
            self.engine_var = tk.StringVar(value=DEFAULT_ENGINE)
            eng_combo = ttk.Combobox(
                eng_frame, textvariable=self.engine_var,
                values=list(SUPPORTED_ENGINES), state="readonly", width=18,
            )
            eng_combo.grid(row=0, column=1, sticky=tk.W, padx=4)
            eng_combo.bind("<<ComboboxSelected>>", self._on_engine_change)

            ttk.Label(eng_frame, text=tr("model")).grid(row=0, column=2, sticky=tk.W, padx=(12, 0))
            self.model_var = tk.StringVar(value="base")
            self.model_combo = ttk.Combobox(
                eng_frame, textvariable=self.model_var,
                values=ENGINE_MODELS.get(DEFAULT_ENGINE, []), width=14,
            )
            self.model_combo.grid(row=0, column=3, sticky=tk.W, padx=4)

            self.dl_btn = ttk.Button(eng_frame, text=tr("download"), command=self._download_model)
            self.dl_btn.grid(row=0, column=4, sticky=tk.W, padx=2)
            self.dl_all_btn = ttk.Button(eng_frame, text=tr("download_all"), command=self._download_all_models)
            self.dl_all_btn.grid(row=0, column=5, sticky=tk.W, padx=2)
            self.dl_everything_btn = ttk.Button(eng_frame, text=tr("download_everything"), command=self._download_everything)
            self.dl_everything_btn.grid(row=0, column=6, sticky=tk.W, padx=2)

            ttk.Label(eng_frame, text=tr("language")).grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
            self.lang_var = tk.StringVar(value="auto")
            ttk.Combobox(
                eng_frame, textvariable=self.lang_var,
                values=LANGUAGE_CODES, width=14,
            ).grid(row=1, column=1, sticky=tk.W, padx=4, pady=(4, 0))
            ttk.Label(eng_frame, text=tr("lang_hint")).grid(
                row=1, column=2, columnspan=4, sticky=tk.W, padx=4, pady=(4, 0),
            )

            self.hf_mirror_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                eng_frame, text=tr("use_hf_mirror"),
                variable=self.hf_mirror_var, command=self._toggle_hf_mirror,
            ).grid(row=2, column=0, sticky=tk.W, pady=(4, 0))
            self.hf_url_var = tk.StringVar(value="https://hf-mirror.com")
            self.hf_entry = ttk.Entry(eng_frame, textvariable=self.hf_url_var, width=30)
            self.hf_entry.grid(row=2, column=1, columnspan=4, sticky=tk.W, padx=4, pady=(4, 0))
            self.hf_entry.config(state="disabled")

            # ── Output Format ──
            fmt_frame = ttk.LabelFrame(main, text=tr("output_format"), padding=6)
            fmt_frame.pack(fill=tk.X, **pad)

            self.format_var = tk.StringVar(value="srt")
            ttk.Radiobutton(fmt_frame, text="SRT", variable=self.format_var, value="srt").pack(side=tk.LEFT, padx=8)
            ttk.Radiobutton(fmt_frame, text="LRC", variable=self.format_var, value="lrc").pack(side=tk.LEFT, padx=8)
            ttk.Radiobutton(fmt_frame, text=tr("transcript_opt"), variable=self.format_var, value="md").pack(side=tk.LEFT, padx=8)

            # ── Auto-Correction & Summarization ──
            extra_frame = ttk.LabelFrame(main, text=tr("auto_correct") + " / " + tr("summarize"), padding=6)
            extra_frame.pack(fill=tk.X, **pad)

            self.autocorrect_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                extra_frame, text=tr("auto_correct"),
                variable=self.autocorrect_var,
            ).grid(row=0, column=0, sticky=tk.W)

            self.summarize_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                extra_frame, text=tr("summarize"),
                variable=self.summarize_var,
                command=self._toggle_summarize,
            ).grid(row=0, column=1, sticky=tk.W, padx=(16, 0))

            ttk.Label(extra_frame, text=tr("keyframe_interval")).grid(
                row=0, column=2, sticky=tk.W, padx=(16, 0),
            )
            self.keyframe_interval_var = tk.StringVar(value="60")
            self.keyframe_interval_entry = ttk.Entry(
                extra_frame, textvariable=self.keyframe_interval_var, width=6,
            )
            self.keyframe_interval_entry.grid(row=0, column=3, sticky=tk.W, padx=4)
            self.keyframe_interval_entry.config(state="disabled")

            ttk.Label(
                extra_frame, text=tr("multimodal_note"), foreground="gray",
            ).grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(4, 0))

            # ── Polishing ──
            pol_frame = ttk.LabelFrame(main, text=tr("subtitle_polishing"), padding=6)
            pol_frame.pack(fill=tk.X, **pad)

            self.polish_var = tk.StringVar(value="none")
            ttk.Radiobutton(
                pol_frame, text=tr("none_opt"), variable=self.polish_var,
                value="none", command=self._toggle_polish,
            ).grid(row=0, column=0, sticky=tk.W)
            ttk.Radiobutton(
                pol_frame, text=tr("llm_opt"), variable=self.polish_var,
                value="llm", command=self._toggle_polish,
            ).grid(row=0, column=1, sticky=tk.W, padx=8)
            ttk.Radiobutton(
                pol_frame, text=tr("nlp_opt"), variable=self.polish_var,
                value="nlp", command=self._toggle_polish,
            ).grid(row=0, column=2, sticky=tk.W, padx=8)

            # LLM sub-panel
            self.llm_frame = ttk.LabelFrame(pol_frame, text=tr("llm_settings"), padding=6)
            self.llm_frame.grid(row=1, column=0, columnspan=4, sticky=tk.EW, pady=(6, 0))

            self.llm_backend_var = tk.StringVar(value="ollama")
            ttk.Radiobutton(
                self.llm_frame, text=tr("ollama"),
                variable=self.llm_backend_var, value="ollama",
                command=self._toggle_llm_backend,
            ).grid(row=0, column=0, sticky=tk.W)
            ttk.Radiobutton(
                self.llm_frame, text=tr("openai_api"),
                variable=self.llm_backend_var, value="api",
                command=self._toggle_llm_backend,
            ).grid(row=0, column=1, columnspan=2, sticky=tk.W, padx=8)

            ttk.Label(self.llm_frame, text=tr("ollama_url")).grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
            self.ollama_url_var = tk.StringVar(value=DEFAULT_OLLAMA_URL)
            self.ollama_url_entry = ttk.Entry(self.llm_frame, textvariable=self.ollama_url_var, width=35)
            self.ollama_url_entry.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=4, pady=(4, 0))

            ttk.Label(self.llm_frame, text=tr("ollama_model")).grid(row=2, column=0, sticky=tk.W, pady=(4, 0))
            self.ollama_model_var = tk.StringVar(value=DEFAULT_OLLAMA_MODEL)
            self.ollama_model_combo = ttk.Combobox(
                self.llm_frame, textvariable=self.ollama_model_var, width=20,
            )
            self.ollama_model_combo.grid(row=2, column=1, sticky=tk.W, padx=4, pady=(4, 0))
            self.ollama_refresh_btn = ttk.Button(
                self.llm_frame, text=tr("refresh"), command=self._refresh_ollama_models,
            )
            self.ollama_refresh_btn.grid(row=2, column=2, sticky=tk.W, padx=2, pady=(4, 0))

            ttk.Separator(self.llm_frame, orient=tk.HORIZONTAL).grid(
                row=3, column=0, columnspan=3, sticky=tk.EW, pady=6,
            )

            ttk.Label(self.llm_frame, text=tr("api_url")).grid(row=4, column=0, sticky=tk.W)
            self.api_url_var = tk.StringVar()
            self.api_url_entry = ttk.Entry(self.llm_frame, textvariable=self.api_url_var, width=35)
            self.api_url_entry.grid(row=4, column=1, columnspan=2, sticky=tk.EW, padx=4)

            ttk.Label(self.llm_frame, text=tr("api_key")).grid(row=5, column=0, sticky=tk.W, pady=(4, 0))
            self.api_key_var = tk.StringVar()
            self.api_key_entry = ttk.Entry(self.llm_frame, textvariable=self.api_key_var, width=35, show="*")
            self.api_key_entry.grid(row=5, column=1, columnspan=2, sticky=tk.EW, padx=4, pady=(4, 0))

            ttk.Label(self.llm_frame, text=tr("api_model")).grid(row=6, column=0, sticky=tk.W, pady=(4, 0))
            self.api_model_var = tk.StringVar()
            self.api_model_entry = ttk.Entry(self.llm_frame, textvariable=self.api_model_var, width=20)
            self.api_model_entry.grid(row=6, column=1, columnspan=2, sticky=tk.W, padx=4, pady=(4, 0))

            self.llm_frame.columnconfigure(2, weight=1)
            pol_frame.columnconfigure(3, weight=1)
            self._toggle_polish()

            # ── Translation ──
            trans_frame = ttk.LabelFrame(main, text=tr("translation"), padding=6)
            trans_frame.pack(fill=tk.X, **pad)

            self.trans_enabled_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                trans_frame, text=tr("enable_trans"),
                variable=self.trans_enabled_var, command=self._toggle_translation,
            ).grid(row=0, column=0, sticky=tk.W)

            # Translation settings sub-frame
            self.trans_settings_frame = ttk.Frame(trans_frame)
            self.trans_settings_frame.grid(row=1, column=0, columnspan=4, sticky=tk.EW, pady=(4, 0))

            ttk.Label(self.trans_settings_frame, text=tr("target_langs")).grid(
                row=0, column=0, sticky=tk.W,
            )
            # Target language checkbuttons
            self.target_lang_vars: Dict[str, tk.BooleanVar] = {}
            tgt_inner = ttk.Frame(self.trans_settings_frame)
            tgt_inner.grid(row=0, column=1, columnspan=3, sticky=tk.W, padx=4)
            tgt_codes = ["zh", "en", "ja", "ko", "fr", "de", "es", "it", "pt", "ru"]
            for ci, code in enumerate(tgt_codes):
                var = tk.BooleanVar(value=False)
                self.target_lang_vars[code] = var
                ttk.Checkbutton(tgt_inner, text=code, variable=var).grid(
                    row=ci // 5, column=ci % 5, sticky=tk.W, padx=2,
                )

            ttk.Label(self.trans_settings_frame, text=tr("trans_backend")).grid(
                row=1, column=0, sticky=tk.W, pady=(4, 0),
            )
            self.trans_backend_var = tk.StringVar(value="nllb")
            ttk.Radiobutton(
                self.trans_settings_frame, text=tr("nllb"),
                variable=self.trans_backend_var, value="nllb",
                command=self._toggle_trans_backend,
            ).grid(row=1, column=1, sticky=tk.W, padx=4, pady=(4, 0))
            ttk.Radiobutton(
                self.trans_settings_frame, text=tr("llm_opt"),
                variable=self.trans_backend_var, value="llm",
                command=self._toggle_trans_backend,
            ).grid(row=1, column=2, sticky=tk.W, padx=4, pady=(4, 0))

            ttk.Label(self.trans_settings_frame, text=tr("nllb_model")).grid(
                row=2, column=0, sticky=tk.W, pady=(4, 0),
            )
            self.nllb_model_var = tk.StringVar(value=DEFAULT_NLLB_MODEL)
            self.nllb_model_combo = ttk.Combobox(
                self.trans_settings_frame, textvariable=self.nllb_model_var,
                values=NLLB_MODELS, width=35,
            )
            self.nllb_model_combo.grid(row=2, column=1, columnspan=3, sticky=tk.W, padx=4, pady=(4, 0))

            trans_frame.columnconfigure(3, weight=1)
            self._toggle_translation()

            # ── Controls ──
            ctrl_frame = ttk.Frame(main)
            ctrl_frame.pack(fill=tk.X, **pad)

            self.start_btn = ttk.Button(ctrl_frame, text=tr("start"), command=self._start)
            self.start_btn.pack(side=tk.LEFT, padx=4)
            self.stop_btn = ttk.Button(ctrl_frame, text=tr("stop"), command=self._stop, state="disabled")
            self.stop_btn.pack(side=tk.LEFT, padx=4)

            # ── Progress ──
            prog_frame = ttk.LabelFrame(main, text=tr("progress"), padding=6)
            prog_frame.pack(fill=tk.X, **pad)

            self.progress_var = tk.DoubleVar(value=0)
            self.progress_bar = ttk.Progressbar(prog_frame, variable=self.progress_var, maximum=100)
            self.progress_bar.pack(fill=tk.X)
            self.progress_label = ttk.Label(prog_frame, text=tr("ready"))
            self.progress_label.pack(anchor=tk.W, pady=(4, 0))

            # ── Log ──
            log_frame = ttk.LabelFrame(main, text=tr("log"), padding=6)
            log_frame.pack(fill=tk.BOTH, expand=True, **pad)

            self.log_text = scrolledtext.ScrolledText(log_frame, height=10, state="disabled", wrap=tk.WORD)
            self.log_text.pack(fill=tk.BOTH, expand=True)

        # ── Callbacks ──

        def _setup_log_handler(self) -> None:
            handler = QueueLogHandler(self.msg_queue)
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        def _poll_queue(self) -> None:
            try:
                while True:
                    msg_type, data = self.msg_queue.get_nowait()
                    if msg_type == "log":
                        self._append_log(data)
                    elif msg_type == "progress":
                        current, total, filename = data
                        if total > 0:
                            pct = (current / total) * 100
                            self.progress_var.set(pct)
                            if current < total:
                                self.progress_label.config(
                                    text=f"[{current + 1}/{total}] {filename}"
                                )
                            else:
                                self.progress_label.config(text=tr("done"))
                    elif msg_type == "done":
                        success, total = data
                        self._append_log(f"[INFO] {tr('done')}: {success}/{total}")
                        self.start_btn.config(state="normal")
                        self.stop_btn.config(state="disabled")
                        self.progress_label.config(text=f"{tr('done')}: {success}/{total}")
                    elif msg_type == "download_done":
                        ok, name = data
                        self.progress_bar.stop()
                        self.progress_bar.config(mode="determinate")
                        self.progress_var.set(100 if ok else 0)
                        self.dl_btn.config(state="normal")
                        self.dl_all_btn.config(state="normal")
                        self.dl_everything_btn.config(state="normal")
                        self.start_btn.config(state="normal")
                        label = tr("download_complete") if ok else tr("download_failed")
                        self.progress_label.config(text=f"{label} {name}")
                    elif msg_type == "download_all_done":
                        success, total = data
                        self.progress_bar.stop()
                        self.progress_bar.config(mode="determinate")
                        self.progress_var.set(100 if success == total else 50)
                        self.dl_btn.config(state="normal")
                        self.dl_all_btn.config(state="normal")
                        self.dl_everything_btn.config(state="normal")
                        self.start_btn.config(state="normal")
                        self.progress_label.config(
                            text=f"{tr('download_complete')} {success}/{total}"
                        )
                    elif msg_type == "download_everything_done":
                        success, total = data
                        self.progress_bar.stop()
                        self.progress_bar.config(mode="determinate")
                        self.progress_var.set(100 if success == total else 50)
                        self.dl_btn.config(state="normal")
                        self.dl_all_btn.config(state="normal")
                        self.dl_everything_btn.config(state="normal")
                        self.start_btn.config(state="normal")
                        self.progress_label.config(
                            text=f"{tr('download_complete')} {success}/{total}"
                        )
                    elif msg_type == "ollama_models":
                        model_list = data
                        self.ollama_refresh_btn.config(state="normal")
                        if model_list:
                            self.ollama_model_combo["values"] = model_list
                            if self.ollama_model_var.get() not in model_list:
                                self.ollama_model_var.set(model_list[0])
                            self._append_log(
                                f"[INFO] Found {len(model_list)} Ollama model(s)."
                            )
                        else:
                            self._append_log(
                                "[WARNING] Could not fetch Ollama models. "
                                "Check the URL and ensure Ollama is running."
                            )
            except Empty:
                pass
            self.root.after(100, self._poll_queue)

        def _append_log(self, text: str) -> None:
            self.log_text.config(state="normal")
            self.log_text.insert(tk.END, text + "\n")
            self.log_text.see(tk.END)
            self.log_text.config(state="disabled")

        def _browse_input(self) -> None:
            d = filedialog.askdirectory(title=tr("input_dir"))
            if d:
                self.input_var.set(d)

        def _browse_output(self) -> None:
            d = filedialog.askdirectory(title=tr("output_dir"))
            if d:
                self.output_var.set(d)

        def _toggle_output_dir(self) -> None:
            state = "disabled" if self.same_dir_var.get() else "normal"
            self.output_entry.config(state=state)

        def _toggle_hf_mirror(self) -> None:
            state = "normal" if self.hf_mirror_var.get() else "disabled"
            self.hf_entry.config(state=state)

        def _toggle_summarize(self) -> None:
            state = "normal" if self.summarize_var.get() else "disabled"
            self.keyframe_interval_entry.config(state=state)

        def _toggle_polish(self) -> None:
            method = self.polish_var.get()
            if method == "llm":
                for child in self.llm_frame.winfo_children():
                    try:
                        child.config(state="normal")
                    except tk.TclError:
                        pass
                self._toggle_llm_backend()
            else:
                for child in self.llm_frame.winfo_children():
                    try:
                        child.config(state="disabled")
                    except tk.TclError:
                        pass

        def _toggle_llm_backend(self) -> None:
            backend = self.llm_backend_var.get()
            if backend == "ollama":
                self.ollama_url_entry.config(state="normal")
                self.ollama_model_combo.config(state="normal")
                self.ollama_refresh_btn.config(state="normal")
                self.api_url_entry.config(state="disabled")
                self.api_key_entry.config(state="disabled")
                self.api_model_entry.config(state="disabled")
            else:
                self.ollama_url_entry.config(state="disabled")
                self.ollama_model_combo.config(state="disabled")
                self.ollama_refresh_btn.config(state="disabled")
                self.api_url_entry.config(state="normal")
                self.api_key_entry.config(state="normal")
                self.api_model_entry.config(state="normal")

        def _toggle_translation(self) -> None:
            enabled = self.trans_enabled_var.get()
            state = "normal" if enabled else "disabled"
            for child in self.trans_settings_frame.winfo_children():
                try:
                    child.config(state=state)
                except tk.TclError:
                    pass
                # Also handle nested frames (target language checkbuttons)
                if hasattr(child, 'winfo_children'):
                    for sub in child.winfo_children():
                        try:
                            sub.config(state=state)
                        except tk.TclError:
                            pass
            if enabled:
                self._toggle_trans_backend()

        def _toggle_trans_backend(self) -> None:
            backend = self.trans_backend_var.get()
            if backend == "nllb":
                self.nllb_model_combo.config(state="normal")
            else:
                self.nllb_model_combo.config(state="disabled")

        def _on_engine_change(self, event=None) -> None:
            engine = self.engine_var.get()
            models = ENGINE_MODELS.get(engine, [])
            self.model_combo["values"] = models
            default = ENGINE_DEFAULT_MODEL.get(engine, "")
            self.model_var.set(default)

        def _refresh_ollama_models(self) -> None:
            url = self.ollama_url_var.get().strip() or DEFAULT_OLLAMA_URL
            self.ollama_refresh_btn.config(state="disabled")
            self._append_log(f"[INFO] Querying Ollama models at {url} ...")

            def worker() -> None:
                models = query_ollama_models(url)
                self.msg_queue.put(("ollama_models", models))

            threading.Thread(target=worker, daemon=True).start()

        def _download_model(self) -> None:
            engine_name = self.engine_var.get()
            model_name = self.model_var.get().strip()
            if not model_name:
                messagebox.showwarning(tr("warning"), tr("select_model_first"))
                return

            hf_mirror = self.hf_url_var.get().strip() if self.hf_mirror_var.get() else ""
            self.dl_btn.config(state="disabled")
            self.dl_all_btn.config(state="disabled")
            self.dl_everything_btn.config(state="disabled")
            self.start_btn.config(state="disabled")
            self.progress_bar.config(mode="indeterminate")
            self.progress_bar.start()
            self.progress_label.config(text=f"{tr('downloading')} {model_name}")

            def worker() -> None:
                def progress_cb(msg: str) -> None:
                    self.msg_queue.put(("log", f"[INFO] {msg}"))

                try:
                    download_model(engine_name, model_name, hf_mirror=hf_mirror, progress_cb=progress_cb)
                    self.msg_queue.put(("download_done", (True, model_name)))
                except Exception as e:
                    logger.error("Download failed: %s", e)
                    self.msg_queue.put(("download_done", (False, model_name)))

            threading.Thread(target=worker, daemon=True).start()

        def _download_all_models(self) -> None:
            engine_name = self.engine_var.get()
            models = ENGINE_MODELS.get(engine_name, [])
            if not models:
                messagebox.showwarning(tr("warning"), tr("no_models"))
                return

            hf_mirror = self.hf_url_var.get().strip() if self.hf_mirror_var.get() else ""
            self.dl_btn.config(state="disabled")
            self.dl_all_btn.config(state="disabled")
            self.dl_everything_btn.config(state="disabled")
            self.start_btn.config(state="disabled")
            self.progress_bar.config(mode="indeterminate")
            self.progress_bar.start()
            self.progress_label.config(text=f"{tr('downloading')} {engine_name} ...")

            def worker() -> None:
                def progress_cb(msg: str) -> None:
                    self.msg_queue.put(("log", f"[INFO] {msg}"))

                try:
                    success, total = download_all_models(
                        engine_name, hf_mirror=hf_mirror, progress_cb=progress_cb,
                    )
                    self.msg_queue.put(("download_all_done", (success, total)))
                except Exception as e:
                    logger.error("Download all failed: %s", e)
                    self.msg_queue.put(("download_all_done", (0, 0)))

            threading.Thread(target=worker, daemon=True).start()

        def _download_everything(self) -> None:
            hf_mirror = self.hf_url_var.get().strip() if self.hf_mirror_var.get() else ""
            self.dl_btn.config(state="disabled")
            self.dl_all_btn.config(state="disabled")
            self.dl_everything_btn.config(state="disabled")
            self.start_btn.config(state="disabled")
            self.progress_bar.config(mode="indeterminate")
            self.progress_bar.start()
            self.progress_label.config(text=f"{tr('downloading')} {tr('download_everything')} ...")

            def worker() -> None:
                def progress_cb(msg: str) -> None:
                    self.msg_queue.put(("log", f"[INFO] {msg}"))

                try:
                    success, total = download_everything(
                        hf_mirror=hf_mirror, progress_cb=progress_cb,
                    )
                    self.msg_queue.put(("download_everything_done", (success, total)))
                except Exception as e:
                    logger.error("Download everything failed: %s", e)
                    self.msg_queue.put(("download_everything_done", (0, 0)))

            threading.Thread(target=worker, daemon=True).start()

        def _build_config(self) -> Optional[JobConfig]:
            input_dir = self.input_var.get().strip()
            if not input_dir:
                messagebox.showerror(tr("error"), tr("select_input"))
                return None
            if not os.path.isdir(input_dir):
                messagebox.showerror(tr("error"), f"{tr('input_not_exist')}\n{input_dir}")
                return None

            output_dir = ""
            if not self.same_dir_var.get():
                output_dir = self.output_var.get().strip()
                if not output_dir:
                    messagebox.showerror(tr("error"), tr("select_output"))
                    return None

            polish = PolishConfig()
            method = self.polish_var.get()
            polish.method = method
            if method == "llm":
                if self.llm_backend_var.get() == "ollama":
                    polish.ollama_url = self.ollama_url_var.get().strip() or DEFAULT_OLLAMA_URL
                    polish.ollama_model = self.ollama_model_var.get().strip() or DEFAULT_OLLAMA_MODEL
                else:
                    polish.api_url = self.api_url_var.get().strip()
                    polish.api_key = self.api_key_var.get().strip()
                    polish.api_model = self.api_model_var.get().strip()
                    if not polish.api_url or not polish.api_key:
                        messagebox.showerror(tr("error"), tr("api_required"))
                        return None

            # Build translation config
            translation = TranslationConfig()
            if self.trans_enabled_var.get():
                translation.enabled = True
                translation.method = self.trans_backend_var.get()
                translation.nllb_model = self.nllb_model_var.get().strip() or DEFAULT_NLLB_MODEL
                translation.ollama_url = self.ollama_url_var.get().strip() or DEFAULT_OLLAMA_URL
                translation.ollama_model = self.ollama_model_var.get().strip() or DEFAULT_OLLAMA_MODEL
                translation.api_url = self.api_url_var.get().strip()
                translation.api_key = self.api_key_var.get().strip()
                translation.api_model = self.api_model_var.get().strip()
                target_langs = [
                    code for code, var in self.target_lang_vars.items()
                    if var.get()
                ]
                if not target_langs:
                    messagebox.showerror(tr("error"), tr("select_target"))
                    return None
                translation.target_languages = target_langs

            lang_raw = self.lang_var.get().strip()
            lang = None if (not lang_raw or lang_raw.lower() == "auto") else lang_raw
            hf_mirror = self.hf_url_var.get().strip() if self.hf_mirror_var.get() else ""

            # Ensure LLM config is populated for auto-correct / summarize / md
            if self.autocorrect_var.get() or self.summarize_var.get() or self.format_var.get() == "md":
                polish.ollama_url = self.ollama_url_var.get().strip() or DEFAULT_OLLAMA_URL
                polish.ollama_model = self.ollama_model_var.get().strip() or DEFAULT_OLLAMA_MODEL
                polish.api_url = polish.api_url or self.api_url_var.get().strip()
                polish.api_key = polish.api_key or self.api_key_var.get().strip()
                polish.api_model = polish.api_model or self.api_model_var.get().strip()

            # Build summarize config
            summarize_cfg = SummarizeConfig()
            if self.summarize_var.get():
                summarize_cfg.enabled = True
                try:
                    summarize_cfg.keyframe_interval = int(self.keyframe_interval_var.get())
                except ValueError:
                    summarize_cfg.keyframe_interval = 60

            return JobConfig(
                input_dir=input_dir,
                output_dir=output_dir,
                format=self.format_var.get(),
                engine_name=self.engine_var.get(),
                model_name=self.model_var.get(),
                language=lang,
                hf_mirror=hf_mirror,
                polish=polish,
                translation=translation,
                recursive=True,
                auto_correct=self.autocorrect_var.get(),
                summarize=summarize_cfg,
            )

        def _start(self) -> None:
            config = self._build_config()
            if config is None:
                return

            self.stop_event.clear()
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.progress_var.set(0)
            self.progress_label.config(text=tr("starting"))

            # Clear log
            self.log_text.config(state="normal")
            self.log_text.delete("1.0", tk.END)
            self.log_text.config(state="disabled")

            def worker() -> None:
                def progress_cb(current: int, total: int, filename: str) -> None:
                    self.msg_queue.put(("progress", (current, total, filename)))

                try:
                    success, total = process_batch(
                        config, progress_cb=progress_cb, stop_event=self.stop_event,
                    )
                    self.msg_queue.put(("done", (success, total)))
                except Exception as e:
                    logger.error("Unexpected error: %s", e)
                    self.msg_queue.put(("done", (0, 0)))

            self.worker_thread = threading.Thread(target=worker, daemon=True)
            self.worker_thread.start()

        def _stop(self) -> None:
            self.stop_event.set()
            self.stop_btn.config(state="disabled")
            self.progress_label.config(text=tr("stopping"))

    root = tk.Tk()
    TingShuoGUI(root)
    root.mainloop()


# ═══════════════════════════════════════════════════════════════════════════════
# §13  Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    # Quick check: if --gui is the only intent, launch GUI directly
    if "--gui" in sys.argv:
        launch_gui()
        return

    parser = build_cli_parser()
    args = parser.parse_args()
    run_cli(args)


if __name__ == "__main__":
    main()
