# -*- coding: utf-8 -*-
#
# TingShuo (听说) - Live Subtitle & Translation Module
#
# Copyright (C) 2024-2025 TingShuo Team <wedonotuse@outlook.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Live system audio capture, real-time transcription, translation,
and floating desktop subtitle overlay for TingShuo (听说).

Provides:
  - AudioCapture: cross-platform system audio loopback
  - VoiceActivityDetector: Silero VAD for speech/non-speech detection
  - LiveTranscriber: streaming STT (faster-whisper + Vosk)
  - LiveTranslator: per-segment real-time translation (NLLB / LLM)
  - SubtitleOverlay: tkinter always-on-top transparent subtitle window
  - SystemTrayController: pystray system tray icon with menu
  - LiveSession: orchestrator wiring capture→transcribe→translate→display
"""

import os
import sys
import json
import time
import wave
import queue
import struct
import logging
import threading
import tempfile
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Callable, Tuple, Generator
from collections import deque

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

logger = logging.getLogger("tingshuo.live")

# ── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000          # 16 kHz mono, same as batch pipeline
BLOCK_DURATION = 0.03        # 30 ms audio blocks for capture
BLOCK_SAMPLES = int(SAMPLE_RATE * BLOCK_DURATION)  # 480 samples
VAD_THRESHOLD = 0.5          # Silero VAD speech probability threshold
SILENCE_TIMEOUT = 1.2        # seconds of silence to end a speech segment
MAX_SEGMENT_DURATION = 30.0  # max seconds of speech before forcing a cut
MIN_SEGMENT_DURATION = 0.3   # ignore very short segments

# Subtitle overlay defaults
OVERLAY_FONT_SIZE = 22
OVERLAY_MAX_LINES = 3
OVERLAY_WIDTH = 800
OVERLAY_HEIGHT = 160
OVERLAY_BG = "#000000"
OVERLAY_FG = "#FFFFFF"
OVERLAY_ALPHA = 0.75          # background opacity

# Language names for auto-detection display
LANG_NAMES: Dict[str, str] = {
    "en": "English", "zh": "中文", "ja": "日本語", "ko": "한국어",
    "fr": "Français", "de": "Deutsch", "es": "Español", "it": "Italiano",
    "pt": "Português", "ru": "Русский", "ar": "العربية", "hi": "हिन्दी",
    "th": "ไทย", "vi": "Tiếng Việt", "tr": "Türkçe", "nl": "Nederlands",
    "pl": "Polski", "uk": "Українська", "sv": "Svenska", "da": "Dansk",
    "fi": "Suomi", "no": "Norsk", "cs": "Čeština", "ro": "Română",
    "el": "Ελληνικά", "hu": "Magyar", "id": "Bahasa Indonesia",
    "ms": "Bahasa Melayu", "tl": "Tagalog", "sw": "Kiswahili",
    "auto": "Auto-detect",
}

# ═══════════════════════════════════════════════════════════════════════════════
# UI Strings for Live Mode (10 languages, same pattern as tingshuo.UI_STRINGS)
# ═══════════════════════════════════════════════════════════════════════════════

LIVE_UI_STRINGS: Dict[str, Dict[str, str]] = {
    "en": {
        "live_mode": "🎤 Live",
        "stop_live": "⏹ Stop Live",
        "live_starting": "Live capture starting... Listening for system audio.",
        "live_listening": "🎤 Live — Listening...",
        "live_stopped": "Live mode stopped.",
        "live_error": "Live mode error",
        "live_import_error": "Live mode requires additional dependencies. Install with: pip install sounddevice numpy",
        "live_no_device": "No audio loopback device found. On Windows, ensure WASAPI is available. On macOS, install BlackHole.",
        "live_select_lang": "Select source language:",
        "live_select_target": "Select target language:",
        "live_show_overlay": "Show Overlay",
        "live_hide_overlay": "Hide Overlay",
    },
    "zh": {
        "live_mode": "🎤 实时",
        "stop_live": "⏹ 停止实时",
        "live_starting": "实时捕获启动中... 正在监听系统音频。",
        "live_listening": "🎤 实时 — 监听中...",
        "live_stopped": "实时模式已停止。",
        "live_error": "实时模式错误",
        "live_import_error": "实时模式需要额外的依赖。安装命令: pip install sounddevice numpy",
        "live_no_device": "未找到音频回采设备。Windows 需确保 WASAPI 可用，macOS 请安装 BlackHole。",
        "live_select_lang": "选择源语言：",
        "live_select_target": "选择目标语言：",
        "live_show_overlay": "显示字幕",
        "live_hide_overlay": "隐藏字幕",
    },
    "ja": {
        "live_mode": "🎤 ライブ",
        "stop_live": "⏹ ライブ停止",
        "live_starting": "ライブキャプチャ開始中... システム音声を監視しています。",
        "live_listening": "🎤 ライブ — 監視中...",
        "live_stopped": "ライブモードを停止しました。",
        "live_error": "ライブモードエラー",
        "live_import_error": "ライブモードには追加の依存関係が必要です。pip install sounddevice numpy",
        "live_no_device": "オーディオループバックデバイスが見つかりません。",
        "live_select_lang": "ソース言語を選択：",
        "live_select_target": "ターゲット言語を選択：",
        "live_show_overlay": "字幕を表示",
        "live_hide_overlay": "字幕を非表示",
    },
    "ko": {
        "live_mode": "🎤 라이브",
        "stop_live": "⏹ 라이브 중지",
        "live_starting": "라이브 캡처 시작 중... 시스템 오디오 청취 중.",
        "live_listening": "🎤 라이브 — 청취 중...",
        "live_stopped": "라이브 모드가 중지되었습니다.",
        "live_error": "라이브 모드 오류",
        "live_import_error": "라이브 모드에 추가 종속성이 필요합니다. pip install sounddevice numpy",
        "live_no_device": "오디오 루프백 장치를 찾을 수 없습니다.",
        "live_select_lang": "소스 언어 선택:",
        "live_select_target": "대상 언어 선택:",
        "live_show_overlay": "자막 표시",
        "live_hide_overlay": "자막 숨기기",
    },
    "fr": {
        "live_mode": "🎤 Direct",
        "stop_live": "⏹ Arrêter le direct",
        "live_starting": "Capture en direct en cours... Écoute de l'audio système.",
        "live_listening": "🎤 Direct — Écoute...",
        "live_stopped": "Mode direct arrêté.",
        "live_error": "Erreur mode direct",
        "live_import_error": "Le mode direct nécessite des dépendances supplémentaires. pip install sounddevice numpy",
        "live_no_device": "Aucun périphérique de bouclage audio trouvé.",
        "live_select_lang": "Choisir la langue source :",
        "live_select_target": "Choisir la langue cible :",
        "live_show_overlay": "Afficher les sous-titres",
        "live_hide_overlay": "Masquer les sous-titres",
    },
    "de": {
        "live_mode": "🎤 Live",
        "stop_live": "⏹ Live stoppen",
        "live_starting": "Live-Aufnahme gestartet... Höre System-Audio.",
        "live_listening": "🎤 Live — Höre zu...",
        "live_stopped": "Live-Modus gestoppt.",
        "live_error": "Live-Modus Fehler",
        "live_import_error": "Live-Modus benötigt zusätzliche Abhängigkeiten. pip install sounddevice numpy",
        "live_no_device": "Kein Audio-Loopback-Gerät gefunden.",
        "live_select_lang": "Quellsprache wählen:",
        "live_select_target": "Zielsprache wählen:",
        "live_show_overlay": "Untertitel anzeigen",
        "live_hide_overlay": "Untertitel ausblenden",
    },
    "es": {
        "live_mode": "🎤 En vivo",
        "stop_live": "⏹ Detener",
        "live_starting": "Captura en vivo iniciada... Escuchando audio del sistema.",
        "live_listening": "🎤 En vivo — Escuchando...",
        "live_stopped": "Modo en vivo detenido.",
        "live_error": "Error en modo en vivo",
        "live_import_error": "El modo en vivo requiere dependencias adicionales. pip install sounddevice numpy",
        "live_no_device": "No se encontró dispositivo de loopback de audio.",
        "live_select_lang": "Seleccionar idioma fuente:",
        "live_select_target": "Seleccionar idioma destino:",
        "live_show_overlay": "Mostrar subtítulos",
        "live_hide_overlay": "Ocultar subtítulos",
    },
    "it": {
        "live_mode": "🎤 Live",
        "stop_live": "⏹ Ferma Live",
        "live_starting": "Acquisizione live avviata... Ascolto audio di sistema.",
        "live_listening": "🎤 Live — In ascolto...",
        "live_stopped": "Modalità live fermata.",
        "live_error": "Errore modalità live",
        "live_import_error": "La modalità live richiede dipendenze aggiuntive. pip install sounddevice numpy",
        "live_no_device": "Nessun dispositivo di loopback audio trovato.",
        "live_select_lang": "Seleziona lingua sorgente:",
        "live_select_target": "Seleziona lingua destinazione:",
        "live_show_overlay": "Mostra sottotitoli",
        "live_hide_overlay": "Nascondi sottotitoli",
    },
    "pt": {
        "live_mode": "🎤 Ao vivo",
        "stop_live": "⏹ Parar",
        "live_starting": "Captura ao vivo iniciada... Ouvindo áudio do sistema.",
        "live_listening": "🎤 Ao vivo — Ouvindo...",
        "live_stopped": "Modo ao vivo parado.",
        "live_error": "Erro no modo ao vivo",
        "live_import_error": "Modo ao vivo requer dependências adicionais. pip install sounddevice numpy",
        "live_no_device": "Nenhum dispositivo de loopback de áudio encontrado.",
        "live_select_lang": "Selecionar idioma fonte:",
        "live_select_target": "Selecionar idioma destino:",
        "live_show_overlay": "Mostrar legendas",
        "live_hide_overlay": "Ocultar legendas",
    },
    "ru": {
        "live_mode": "🎤 Лайв",
        "stop_live": "⏹ Остановить",
        "live_starting": "Запись в реальном времени... Прослушивание системного аудио.",
        "live_listening": "🎤 Лайв — Прослушивание...",
        "live_stopped": "Режим реального времени остановлен.",
        "live_error": "Ошибка режима реального времени",
        "live_import_error": "Для живого режима нужны дополнительные зависимости. pip install sounddevice numpy",
        "live_no_device": "Устройство аудио loopback не найдено.",
        "live_select_lang": "Выберите исходный язык:",
        "live_select_target": "Выберите целевой язык:",
        "live_show_overlay": "Показать субтитры",
        "live_hide_overlay": "Скрыть субтитры",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LiveSegment:
    """A transcribed segment with optional translation."""
    text: str
    start: float       # seconds since session start
    end: float
    language: str = ""
    translation: str = ""
    is_partial: bool = False  # True for in-progress speech


@dataclass
class LiveConfig:
    """Configuration for a live session."""
    # Audio capture
    capture_device: Optional[int] = None   # sounddevice device index, None=default loopback
    sample_rate: int = SAMPLE_RATE

    # STT
    engine_name: str = "faster-whisper"    # "faster-whisper" or "vosk"
    model_name: str = "base"
    language: Optional[str] = None         # None = auto-detect

    # Translation
    translate_enabled: bool = False
    translate_method: str = "nllb"         # "nllb" or "llm"
    target_languages: List[str] = field(default_factory=list)
    nllb_model: str = "facebook/nllb-200-distilled-600M"
    # LLM translation settings
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5"
    api_url: str = ""
    api_key: str = ""
    api_model: str = ""

    # Overlay
    overlay_enabled: bool = True
    overlay_font_size: int = OVERLAY_FONT_SIZE
    overlay_max_lines: int = OVERLAY_MAX_LINES
    overlay_bg: str = OVERLAY_BG
    overlay_fg: str = OVERLAY_FG
    overlay_alpha: float = OVERLAY_ALPHA
    overlay_pos_x: Optional[int] = None    # None = center-bottom
    overlay_pos_y: Optional[int] = None
    show_original: bool = True             # show original text in overlay
    show_translation: bool = True          # show translation in overlay

    # System tray
    tray_enabled: bool = True

    # Advanced
    vad_threshold: float = VAD_THRESHOLD
    silence_timeout: float = SILENCE_TIMEOUT
    max_segment_duration: float = MAX_SEGMENT_DURATION
    min_segment_duration: float = MIN_SEGMENT_DURATION


# ═══════════════════════════════════════════════════════════════════════════════
# Audio Capture — cross-platform system loopback
# ═══════════════════════════════════════════════════════════════════════════════


class AudioCapture:
    """Capture system audio output (loopback) using sounddevice.

    Supports:
      - Windows: WASAPI loopback
      - macOS:    BlackHole / Soundflower virtual device
      - Linux:    PulseAudio monitor or ALSA loopback
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE,
                 device: Optional[int] = None,
                 block_duration: float = BLOCK_DURATION):
        self._sample_rate = sample_rate
        self._device = device
        self._block_samples = int(sample_rate * block_duration)
        self._stream = None
        self._sd = None
        self._running = False
        self._lock = threading.Lock()

    def _ensure_sounddevice(self):
        if self._sd is not None:
            return
        if np is None:
            raise ImportError(
                "numpy is required for live capture. "
                "Install with: pip install numpy"
            )
        try:
            import sounddevice as sd
            self._sd = sd
        except ImportError:
            raise ImportError(
                "sounddevice is required for live capture. "
                "Install with: pip install sounddevice"
            )

    @staticmethod
    def list_devices() -> List[Dict]:
        """List available audio devices with loopback capability."""
        try:
            import sounddevice as sd
        except ImportError:
            return []
        devices = []
        for idx, dev in enumerate(sd.query_devices()):
            hostapi = sd.query_hostapis(dev['hostapi'])
            info = {
                "index": idx,
                "name": dev['name'],
                "channels": dev['max_input_channels'],
                "sample_rate": int(dev['default_samplerate']),
                "hostapi": hostapi['name'],
                "is_loopback": "loopback" in dev['name'].lower()
                               or "wasapi" in hostapi['name'].lower(),
            }
            devices.append(info)
        return devices

    @staticmethod
    def find_loopback_device() -> Optional[int]:
        """Auto-detect the best loopback device. Returns device index or None."""
        devices = AudioCapture.list_devices()
        # Preference: explicit loopback > BlackHole > WASAPI input > PulseAudio monitor > first input
        for dev in devices:
            if "loopback" in dev['name'].lower():
                return dev['index']
        for dev in devices:
            if "blackhole" in dev['name'].lower():
                return dev['index']
        for dev in devices:
            if dev['hostapi'] and "wasapi" in dev['hostapi'].lower() \
               and dev['channels'] > 0:
                return dev['index']
        for dev in devices:
            if "monitor" in dev['name'].lower():
                return dev['index']
        # Fall back to default input
        try:
            import sounddevice as sd
            default = sd.default.device[0]  # input device
            if default is not None:
                return default
        except Exception:
            pass
        return None

    def start(self, callback: Callable[["np.ndarray"], None]) -> None:
        """Start capturing. `callback(audio_chunk: np.ndarray)` is called
        with float32 audio blocks (shape: [samples, 1])."""
        self._ensure_sounddevice()

        # Try WASAPI loopback on Windows
        device = self._device
        if device is None:
            device = self.find_loopback_device()
            if device is None:
                raise RuntimeError(
                    "No suitable loopback device found. "
                    "On Windows, make sure WASAPI is available. "
                    "On macOS, install BlackHole. "
                    "On Linux, use PulseAudio monitor."
                )

        # On Windows WASAPI, we need to set extra params for loopback
        extra_settings = None
        try:
            dev_info = self._sd.query_devices(device)
            hostapi = self._sd.query_hostapis(dev_info['hostapi'])
            if 'WASAPI' in hostapi['name'].upper():
                extra_settings = self._sd.WasapiSettings(loopback=True)
        except Exception:
            pass

        logger.info("Starting audio capture: device=%s, sr=%d, block=%d samples",
                    device, self._sample_rate, self._block_samples)

        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.debug("Audio status: %s", status)
            with self._lock:
                if self._running:
                    # Convert to mono if needed, float32
                    if indata.ndim > 1 and indata.shape[1] > 1:
                        chunk = indata.mean(axis=1).astype(np.float32)
                    else:
                        chunk = indata.ravel().astype(np.float32)
                    try:
                        callback(chunk)
                    except Exception as e:
                        logger.error("Audio callback error: %s", e)

        try:
            self._stream = self._sd.InputStream(
                device=device,
                channels=1,  # capture as mono
                samplerate=self._sample_rate,
                blocksize=self._block_samples,
                callback=audio_callback,
                dtype=np.float32,
                extra_settings=extra_settings,
            )
        except Exception as e:
            # Fallback: try stereo then mix down
            logger.warning("Mono capture failed, trying stereo: %s", e)
            self._stream = self._sd.InputStream(
                device=device,
                channels=2,
                samplerate=self._sample_rate,
                blocksize=self._block_samples,
                callback=audio_callback,
                dtype=np.float32,
                extra_settings=extra_settings,
            )

        self._running = True
        self._stream.start()

    def stop(self) -> None:
        """Stop capturing."""
        with self._lock:
            self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        logger.info("Audio capture stopped.")


# ═══════════════════════════════════════════════════════════════════════════════
# Voice Activity Detection (VAD)
# ═══════════════════════════════════════════════════════════════════════════════


class VoiceActivityDetector:
    """Silero VAD-based voice activity detection.

    Uses the lightweight Silero VAD model (onnx) to classify each audio
    frame as speech or non-speech. Accumulates speech frames into segments.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE,
                 threshold: float = VAD_THRESHOLD,
                 silence_timeout: float = SILENCE_TIMEOUT,
                 max_duration: float = MAX_SEGMENT_DURATION,
                 min_duration: float = MIN_SEGMENT_DURATION):
        self._sample_rate = sample_rate
        self._threshold = threshold
        self._silence_timeout = silence_timeout
        self._max_duration = max_duration
        self._min_duration = min_duration

        self._model = None
        self._speech_buffer: List["np.ndarray"] = []
        self._speech_start: Optional[float] = None
        self._last_speech_time: Optional[float] = None
        self._total_samples = 0
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            import torch
            torch.set_num_threads(1)  # minimize CPU usage for VAD
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True,
            )
            self._model = model
            self._get_speech_ts = utils[0]  # get_speech_timestamps
        except ImportError:
            raise ImportError(
                "torch is required for VAD. Install with: pip install torch"
            )
        except Exception as e:
            # Try loading from local cache
            logger.warning("Failed to load Silero VAD from hub: %s", e)
            raise RuntimeError(
                "Silero VAD model is required for live transcription. "
                "Install with: pip install torch torchaudio"
            )

    def process_frame(self, audio_frame: "np.ndarray",
                      timestamp: float) -> Optional[Tuple["np.ndarray", float, float]]:
        """Process one audio frame. Returns (audio_segment, start, end) when
        a complete speech segment is detected, or None otherwise."""
        if np is None:
            raise ImportError("numpy is required for VAD. Install with: pip install numpy")
        self._ensure_model()

        with self._lock:
            self._total_samples += len(audio_frame)

            # Convert to tensor for VAD
            import torch
            audio_tensor = torch.from_numpy(audio_frame.copy()).float()

            try:
                speech_prob = self._model(audio_tensor, self._sample_rate).item()
            except Exception:
                # Fallback: simple energy-based detection
                energy = np.sqrt(np.mean(audio_frame ** 2))
                speech_prob = min(1.0, energy * 20.0)

            is_speech = speech_prob >= self._threshold
            current_time = self._total_samples / self._sample_rate

            if is_speech:
                if self._speech_start is None:
                    self._speech_start = current_time
                self._last_speech_time = current_time
                self._speech_buffer.append(audio_frame.copy())

                # Check max duration
                seg_dur = current_time - self._speech_start
                if seg_dur >= self._max_duration:
                    return self._flush_segment()

            else:
                # Silence — check if speech just ended
                if self._speech_buffer and self._last_speech_time is not None:
                    silence_dur = current_time - self._last_speech_time
                    if silence_dur >= self._silence_timeout:
                        return self._flush_segment()

            return None

    def _flush_segment(self) -> Optional[Tuple["np.ndarray", float, float]]:
        """Emit the current speech buffer as a segment."""
        if not self._speech_buffer:
            self._speech_start = None
            self._last_speech_time = None
            return None

        audio = np.concatenate(self._speech_buffer)
        start = self._speech_start or 0.0
        end = start + len(audio) / self._sample_rate

        self._speech_buffer.clear()
        self._speech_start = None
        self._last_speech_time = None

        # Check minimum duration
        duration = end - start
        if duration < self._min_duration:
            return None

        logger.debug("VAD segment: %.2f-%.2f (%.2fs)", start, end, duration)
        return audio, start, end

    def flush(self) -> Optional[Tuple["np.ndarray", float, float]]:
        """Force-flush any remaining speech in buffer."""
        with self._lock:
            return self._flush_segment()

    def reset(self):
        with self._lock:
            self._speech_buffer.clear()
            self._speech_start = None
            self._last_speech_time = None
            self._total_samples = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming Transcribers
# ═══════════════════════════════════════════════════════════════════════════════


class StreamingTranscriber(ABC):
    """Abstract base for streaming speech-to-text engines."""

    name: str = ""

    @abstractmethod
    def transcribe_chunk(self, audio: "np.ndarray",
                         language: Optional[str] = None) -> str:
        """Transcribe a single audio chunk (numpy float32 array, 16kHz mono).
        Returns the transcribed text."""
        ...

    @abstractmethod
    def check_available(cls) -> bool:
        ...

    @abstractmethod
    def get_models(cls) -> List[str]:
        ...


class FasterWhisperStreaming(StreamingTranscriber):
    """Streaming transcription using faster-whisper.

    Writes each audio segment to a temporary WAV file and transcribes it
    via faster-whisper. Not truly streaming at the model level, but uses
    VAD to chunk the continuous audio stream so that segments are processed
    incrementally with low latency.
    """

    name = "faster-whisper"

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._model = None
        self._device = "auto"
        self._compute_type = "auto"

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is required. Install with: pip install faster-whisper"
            )
        logger.info("Loading faster-whisper model '%s' for live mode ...",
                    self.model_name)
        self._model = WhisperModel(
            self.model_name, device=self._device, compute_type=self._compute_type,
        )

    def transcribe_chunk(self, audio: "np.ndarray",
                         language: Optional[str] = None) -> str:
        self._load_model()

        # Write audio to temp WAV
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="tingshuo_live_")
            os.close(fd)

            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit PCM
                wf.setframerate(SAMPLE_RATE)
                # Convert float32 [-1,1] → int16
                audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())

            kwargs = {"beam_size": 5}
            if language:
                kwargs["language"] = language

            segments_gen, _ = self._model.transcribe(tmp_path, **kwargs)
            texts = [seg.text.strip() for seg in segments_gen if seg.text.strip()]
            return " ".join(texts)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @classmethod
    def check_available(cls) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def get_models(cls) -> List[str]:
        return ["tiny", "tiny.en", "base", "base.en", "small", "small.en",
                "medium", "medium.en", "large-v2", "large-v3"]


class VoskStreaming(StreamingTranscriber):
    """True streaming transcription using Vosk KaldiRecognizer.

    Feeds audio frames incrementally to Vosk and returns partial/final results.
    Lower latency than faster-whisper but lower accuracy.
    """

    name = "vosk"

    def __init__(self, model_name: str = ""):
        self.model_name = model_name
        self._model = None
        self._recognizer = None
        self._sample_rate = SAMPLE_RATE

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from vosk import Model as VoskModel, KaldiRecognizer, SetLogLevel
        except ImportError:
            raise ImportError(
                "vosk is required. Install with: pip install vosk"
            )
        SetLogLevel(-1)

        if self.model_name and os.path.isdir(self.model_name):
            logger.info("Loading Vosk model from path: %s", self.model_name)
            self._model = VoskModel(model_path=self.model_name)
        elif self.model_name:
            logger.info("Loading Vosk model: %s", self.model_name)
            self._model = VoskModel(model_name=self.model_name)
        else:
            logger.info("Loading default Vosk model (English) ...")
            self._model = VoskModel(lang="en")

        self._recognizer = KaldiRecognizer(self._model, self._sample_rate)
        self._recognizer.SetWords(True)

    def transcribe_chunk(self, audio: "np.ndarray",
                         language: Optional[str] = None) -> str:
        self._load_model()

        # Vosk expects 16-bit PCM bytes
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        data = audio_int16.tobytes()

        if self._recognizer.AcceptWaveform(data):
            result = json.loads(self._recognizer.Result())
            return result.get("text", "")
        else:
            partial = json.loads(self._recognizer.PartialResult())
            return partial.get("partial", "")

    def reset_recognizer(self):
        """Reset the recognizer for a new utterance."""
        if self._model is not None:
            from vosk import KaldiRecognizer
            self._recognizer = KaldiRecognizer(self._model, self._sample_rate)
            self._recognizer.SetWords(True)

    @classmethod
    def check_available(cls) -> bool:
        try:
            import vosk  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def get_models(cls) -> List[str]:
        from tingshuo import ENGINE_MODELS
        return ENGINE_MODELS.get("vosk", [])


# ═══════════════════════════════════════════════════════════════════════════════
# Live Translator
# ═══════════════════════════════════════════════════════════════════════════════


class LiveTranslator:
    """Real-time segment-by-segment translation.

    Caches the NLLB model/tokenizer so it's only loaded once.
    For LLM-based translation, calls the Ollama or OpenAI-compatible API.
    """

    def __init__(self, config: LiveConfig):
        self._config = config
        self._nllb_model = None
        self._nllb_tokenizer = None
        self._lock = threading.Lock()

    def translate(self, text: str, source_lang: str,
                  target_lang: str) -> str:
        if not text.strip():
            return ""

        if self._config.translate_method == "nllb":
            return self._translate_nllb(text, source_lang, target_lang)
        else:
            return self._translate_llm(text, source_lang, target_lang)

    def _translate_nllb(self, text: str, source_lang: str,
                        target_lang: str) -> str:
        with self._lock:
            if self._nllb_model is None:
                try:
                    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                except ImportError:
                    raise ImportError(
                        "transformers is required for NLLB translation. "
                        "Install with: pip install transformers sentencepiece"
                    )
                model_name = self._config.nllb_model
                logger.info("Loading NLLB model for live translation: %s", model_name)
                self._nllb_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                # Move to GPU if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        self._nllb_model = self._nllb_model.to("cuda")
                except ImportError:
                    pass

        # Map language codes to NLLB codes
        nllb_src = self._to_nllb_code(source_lang)
        nllb_tgt = self._to_nllb_code(target_lang)

        self._nllb_tokenizer.src_lang = nllb_src
        inputs = self._nllb_tokenizer(text, return_tensors="pt")

        try:
            import torch
            if torch.cuda.is_available() and self._nllb_model.device.type == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
        except Exception:
            pass

        forced_bos = self._nllb_tokenizer.lang_code_to_id.get(
            nllb_tgt,
            self._nllb_tokenizer.lang_code_to_id.get("eng_Latn", 0),
        )
        outputs = self._nllb_model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_length=256,
            num_beams=1,
        )
        result = self._nllb_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result

    def _translate_llm(self, text: str, source_lang: str,
                       target_lang: str) -> str:
        """Use LLM (Ollama or OpenAI API) for translation."""
        src_name = LANG_NAMES.get(source_lang, source_lang)
        tgt_name = LANG_NAMES.get(target_lang, target_lang)

        if self._config.api_url and self._config.api_key:
            return self._call_api(text, src_name, tgt_name)
        elif self._config.ollama_model:
            return self._call_ollama(text, src_name, tgt_name)
        return ""

    def _call_ollama(self, text: str, src_name: str, tgt_name: str) -> str:
        from urllib.request import urlopen, Request
        from urllib.error import URLError, HTTPError

        prompt = (
            f"Translate this from {src_name} to {tgt_name}. "
            f"Return ONLY the translation, nothing else.\n\n{text}"
        )
        url = self._config.ollama_url.rstrip("/") + "/api/generate"
        body = json.dumps({
            "model": self._config.ollama_model,
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")
        req = Request(url, data=body)
        req.add_header("Content-Type", "application/json")
        try:
            with urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "").strip()
        except (URLError, HTTPError, TimeoutError) as e:
            logger.warning("Ollama translation failed: %s", e)
            return ""

    def _call_api(self, text: str, src_name: str, tgt_name: str) -> str:
        from urllib.request import urlopen, Request
        from urllib.error import URLError, HTTPError

        base = self._config.api_url.rstrip("/")
        if not base.endswith("/v1"):
            url = base + "/v1/chat/completions"
        else:
            url = base + "/chat/completions"

        prompt = (
            f"Translate this from {src_name} to {tgt_name}. "
            f"Return ONLY the translation, nothing else."
        )
        body = json.dumps({
            "model": self._config.api_model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            "temperature": 0.1,
        }).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
        }
        req = Request(url, data=body, headers=headers)
        try:
            with urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                choices = result.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
                return ""
        except (URLError, HTTPError, TimeoutError) as e:
            logger.warning("API translation failed: %s", e)
            return ""

    @staticmethod
    def _to_nllb_code(lang: str) -> str:
        """Convert short language code to NLLB-200 code."""
        mapping = {
            "en": "eng_Latn", "zh": "zho_Hans", "ja": "jpn_Jpan",
            "ko": "kor_Hang", "fr": "fra_Latn", "de": "deu_Latn",
            "es": "spa_Latn", "it": "ita_Latn", "pt": "por_Latn",
            "ru": "rus_Cyrl", "ar": "arb_Arab", "hi": "hin_Deva",
            "th": "tha_Thai", "vi": "vie_Latn", "tr": "tur_Latn",
            "nl": "nld_Latn", "pl": "pol_Latn", "uk": "ukr_Cyrl",
            "sv": "swe_Latn", "da": "dan_Latn", "fi": "fin_Latn",
            "no": "nob_Latn", "cs": "ces_Latn", "ro": "ron_Latn",
            "el": "ell_Grek", "hu": "hun_Latn", "id": "ind_Latn",
            "ms": "msa_Latn", "sw": "swh_Latn",
        }
        return mapping.get(lang, "eng_Latn")


# ═══════════════════════════════════════════════════════════════════════════════
# Subtitle Overlay — floating always-on-top window
# ═══════════════════════════════════════════════════════════════════════════════


class SubtitleOverlay:
    """A floating, semi-transparent, always-on-top subtitle window using tkinter.

    Features:
      - Frameless, click-through capable
      - Draggable via title bar area
      - Configurable font, colors, opacity
      - Shows original + translated text
      - Auto-hides when no active speech
    """

    def __init__(self, config: LiveConfig):
        self._config = config
        self._root = None
        self._label_original = None
        self._label_translation = None
        self._label_status = None
        self._visible = False
        self._lines: deque = deque(maxlen=config.overlay_max_lines)
        self._hovering = False
        self._drag_x = 0
        self._drag_y = 0
        self._started = False

    def start(self):
        """Create and show the overlay window."""
        if self._started:
            return
        self._started = True

        import tkinter as tk

        self._root = tk.Toplevel()
        self._root.title("TingShuo Live Subtitles")
        self._root.overrideredirect(True)           # frameless
        self._root.wm_attributes("-topmost", True)   # always on top
        self._root.wm_attributes("-alpha", self._config.overlay_alpha)

        # Set background color
        bg = self._config.overlay_bg
        self._root.configure(bg=bg)

        # Window size
        w = OVERLAY_WIDTH
        h = OVERLAY_HEIGHT
        self._root.geometry(f"{w}x{h}")

        # Position: center-bottom by default
        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()
        x = self._config.overlay_pos_x
        y = self._config.overlay_pos_y
        if x is None:
            x = (screen_w - w) // 2
        if y is None:
            y = screen_h - h - 60  # above taskbar

        self._root.geometry(f"+{x}+{y}")
        self._root.resizable(True, True)
        self._root.minsize(400, 60)

        # ── Title bar (drag handle) ──
        title_bar = tk.Frame(self._root, bg="#333333", height=24)
        title_bar.pack(fill=tk.X, side=tk.TOP)
        title_bar.bind("<ButtonPress-1>", self._on_drag_start)
        title_bar.bind("<B1-Motion>", self._on_drag_move)
        title_bar.bind("<Enter>", lambda e: setattr(self, '_hovering', True))
        title_bar.bind("<Leave>", lambda e: setattr(self, '_hovering', False))

        close_btn = tk.Button(
            title_bar, text="✕", command=self.hide,
            bg="#333333", fg="#cccccc", bd=0, font=("", 10),
            activebackground="#555555", activeforeground="white",
            cursor="hand2",
        )
        close_btn.pack(side=tk.RIGHT, padx=4)

        title_lbl = tk.Label(
            title_bar, text="TingShuo 听说 · Live",
            bg="#333333", fg="#aaaaaa", font=("", 9),
        )
        title_lbl.pack(side=tk.LEFT, padx=6)

        # ── Content area ──
        content = tk.Frame(self._root, bg=bg)
        content.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 8))

        font_size = self._config.overlay_font_size

        self._label_status = tk.Label(
            content, text="🎤 Listening...",
            bg=bg, fg="#888888", font=("", max(9, font_size - 4)),
            anchor="w",
        )
        self._label_status.pack(fill=tk.X, side=tk.TOP)

        self._label_original = tk.Label(
            content, text="",
            bg=bg, fg=self._config.overlay_fg,
            font=("", font_size, "bold"),
            wraplength=w - 30, justify="left", anchor="w",
        )
        self._label_original.pack(fill=tk.X, side=tk.TOP, pady=(2, 0))

        self._label_translation = tk.Label(
            content, text="",
            bg=bg, fg="#ffcc66",
            font=("", max(10, font_size - 2)),
            wraplength=w - 30, justify="left", anchor="w",
        )
        self._label_translation.pack(fill=tk.X, side=tk.TOP, pady=(1, 0))

        # Right-click menu
        self._menu = tk.Menu(self._root, tearoff=0)
        self._menu.add_command(label="Hide", command=self.hide)
        self._menu.add_command(label="Font +", command=self._increase_font)
        self._menu.add_command(label="Font -", command=self._decrease_font)
        self._menu.add_separator()
        self._menu.add_command(label="Exit Live Mode", command=self._exit_live)
        self._root.bind("<Button-3>", lambda e: self._menu.post(e.x_root, e.y_root))

        self._root.withdraw()  # hidden until first subtitle
        self._visible = False

    def _on_drag_start(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_drag_move(self, event):
        x = self._root.winfo_x() + event.x - self._drag_x
        y = self._root.winfo_y() + event.y - self._drag_y
        self._root.geometry(f"+{x}+{y}")

    def _increase_font(self):
        self._config.overlay_font_size = min(48, self._config.overlay_font_size + 2)
        self._label_original.config(font=("", self._config.overlay_font_size, "bold"))

    def _decrease_font(self):
        self._config.overlay_font_size = max(10, self._config.overlay_font_size - 2)
        self._label_original.config(font=("", self._config.overlay_font_size, "bold"))

    def _exit_live(self):
        """Signal exit. The LiveSession checks this."""
        if hasattr(self, '_on_exit_callback') and self._on_exit_callback:
            self._on_exit_callback()

    def set_exit_callback(self, cb: Callable):
        self._on_exit_callback = cb

    def update_subtitle(self, text: str, translation: str = "",
                        language: str = "", is_partial: bool = False):
        """Update the displayed subtitle text."""
        if not self._started:
            return

        self._root.after(0, lambda: self._do_update(text, translation,
                                                     language, is_partial))

    def _do_update(self, text: str, translation: str,
                   language: str, is_partial: bool):
        if not text.strip() and not translation.strip():
            # Fade out after a short delay
            self._root.after(3000, self._maybe_hide)
            return

        if not self._visible:
            self._root.deiconify()
            self._visible = True

        if is_partial:
            self._label_status.config(text=f"🎤 {language} (partial)...")
        else:
            lang_display = LANG_NAMES.get(language, language) if language else ""
            self._label_status.config(text=f"🎤 {lang_display}")

        self._label_original.config(text=text)
        if translation:
            self._label_translation.config(text=translation)
        else:
            self._label_translation.config(text="")

        # Update wraplength on resize
        w = self._root.winfo_width()
        if w > 100:
            self._label_original.config(wraplength=w - 30)
            self._label_translation.config(wraplength=w - 30)

    def _maybe_hide(self):
        """Hide overlay if no new text has arrived."""
        if not self._hovering:
            self._root.withdraw()
            self._visible = False

    def hide(self):
        if self._root:
            self._root.withdraw()
            self._visible = False

    def show(self):
        if self._root:
            self._root.deiconify()
            self._visible = True

    def stop(self):
        """Destroy the overlay window."""
        self._started = False
        if self._root:
            try:
                self._root.destroy()
            except Exception:
                pass
            self._root = None

    def is_alive(self) -> bool:
        if not self._root:
            return False
        try:
            self._root.winfo_exists()
            return True
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# System Tray Controller
# ═══════════════════════════════════════════════════════════════════════════════


class SystemTrayController:
    """System tray icon with context menu for controlling live mode.

    Uses pystray (cross-platform). Provides:
      - Start / Stop live capture
      - Language selection
      - Translation toggle
      - Show / Hide overlay
      - Exit
    """

    def __init__(self, config: LiveConfig):
        self._config = config
        self._icon = None
        self._running = False
        self._callbacks: Dict[str, Callable] = {}
        self._icon_image = None

    def set_callback(self, name: str, cb: Callable):
        self._callbacks[name] = cb

    def start(self):
        """Start the system tray icon."""
        if self._running:
            return
        self._running = True

        try:
            import pystray
            from PIL import Image, ImageDraw
        except ImportError:
            logger.warning(
                "pystray or Pillow not installed. System tray disabled. "
                "Install with: pip install pystray Pillow"
            )
            return

        # Create a simple icon
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # Draw a speech-bubble-like icon
        draw.rounded_rectangle([8, 8, 56, 56], radius=12,
                               fill=(60, 140, 230, 255))
        draw.text((20, 18), "听", fill=(255, 255, 255, 255))
        self._icon_image = img

        def on_clicked(icon, item):
            action = getattr(item, 'action', None)
            if action and action in self._callbacks:
                self._callbacks[action]()

        # Build menu
        from pystray import Menu, MenuItem

        menu = Menu(
            MenuItem("🎤 Start Capture", self._on_start, enabled=True),
            MenuItem("⏹ Stop Capture", self._on_stop, enabled=False),
            Menu.SEPARATOR,
            MenuItem("📺 Show Overlay", self._on_show_overlay),
            MenuItem("📺 Hide Overlay", self._on_hide_overlay),
            Menu.SEPARATOR,
            MenuItem("Language ▼", Menu(
                MenuItem("Auto-detect", lambda: self._on_lang(None)),
                MenuItem("English", lambda: self._on_lang("en")),
                MenuItem("中文", lambda: self._on_lang("zh")),
                MenuItem("日本語", lambda: self._on_lang("ja")),
                MenuItem("한국어", lambda: self._on_lang("ko")),
                MenuItem("Français", lambda: self._on_lang("fr")),
                MenuItem("Deutsch", lambda: self._on_lang("de")),
                MenuItem("Español", lambda: self._on_lang("es")),
                MenuItem("Português", lambda: self._on_lang("pt")),
                MenuItem("Русский", lambda: self._on_lang("ru")),
            )),
            MenuItem("Translation ▼", Menu(
                MenuItem("Off", lambda: self._on_trans_toggle(False)),
                *[MenuItem(
                    name, lambda c=code: self._on_target_lang(c)
                ) for code, name in [
                    ("zh", "→ 中文"), ("en", "→ English"),
                    ("ja", "→ 日本語"), ("ko", "→ 한국어"),
                    ("fr", "→ Français"), ("de", "→ Deutsch"),
                    ("es", "→ Español"),
                ]],
            )),
            Menu.SEPARATOR,
            MenuItem("❌ Exit", self._on_exit),
        )

        self._icon = pystray.Icon("tingshuo", img, "TingShuo 听说 · Live", menu)

        # Run in a separate thread
        threading.Thread(target=self._icon.run, daemon=True).start()

    def _on_start(self):
        if "start" in self._callbacks:
            self._callbacks["start"]()

    def _on_stop(self):
        if "stop" in self._callbacks:
            self._callbacks["stop"]()

    def _on_show_overlay(self):
        if "show_overlay" in self._callbacks:
            self._callbacks["show_overlay"]()

    def _on_hide_overlay(self):
        if "hide_overlay" in self._callbacks:
            self._callbacks["hide_overlay"]()

    def _on_lang(self, lang):
        if "set_language" in self._callbacks:
            self._callbacks["set_language"](lang)

    def _on_target_lang(self, lang):
        if "set_target_language" in self._callbacks:
            self._callbacks["set_target_language"](lang)

    def _on_trans_toggle(self, enabled):
        if "toggle_translation" in self._callbacks:
            self._callbacks["toggle_translation"](enabled)

    def _on_exit(self):
        if "exit" in self._callbacks:
            self._callbacks["exit"]()
        self.stop()

    def stop(self):
        self._running = False
        if self._icon:
            try:
                self._icon.stop()
            except Exception:
                pass
            self._icon = None


# ═══════════════════════════════════════════════════════════════════════════════
# Live Session — orchestrator
# ═══════════════════════════════════════════════════════════════════════════════


class LiveSession:
    """Orchestrates the full live pipeline:

    AudioCapture → VAD → StreamingTranscriber → LiveTranslator → SubtitleOverlay

    Runs capture on a high-priority thread; transcription and translation
    happen on worker threads pulled from a queue so slow processing doesn't
    drop audio frames.
    """

    def __init__(self, config: LiveConfig,
                 on_log: Optional[Callable[[str], None]] = None):
        self.config = config
        self._on_log = on_log or (lambda msg: logger.info(msg))

        # Components
        self._capture: Optional[AudioCapture] = None
        self._vad: Optional[VoiceActivityDetector] = None
        self._transcriber: Optional[StreamingTranscriber] = None
        self._translator: Optional[LiveTranslator] = None
        self._overlay: Optional[SubtitleOverlay] = None
        self._tray: Optional[SystemTrayController] = None

        # State
        self._running = False
        self._stop_event = threading.Event()
        self._segment_queue: queue.Queue = queue.Queue(maxsize=50)
        self._result_queue: queue.Queue = queue.Queue(maxsize=100)
        self._language: Optional[str] = config.language
        self._detected_language: str = ""
        self._session_start: float = 0.0
        self._worker_threads: List[threading.Thread] = []

    def prepare_overlay(self):
        """Create the overlay window on the calling thread (must be main/GUI thread).
        Call this before start() when overlay is enabled."""
        if not self.config.overlay_enabled or self._overlay is not None:
            return
        self._overlay = SubtitleOverlay(self.config)
        self._overlay.set_exit_callback(self.stop)
        self._overlay.start()
        return self._overlay

    def start(self):
        """Start the live session. Call prepare_overlay() first on the main thread
        if overlay is enabled."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._session_start = time.time()

        self._on_log("🚀 Starting live session...")

        # ── Initialize components ──

        # VAD
        self._vad = VoiceActivityDetector(
            sample_rate=self.config.sample_rate,
            threshold=self.config.vad_threshold,
            silence_timeout=self.config.silence_timeout,
            max_duration=self.config.max_segment_duration,
            min_duration=self.config.min_segment_duration,
        )

        # Transcriber
        if self.config.engine_name == "vosk":
            self._transcriber = VoskStreaming(model_name=self.config.model_name)
        else:
            self._transcriber = FasterWhisperStreaming(
                model_name=self.config.model_name,
            )
        self._on_log(f"🔊 Engine: {self.config.engine_name} / {self.config.model_name}")
        self._on_log(f"🌐 Language: {self.config.language or 'auto-detect'}")

        # Translator
        if self.config.translate_enabled and self.config.target_languages:
            self._translator = LiveTranslator(self.config)
            self._on_log(f"📝 Translation: → {', '.join(self.config.target_languages)}")

        # Overlay — must be pre-created on main thread via prepare_overlay()
        if self.config.overlay_enabled:
            if self._overlay is None:
                # Create now if caller forgot to call prepare_overlay()
                self._overlay = SubtitleOverlay(self.config)
                self._overlay.set_exit_callback(self.stop)
                self._overlay.start()
            self._on_log("📺 Overlay window ready")

        # System tray
        if self.config.tray_enabled:
            self._tray = SystemTrayController(self.config)
            self._tray.set_callback("start", self.start)
            self._tray.set_callback("stop", self.stop)
            self._tray.set_callback("show_overlay",
                                    lambda: self._overlay.show() if self._overlay else None)
            self._tray.set_callback("hide_overlay",
                                    lambda: self._overlay.hide() if self._overlay else None)
            self._tray.set_callback("set_language", self.set_language)
            self._tray.set_callback("set_target_language", self.set_target_language)
            self._tray.set_callback("toggle_translation", self.toggle_translation)
            self._tray.set_callback("exit", self.stop)
            self._tray.start()

        # ── Start worker threads ──

        # Transcription worker: takes VAD segments, runs STT, pushes results
        n_transcribers = max(1, min(2, (os.cpu_count() or 4) // 2))
        for i in range(n_transcribers):
            t = threading.Thread(
                target=self._transcribe_worker,
                name=f"ts-transcribe-{i}",
                daemon=True,
            )
            t.start()
            self._worker_threads.append(t)

        # Display worker: takes results, runs translation, updates overlay
        t = threading.Thread(
            target=self._display_worker,
            name="ts-display",
            daemon=True,
        )
        t.start()
        self._worker_threads.append(t)

        # ── Start audio capture on main thread ──
        self._capture = AudioCapture(
            sample_rate=self.config.sample_rate,
            device=self.config.capture_device,
        )

        def on_audio(chunk: np.ndarray):
            if self._stop_event.is_set():
                return
            timestamp = time.time() - self._session_start
            result = self._vad.process_frame(chunk, timestamp)
            if result is not None:
                audio_seg, start_t, end_t = result
                try:
                    self._segment_queue.put_nowait((audio_seg, start_t, end_t))
                except queue.Full:
                    logger.debug("Segment queue full, dropping oldest segment.")

        self._capture.start(on_audio)
        self._on_log("🎤 Live capture active — listening for speech...")

    def stop(self):
        """Stop the live session gracefully."""
        if not self._running:
            return
        self._on_log("Stopping live session...")
        self._stop_event.set()
        self._running = False

        # Stop capture
        if self._capture:
            self._capture.stop()

        # Flush VAD
        if self._vad:
            remaining = self._vad.flush()
            if remaining:
                audio_seg, start_t, end_t = remaining
                try:
                    self._segment_queue.put_nowait((audio_seg, start_t, end_t))
                except queue.Full:
                    pass

        # Signal workers to finish
        for _ in self._worker_threads:
            try:
                self._segment_queue.put_nowait(None)  # sentinel
            except queue.Full:
                pass

        # Wait for workers
        for t in self._worker_threads:
            t.join(timeout=5.0)

        # Stop overlay
        if self._overlay:
            self._overlay.stop()

        # Stop tray
        if self._tray:
            self._tray.stop()

        self._on_log("Live session stopped.")

    def _transcribe_worker(self):
        """Worker: pull VAD segments, transcribe, push to result queue."""
        while not self._stop_event.is_set():
            try:
                item = self._segment_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:  # sentinel
                break

            audio_seg, start_t, end_t = item
            try:
                text = self._transcriber.transcribe_chunk(
                    audio_seg, language=self._language,
                )

                # Detect language on first successful result
                if not self._detected_language and self._language is None:
                    self._detected_language = self._guess_language(text)

                lang = self._language or self._detected_language or ""
                if text.strip():
                    segment = LiveSegment(
                        text=text.strip(),
                        start=start_t,
                        end=end_t,
                        language=lang,
                    )
                    try:
                        self._result_queue.put_nowait(segment)
                    except queue.Full:
                        logger.debug("Result queue full, dropping.")
            except Exception as e:
                logger.error("Transcription error: %s", e)

    def _display_worker(self):
        """Worker: pull results, translate, update overlay."""
        while not self._stop_event.is_set():
            try:
                segment = self._result_queue.get(timeout=0.3)
            except queue.Empty:
                continue

            if segment is None:
                break

            # Translate if enabled
            translation = ""
            if self._translator and segment.language:
                target_langs = self.config.target_languages
                if target_langs:
                    tgt = target_langs[0]  # primary target language
                    try:
                        translation = self._translator.translate(
                            segment.text, segment.language, tgt,
                        )
                    except Exception as e:
                        logger.warning("Translation error: %s", e)

            segment.translation = translation

            # Update overlay
            if self._overlay:
                self._overlay.update_subtitle(
                    text=segment.text,
                    translation=translation,
                    language=segment.language,
                    is_partial=False,
                )

            # Log to terminal with clear formatting
            lang_name = LANG_NAMES.get(segment.language, segment.language or "??")
            timestamp = f"{segment.start:5.1f}s"
            log_line = f"🎤 [{lang_name}] {segment.text}"
            if translation:
                tgt_name = LANG_NAMES.get(
                    self.config.target_languages[0] if self.config.target_languages else "",
                    self.config.target_languages[0] if self.config.target_languages else "")
                log_line += f"\n📝 [{tgt_name}] {translation}"
            self._on_log(log_line)

    def set_language(self, language: Optional[str]):
        """Set source language. None = auto-detect."""
        self._language = language
        self._detected_language = ""
        self._on_log(f"Language set to: {language or 'auto-detect'}")

    def set_target_language(self, language: str):
        """Set translation target language."""
        self.config.target_languages = [language] if language else []
        self.config.translate_enabled = bool(language)
        if language:
            self._translator = LiveTranslator(self.config)
            self._on_log(f"Translation target: {language}")
        else:
            self._translator = None
            self._on_log("Translation disabled")

    def toggle_translation(self, enabled: bool):
        """Enable/disable translation."""
        self.config.translate_enabled = enabled
        if enabled and self.config.target_languages:
            self._translator = LiveTranslator(self.config)
        else:
            self._translator = None
        self._on_log(f"Translation: {'ON' if enabled else 'OFF'}")

    @staticmethod
    def _guess_language(text: str) -> str:
        """Simple language guessing based on character ranges."""
        # CJK detection
        cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff'
                        or '\u3040' <= c <= '\u309f' or '\uac00' <= c <= '\ud7af')
        if cjk_count > len(text) * 0.3:
            # Distinguish Chinese, Japanese, Korean
            hiragana = sum(1 for c in text if '\u3040' <= c <= '\u309f')
            hangul = sum(1 for c in text if '\uac00' <= c <= '\ud7af')
            if hangul > hiragana:
                return "ko"
            if hiragana > 0:
                return "ja"
            return "zh"

        # Cyrillic
        cyrillic = sum(1 for c in text if '\u0400' <= c <= '\u04ff')
        if cyrillic > len(text) * 0.3:
            return "ru"

        # Latin script → likely English but could be many
        return "en"

    @property
    def is_running(self) -> bool:
        return self._running


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════════════════════


def create_live_config_from_gui(config_dict: dict) -> LiveConfig:
    """Create a LiveConfig from GUI settings dict (compatible with existing GUI)."""
    lc = LiveConfig()
    for key, val in config_dict.items():
        if hasattr(lc, key):
            setattr(lc, key, val)
    return lc


def run_live_session(config: LiveConfig,
                     on_log: Optional[Callable[[str], None]] = None) -> LiveSession:
    """Create and start a live session. Returns the session object.

    The caller is responsible for keeping the main thread alive
    (e.g. via tkinter mainloop).
    """
    session = LiveSession(config, on_log=on_log)

    # If overlay is enabled, start it on the main thread now
    if config.overlay_enabled:
        import tkinter as tk
        # We need a hidden root if there's no existing tkinter loop
        try:
            root = tk.Tk()
            root.withdraw()
        except Exception:
            pass

    session.start()
    return session
