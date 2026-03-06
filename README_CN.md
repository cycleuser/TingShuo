# TingShuo 听说

**使用多种语音识别引擎从音频/视频文件生成 SRT/LRC 字幕，支持 LLM 润色。**

TingShuo 可以递归扫描目录中的所有媒体文件，使用您选择的语音识别引擎进行转录，并输出 SRT 或 LRC 格式的字幕文件。还可以通过 LLM（Ollama 或 OpenAI 兼容 API）或 NLP 句子分割（nltk）对字幕进行润色，生成自然、完整的句子。

## 功能特点

- **4 种语音引擎**：faster-whisper、Vosk、OpenAI Whisper、whisper.cpp
- **2 种输出格式**：SRT（字幕）和 LRC（歌词）
- **字幕翻译**：使用 NLLB 或 LLM 将字幕翻译为多种目标语言
- **多语言界面**：支持中文、英文、日文、韩文、法文、德文、西班牙文、意大利文、葡萄牙文、俄文
- **LLM 润色**：通过 Ollama 或 OpenAI 兼容 API 将碎片化字幕合并为自然句子
- **NLP 润色**：通过 nltk 进行句子边界检测（无需 LLM）
- **命令行 + 图形界面**：完整的 CLI 和 tkinter GUI
- **递归扫描**：处理整个目录树中的媒体文件
- **HuggingFace 镜像**：内置 HF 镜像支持（解决国内下载模型困难的问题）
- **灵活输出**：字幕可保存在源文件目录或自定义目录
- **设置持久化**：界面语言和偏好设置保存到 `~/.config/tingshuo/settings.json`

## 安装

### 从 PyPI 安装

```bash
# 基础安装（不含语音引擎）
pip install tingshuo

# 安装特定引擎：
pip install tingshuo[faster-whisper]   # 推荐
pip install tingshuo[vosk]
pip install tingshuo[whisper]
pip install tingshuo[whisper-cpp]

# 安装 NLP 润色支持：
pip install tingshuo[nlp]

# 安装所有组件：
pip install tingshuo[all]
```

### 从源码安装

```bash
git clone https://github.com/cycleuser/TingShuo.git
cd tingshuo
pip install -e .[faster-whisper,nlp]
```

## 前置要求

- **Python 3.9+**
- **ffmpeg** 必须已安装并在系统 PATH 中
  - Linux: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: 从 [ffmpeg.org](https://ffmpeg.org/download.html) 下载，解压后将 `bin` 目录添加到系统环境变量 PATH 中

## 快速开始

### 命令行模式

**基本转录（生成 SRT）：**
```bash
tingshuo -i ./videos -e faster-whisper -f srt
```

**生成 LRC 到指定目录：**
```bash
tingshuo -i ./audio -e vosk -f lrc -o ./subtitles
```

**LLM 润色（本地 Ollama）：**
```bash
tingshuo -i ./media --polish-llm --ollama-model qwen2.5
```

**LLM 润色（OpenAI 兼容 API）：**
```bash
tingshuo -i ./media --polish-llm --api-url https://api.example.com --api-key sk-xxx --api-model gpt-4o-mini
```

**NLP 润色：**
```bash
tingshuo -i ./media --polish-nlp -l zh
```

**指定语言和模型：**
```bash
tingshuo -i ./videos -e faster-whisper -m large-v3 -l zh
```

**使用 HuggingFace 镜像（国内用户）：**
```bash
tingshuo -i ./videos -e faster-whisper --hf-mirror https://hf-mirror.com
```

**翻译字幕到多种语言（NLLB）：**
```bash
tingshuo -i ./videos -e faster-whisper --translate --target-lang zh,ja,ko
```

**使用 LLM 翻译字幕：**
```bash
tingshuo -i ./videos -e faster-whisper --translate --target-lang zh --trans-backend llm --ollama-model qwen2.5
```

**预下载模型：**
```bash
tingshuo --download -e faster-whisper -m large-v3
tingshuo --download -e faster-whisper -m large-v3 --hf-mirror https://hf-mirror.com
```

**批量下载某引擎的所有模型：**
```bash
tingshuo --download-all -e faster-whisper
```

**列出已安装的 Ollama 模型：**
```bash
tingshuo --list-ollama-models
tingshuo --list-ollama-models --ollama-url http://192.168.1.100:11434
```

### 图形界面模式

```bash
tingshuo --gui
```

图形界面提供：
- 输入/输出目录选择（带浏览按钮）
- 引擎和模型选择下拉框
- **语言下拉选择框**，包含常用语言（自动检测、zh、en、ja、ko 等），也可手动输入语言代码
- **模型下载按钮**（下载 / 全部下载），带进度反馈
- 格式切换（SRT/LRC）
- 润色选项（无 / LLM / NLP），含配置面板
- **翻译面板**：启用翻译、选择目标语言、选择翻译后端（NLLB 或 LLM）
- **Ollama 模型下拉框**，带刷新按钮，可从服务器查询已安装模型
- **菜单栏**：帮助 > 设置（界面语言）、帮助 > 关于（版本信息）
- **多语言界面**：设置中可切换 10 种界面语言
- HuggingFace 镜像开关
- 进度条和实时日志输出
- 开始/停止控制按钮

## 命令行参数说明

```
用法: tingshuo [-h] [--version] [--gui] [-i DIR] [-o DIR] [-f {srt,lrc}]
               [--no-recursive] [-e ENGINE] [-m NAME] [-l CODE]
               [--hf-mirror URL] [--download] [--download-all]
               [--list-ollama-models] [--polish-llm | --polish-nlp]
               [--ollama-url URL] [--ollama-model NAME] [--api-url URL]
               [--api-key KEY] [--api-model NAME] [-v]
               [--translate] [--target-lang CODES]
               [--trans-backend {nllb,llm}] [--nllb-model NAME]
```

### 输入/输出

| 参数 | 说明 |
|------|------|
| `-i`, `--input DIR` | 输入目录，包含音频/视频文件（必需） |
| `-o`, `--output DIR` | 输出目录（默认：与源文件相同目录） |
| `-f`, `--format {srt,lrc}` | 字幕格式（默认：srt） |
| `--no-recursive` | 不递归扫描子目录 |

### 语音引擎

| 参数 | 说明 |
|------|------|
| `-e`, `--engine` | 引擎：`faster-whisper`、`vosk`、`whisper`、`whisper-cpp`（默认：faster-whisper） |
| `-m`, `--model NAME` | 模型名称或路径（默认：引擎默认值，通常为 "base"） |
| `-l`, `--language CODE` | 语言代码：zh、en、ja 等，使用 "auto" 自动检测（默认：auto） |

### HuggingFace 镜像

| 参数 | 说明 |
|------|------|
| `--hf-mirror URL` | HuggingFace 镜像地址，例如 `https://hf-mirror.com` |

### 模型管理

| 参数 | 说明 |
|------|------|
| `--download` | 下载 `-e` 和 `-m` 指定的模型，然后退出 |
| `--download-all` | 下载 `-e` 指定引擎的所有已知模型，然后退出 |
| `--list-ollama-models` | 列出 Ollama 服务器上已安装的模型（使用 `--ollama-url`），然后退出 |

### 字幕润色

| 参数 | 说明 |
|------|------|
| `--polish-llm` | 使用 LLM 润色（Ollama 或 OpenAI 兼容 API） |
| `--polish-nlp` | 使用 NLP 句子分割润色（nltk） |
| `--ollama-url URL` | Ollama API 地址（默认：http://localhost:11434） |
| `--ollama-model NAME` | Ollama 模型名称（默认：qwen2.5） |
| `--api-url URL` | OpenAI 兼容 API 基础地址 |
| `--api-key KEY` | API 密钥 |
| `--api-model NAME` | API 模型名称 |

### 其他

| 参数 | 说明 |
|------|------|
| `--gui` | 启动图形界面 |
| `-v`, `--verbose` | 启用详细日志 |
| `--version` | 显示版本号 |

### 字幕翻译

| 参数 | 说明 |
|------|------|
| `--translate` | 启用字幕翻译到目标语言 |
| `--target-lang CODES` | 逗号分隔的目标语言代码，例如 `zh,en,ja` |
| `--trans-backend {nllb,llm}` | 翻译后端：`nllb`（Helsinki-NLP/NLLB）或 `llm`（默认：nllb） |
| `--nllb-model NAME` | NLLB 模型名称（默认：facebook/nllb-200-distilled-600M） |

## 支持的格式

### 输入（音频/视频）

**音频**：mp3、wav、flac、aac、ogg、wma、m4a、opus

**视频**：mp4、mkv、avi、mov、wmv、flv、webm、ts、m4v、mpg、mpeg

### 输出

**SRT**（SubRip 字幕格式）：
```
1
00:00:01,500 --> 00:00:04,200
这是第一行字幕。

2
00:00:05,000 --> 00:00:08,300
这是第二行字幕。
```

**LRC**（歌词格式）：
```
[ti:文件名]
[re:TingShuo v0.1.0]

[00:01.50]这是第一行字幕。
[00:05.00]这是第二行字幕。
```

## 语音识别引擎

### faster-whisper（推荐）

基于 CTranslate2 的 Whisper 实现。速度快，支持 GPU 加速。

```bash
pip install faster-whisper
```

**可用模型**：tiny、base、small、medium、large-v2、large-v3

### Vosk

轻量级离线语音识别。精度较低但 CPU 上运行非常快。

```bash
pip install vosk
```

**模型**：按语言自动下载，或使用 `-m /path/to/model` 指定本地模型路径。

### OpenAI Whisper

OpenAI 官方 Whisper 模型。

```bash
pip install openai-whisper
```

**可用模型**：tiny、base、small、medium、large

### whisper.cpp

Whisper 的 C++ 实现，通过 Python 绑定使用。CPU 上非常快速。

```bash
pip install pywhispercpp
```

**可用模型**：tiny、base、small、medium、large

## 字幕润色

### LLM 润色

将字幕片段发送给 LLM，合并为完整、自然的句子。

**使用本地 Ollama：**

1. 安装并启动 [Ollama](https://ollama.com)
2. 拉取模型：`ollama pull qwen2.5`
3. 运行：`tingshuo -i ./media --polish-llm --ollama-model qwen2.5`

**使用局域网 Ollama：**
```bash
tingshuo -i ./media --polish-llm --ollama-url http://192.168.1.100:11434 --ollama-model qwen2.5
```

**使用 OpenAI 兼容 API：**
```bash
tingshuo -i ./media --polish-llm --api-url https://api.openai.com --api-key sk-xxx --api-model gpt-4o-mini
```

支持任何 OpenAI 兼容的 API 服务（如各种国内 AI API 服务商）。

### NLP 润色

使用 nltk 句子分割来检测句子边界并合并碎片。无需 LLM，无需网络连接。

```bash
pip install nltk
tingshuo -i ./media --polish-nlp -l zh
```

英文等语言使用 nltk 内置分句器；中文/日文/韩文使用标点符号进行句子分割（`。！？`等）。

## 字幕翻译

TingShuo 可以自动将生成的字幕翻译为多种目标语言。翻译后的字幕保存为带语言代码的独立文件（例如 `video.zh.srt`、`video.ja.srt`）。

### NLLB 翻译（推荐）

使用 Helsinki-NLP/NLLB 模型进行高质量离线翻译，支持 200+ 种语言。

```bash
# 安装依赖
pip install transformers sentencepiece

# 翻译为中文和日文
tingshuo -i ./videos -e faster-whisper --translate --target-lang zh,ja

# 使用更大的 NLLB 模型以获得更好的翻译质量
tingshuo -i ./videos --translate --target-lang zh --nllb-model facebook/nllb-200-distilled-1.3B
```

可用的 NLLB 模型：`facebook/nllb-200-distilled-600M`（默认）、`facebook/nllb-200-distilled-1.3B`、`facebook/nllb-200-3.3B`

### LLM 翻译

使用 Ollama 或 OpenAI 兼容 API 进行翻译。

```bash
# 使用 Ollama 翻译
tingshuo -i ./videos --translate --target-lang zh --trans-backend llm --ollama-model qwen2.5

# 使用 OpenAI API 翻译
tingshuo -i ./videos --translate --target-lang zh --trans-backend llm --api-url https://api.openai.com --api-key sk-xxx --api-model gpt-4o-mini
```

## HuggingFace 镜像（国内用户）

国内用户下载模型时可能遇到网络问题，可使用 HF 镜像：

```bash
# 方式一：命令行参数
tingshuo -i ./videos -e faster-whisper --hf-mirror https://hf-mirror.com

# 方式二：环境变量
export HF_ENDPOINT=https://hf-mirror.com
tingshuo -i ./videos -e faster-whisper
```

图形界面中也可以勾选 "Use HF Mirror" 选项并填入镜像地址。

## 常见问题

### ffmpeg 未找到

确保 ffmpeg 已安装并在 PATH 中：
```bash
ffmpeg -version
```

Windows 用户注意：下载 ffmpeg 后需要将其 `bin` 目录（包含 `ffmpeg.exe`）添加到系统环境变量 PATH。

### CUDA / GPU 支持

faster-whisper 和 OpenAI Whisper 支持 CUDA GPU 加速。需要安装对应的 CUDA 版本的 PyTorch：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 模型下载慢

使用 `--hf-mirror https://hf-mirror.com` 参数或设置 `HF_ENDPOINT` 环境变量。

## 许可证

本项目基于 GNU 通用公共许可证 v3.0 发布。详见 [LICENSE](LICENSE)。
