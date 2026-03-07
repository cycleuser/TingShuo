# TingShuo 听说

**使用多种语音识别引擎从音频/视频文件生成 SRT/LRC 字幕和 Markdown 讲稿，支持自动纠错、LLM 润色和多模态内容总结。**

TingShuo 可以递归扫描目录中的所有媒体文件，使用您选择的语音识别引擎进行转录，并输出 SRT、LRC 或 Markdown 讲稿格式的文件。功能包括基于 LLM 的自动纠错（错别字和口误）、字幕润色（LLM 或 NLP）以及带多模态视频分析的内容总结。

## 功能特点

- **4 种语音引擎**：faster-whisper、Vosk、OpenAI Whisper、whisper.cpp
- **3 种输出格式**：SRT（字幕）、LRC（歌词）和 MD（Markdown 讲稿）
- **Markdown 讲稿**：从演讲、讲座等生成结构清晰的讲稿文档
- **自动纠错**：通过 LLM 自动修复错别字、口误等转录错误
- **内容总结**：对音频/视频内容进行总结，视频支持多模态分析（关键帧提取 + 视觉 LLM）
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

**生成 Markdown 讲稿（讲座/演讲）：**
```bash
tingshuo -i ./lectures -f md --polish-llm --ollama-model qwen2.5
```

**自动纠正错别字和口误：**
```bash
tingshuo -i ./media --auto-correct --ollama-model qwen2.5
```

**自动纠错 + LLM 润色组合使用：**
```bash
tingshuo -i ./media --auto-correct --polish-llm --ollama-model qwen2.5
```

**生成内容总结：**
```bash
tingshuo -i ./media --summarize --ollama-model qwen2.5
```

**多模态视频总结（OpenAI 兼容 API）：**
```bash
tingshuo -i ./videos --summarize --api-url https://api.example.com --api-key sk-xxx --api-model gpt-4o-mini
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
- 格式切换（SRT/LRC/MD）
- **自动纠错复选框**：启用 LLM 自动纠错转录错误
- **内容总结复选框**：生成总结文件，可设置关键帧间隔
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
用法: tingshuo [-h] [--version] [--gui] [-i DIR] [-o DIR] [-f {srt,lrc,md}]
               [--no-recursive] [-e ENGINE] [-m NAME] [-l CODE]
               [--hf-mirror URL] [--download] [--download-all]
               [--list-ollama-models] [--auto-correct]
               [--polish-llm | --polish-nlp]
               [--ollama-url URL] [--ollama-model NAME] [--api-url URL]
               [--api-key KEY] [--api-model NAME] [-v]
               [--translate] [--target-lang CODES]
               [--trans-backend {nllb,llm}] [--nllb-model NAME]
               [--summarize] [--keyframe-interval SECONDS]
```

### 输入/输出

| 参数 | 说明 |
|------|------|
| `-i`, `--input DIR` | 输入目录，包含音频/视频文件（必需） |
| `-o`, `--output DIR` | 输出目录（默认：与源文件相同目录） |
| `-f`, `--format {srt,lrc,md}` | 输出格式：srt、lrc 或 md（Markdown 讲稿）（默认：srt） |
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

### 自动纠错

| 参数 | 说明 |
|------|------|
| `--auto-correct` | 使用 LLM 自动纠正错别字、口误等转录错误 |

### LLM 设置

| 参数 | 说明 |
|------|------|
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

### 内容总结

| 参数 | 说明 |
|------|------|
| `--summarize` | 生成内容总结文件（.summary.md） |
| `--keyframe-interval SECONDS` | 视频总结的关键帧提取间隔（秒）（默认：60） |

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
[re:TingShuo v0.1.3]

[00:01.50]这是第一行字幕。
[00:05.00]这是第二行字幕。
```

**MD**（Markdown 讲稿格式）：
```markdown
## 引言

这是演讲的开头部分，LLM 将内容组织成自然的段落。

## 主要内容

演讲者接下来讨论了主要议题，关键要点被整理为可读的段落。
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

## Markdown 讲稿生成

TingShuo 可以从演讲、讲座和报告中生成结构清晰的 Markdown 讲稿。与带时间戳的字幕不同，MD 格式输出的是按主题组织的流畅文本。

```bash
# 生成 Markdown 讲稿（使用 LLM 组织段落）
tingshuo -i ./lectures -f md --polish-llm --ollama-model qwen2.5

# 配合自动纠错获得更干净的输出
tingshuo -i ./lectures -f md --auto-correct --polish-llm --ollama-model qwen2.5
```

LLM 会将原始转录内容组织成带 Markdown 标题和段落的逻辑结构。如果未配置 LLM，则使用简单的段落分组作为后备方案。

## 自动纠错

TingShuo 可以在润色或输出之前自动修复转录错误，包括：

- **错别字**：语音识别引擎常见的错误识别
- **口误**：说话时的口误和失言
- **语气词**：去除"嗯"、"那个"、"um"、"uh"等无意义的语气词

```bash
# 仅自动纠错
tingshuo -i ./media --auto-correct --ollama-model qwen2.5

# 自动纠错 + LLM 润色（先纠错再润色）
tingshuo -i ./media --auto-correct --polish-llm --ollama-model qwen2.5

# 使用 OpenAI 兼容 API 进行自动纠错
tingshuo -i ./media --auto-correct --api-url https://api.example.com --api-key sk-xxx --api-model gpt-4o-mini
```

自动纠错保留片段边界（时间戳不变），适用于所有输出格式（SRT、LRC、MD）。

## 内容总结

TingShuo 可以在正常输出之外额外生成内容总结文件（`.summary.md`）。对于视频文件，支持通过关键帧提取和视觉 LLM 进行多模态分析。

### 纯文本总结（音频或视频）

```bash
# 使用 Ollama 总结
tingshuo -i ./media --summarize --ollama-model qwen2.5

# 使用 OpenAI 兼容 API 总结
tingshuo -i ./media --summarize --api-url https://api.example.com --api-key sk-xxx --api-model gpt-4o-mini
```

### 多模态视频总结

对于视频文件，TingShuo 使用 ffmpeg 提取关键帧，并将其与转录文本一起发送给支持视觉的 LLM 进行综合分析：

```bash
# 多模态总结，提取关键帧（默认间隔 60 秒）
tingshuo -i ./videos --summarize --api-url https://api.example.com --api-key sk-xxx --api-model gpt-4o-mini

# 自定义关键帧间隔（每 30 秒）
tingshuo -i ./videos --summarize --keyframe-interval 30 --api-url https://api.example.com --api-key sk-xxx --api-model gpt-4o-mini

# 使用 Ollama 多模态模型（如 llava、llama3.2-vision）
tingshuo -i ./videos --summarize --ollama-model llava
```

多模态总结整合了：
- 转录文本中的语音内容
- 视觉元素：幻灯片、图表、图示、演示等
- 补充语音内容的关键视觉信息

如果 LLM 不支持视觉功能，TingShuo 会自动回退到纯文本总结。

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
