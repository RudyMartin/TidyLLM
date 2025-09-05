# 🤖 Simple RAG Demo

Upload documents and chat with AI.

## 🚀 Quick Start

```bash
python start_simple_demo.py
```

That's it! The launcher will:
1. ✅ Create a virtual environment automatically
2. ✅ Install all required dependencies
3. ✅ Launch the demo on port 8555
4. ✅ Open your browser to the application

## 📋 What You'll See

1. **Upload Section**: Drag and drop up to 5 documents (PDF or TXT files)
2. **Process Button**: Click to process your documents
3. **Chat Interface**: Ask questions about your documents
4. **AI Responses**: Get intelligent answers based on your documents

## 🎯 How to Use

### Step 1: Launch
```bash
python start_simple_demo.py
```

### Step 2: Upload Documents
- Click "Browse files" or drag and drop documents
- Select up to 5 PDF or TXT files
- Click "🔄 Process Documents"

### Step 3: Start Chatting
- Type your question in the chatbox
- Click "🚀 Send"
- Get AI-powered answers based on your documents

## 🔧 Options

### Custom Port
```bash
python start_simple_demo.py --port 8560
```

### Reinstall Dependencies
```bash
python start_simple_demo.py --reinstall
```

## 📚 Detailed Guide

For complete instructions, troubleshooting, and advanced features, see:
**[Complete User Guide](docs/user-guide/README.md)**

## 🛠️ Technical Details

- **Requirements**: Python 3.7+
- **Dependencies**: Installed automatically
- **Virtual Environment**: Created automatically
- **AI Gateway**: ZLLM (localhost:11434)

## 🚨 Troubleshooting

### "Python not found"
- Make sure Python 3.7+ is installed

### "Port already in use"
```bash
python start_simple_demo.py --port 8560
```

### "Dependencies failed"
```bash
python start_simple_demo.py --reinstall
```

For more help, see the [Complete User Guide](docs/user-guide/README.md).
