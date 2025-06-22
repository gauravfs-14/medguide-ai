# 🧠 MedGuide AI — A Private AI Agent for Understanding Health Reports and Medications

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.46.0+-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.26+-green.svg)](https://langchain.dev/)
[![Model Context Protocol](https://img.shields.io/badge/MCP-Enabled-purple.svg)](https://modelcontextprotocol.io/)

## 🩺 **What it Does**

**MedGuide AI** is an intelligent medical assistant that helps everyday people understand their health information through AI-powered document analysis and conversation. It provides:

- **Lab test interpretation** - Understand blood work, thyroid panels, and other diagnostic results
- **Medication guidance** - Learn about prescriptions, dosages, and side effects  
- **Medical document analysis** - Parse reports, visit summaries, and clinical notes
- **Health education** - Get explanations for medical terminology and concepts

All processing happens **locally** with **privacy-first** design and **offline capabilities**.

---

## ⚠️ **Problem it Solves**

Healthcare information is often confusing and inaccessible:

- **Lab results** like "TSH = 4.9 mIU/L" leave patients confused
- **Medication instructions** like "Losartan 50mg BID" aren't clear to non-medical people  
- **Clinical documents** are filled with jargon that's hard to understand
- **Online searches** lead to information overload and misinterpretation

MedGuide AI bridges this gap by providing personalized, contextual explanations in plain language.

---

## 🏗️ **Architecture Overview**

MedGuide AI uses a modern **Model Context Protocol (MCP)** architecture for modular, extensible AI capabilities:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │◄──►│   LangGraph      │◄──►│  MCP Servers    │
│   Frontend      │    │   Agent          │    │                 │
│                 │    │                  │    │ • Memory        │
│ • Chat UI       │    │ • Tool Calling   │    │ • Vector Store  │
│ • File Upload   │    │ • Reasoning      │    │ • PDF Parser    │
│ • Document      │    │ • Context Mgmt   │    │ • Embeddings    │
│   Visualization │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **Core Components**

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit | Interactive chat interface with file upload |
| **AI Agent** | LangGraph + Google Gemini | Orchestrates reasoning and tool usage |
| **Memory System** | MCP Memory Server | Persistent conversation and document history |
| **Vector Database** | ChromaDB via MCP | Document embedding and semantic search |
| **PDF Processing** | PyMuPDF + Custom MCP Server | Extract and vectorize medical documents |
| **Embeddings** | Ollama (nomic-embed-text) | Local text embeddings for RAG |

---

## 🚀 **Key Features**

### **Intelligent Document Processing**
- **Automatic PDF analysis** - Upload documents and get instant insights
- **Smart vectorization** - Documents are automatically embedded and stored
- **Memory integration** - Past documents are remembered across sessions
- **Multi-format support** - PDF, images, text files, and Word documents

### **Conversational AI**
- **Context-aware responses** - References your previous documents and medical history
- **Reasoning transparency** - See the AI's thought process with expandable reasoning blocks
- **Tool integration** - Seamlessly uses multiple AI tools for comprehensive analysis
- **Streaming responses** - Real-time response generation for better UX

### **Privacy & Security**
- **Local processing** - All document analysis happens on your machine
- **No data sharing** - Your medical information never leaves your environment
- **Offline capable** - Works without internet once models are downloaded
- **Encrypted storage** - ChromaDB provides secure document storage

---

## 💡 **How It Works**

1. **Document Upload** 📄
   - Upload lab reports, prescriptions, or medical documents
   - AI automatically extracts filename and creates document collections
   - PDF content is vectorized and stored in ChromaDB

2. **Intelligent Analysis** 🧠  
   - LangGraph agent coordinates multiple MCP tools
   - Memory system recalls relevant past medical information
   - Vector search finds related document sections

3. **Expert Explanations** 💬
   - Gemini 2.0 Flash provides medical explanations in plain language
   - Responses reference specific document content and medical history
   - Suggestions for follow-up questions and next steps

4. **Persistent Memory** 💾
   - All interactions and documents are remembered
   - Cross-session context maintains continuity
   - Build comprehensive medical history over time

---

## 🛠️ **Technology Stack**

### **AI & ML**
- **[LangGraph](https://langchain.dev/langgraph)** - Agent orchestration and tool calling
- **[Google Gemini 2.0 Flash](https://deepmind.google/technologies/gemini/)** - Large language model for reasoning
- **[Ollama](https://ollama.ai/)** - Local embeddings (nomic-embed-text)
- **[ChromaDB](https://www.trychroma.com/)** - Vector database for document storage

### **Document Processing**
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** - PDF text extraction
- **[LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)** - Intelligent text chunking

### **Framework & Infrastructure**
- **[Model Context Protocol (MCP)](https://modelcontextprotocol.io/)** - Modular AI tool architecture
- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[UV](https://docs.astral.sh/uv/)** - Fast Python package management
- **[Docker](https://www.docker.com/)** - Containerized deployment

---

## 📋 **Prerequisites**

### **Required Software**
- **Python 3.11+** - Core runtime
- **Node.js 22+** - For MCP memory server
- **Ollama** - Local embedding model hosting
- **UV** - Python package manager

### **Required Models**
```bash
# Install Ollama embedding model
ollama pull nomic-embed-text:latest
```

### **Environment Variables**
```bash
# Google Gemini API (required)
GOOGLE_API_KEY=your_google_api_key_here

# Optional customization
CHROMA_DB_PATH=./data  # ChromaDB storage location
MEMORY_FILE_PATH=./memory.json  # Memory persistence file
```

---

## 🚀 **Quick Start**

### **1. Clone and Setup**
```bash
git clone https://github.com/gauravfs-14/medguide-ai.git
cd medguide-ai

# Install dependencies with UV
uv sync
```

### **2. Configure Environment**
```bash
# Copy environment template
cp .env.example .env

# Add your Google API key
echo "GOOGLE_API_KEY=your_api_key_here" >> .env
```

### **3. Install Ollama Model**
```bash
# Install embedding model
ollama pull nomic-embed-text:latest
```

### **4. Run the Application**
```bash
# Start the Streamlit app
uv run streamlit run mcp-client/client.py
```

Visit `http://localhost:8501` to access the application.

---

## 🐳 **Docker Deployment**

### **Build and Run**
```bash
# Build Docker image
docker build -t medguide-ai .

# Run container
docker run -p 8501:8501 \
  -e GOOGLE_API_KEY=your_api_key \
  -v $(pwd)/data:/medguide-ai/data \
  medguide-ai
```

### **Docker Compose** (Recommended)
```yaml
version: '3.8'
services:
  medguide-ai:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    volumes:
      - ./data:/medguide-ai/data
      - ./memory.json:/medguide-ai/memory.json
```

---

## 📚 **Usage Examples**

### **Lab Report Analysis**
```
User: "Can you analyze my recent blood work?"
[Upload: blood_test_results.pdf]

AI: "I've analyzed your blood test results. Here are the key findings:

🔬 **Complete Blood Count (CBC)**
- White Blood Cells: 6.2 K/μL (Normal: 4.0-10.0)
- Red Blood Cells: 4.8 M/μL (Normal: 4.2-5.4) 
- Hemoglobin: 14.2 g/dL (Normal: 12.0-15.5)

✅ **Overall Assessment**: Your blood counts are within normal ranges, indicating good general health..."
```

### **Medication Questions**
```
User: "What does 'Metformin 500mg BID' mean?"

AI: "**Metformin 500mg BID** means:

💊 **Medication**: Metformin - a diabetes medication that helps control blood sugar
📏 **Dosage**: 500 milligrams per dose  
⏰ **Frequency**: BID = 'bis in die' (twice daily)

**How to take**: Take one 500mg tablet twice per day, typically with meals to reduce stomach upset..."
```

---

## 🔧 **Configuration**

### **MCP Server Configuration**
The application uses `mcp-client/mcp_config.json` to configure AI tools:

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"],
      "transport": "stdio",
      "env": {
        "MEMORY_FILE_PATH": "memory.json"
      }
    },
    "vectorize_mcp": {
      "command": "uv",
      "args": ["--directory", "./mcp-servers", "run", "vectorize_mcp.py"],
      "transport": "stdio"
    }
  }
}
```

### **Custom MCP Servers**
Add your own MCP servers by:
1. Creating a new server in `mcp-servers/`
2. Adding configuration to `mcp_config.json`
3. Implementing tools using FastMCP or similar

---

## 🤝 **Contributing**

We welcome contributions! Here's how to get started:

### **Development Setup**
```bash
# Clone repository
git clone https://github.com/gauravfs-14/medguide-ai.git
cd medguide-ai

# Install development dependencies  
uv sync --dev

# Install pre-commit hooks
pre-commit install
```

### **Project Structure**
```
medguide-ai/
├── mcp-client/           # Streamlit frontend and agent
│   ├── client.py         # Main Streamlit application
│   ├── graph.py          # LangGraph agent definition
│   └── mcp_config.json   # MCP server configuration
├── mcp-servers/          # Custom MCP tool servers
│   └── vectorize_mcp.py  # PDF vectorization server
├── data/                 # ChromaDB and document storage
├── Dockerfile            # Container configuration
├── pyproject.toml        # Python dependencies
└── README.md             # This file
```

### **Adding New Features**
1. **New MCP Tools** - Add servers in `mcp-servers/`
2. **UI Improvements** - Modify `mcp-client/client.py`
3. **Agent Logic** - Update `mcp-client/graph.py`
4. **Documentation** - Update README and docstrings

---

## 🔮 **Roadmap**

### **Current Features** ✅
- [x] PDF document processing and vectorization
- [x] Intelligent chat interface with memory
- [x] MCP-based modular architecture  
- [x] Local embedding and vector storage
- [x] Docker containerization

### **Upcoming Features** 🚧
- [ ] **OCR Support** - Process scanned documents and images
- [ ] **Drug Database Integration** - Comprehensive medication information
- [ ] **Calendar Integration** - Medication reminders and appointments
- [ ] **Multi-language Support** - Support for non-English medical documents
- [ ] **Advanced Visualizations** - Charts and graphs for lab trends
- [ ] **Mobile App** - React Native or Flutter mobile interface

### **Future Enhancements** 🎯
- [ ] **FHIR Integration** - Connect with electronic health records
- [ ] **Clinical Decision Support** - AI-powered health recommendations
- [ ] **Telemedicine Integration** - Connect with healthcare providers
- [ ] **Wearable Data** - Integration with fitness trackers and health devices

---

## ⚠️ **Important Medical Disclaimer**

> **MedGuide AI provides educational information only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult your healthcare provider for medical decisions. In case of medical emergencies, contact emergency services immediately.**

---

## 📄 **License**

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙋‍♂️ **Support**

- **Documentation**: [Wiki](https://github.com/gauravfs-14/medguide-ai/wiki)
- **Issues**: [GitHub Issues](https://github.com/gauravfs-14/medguide-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/gauravfs-14/medguide-ai/discussions)

---

## 🌟 **Acknowledgments**

- **[Model Context Protocol](https://modelcontextprotocol.io/)** - For the modular AI architecture
- **[LangChain](https://langchain.dev/)** - For the AI agent framework
- **[Google](https://deepmind.google/technologies/gemini/)** - For the Gemini language model
- **[Anthropic](https://www.anthropic.com/)** - For MCP specification and inspiration

---

<div align="center">

**Made with ❤️ for better healthcare accessibility**

[⭐ Star this repo](https://github.com/gauravfs-14/medguide-ai) | [🐛 Report Bug](https://github.com/gauravfs-14/medguide-ai/issues) | [💡 Request Feature](https://github.com/gauravfs-14/medguide-ai/issues)

</div>
