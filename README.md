# 🧠 MedGuide AI — A Private AI Agent for Understanding Health Reports and Medications

## 🩺 **What it Does**

**MedGuide AI** helps everyday people understand:

* Lab test results (e.g., blood work, thyroid reports)
* Prescription medications (what they are, how to take them, side effects)
* Medical jargon in doctor's notes or visit summaries

And it does all this **locally**, ensuring **privacy** and **offline accessibility**.

---

## ⚠️ Problem it Solves

Most people:

* Get confused by lab results like: “TSH = 4.9 mIU/L”
* Don’t know how to interpret medications like “Losartan 50mg BID”
* Avoid reading clinical documents because they’re filled with jargon

They Google everything, get overwhelmed, and sometimes misinterpret crucial information.

---

### 💡 How It Works

1. User uploads a PDF or photo of a lab report or prescription

2. The agent extracts structured data from the document

3. It uses a **local LLM** (via Ollama) + RAG (medical glossary, drugs DB)

4. Outputs an easy-to-understand explanation like:

   > “Your TSH level is slightly high. This may suggest underactive thyroid (hypothyroidism). You may want to consult your doctor.”

   or:

   > “Losartan is a medication used to lower blood pressure. ‘BID’ means twice daily.”

5. Optionally, the agent can:

   * Set a **reminder to take meds**
   * Generate a **summary for a family member**
   * Offer a **list of follow-up questions** to ask the doctor

---

## 🔧 Stack Overview

| Feature          | Tool                                                |
| ---------------- | --------------------------------------------------- |
| OCR (for photos) | `pytesseract` or `easyocr`                          |
| PDF parsing      | `pdfplumber`, `PyMuPDF`                             |
| LLM              | Ollama (LLaMA 3)                                    |
| RAG              | Local FAISS + Drug DB + Mayo Clinic scraped content |
| Interface        | React frontend or mobile-friendly webapp            |
| Tool calling     | For reminders, calendar, note generation            |

---

## Tasks

- [ ] Find out existing MCP servers for different tasks.
- [ ] Build an MCP client that can handle tool calling and LLM interactions (streamlit)
- [ ] Find out a database for drug information.
- [ ] Implement PDF parsing and or OCR for document uploads.
- [ ] Embedding and RAG setup for medical glossary and drug info, and report parsing.

### MCP Server Lists

- Chroma DB MCP Server for RAG. (https://github.com/chroma-core/chroma-mcp)
- Embedding MCP Server for medical glossary and drug info.
- OCR, PDF parsing MCP Server for document uploads.
- Calendar and reminder MCP Server for scheduling and reminders.
- Note generation MCP Server for generating summaries and follow-up questions.
