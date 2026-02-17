# Enterprise Document Intelligence System

Multi-agent AI system for contract analysis, risk detection and document Q&A, built with LangGraph and Amazon Bedrock.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-purple?style=flat-square)
![AWS](https://img.shields.io/badge/Amazon-Bedrock-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

Upload any business contracts or documents. Four specialised AI agents work together to analyse them instantly:

- Answer specific questions with exact citations from source documents
- Summarise key terms and obligations in plain language
- Identify risks rated CRITICAL, HIGH, MEDIUM, or LOW
- Generate downloadable intelligence reports

This reduces a 4-hour manual contract review to under 10 seconds at a fraction of the cost.

---

## Architecture

```
User uploads PDF documents
           |
    Document Loader
           |
    Text Chunker (1000 chars, 200 overlap)
           |
    Titan Embeddings --> FAISS Vector Store
           |
    +------+----------------------------------------------+
    |              LangGraph Workflow                      |
    |                                                      |
    |   Router Agent (Coordinator)                         |
    |        |                                             |
    |   +----+------------------+------------------+      |
    |   |                       |                  |      |
    |  RAG Agent          Summarizer         Risk Analyzer |
    |  (Q&A with          (Executive         (CRITICAL /   |
    |   citations)         summary)           HIGH / LOW)  |
    |        |                                             |
    |    Finalizer (Full Report)                           |
    +------------------------------------------------------+
           |
    Downloadable Intelligence Report
```

---

## Agent Roles

| Agent | Responsibility |
|-------|---------------|
| Router | Coordinator that decides which agent handles each task |
| RAG Agent | Searches document chunks and answers questions with citations |
| Summarizer | Creates concise executive summaries of all documents |
| Risk Analyzer | Scans for risk clauses and assigns severity levels |
| Finalizer | Combines all results into a structured report |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Amazon Bedrock - Claude 3 Haiku |
| Embeddings | Amazon Bedrock - Titan Embeddings G1 |
| Agent Orchestration | LangGraph |
| RAG Pipeline | LangChain |
| Vector Store | FAISS |
| Document Loading | PyPDF + LangChain Community |
| UI | Streamlit |
| Language | Python 3.11 |
| Cloud | AWS us-east-1 |

---

## Quick Start

### Prerequisites

- Python 3.11 or higher
- AWS account with Amazon Bedrock access enabled
- Claude 3 Haiku model accessible in your AWS region

### Installation

```bash
git clone https://github.com/Negbepierre/enterprise-doc-intelligence.git
cd enterprise-doc-intelligence

python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Open .env and add your AWS credentials:

```
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
```

### Add Documents

Place PDF files in the following folder:

```
data/sample_contracts/
```

### Run

```bash
# Test Bedrock connection
python sprint_bedrock.py

# Test RAG system
python sprint_rag.py

# Launch Streamlit UI
streamlit run sprint_app_final.py
```

Open your browser at http://localhost:8501

---

## Project Structure

```
enterprise-doc-intelligence/
|-- sprint_bedrock.py        AWS Bedrock client (LLM + Embeddings)
|-- sprint_rag.py            RAG system with FAISS vector store
|-- sprint_agents.py         Multi-agent system using LangGraph
|-- sprint_app_final.py      Streamlit UI
|-- requirements.txt         Python dependencies
|-- .env.example             Environment variable template
|-- data/
    |-- sample_contracts/    Place your PDF documents here
```

---

## How It Works

### Step 1 - Document Processing

PDF files are loaded and split into overlapping chunks of 1000 characters. This allows the system to search by meaning rather than exact keywords.

### Step 2 - Semantic Indexing

Each chunk is converted into a 1536-dimensional vector using Amazon Titan Embeddings. These vectors capture the meaning of the text and are stored in a FAISS index for fast retrieval.

### Step 3 - RAG Question Answering

When you ask a question, the system finds the three most relevant chunks using semantic search, then passes them to Claude with your question. The answer is grounded in your actual documents and includes source citations.

### Step 4 - Multi-Agent Analysis

LangGraph orchestrates four specialised agents in sequence. Each agent receives the state from the previous one, ensuring the final report combines accurate answers, a readable summary, and a detailed risk assessment.

---

## Performance

| Metric | Value |
|--------|-------|
| Answer latency | 3 to 5 seconds |
| Cost per full analysis | approximately 0.04 GBP |
| Source citations | Included with every answer |

---

## Business Value

Contract review by a consultant typically takes 4 hours at 150 GBP per hour, totalling 600 GBP per contract. For an organisation processing 50 contracts per month, that is 30,000 GBP in review costs.

This system performs the same analysis in under 10 seconds at approximately 0.04 GBP per contract, representing a saving of over 29,998 GBP per month on a comparable workload.

Beyond cost, the system provides consistent analysis quality regardless of document volume, eliminates the risk of human oversight on complex clauses, and produces an auditable report for every analysis.

---

## Sample Contracts Included

Five realistic contracts are included for demonstration purposes:

1. Software Development Agreement - low risk, standard commercial terms
2. Cloud Services SLA Agreement - medium risk, auto-renewal and price escalation clauses
3. AI Consulting Services Agreement - high risk, unfavourable IP and liability terms
4. Data Processing Agreement GDPR - compliance focus, healthcare data obligations
5. Vendor Supply Agreement - critical risk, no liability cap and non-cancellable terms

---

## Security Notes

- Never commit your .env file to version control
- The .gitignore in this repository excludes .env by default
- Use IAM users with least-privilege permissions rather than root credentials
- Rotate access keys regularly
- For production deployments, use IAM roles instead of long-term access keys

---

## Author

Inegbenose Pierre - Cloud and AI Engineer

Website: https://inegbenose.xyz
GitHub: https://github.com/Negbepierre

---

## License

MIT License. See LICENSE for details.

---

Built as a portfolio project demonstrating agentic AI engineering with Amazon Bedrock and LangGraph.
