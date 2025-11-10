# 🤖 Q&A Multi-Model Chatbot using OpenRouter

## 🧠 Overview
**Q&A Multi-Model Chatbot** is an advanced **Streamlit-based application** designed to compare responses from **multiple Large Language Models (LLMs)** via the **OpenRouter API**.  
It supports **contextual memory**, **semantic search (FAISS-like)**, and **side-by-side model comparison**, making it ideal for analyzing response quality, accuracy, and reasoning across AI models.

---

## 🚀 Key Features

- 🔀 **Dual Modes**
  - **Single Model Chat** – Chat with one selected LLM interactively.  
  - **Comparison Mode** – View and compare responses from two models simultaneously.

- 🧩 **Multi-Model Support**
  - **GPT-4o-mini (OpenAI)**
  - **GPT-4o (OpenAI)**
  - **LLaMA-3-8B Instruct (Meta)**
  - **LLaMA-3-13B Instruct (Meta)**

- 🧠 **Contextual Memory**
  - Stores all user queries and responses.
  - Uses **SentenceTransformer embeddings** (`all-MiniLM-L6-v2`) for semantic similarity.
  - Retrieves relevant past contexts using **cosine similarity**, functioning like a **FAISS-style vector store**.

- 💬 **Modern Chat UI**
  - Built with **Streamlit**.
  - Model selector, mode toggle, and clean side-by-side layout.
  - Persistent memory and contextual display.
  - Bottom input box for natural chat flow.

- 🔒 **Secure API Handling**
  - `.env` file used to store **OpenRouter API keys** safely.

---

## 🛠️ Tech Stack

| Component | Technology Used |
|------------|----------------|
| **UI Framework** | Streamlit |
| **Backend Language** | Python |
| **API Gateway** | OpenRouter |
| **Embeddings Model** | `all-MiniLM-L6-v2` (SentenceTransformers) |
| **Vector Search** | Cosine Similarity (FAISS-like memory) |
| **Env Handling** | python-dotenv |
| **HTTP Requests** | requests |
| **LLMs Supported** | OpenAI GPT-4o, GPT-4o-mini, LLaMA-3 8B, LLaMA-3 13B |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/qa-multimodel-chatbot.git
cd qa-multimodel-chatbot
```

### 2️⃣ Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate       # For macOS/Linux
venv\Scripts\activate          # For Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables
Create a `.env` file in the project root directory:
```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

You can get your API key from [https://openrouter.ai](https://openrouter.ai).

### 5️⃣ Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at [http://localhost:8501](http://localhost:8501).

---

## 📁 Project Structure
```
📦 Q&A-MultiModel-Chatbot
├── app.py
├── requirements.txt
├── config.py
├── utils/
├── models/
├── .env
├── screenshots/
│   ├── openai_models/
│   ├── llama_models/
│   └── comparison/
└── README.md
```

---

## 🧮 How Contextual Memory Works
1. User queries are converted to embeddings using **SentenceTransformer** (`all-MiniLM-L6-v2`).  
2. These vectors are stored in memory along with corresponding responses.  
3. On each new query, cosine similarity identifies the most relevant past entries.  
4. Retrieved contexts are included in the next model prompt for improved coherence and accuracy.  

> This architecture mimics **FAISS (Facebook AI Similarity Search)** but is implemented in-memory for simplicity and portability.

---

## 🧠 Model Behavior Customization

### 🧩 GPT-4o / GPT-4o-mini
- Provides structured, well-organized answers.
- Uses markdown formatting and factual style.

### 🦙 LLaMA-3 (8B / 13B)
- Concise, logic-driven responses.
- Clearly states uncertainty when unsure.

---

## 🎨 User Interface Highlights
- Two-column layout for model comparison.
- Prompt box positioned at the bottom.
- Responsive, clean, and minimal dark UI.

---

## 📸 Screenshots

All screenshots are stored in the **`/screenshots/`** folder.

| Folder | Description |
|---------|-------------|
| `/screenshots/openai_models/` | GPT-4o and GPT-4o-mini outputs |
| `/screenshots/llama_models/` | LLaMA-3 model outputs |
| `/screenshots/comparison/` | Side-by-side comparisons |

---

## 🧾 Example Usage

**Prompt:**  
> “What is the Model Context Protocol (MCP)?”

**🧠 GPT-4o-mini (OpenAI):**
- Structured, academic-style explanation with headers.

**🦙 LLaMA-3-8B:**
- Concise bullet-point summary with uncertainty notes.

---

## 🔒 Environment Variables

| Variable | Description |
|-----------|-------------|
| `OPENROUTER_API_KEY` | API key from OpenRouter |

---

## 📦 Dependencies

```
streamlit
requests
sentence-transformers
scikit-learn
python-dotenv
numpy
```

---

## 👨‍💻 Author

**Amogh**  
Built with ❤️ using Streamlit, OpenRouter, and SentenceTransformers.

---

## 🪪 License
Licensed under the **MIT License**.

---
