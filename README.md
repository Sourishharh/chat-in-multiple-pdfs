# 📚 Chat with Multiple PDFs

> ⚡ Chat with Multiple PDFs is a Streamlit-based application that allows users to interactively query and retrieve information from multiple PDF files. This tool uses state-of-the-art AI models for natural language understanding, making it an ideal companion for researchers, students, and professionals who need quick answers from large documents.

---

## ✨ Features

* 📂 **Upload and Process PDFs** → Upload multiple PDFs for text extraction & indexing.
* 🔎 **Semantic Search** → Uses **Google Generative AI Embeddings** with FAISS.
* 🤖 **Conversational AI** → Ask questions in natural language, get answers from your PDFs.
* 🚨 **Error Handling** → Detects invalid files, empty queries, and shows feedback.

---

## 🛠️ Tech Stack

| Component             | Technology                                       |
| --------------------- | ------------------------------------------------ |
| 🎨 **Frontend**       | Streamlit (Interactive UI)                       |
| ⚙️ **Backend**        | PyPDF2 (Text Extraction), LangChain (Processing) |
| 🧾 **Indexing**       | FAISS (Semantic Indexing)                        |
| 🧠 **AI Models**      | Google Generative AI (Embeddings + Chatbot)      |
| 🔑 **Env Management** | python-dotenv                                    |

---

## 🚀 Installation

```bash
# 1. Clone the Repository
git clone https://github.com/Sourishharh/chat-in-multiple-pdfs.git

# 2. Create & Activate Virtual Environment
python -m venv venv

# For Mac/Linux
source venv/bin/activate

# For Windows
venv\Scripts\activate

# 3. Install Dependencies
pip install -r requirements.txt
```

👉 Create a `.env` file and add your API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

---

## ▶️ Usage

1. **Run the App**

   ```bash
   streamlit run Start.py
   ```

2. **Upload PDFs**

   * 📥 Use sidebar to upload files.
   * ⚡ Click `Submit & Process` to extract and index.

3. **Ask Questions**

   * ❓ Type your query in the input box.
   * 💬 View answers + conversation history.

---

## 📂 File Structure

```
chat-with-pdfs/
├── app.py                 
├── requirements.txt       
├── .env                   
├── README.md              
└── ... (other files)
```

---

## 📜 License

📝 This project is licensed under the **MIT License**.
See the [LICENSE](./LICENSE) file for details.

---

## 🙌 Acknowledgments

* 🎨 **Streamlit** → For the interactive UI framework.
* 🔗 **LangChain** → For managing complex chains of processing.
* 🤖 **Google Generative AI** → For embeddings & conversational AI.
* ⚡ **FAISS** → For efficient semantic indexing.
