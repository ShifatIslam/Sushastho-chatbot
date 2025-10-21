
#  Susastho.AI Chatbot

**Susastho.AI** is an intelligent Bengali-language chatbot designed to support adolescents by providing accurate, accessible, and empathetic answers to questions related to **sexual, reproductive, and mental health**.  
Built using advanced Natural Language Processing (NLP) and different large language models, the chatbot enables young users to freely seek reliable information and guidance in a culturally sensitive and linguistically inclusive manner.  

This project combines AI-driven language understanding with a secure, scalable Python web backend to deliver a responsive and trustworthy health education platform.

---

##  Features
-  NLP-powered question-answering system.
-  Context-aware response generation using vector databases.
-  Modular backend with scalable architecture.
-  Ready for cloud deployment (Docker-compatible).
-  Data-driven context management with CSV and FAISS index support.

---

## Project Structure
```bash
Susastho.AI-main/
│
├── Susastho-AI-backend/
│   ├── push-win.bat               # Script for quick Git push on Windows
│   └── src/
│       ├── app.py                 # Main backend entry point
│       ├── requirements.txt       # Python dependencies
│       ├── run.sh                 # Script to start app (Linux/Mac)
│       ├── docx_context.csv       # Document context for NLP
│       ├── duplicates.csv         # Preprocessed data file
│       ├── nlp_api/
│       │   ├── app.py             # NLP API logic
│       │   ├── vectordb.py        # Vector database integration
│       │   ├── data/              # Knowledge or embedding data
│       │   └── prompts/           # AI prompt templates
│       └── script                 # Helper file or executable
│
└── README.md                      # Documentation
```

---

##  Installation and Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/Susastho.AI.git
cd Susastho.AI-main/Susastho-AI-backend/src
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

### Run on Windows:
```bash
python app.py
```

### Run on Linux/Mac:
```bash
bash run.sh
```

The app will start on:
```
http://localhost:5000
```
(or the port defined in your app configuration)

---

##  Core Modules
| Module | Description |
|--------|--------------|
| **nlp_api/app.py** | Defines API endpoints and request handling |
| **nlp_api/vectordb.py** | Handles FAISS or vector-based data retrieval |
| **docx_context.csv** | Stores reference context for healthcare dialogues |
| **prompts/** | Contains structured AI prompt templates |

---

##  Technology Stack
- **Backend:** Python (Flask/FastAPI)
- **AI/NLP:** FAISS, Sentence Transformers, Hugging Face
- **Database:** VectorDB / Local Storage
- **Deployment:** Docker, Shell scripts
- **Version Control:** GitHub

---

##  Dependencies
Listed in `requirements.txt`. Install using:
```bash
pip install -r requirements.txt
```

---

##  Contributing
1. Fork the repository  
2. Create your feature branch (`git checkout -b feature-name`)  
3. Commit your changes (`git commit -m 'Add feature name'`)  
4. Push to the branch (`git push origin feature-name`)  
5. Open a Pull Request  

---


