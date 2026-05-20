# RAG Document Chat

Eine Streamlit-basierte Web-App mit FastAPI-Backend für PDF-basiertes Chatten.

## Funktionen

- Mehrere PDFs gleichzeitig hochladen
- Chunking pro Seite mit Satzüberlappung
- Embeddings über TF-IDF + SVD und Vektorsuche mit FAISS
- Chat-Antworten mit Quellenangabe inklusive Dokumentname und Seite
- 5 hartcodierte Benchmark-Fragen mit erwarteten Antworten und Retrieval-Score im UI
- Laufbar mit Docker Compose

## Architektur

- Backend: FastAPI unter `/chat`, `/documents/upload`, `/benchmark`
- Frontend: Streamlit mit Chat-Ansicht und Benchmark-Tabelle
- Index: pro Konversation im Speicher gehalten; Uploads werden beim Indexieren zusammengeführt

## Chunking-Strategie

Jede PDF-Seite wird zunächst mit PyMuPDF in Markdown umgewandelt. Danach wird der Markdown-Text mit einem Markdown-sensitiven Splitter in Chunks von ungefähr 2200 Zeichen mit Overlap zerlegt.

Das hat zwei Effekte auf die Retrieval-Qualität:

- Besserer Recall bei Fragen, deren Antwort in Überschriften, Listen oder Absätzen steckt.
- Mehr Strukturtreue, weil Markdown-Elemente beim Splitten erhalten bleiben.

Die Seitengrenze bleibt erhalten, damit die Antwort sauber mit einer Quelle zitiert werden kann.

Benötigt werden dafür `pymupdf` und `langchain-text-splitters`.

## Lokaler Start

Backend:

```bash
uvicorn backend.main:app --reload --port 8000
```

Frontend:

```bash
streamlit run app.py
```

## Start mit Docker Compose

```bash
docker compose up --build
```

Dann:

- Frontend: http://localhost:8501
- Backend: http://localhost:8000

## Benchmark

Die Benchmark-Funktion nutzt ein hartcodiertes Demo-Dokument mit fünf Fragen und erwarteten Antworten. Der sichtbare Retrieval-Score ist eine kombinierte Kennzahl aus Keyword-Recall, Antwortüberlappung und lexikalischer Ähnlichkeit zwischen erwarteter Antwort und gefundenem Chunk.

## Erweiterungen

- Streaming-Antworten lassen sich als nächster Schritt ergänzen.
- Re-Ranking kann vor dem Antwortbau auf die Top-Chunks angewendet werden.
- Für echtes Deployment kann der Stack unverändert auf eine öffentliche Plattform gesetzt werden.