import os
import uuid
from dotenv import load_dotenv

import pandas as pd
import requests
import streamlit as st

load_dotenv()  # Lädt die .env Datei


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def _api(path: str) -> str:
	return f"{BACKEND_URL.rstrip('/')}{path}"


def _ensure_session_id() -> str:
	if "conversation_id" not in st.session_state:
		st.session_state.conversation_id = str(uuid.uuid4())
	return st.session_state.conversation_id


def _upload_documents(conversation_id: str, files) -> dict:
	multipart = [
		("files", (file.name, file.getvalue(), "application/pdf"))
		for file in files
	]
	response = requests.post(
		_api("/documents/upload"),
		data={"conversation_id": conversation_id},
		files=multipart,
		timeout=120,
	)
	response.raise_for_status()
	return response.json()


def _ask_chat(conversation_id: str, question: str) -> dict:
	response = requests.post(
		_api("/chat"),
		json={"conversation_id": conversation_id, "question": question},
		timeout=120,
	)
	response.raise_for_status()
	return response.json()


def _run_benchmark() -> dict:
	response = requests.get(_api("/benchmark"), timeout=1000)
	response.raise_for_status()
	print("Benchmark response: ", response.json())
	return response.json()


st.set_page_config(page_title="RAG Dokument Chat", page_icon="📄", layout="wide")

conversation_id = _ensure_session_id()
st.title("RAG Dokument Chat")
st.caption("PDFs hochladen, indexieren und direkt im Chat mit Quellenangabe abfragen.")

with st.sidebar:
	st.subheader("Dokumente")
	uploaded_files = st.file_uploader(
		"PDFs hochladen",
		type=["pdf"],
		accept_multiple_files=True,
	)
	if st.button("Dokumente indexieren", use_container_width=True, disabled=not uploaded_files):
		try:
			with st.spinner("Dokumente werden verarbeitet..."):
				result = _upload_documents(conversation_id, uploaded_files)
			st.success(f"{result['document_count']} Dokument(e) mit {result['chunk_count']} Chunks indexiert.")
			st.session_state.index_summary = result
		except Exception as exc:  # pragma: no cover - UI feedback
			st.error(f"Upload fehlgeschlagen: {exc}")

	if st.button("Benchmark ausführen", use_container_width=True):
		try:
			with st.spinner("Testfragen laufen..."):
				st.session_state.benchmark = _run_benchmark()
		except Exception as exc:  # pragma: no cover - UI feedback
			st.error(f"Benchmark fehlgeschlagen: {exc}")

	if "index_summary" in st.session_state:
		summary = st.session_state.index_summary
		st.metric("Dokumente", summary["document_count"])
		st.metric("Chunks", summary["chunk_count"])


col_chat, col_eval = st.columns([1.35, 1])

with col_chat:
	st.subheader("Chat")
	if "messages" not in st.session_state:
		st.session_state.messages = []

	for message in st.session_state.messages:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])
			if message.get("citations"):
				with st.expander("Quellen"):
					for citation in message["citations"]:
						st.markdown(
							f"**{citation['document_name']}** · Seite {citation['page_number']} · Chunk {citation['chunk_index']}"
						)
						st.write(citation["text_excerpt"])

	question = st.chat_input("Stelle eine Frage zu deinen PDFs")
	if question:
		st.session_state.messages.append({"role": "user", "content": question})
		with st.chat_message("user"):
			st.markdown(question)

		try:
			with st.chat_message("assistant"):
				with st.spinner("Antwort wird generiert..."):
					answer = _ask_chat(conversation_id, question)
				st.markdown(answer["answer"])
				if answer.get("citations"):
					with st.expander("Quellen"):
						st.markdown(f"**Gefundene Chunks insgesamt:** {len(answer['citations'])}")
						for citation in answer["citations"]:
							st.markdown(
								f"""
								<p style="font-size:20px; margin-bottom:4px; color:red;">
									<b>{citation['document_name']}</b> ·
									Seite {citation['page_number']} ·
									Chunk {citation['chunk_index']}
								</p>
								""",
								unsafe_allow_html=True
							)
							st.write(citation["text_excerpt"])
			st.session_state.messages.append(
				{
					"role": "assistant",
					"content": answer["answer"],
					"citations": answer.get("citations", []),
				}
			)
		except Exception as exc:  # pragma: no cover - UI feedback
			st.error(f"Chat fehlgeschlagen: {exc}")


with col_eval:
	st.subheader("Testfragen")
	benchmark = st.session_state.get("benchmark")
	if benchmark:
		scores = benchmark["average_scores"]
		col1, col2 = st.columns(2)
		col3, col4 = st.columns(2)
		col1.metric("Faithfulness", scores["faithfulness"])
		col2.metric("Answer Relevancy", scores["answer_relevancy"])
		col3.metric("Context Precision", scores["context_precision"])
		col4.metric("Context Recall", scores["context_recall"])
		rows = []
		print("Benchmark items: ", benchmark["items"])
		for item in benchmark["items"]:
			rows.append(
				{
					"Frage": item["user_input"],
					"Antwort": item["response"],
					"Expected Answer": item["reference"],
					"Faithfulness": item.get("faithfulness"),
					"Answer Relevancy": item.get("answer_relevancy"),
					"Context Precision": item.get("context_precision"),
					"Context Recall": item.get("context_recall"),
				}
			)

		st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
	else:
		st.info("Starte den Benchmark, um die 5 hartcodierten Testfragen und den Retrieval-Score anzuzeigen.")
