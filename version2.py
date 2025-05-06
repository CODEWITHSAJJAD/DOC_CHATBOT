import json
import hashlib
import sqlite3
import re
import faiss
import numpy as np
from tkinter import *
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from datetime import datetime
from tkinter import scrolledtext
import threading
import os
import io
import PyPDF2
from queue import Queue
import fitz
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import docx
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor
import time
from ttkthemes import ThemedStyle
import customtkinter as ctk
from PIL import ImageEnhance, ImageDraw
import colorsys
import mammoth  # For DOCX processing
import chardet  # For text file encoding detection
import pytesseract  # For OCR
from pdf2image import convert_from_path  # For converting PDF to images for OCR
import tkinter as tk
import sys
import shutil

# Configure Tesseract path (you need to install Tesseract-OCR and set this path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Download NLTK resources
nltk.data.path.append(r'C:\Users\SUQOON\AppData\Roaming\nltk_data')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
import en_core_web_lg
nlp = en_core_web_lg.load()  # CORRECT: this loads the model

# Initialize NLP components with increased text length limit
nlp.max_length = 10000000  # Increased from 5,000,000

# Load better transformer models
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
qa_model_name = "deepset/roberta-large-squad2"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

# Initialize T5 model for answer generation
# Add legacy=False to silence warning
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base', legacy=False)
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Move models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
qa_model = qa_model.to(device)
t5_model = t5_model.to(device)

# Application paths
APP_DATA = "app_data"
DB_FILE = os.path.join(APP_DATA, "docqna.db")
CHATS_DIR = os.path.join(APP_DATA, "chats")
CACHE_DIR = os.path.join(APP_DATA, "cache")
PROFILES_DIR = os.path.join(APP_DATA, "profiles")
os.makedirs(PROFILES_DIR, exist_ok=True)

# Create directories if they don't exist
for directory in [APP_DATA, CHATS_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)


# Initialize database
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            email TEXT,
            contact TEXT,
            created_at TEXT NOT NULL,
            last_login TEXT,
            profile_image TEXT
        )
        ''')

    # Create documents table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        filename TEXT NOT NULL,
        filepath TEXT NOT NULL,
        uploaded_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')

    # Create chats table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        document_id INTEGER,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id),
        FOREIGN KEY (document_id) REFERENCES documents (id)
    )
    ''')

    # Create messages table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        is_relevant INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (chat_id) REFERENCES chats (id)
    )
    ''')

    # Create sources table (for answer references)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id INTEGER NOT NULL,
        page_number INTEGER NOT NULL,
        text TEXT NOT NULL,
        start_char INTEGER NOT NULL,
        end_char INTEGER NOT NULL,
        FOREIGN KEY (message_id) REFERENCES messages (id)
    )
    ''')

    conn.commit()
    conn.close()


init_db()
class SonnerToast:
    def __init__(self, root, message, buttons=None, progressbar=None, variant="info"):
        self.root = root
        self.message = message
        self.buttons = buttons or []
        self.progressbar = progressbar or []
        self.variant = variant

        # Color and icon scheme
        variants = {
            "success": {
                "bg": "#e6fbe6", "border": "#4caf50", "text": "#222", "icon": "\u2714", "icon_bg": "#4caf50"
            },
            "info": {
                "bg": "#eaf3ff", "border": "#2196f3", "text": "#222", "icon": "\u2139", "icon_bg": "#2196f3"
            },
            "warning": {
                "bg": "#fff8e1", "border": "#ff9800", "text": "#222", "icon": "\u26A0", "icon_bg": "#ff9800"
            },
            "error": {
                "bg": "#ffeaea", "border": "#f44336", "text": "#222", "icon": "\u2716", "icon_bg": "#f44336"
            },
        }
        style = variants.get(self.variant, variants["info"])
        bg_color = style["bg"]
        border_color = style["border"]
        text_color = style["text"]
        icon_char = style["icon"]
        icon_bg = style["icon_bg"]

        base_width = 400
        if self.buttons:
            width = base_width + 80 * len(self.buttons)
        else:
            width = base_width
        height = 56
        self.toast = tk.Toplevel(root)
        self.toast.overrideredirect(True)
        self.toast.geometry(f"{width}x{height}")
        self.toast.attributes("-alpha", 0.0)
        self.toast.attributes("-topmost", True)
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        self.toast.geometry(f"+{screen_width - width - 20}+{screen_height - height - 60}")
        try:
            self.toast.iconbitmap(resource_path("logo.ico"))
        except Exception as e:
            print(f"Error setting toast icon: {e}")

        self.canvas = tk.Canvas(self.toast, bg=bg_color, highlightthickness=0, bd=0, width=width, height=height)
        self.canvas.pack(fill="both", expand=True)
        # Very rounded (radius 40)
        self.canvas.create_rounded_rect(2, 2, width-2, height-2, 40, fill=bg_color, outline=border_color, width=2)

        # Row frame for icon, message, close button
        row_frame = tk.Frame(self.canvas, bg=bg_color)
        row_frame.pack_propagate(False)
        row_frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Icon (circle background)
        icon_frame = tk.Frame(row_frame, bg=bg_color)
        icon_canvas = tk.Canvas(icon_frame, width=32, height=32, bg=bg_color, highlightthickness=0, bd=0)
        icon_canvas.create_oval(2, 2, 30, 30, fill=icon_bg, outline=icon_bg)
        icon_canvas.create_text(16, 16, text=icon_char, fill="#fff", font=("Segoe UI", 16, "bold"))
        icon_canvas.pack()
        icon_frame.pack(side="left", padx=(18, 8), pady=0)

        # Message label
        self.label = tk.Label(row_frame, text=message, fg=text_color, bg=bg_color, font=("Segoe UI", 11, "bold"), anchor="w")
        self.label.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Close (X) button (always present)
        close_btn = tk.Button(row_frame, text="\u2716", command=self.fade_out, relief="flat", bg="#fff", fg="#888", font=("Segoe UI", 12, "bold"), padx=0, pady=0, borderwidth=0, activebackground="#eee", activeforeground="#f44336", width=2, height=1, cursor="hand2")
        close_btn.pack(side="right", padx=(0, 18), pady=0)

        # Buttons (OK/Yes/No) if provided
        if self.buttons:
            btn_frame = tk.Frame(row_frame, bg=bg_color)
            for i, btn in enumerate(self.buttons):
                b = tk.Button(
                    btn_frame, text=btn["text"],
                    command=lambda cmd=btn["command"]: self._on_button(cmd),
                    relief="flat", bg=icon_bg, fg="#fff",
                    font=("Segoe UI", 9, "bold"), padx=10, pady=2, borderwidth=0,
                    activebackground=border_color, activeforeground="#fff"
                )
                b.pack(side="right", padx=8)
            btn_frame.pack(side="right", padx=10)
        self.fade_in()
        self.toast.after(10000, self.fade_out)

    def _on_button(self, cmd):
        self.fade_out()
        if cmd:
            cmd()

    def fade_in(self):
        for i in range(0, 101, 10):
            self.toast.attributes("-alpha", i / 100)
            self.toast.update()
            time.sleep(0.01)

    def fade_out(self):
        for i in range(100, -1, -10):
            self.toast.attributes("-alpha", i / 100)
            self.toast.update()
            time.sleep(0.01)
        self.toast.destroy()



# Patch tk.Canvas to support rounded rectangles
def _create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
    points = [
        x1 + r, y1,
        x1 + r, y1,
        x2 - r, y1,
        x2 - r, y1,
        x2, y1,
        x2, y1 + r,
        x2, y1 + r,
        x2, y2 - r,
        x2, y2 - r,
        x2, y2,
        x2 - r, y2,
        x2 - r, y2,
        x1 + r, y2,
        x1 + r, y2,
        x1, y2,
        x1, y2 - r,
        x1, y2 - r,
        x1, y1 + r,
        x1, y1 + r,
        x1, y1
    ]
    return self.create_polygon(points, **kwargs, smooth=True)


tk.Canvas.create_rounded_rect = _create_rounded_rect

class DatabaseManager:
    @staticmethod
    def execute_query(query, params=(), fetch=False):
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(query, params)

        if fetch:
            result = cursor.fetchall()
        else:
            conn.commit()
            result = None

        conn.close()
        return result


class DocumentAnalyzer:
    def __init__(self):
        self.documents = []
        self.index = None
        self.text_chunks = []
        self.chunk_sources = []
        self.current_doc = None
        self.progress_queue = Queue()
        self.cancel_processing = False
        self.current_filepath = None
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.document_text = ""
        self.document_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.chunk_size = 2000  # Characters per chunk

        # Initialize NLP models with better configurations
        self.qa_model = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=0 if torch.cuda.is_available() else -1
        )
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/all-mpnet-base-v2',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.nlp = nlp
        self.nlp.max_length = 1000000  # Increase max length for better processing

        # Supported document formats and their processors
        self.supported_formats = {
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.doc': self._process_doc,
            '.txt': self._process_txt,
            '.csv': self._process_csv,
            '.xlsx': self._process_excel
        }

        self.chat_history = []
        self.user_records = {}
        self.current_user = None

    def _process_pdf(self, filepath):
        """Process PDF documents with improved text extraction and page tracking"""
        try:
            self.current_doc = fitz.open(filepath)
            total_pages = len(self.current_doc)
            full_text = ""
            page_texts = {}  # Store text for each page

            for page_num in range(total_pages):
                if self.cancel_processing:
                    break
                try:
                    page = self.current_doc[page_num]
                    text = page.get_text()
                    full_text += text + "\n"
                    page_texts[page_num + 1] = text  # Store page text with 1-based indexing

                    # Process text with better sentence splitting
                    doc = self.nlp(text)
                    for sent in doc.sents:
                        if len(sent.text.strip()) > 0:
                            # Store more context for better reference
                            self.text_chunks.append({
                                'page': page_num + 1,
                                'text': sent.text.strip(),
                                'filename': os.path.basename(filepath),
                                'char_span': (sent.start_char, sent.end_char),
                                'page_text': text,
                                'context': self._get_context(sent, doc)
                            })
                            self.chunk_sources.append({
                                'filename': os.path.basename(filepath),
                                'page': page_num + 1,
                                'sentence': sent.text.strip(),
                                'char_span': (sent.start_char, sent.end_char),
                                'page_text': text,
                                'context': self._get_context(sent, doc)
                            })
                    progress = int((page_num + 1) / total_pages * 100)
                    self.progress_queue.put(progress)
                except Exception as e:
                    print(f"Error processing page {page_num}: {e}")
                    continue

            self.documents.append({
                'filename': os.path.basename(filepath),
                'path': filepath,
                'full_text': full_text,
                'page_texts': page_texts  # Store page texts for better reference
            })
            self.document_text = full_text
            self.tfidf_vectorizer.fit_transform([self.preprocess_text(full_text[:10000])])
            return True
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return False

    def _get_context(self, sent, doc):
        """Get surrounding context for a sentence"""
        start_idx = sent.start
        end_idx = sent.end
        context_start = max(0, start_idx - 2)
        context_end = min(len(doc), end_idx + 2)
        return ' '.join([token.text for token in doc[context_start:context_end]])

    def query_documents(self, question, k=5):
        """Improved document querying with better answer generation and citation"""
        try:
            if not self.text_chunks or self.index is None:
                return {
                    'answer': "No documents loaded. Please load a document first.",
                    'confidence': 0.0,
                    'source': None,
                    'highlighted_image': None
                }
            # Check if question is relevant
            if not self.is_relevant_question(question, self.document_text):
                return {
                    'answer': "This question does not appear to be related to the document. Please ask something relevant to the document content.",
                    'confidence': 0.0,
                    'source': None,
                    'highlighted_image': None
                }
            # Preprocess question for better matching
            processed_question = self.preprocess_question(question)
            question_embedding = self.embedding_model.encode(processed_question)
            D, I = self.index.search(np.array([question_embedding]), k)

            # Get relevant chunks with their original indices
            relevant_chunks = [(self.text_chunks[i], i) for i in I[0] if i < len(self.text_chunks)]

            # Use multiple chunks for better context
            context_chunks = []
            for chunk, _ in relevant_chunks[:5]:  # Use top 5 chunks for context
                context_chunks.append(chunk['text'])
                if chunk.get('context'):
                    context_chunks.append(chunk['context'])
            context = ' '.join(context_chunks)

            # Detect question type
            q_lower = question.lower()
            if any(w in q_lower for w in ['what', 'who', 'when', 'where']):
                qtype = 'factoid'
            elif any(w in q_lower for w in ['why', 'how', 'explain', 'describe']):
                qtype = 'descriptive'
            elif any(w in q_lower for w in ['compare', 'difference', 'versus', 'vs']):
                qtype = 'comparative'
            else:
                qtype = 'general'

            # Use both extractive and generative models for best answer
            answer_text = None
            confidence = 0.0
            if qtype == 'factoid':
                # Use both QA and T5, pick the best
                try:
                    qa_result = self.qa_model(question=question, context=context)
                    qa_answer = qa_result['answer']
                    qa_score = qa_result['score']
                except Exception as e:
                    print(f"QA model error: {e}")
                    qa_answer, qa_score = '', 0.0
                try:
                    t5_answer = self.generate_factoid_answer(question, context, relevant_chunks)
                except Exception as e:
                    print(f"T5 model error: {e}")
                    t5_answer = ''
                # Prefer QA if confident, else T5
                if qa_score > 0.3 and len(qa_answer) > 5:
                    answer_text = qa_answer
                    confidence = qa_score
                elif t5_answer:
                    answer_text = t5_answer
                    confidence = 0.5
                else:
                    answer_text = qa_answer or t5_answer or "I couldn't find a specific answer."
            elif qtype == 'descriptive':
                try:
                    answer_text = self.generate_descriptive_answer(question, context, relevant_chunks)
                    confidence = 0.5
                except Exception as e:
                    print(f"Descriptive T5 error: {e}")
                    answer_text = self.generate_general_answer(question, context, relevant_chunks)
                    confidence = 0.3
            elif qtype == 'comparative':
                try:
                    answer_text = self.generate_comparative_answer(question, context, relevant_chunks)
                    confidence = 0.5
                except Exception as e:
                    print(f"Comparative T5 error: {e}")
                    answer_text = self.generate_general_answer(question, context, relevant_chunks)
                    confidence = 0.3
            else:
                try:
                    answer_text = self.generate_general_answer(question, context, relevant_chunks)
                    confidence = 0.4
                except Exception as e:
                    print(f"General T5 error: {e}")
                    answer_text = "I couldn't generate a proper answer."
                    confidence = 0.1

            # Find the best matching source
            best_source = None
            best_score = 0
            for chunk, _ in relevant_chunks:
                score = self.calculate_relevance_score(answer_text, chunk['text'])
                if score > best_score:
                    best_score = score
                    best_source = chunk

            # Generate highlighted image if possible (PDF only)
            highlighted_image = None
            if best_source and self.current_doc and hasattr(self.current_doc, 'load_page'):
                try:
                    highlighted_image = self.highlight_reference(best_source, answer_text=answer_text)
                except Exception as e:
                    print(f"Error generating highlight: {e}")

            # Format the answer with reference
            if best_source:
                answer_text = f"{self.format_answer(answer_text).strip('. ')}. [Page {best_source.get('page', 1)}]"

            # Only save if not a 'not relevant' message
            if self.current_user and not answer_text.startswith("This question does not appear to be related"):
                self.add_to_chat_history(
                    self.current_user,
                    question,
                    answer_text,
                    best_source
                )

            return {
                'answer': answer_text,
                'confidence': confidence,
                'source': best_source,
                'highlighted_image': highlighted_image
            }

        except Exception as e:
            print(f"Error querying documents: {e}")
            return {
                'answer': "Sorry, I encountered an error while processing your question.",
                'confidence': 0.0,
                'source': None,
                'highlighted_image': None
            }

    def calculate_relevance_score(self, answer, source_text):
        """Calculate relevance score between answer and source text"""
        # Calculate semantic similarity
        answer_embedding = self.embedding_model.encode([answer])
        source_embedding = self.embedding_model.encode([source_text])
        semantic_score = util.pytorch_cos_sim(answer_embedding, source_embedding).item()

        # Calculate word overlap
        answer_words = set(word_tokenize(answer.lower()))
        source_words = set(word_tokenize(source_text.lower()))
        overlap_score = len(answer_words.intersection(source_words)) / len(answer_words)

        # Combine scores
        return (semantic_score + overlap_score) / 2

    def highlight_reference(self, source, zoom=1.5, answer_text=None):
        """Improved reference highlighting with better text matching"""
        if not self.current_doc or not source:
            return None
        if not hasattr(self.current_doc, 'load_page'):
            return None
        try:
            page_num = source.get('page', 1) - 1
            if page_num < 0 or page_num >= len(self.current_doc):
                print(f"[DEBUG] Invalid page number for highlight: {page_num + 1}")
                return None
            doc = fitz.open(self.current_filepath)
            page = doc.load_page(page_num)

            # Try multiple text matching strategies
            candidates = []
            if answer_text:
                # Try full answer
                candidates.append(answer_text)
                # Try first sentence
                sentences = nltk.sent_tokenize(answer_text)
                if sentences:
                    candidates.append(sentences[0])
                # Try key phrases
                key_phrases = self._extract_key_phrases(answer_text)
                candidates.extend(key_phrases)

            if 'text' in source:
                candidates.append(source['text'])
                if source.get('context'):
                    candidates.append(source['context'])

            areas = []
            for text in candidates:
                if text:
                    found = page.search_for(text)
                    if found:
                        areas = found
                        break

            # Merge overlapping rectangles
            merged_areas = self.merge_rectangles(areas)

            # Apply highlights
            for area in merged_areas:
                highlight = page.add_highlight_annot(area)
                highlight.set_colors(stroke=[1, 1, 0])  # Yellow highlight
                highlight.update()

            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            return img
        except Exception as e:
            print(f"Error highlighting reference: {e}")
            return None

    def _extract_key_phrases(self, text):
        """Extract key phrases from text for better matching"""
        doc = self.nlp(text)
        phrases = []
        for chunk in doc.noun_chunks:
            phrases.append(chunk.text)
        for ent in doc.ents:
            phrases.append(ent.text)
        return phrases

    def _process_docx(self, filepath):
        """Process DOCX documents with robust chunking"""
        try:
            doc = docx.Document(filepath)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            # Group at least 5 paragraphs or 500+ chars per chunk
            chunks = []
            current = []
            current_len = 0
            for i, para in enumerate(paragraphs):
                current.append(para)
                current_len += len(para)
                if (len(current) >= 5 or current_len >= 500):
                    chunks.append(' '.join(current))
                    current = []
                    current_len = 0
            if current:
                chunks.append(' '.join(current))
            # Merge short chunks
            merged = []
            for chunk in chunks:
                if merged and len(chunk) < 100:
                    merged[-1] += ' ' + chunk
                else:
                    merged.append(chunk)
            return [{'page': i + 1, 'text': chunk, 'images': 0} for i, chunk in enumerate(merged)]
        except Exception as e:
            print(f"Error processing DOCX: {e}")
            return None

    def _process_doc(self, filepath):
        """Process legacy DOC documents using a combination of methods"""
        try:
            # First try to convert to DOCX using LibreOffice if available
            docx_path = filepath + '.docx'
            try:
                import subprocess
                subprocess.run(
                    ['soffice', '--headless', '--convert-to', 'docx', filepath, '--outdir', os.path.dirname(filepath)])
                if os.path.exists(docx_path):
                    chunks = self._process_docx(docx_path)
                    os.remove(docx_path)  # Clean up
                    return chunks
            except:
                pass

            # If conversion fails, try reading as plain text
            try:
                with open(filepath, 'rb') as file:
                    raw = file.read()
                    result = chardet.detect(raw)
                    text = raw.decode(result['encoding'])
                chunks = self._split_text_into_chunks(text)
                return [{'page': i + 1, 'text': chunk, 'images': 0} for i, chunk in enumerate(chunks)]
            except:
                # If all else fails, try OCR
                return self._process_with_ocr(filepath)
        except Exception as e:
            print(f"Error processing DOC: {e}")
            return None

    def _process_txt(self, filepath):
        """Process TXT documents with improved chunking"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            # Group lines into larger chunks (e.g., 300+ chars)
            chunks = []
            current = []
            current_len = 0
            for line in lines:
                if current_len + len(line) > 300 and current:
                    chunks.append(' '.join(current))
                    current = []
                    current_len = 0
                current.append(line)
                current_len += len(line)
            if current:
                chunks.append(' '.join(current))
            return [{'page': i + 1, 'text': chunk, 'images': 0} for i, chunk in enumerate(chunks)]
        except Exception as e:
            print(f"Error processing TXT: {e}")
            return None

    def _process_csv(self, filepath):
        """Process CSV documents with improved chunking"""
        try:
            df = pd.read_csv(filepath)
            text = df.to_string()
            lines = text.split('\n')
            # Group lines into larger chunks (e.g., 300+ chars)
            chunks = []
            current = []
            current_len = 0
            for line in lines:
                if current_len + len(line) > 300 and current:
                    chunks.append(' '.join(current))
                    current = []
                    current_len = 0
                current.append(line)
                current_len += len(line)
            if current:
                chunks.append(' '.join(current))
            return [{'page': i + 1, 'text': chunk, 'images': 0} for i, chunk in enumerate(chunks)]
        except Exception as e:
            print(f"Error processing CSV: {e}")
            return None

    def _process_excel(self, filepath):
        """Process Excel documents with improved chunking"""
        try:
            df = pd.read_excel(filepath)
            text = df.to_string()
            lines = text.split('\n')
            # Group lines into larger chunks (e.g., 300+ chars)
            chunks = []
            current = []
            current_len = 0
            for line in lines:
                if current_len + len(line) > 300 and current:
                    chunks.append(' '.join(current))
                    current = []
                    current_len = 0
                current.append(line)
                current_len += len(line)
            if current:
                chunks.append(' '.join(current))
            return [{'page': i + 1, 'text': chunk, 'images': 0} for i, chunk in enumerate(chunks)]
        except Exception as e:
            print(f"Error processing Excel: {e}")
            return None

    def _split_text_into_chunks(self, text, min_chunk_size=100):
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)
            if current_size + para_size > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
            current_chunk.append(para)
            current_size += para_size

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        # If still too small, merge to at least 3 sentences per chunk
        final_chunks = []
        for chunk in chunks:
            sentences = sent_tokenize(chunk)
            for i in range(0, len(sentences), 3):
                final_chunks.append(' '.join(sentences[i:i + 3]))
        return final_chunks

    def load_document(self, filepath, progress_callback=None):
        """Load and process any supported document type, always set current_doc and update reference window."""
        try:
            self.cancel_processing = False
            self.current_filepath = filepath
            self.text_chunks = []
            self.chunk_sources = []
            self.documents = []
            self.index = None
            if self.current_doc:
                try:
                    self.current_doc.close()
                except:
                    pass
                self.current_doc = None
            _, ext = os.path.splitext(filepath)
            ext = ext.lower()
            if ext not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {ext}")
            cached_chunks = self._get_cached_document(filepath)
            if cached_chunks:
                self.text_chunks = cached_chunks
                self.chunk_sources = [
                    {'filename': os.path.basename(filepath), 'page': chunk['page'], 'sentence': chunk['text']} for chunk
                    in cached_chunks]
                if ext == '.pdf':
                    self.current_doc = fitz.open(filepath)
                else:
                    self.current_doc = None
                # Build FAISS index for all types
                if self.text_chunks:
                    embeddings = self.embedding_model.encode([chunk['text'] for chunk in self.text_chunks])
                    self.index = faiss.IndexFlatL2(embeddings.shape[1])
                    self.index.add(embeddings)
                # Always set document_text for all types
                self.document_text = '\n'.join([chunk['text'] for chunk in self.text_chunks])
                if hasattr(self, 'update_ref_pdf_viewer'):
                    self.update_ref_pdf_viewer()
                return True
            processor = self.supported_formats[ext]
            result = processor(filepath)
            # For PDF, processor returns True/False; for others, returns chunks
            if ext == '.pdf':
                success = result
            else:
                success = result is not None
                if success:
                    self.text_chunks = result
                    self.chunk_sources = [
                        {'filename': os.path.basename(filepath), 'page': chunk['page'], 'sentence': chunk['text']} for
                        chunk in result]
                    self.current_doc = None
            if success:
                self._cache_document(filepath, self.text_chunks)
                # Build FAISS index for all types
                if self.text_chunks:
                    embeddings = self.embedding_model.encode([chunk['text'] for chunk in self.text_chunks])
                    self.index = faiss.IndexFlatL2(embeddings.shape[1])
                    self.index.add(embeddings)
                # Always set document_text for all types
                self.document_text = '\n'.join([chunk['text'] for chunk in self.text_chunks])
                if ext == '.pdf':
                    self.current_doc = fitz.open(filepath)
                if hasattr(self, 'update_ref_pdf_viewer'):
                    self.update_ref_pdf_viewer()
                return True
            return False
        except Exception as e:
            print(f"Error loading document: {e}")
            return False

    def _cache_document(self, filepath, chunks):
        """Cache processed document chunks"""
        cache_path = os.path.join(CACHE_DIR, hashlib.md5(filepath.encode()).hexdigest() + '.json')
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({
                'filepath': filepath,
                'chunks': chunks,
                'timestamp': time.time()
            }, f)

    def _get_cached_document(self, filepath):
        """Retrieve cached document if available"""
        cache_path = os.path.join(CACHE_DIR, hashlib.md5(filepath.encode()).hexdigest() + '.json')
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            if time.time() - cache['timestamp'] < 86400:  # 24 hour cache
                return cache['chunks']
        return None

    def _process_with_ocr(self, filepath):
        # Implement OCR processing logic here
        pass

    def preprocess_text(self, text):
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Tokenize and remove stopwords
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in stop_words]
        return ' '.join(filtered_text)

    def is_relevant_question(self, question, document_text):
        # Use semantic similarity for relevance check
        preprocessed_question = self.preprocess_text(question)
        preprocessed_doc = self.preprocess_text(document_text[:20000])  # Use first 20k chars
        chunk_count = len(self.text_chunks) if hasattr(self, 'text_chunks') else 0
        print(f"[DEBUG] Doc length: {len(document_text)}, Chunk count: {chunk_count}")
        if not preprocessed_question or not preprocessed_doc:
            return False
        # Always allow if doc is short or has few chunks
        if len(document_text) < 1000 or chunk_count < 5:
            print("[DEBUG] Relevance: doc is short or few chunks, allowing question.")
            return True
        # Allow if any word in question is in document
        doc_words = set(preprocessed_doc.split())
        q_words = set(preprocessed_question.split())
        if doc_words.intersection(q_words):
            print("[DEBUG] Relevance: direct word match")
            return True
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([preprocessed_question, preprocessed_doc])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            question_emb = self.embedding_model.encode([question])[0]
            doc_emb = self.embedding_model.encode([document_text[:20000]])[0]
            sem_sim = np.dot(question_emb, doc_emb) / (np.linalg.norm(question_emb) * np.linalg.norm(doc_emb))
            print(f"[DEBUG] TF-IDF sim: {similarity:.3f}, Embedding sim: {sem_sim:.3f}")
            if similarity > 0.12 or sem_sim > 0.45:
                return True
            return False
        except Exception as e:
            print(f"Relevance check error: {e}")
            return False

    def generate_factoid_answer(self, question, context, chunks):
        """Generate precise answers for factoid questions"""
        try:
            # Prepare input for T5
            input_text = f"question: {question} context: {context}"
            input_ids = t5_tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)

            # Generate answer
            outputs = t5_model.generate(
                input_ids,
                max_length=64,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Verify answer with QA model
            qa_inputs = qa_tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True).to(device)
            qa_outputs = qa_model(**qa_inputs)

            start_scores = qa_outputs.start_logits
            end_scores = qa_outputs.end_logits

            # Get the most likely answer span
            answer_start = torch.argmax(start_scores)
            answer_end = torch.argmax(end_scores) + 1

            qa_answer = qa_tokenizer.decode(qa_inputs["input_ids"][0][answer_start:answer_end])

            # Choose the better answer based on confidence
            final_answer = qa_answer if len(qa_answer) > 10 else answer
            return self.format_answer(final_answer)

        except Exception as e:
            print(f"Error generating factoid answer: {e}")
            return self.generate_general_answer(question, context, chunks)

    def generate_descriptive_answer(self, question, context, chunks):
        """Generate detailed explanatory answers"""
        try:
            # Extract key concepts from question
            doc = nlp(question)
            key_concepts = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB']]

            # Find relevant sentences in context
            relevant_sentences = []
            for chunk, _ in chunks:
                doc = nlp(chunk[0]['text'])
                for sent in doc.sents:
                    if any(concept.lower() in sent.text.lower() for concept in key_concepts):
                        relevant_sentences.append(sent.text)

            # Generate comprehensive answer
            if relevant_sentences:
                input_text = f"explain: {question} context: {' '.join(relevant_sentences)}"
                input_ids = t5_tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True).to(
                    device)

                outputs = t5_model.generate(
                    input_ids,
                    max_length=150,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True
                )

                answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                return self.format_answer(answer)

            return self.generate_general_answer(question, context, chunks)

        except Exception as e:
            print(f"Error generating descriptive answer: {e}")
            return self.generate_general_answer(question, context, chunks)

    def generate_comparative_answer(self, question, context, chunks):
        """Generate answers for comparison questions"""
        try:
            # Extract comparison elements
            doc = nlp(question)
            comparison_items = []
            for token in doc:
                if token.dep_ in ['nsubj', 'dobj', 'pobj'] and token.pos_ == 'NOUN':
                    comparison_items.append(token.text)

            if len(comparison_items) >= 2:
                # Find sentences containing comparison items
                relevant_sentences = []
                for chunk, _ in chunks:
                    doc = nlp(chunk[0]['text'])
                    for sent in doc.sents:
                        if all(item.lower() in sent.text.lower() for item in comparison_items):
                            relevant_sentences.append(sent.text)

                if relevant_sentences:
                    input_text = f"compare: {question} context: {' '.join(relevant_sentences)}"
                    input_ids = t5_tokenizer.encode(input_text, return_tensors='pt', max_length=512,
                                                    truncation=True).to(device)

                    outputs = t5_model.generate(
                        input_ids,
                        max_length=150,
                        num_beams=4,
                        length_penalty=2.0,
                        early_stopping=True
                    )

                    answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    return self.format_answer(answer)

            return self.generate_general_answer(question, context, chunks)

        except Exception as e:
            print(f"Error generating comparative answer: {e}")
            return self.generate_general_answer(question, context, chunks)

    def generate_general_answer(self, question, context, chunks):
        """Generate general purpose answers"""
        try:
            input_text = f"answer: {question} context: {context}"
            input_ids = t5_tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)

            outputs = t5_model.generate(
                input_ids,
                max_length=150,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )

            answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self.format_answer(answer)

        except Exception as e:
            print(f"Error generating general answer: {e}")
            return "I apologize, but I couldn't generate a proper answer from the available information."

    def find_highlight_spans(self, text, answer):
        """Find spans in the text that should be highlighted"""
        try:
            # Tokenize text and answer
            text_tokens = word_tokenize(text.lower())
            answer_tokens = word_tokenize(answer.lower())

            # Find matching sequences
            spans = []
            for i in range(len(text_tokens)):
                for j in range(i + 1, len(text_tokens) + 1):
                    window = text_tokens[i:j]
                    if len(window) > 2 and all(token in answer_tokens for token in window):
                        # Convert token indices to character spans
                        start_char = text.lower().find(' '.join(window))
                        if start_char != -1:
                            end_char = start_char + len(' '.join(window))
                            spans.append((start_char, end_char))

            # Merge overlapping spans
            if spans:
                spans.sort(key=lambda x: x[0])
                merged = [spans[0]]
                for current in spans[1:]:
                    previous = merged[-1]
                    if current[0] <= previous[1]:
                        merged[-1] = (previous[0], max(previous[1], current[1]))
                    else:
                        merged.append(current)
                return merged

            return []

        except Exception as e:
            print(f"Error finding highlight spans: {e}")
            return []

    def preprocess_question(self, question):
        """Preprocess the question to improve matching"""
        # Remove extra whitespace and lowercase
        question = question.strip().lower()

        # Extract key phrases for specific types of questions
        if "what is" in question:
            # For definition questions, focus on the term being asked about
            term = question.split("what is")[-1].strip()
            question = f"define {term}"
        elif "how to" in question:
            # For procedural questions, focus on the action
            action = question.split("how to")[-1].strip()
            question = f"steps to {action}"
        elif "explain" in question:
            # For explanatory questions, focus on the concept
            concept = question.split("explain")[-1].strip()
            question = f"explain concept {concept}"

        return question

    def calculate_chunk_relevance(self, chunk, question):
        """Calculate relevance score for a chunk"""
        # Calculate semantic similarity
        chunk_embedding = embedding_model.encode([chunk[0]['text']])
        question_embedding = embedding_model.encode([question])
        semantic_score = util.pytorch_cos_sim(question_embedding, chunk_embedding).item()

        # Calculate concept overlap
        chunk_doc = nlp(chunk[0]['text'].lower())
        chunk_concepts = set()
        for token in chunk_doc:
            if token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                chunk_concepts.add(token.lemma_.lower())

        concept_overlap = len(chunk_concepts) / len(chunk_concepts) if chunk_concepts else 0

        # Check for exact phrase matches
        exact_match_score = 1.0 if question.lower() in chunk[0]['text'].lower() else 0.0

        # Combine scores with weights
        total_score = (semantic_score * 0.4) + (concept_overlap * 0.4) + (exact_match_score * 0.2)
        return total_score

    def merge_rectangles(self, rectangles):
        """Merge overlapping rectangles to create continuous highlights"""
        if not rectangles:
            return []

        # Sort rectangles by their y0 coordinate (top)
        sorted_rects = sorted(rectangles, key=lambda r: (r.y0, r.x0))
        merged = [sorted_rects[0]]

        for rect in sorted_rects[1:]:
            last = merged[-1]

            # Check if rectangles are on the same line (similar y-coordinates)
            if abs(rect.y0 - last.y0) < 5 and abs(rect.y1 - last.y1) < 5:
                # Check if rectangles overlap or are very close horizontally
                if rect.x0 <= last.x1 + 5:
                    # Merge rectangles
                    merged[-1] = fitz.Rect(
                        last.x0,
                        min(last.y0, rect.y0),
                        max(last.x1, rect.x1),
                        max(last.y1, rect.y1)
                    )
                else:
                    merged.append(rect)
            else:
                merged.append(rect)

        return merged

    def add_to_chat_history(self, user_id, message, response, source=None):
        """Add a message and response to chat history"""
        timestamp = time.time()
        chat_entry = {
            'timestamp': timestamp,
            'user_id': user_id,
            'message': message,
            'response': response,
            'source': source
        }
        self.chat_history.append(chat_entry)

        # Update user records
        if user_id not in self.user_records:
            self.user_records[user_id] = {
                'chat_count': 0,
                'last_active': timestamp,
                'documents_viewed': set()
            }

        self.user_records[user_id]['chat_count'] += 1
        self.user_records[user_id]['last_active'] = timestamp

        if source and source.get('filename'):
            self.user_records[user_id]['documents_viewed'].add(source['filename'])

        # Save chat history to file
        self._save_chat_history()

    def get_chat_history(self, user_id=None, limit=50):
        """Get chat history for a specific user or all users"""
        if user_id:
            return [entry for entry in self.chat_history if entry['user_id'] == user_id][-limit:]
        return self.chat_history[-limit:]

    def _save_chat_history(self):
        """Save chat history to a file"""
        try:
            with open('chat_history.json', 'w', encoding='utf-8') as f:
                json.dump({
                    'chat_history': self.chat_history,
                    'user_records': {
                        user_id: {
                            'chat_count': data['chat_count'],
                            'last_active': data['last_active'],
                            'documents_viewed': list(data['documents_viewed'])
                        }
                        for user_id, data in self.user_records.items()
                    }
                }, f, indent=2)
        except Exception as e:
            print(f"Error saving chat history: {e}")

    def load_chat_history(self):
        """Load chat history from file"""
        try:
            if os.path.exists('chat_history.json'):
                with open('chat_history.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.chat_history = data.get('chat_history', [])
                    self.user_records = {
                        user_id: {
                            'chat_count': data['chat_count'],
                            'last_active': data['last_active'],
                            'documents_viewed': set(data['documents_viewed'])
                        }
                        for user_id, data in data.get('user_records', {}).items()
                    }
        except Exception as e:
            print(f"Error loading chat history: {e}")

    def get_user_stats(self, user_id):
        """Get statistics for a specific user"""
        if user_id not in self.user_records:
            return None

        stats = self.user_records[user_id].copy()
        stats['documents_viewed'] = list(stats['documents_viewed'])
        return stats

    def set_current_user(self, user_id):
        """Set the current active user"""
        self.current_user = user_id
        if user_id not in self.user_records:
            self.user_records[user_id] = {
                'chat_count': 0,
                'last_active': time.time(),
                'documents_viewed': set()
            }

    def get_page_image(self, page_num, zoom=1.5):
        """Get a rendered PDF page as PIL image"""
        if not self.current_doc or page_num < 0 or page_num >= len(self.current_doc):
            return None

        try:
            page = self.current_doc.load_page(page_num)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return img
        except Exception as e:
            print(f"Error rendering page {page_num}: {e}")
            return None

    def find_text_in_document(self, text):
        """Find the page number where the given text appears"""
        if not self.current_doc:
            return None

        try:
            for page_num in range(len(self.current_doc)):
                page = self.current_doc[page_num]
                page_text = page.get_text()
                if text in page_text:
                    return page_num + 1
            return None
        except Exception as e:
            print(f"Error finding text in document: {e}")
            return None

    def format_answer(self, answer):
        """Format the answer for better readability"""
        import nltk
        sentences = nltk.sent_tokenize(answer)
        sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
        formatted_answer = ' '.join(sentences)
        if not formatted_answer.endswith(('.', '!', '?')):
            formatted_answer += '.'
        return formatted_answer


class UserManager:
    @staticmethod
    def hash_password(password):
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def register_user(username, password):
        # Check if username exists
        result = DatabaseManager.execute_query(
            "SELECT id FROM users WHERE username = ?",
            (username,),
            fetch=True
        )

        if result:
            return False, "Username already exists"

        # Create new user
        DatabaseManager.execute_query(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, UserManager.hash_password(password), datetime.now().isoformat())
        )

        # Create user chat directory
        os.makedirs(os.path.join(CHATS_DIR, username), exist_ok=True)
        return True, "Registration successful"

    @staticmethod
    def authenticate_user(username, password):
        result = DatabaseManager.execute_query(
            "SELECT password_hash FROM users WHERE username = ?",
            (username,),
            fetch=True
        )

        if not result:
            return False, "User not found"

        stored_hash = result[0][0]
        if stored_hash == UserManager.hash_password(password):
            return True, "Authentication successful"
        else:
            return False, "Incorrect password"


class ChatManager:
    @staticmethod
    def save_chat(user_id, document_id, messages):
        # Create a new chat record
        DatabaseManager.execute_query(
            "INSERT INTO chats (user_id, document_id, created_at) VALUES (?, ?, ?)",
            (user_id, document_id, datetime.now().isoformat())
        )
        chat_id = DatabaseManager.execute_query(
            "SELECT id FROM chats WHERE user_id = ? AND document_id = ? ORDER BY created_at DESC LIMIT 1",
            (user_id, document_id),
            fetch=True
        )[0][0]
        # Save all messages
        for msg in messages:
            DatabaseManager.execute_query(
                '''INSERT INTO messages 
                (chat_id, question, answer, is_relevant, created_at) 
                VALUES (?, ?, ?, ?, ?)''',
                (chat_id, msg['question'], msg['answer'], int(msg['is_relevant']), msg['timestamp'])
            )
            message_id = DatabaseManager.execute_query(
                "SELECT id FROM messages WHERE chat_id = ? ORDER BY created_at DESC LIMIT 1",
                (chat_id,),
                fetch=True
            )[0][0]
            # Save sources if available
            if msg.get('sources'):
                for source in msg['sources']:
                    DatabaseManager.execute_query(
                        '''INSERT INTO sources 
                        (message_id, page_number, text, start_char, end_char) 
                        VALUES (?, ?, ?, ?, ?)''',
                        (
                            message_id,
                            source.get('page', 1),
                            source.get('sentence', ''),
                            source.get('char_span', (0, 0))[0],
                            source.get('char_span', (0, 0))[1]
                        )
                    )
        # Debug print: show all chats for this user
        print("[DEBUG] Chats in DB for user:", ChatManager.load_chat_history(user_id))
        return chat_id

    @staticmethod
    def load_chat_history(user_id, limit=5):
        result = DatabaseManager.execute_query(
            '''SELECT c.id, c.created_at, d.filename 
               FROM chats c 
               LEFT JOIN documents d ON c.document_id = d.id 
               WHERE c.user_id = ? 
               ORDER BY c.created_at DESC 
               LIMIT ?''',
            (user_id, limit),
            fetch=True
        )

        history = []
        for row in result:
            chat_id, created_at, filename = row
            # Get first question as preview
            preview = DatabaseManager.execute_query(
                "SELECT question FROM messages WHERE chat_id = ? ORDER BY created_at LIMIT 1",
                (chat_id,),
                fetch=True
            )

            history.append({
                'id': chat_id,
                'timestamp': created_at.replace('T', ' ').split('.')[0],
                'document': filename or 'No document',
                'preview': preview[0][0][:50] + "..." if preview else "No messages"
            })

        return history

    @staticmethod
    def load_chat_messages(chat_id):
        # Get messages
        messages = []
        msg_rows = DatabaseManager.execute_query(
            "SELECT id, question, answer, is_relevant, created_at FROM messages WHERE chat_id = ? ORDER BY created_at",
            (chat_id,),
            fetch=True
        )

        for msg_row in msg_rows:
            msg_id, question, answer, is_relevant, created_at = msg_row

            # Get sources if available
            sources = []
            source_rows = DatabaseManager.execute_query(
                "SELECT page_number, text, start_char, end_char FROM sources WHERE message_id = ?",
                (msg_id,),
                fetch=True
            )

            for src_row in source_rows:
                page_num, text, start_char, end_char = src_row
                sources.append({
                    'page': page_num,
                    'sentence': text,
                    'char_span': (start_char, end_char)
                })

            messages.append({
                'question': question,
                'answer': answer,
                'is_relevant': bool(is_relevant),
                'sources': sources,
                'timestamp': created_at
            })

        return messages

    @staticmethod
    def save_document(user_id, filename, filepath):
        # Always insert a new document for this user and path if not exists
        result = DatabaseManager.execute_query(
            "SELECT id FROM documents WHERE user_id = ? AND filepath = ?",
            (user_id, filepath),
            fetch=True
        )
        if result:
            return result[0][0]  # Return existing document ID
        DatabaseManager.execute_query(
            "INSERT INTO documents (user_id, filename, filepath, uploaded_at) VALUES (?, ?, ?, ?)",
            (user_id, filename, filepath, datetime.now().isoformat())
        )
        doc_id = DatabaseManager.execute_query(
            "SELECT id FROM documents WHERE user_id = ? AND filepath = ?",
            (user_id, filepath),
            fetch=True
        )[0][0]
        # Debug print: show all documents for this user
        print("[DEBUG] Documents in DB for user:", DatabaseManager.execute_query(
            "SELECT id, filename, filepath FROM documents WHERE user_id = ?", (user_id,), fetch=True))
        return doc_id


class DocumentQnAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Document Analysis & QnA System")
        self.root.geometry("1100x650")
        self.root.minsize(100, 650)

        # App state
        self.current_user = None
        self.current_user_id = None
        self.analyzer = DocumentAnalyzer()
        self.current_document = None
        self.current_document_id = None
        self.current_chat = []
        self.current_chat_id = None
        self.current_page = 0
        self.progress_window = None
        self.current_highlighted_source = None
        self.file_label = ttk.Label(self.root)
        self.file_label.pack_forget()
        self.upload_status = ttk.Label(self.root)
        self.upload_status.pack_forget()

        # Configure styles
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Main colors
        self.primary_color = '#4e73df'
        self.secondary_color = '#1cc88a'
        self.dark_color = '#5a5c69'
        self.light_color = '#f8f9fc'
        self.dark_mode = False

        # Configure styles
        self.style.configure('TFrame', background=self.light_color)
        self.style.configure('TLabel', background=self.light_color, font=('Helvetica', 10))
        self.style.configure('TEntry', font=('Helvetica', 10), padding=6)

        # Button styles
        self.style.configure('TButton', font=('Helvetica', 10), padding=6)
        self.style.configure('Accent.TButton', background=self.primary_color, foreground='white')
        self.style.configure('Secondary.TButton', background=self.secondary_color, foreground='white')
        self.style.configure('Danger.TButton', background='#e74a3b', foreground='white')
        self.style.configure('Small.TButton', font=('Helvetica', 9), padding=3)

        # Map button states
        self.style.map('Accent.TButton',
                       background=[('active', '#3a56b4'), ('disabled', '#cccccc')],
                       foreground=[('active', 'white'), ('disabled', '#888888')])
        self.style.map('Secondary.TButton',
                       background=[('active', '#17a673'), ('disabled', '#cccccc')],
                       foreground=[('active', 'white'), ('disabled', '#888888')])

        # Configure treeview style
        self.style.configure('Treeview', rowheight=25, fieldbackground=self.light_color)
        self.style.configure('Treeview.Heading', font=('Helvetica', 10, 'bold'))
        self.style.map('Treeview', background=[('selected', self.primary_color)])

        # Initialize UI
        self.show_login_screen()

        self.zoom_var = DoubleVar(value=1.5)  # Add zoom_var for PDF viewer

    def show_login_screen(self):
        self.clear_window()

        # Main frame
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(expand=True, fill=BOTH)

        # Logo
        logo_frame = ttk.Frame(main_frame)
        logo_frame.pack(pady=20)

        try:
            logo_img = Image.open("logo.png").resize((100, 100), Image.LANCZOS)
            self.logo = ImageTk.PhotoImage(logo_img)
            ttk.Label(logo_frame, image=self.logo).pack()
        except:
            ttk.Label(logo_frame, text="📚", font=('Helvetica', 48)).pack()

        ttk.Label(logo_frame, text="Advanced Document QnA", font=('Helvetica', 16, 'bold')).pack(pady=10)

        # Login form
        form_frame = ttk.Frame(main_frame)
        form_frame.pack(pady=20, ipadx=20, ipady=20)

        ttk.Label(form_frame, text="Username:").grid(row=0, column=0, padx=5, pady=5, sticky=W)
        self.username_entry = ttk.Entry(form_frame, width=30)
        self.username_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(form_frame, text="Password:").grid(row=1, column=0, padx=5, pady=5, sticky=W)
        self.password_entry = ttk.Entry(form_frame, width=30, show="*")
        self.password_entry.grid(row=1, column=1, padx=5, pady=5)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)

        ttk.Button(button_frame, text="Login", command=self.handle_login, style='Secondary.TButton').pack(side=LEFT,
                                                                                                        padx=10)
        ttk.Button(button_frame, text="Register", command=self.show_register_screen,style='Accent.TButton').pack(side=LEFT, padx=10)
        copyright_label = ttk.Label(main_frame, text="© All rights reserved to @codewithsajjad",
                                    font=("Helvetica", 8), anchor="center")
        copyright_label.pack(side=BOTTOM, fill=X, pady=2)
    def show_register_screen(self):
        self.clear_window()

        # Configure root window for full expansion
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Main container with gradient background
        main_frame = ttk.Frame(self.root, padding=0)
        main_frame.grid(row=0, column=0, sticky='nsew')
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)

        # Left Column - Personal Information (with decorative elements)
        left_frame = tk.Frame(main_frame, bg='#e6f2ff', padx=30, pady=30)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))

        # Decorative header for left column
        header_left = ttk.Frame(left_frame, style='Header.TFrame')
        header_left.pack(fill=X, pady=(0, 20))
        ttk.Label(header_left, text="Personal Details", font=('Helvetica', 14, 'bold'),
                  style='Header.TLabel').pack(pady=5)
        ttk.Label(header_left, text="Tell us about yourself", font=('Helvetica', 10),
                  style='Subheader.TLabel').pack()

        # Personal info fields with icons
        self.create_form_field(left_frame, "👤 Full Name:", "reg_name_entry", bg_color='#e6f2ff')
        self.create_form_field(left_frame, "📧 Email:", "reg_email_entry",bg_color='#e6f2ff')
        self.create_form_field(left_frame, "📱 Contact:", "reg_contact_entry", bg_color='#e6f2ff')

        # Middle Column - Account Information (with visual divider)
        middle_frame = tk.Frame(main_frame, bg='#f0e6ff', padx=30, pady=30)
        middle_frame.grid(row=0, column=1, sticky='nsew', padx=5)

        # Decorative header for middle column
        header_middle = ttk.Frame(middle_frame, style='Header.TFrame')
        header_middle.pack(fill=X, pady=(0, 20))
        ttk.Label(header_middle, text="Account Setup", font=('Helvetica', 14, 'bold'),
                  style='Header.TLabel').pack(pady=5)
        ttk.Label(header_middle, text="Create your credentials", font=('Helvetica', 10),
                  style='Subheader.TLabel').pack()

        # Account info fields with interactive feedback
        self.create_form_field(middle_frame, "🔐 Username:", "reg_username_entry",
                               feedback_label="username_status_label", bg_color='#f0e6ff')
        self.reg_username_entry.bind('<KeyRelease>', self.check_username_availability)

        self.create_form_field(middle_frame, "🔑 Password:", "reg_password_entry",
                               show="*", feedback_label="password_strength", bg_color='#f0e6ff')
        self.reg_password_entry.bind('<KeyRelease>', self.update_password_strength)

        self.create_form_field(middle_frame, "✓ Confirm Password:", "reg_confirm_entry",
                               show="*", feedback_label="password_match_label", bg_color='#f0e6ff')
        self.reg_confirm_entry.bind('<KeyRelease>', self.check_password_match)

        # Right Column - Action Buttons (with decorative background)
        right_frame = ttk.Frame(main_frame, padding=30, style='RightPanel.TFrame')
        right_frame.grid(row=0, column=2, sticky='nsew', padx=(5, 0))

        # Decorative centerpiece for right column
        centerpiece = ttk.Frame(right_frame, style='Centerpiece.TFrame')
        centerpiece.pack(expand=True, fill=BOTH, pady=50)

        # Logo in the center
        try:
            logo_img = Image.open("logo.png").resize((150, 150), Image.LANCZOS)
            self.logo = ImageTk.PhotoImage(logo_img)
            ttk.Label(centerpiece, image=self.logo, style='Logo.TLabel').pack(pady=20)
        except:
            ttk.Label(centerpiece, text="📚", font=('Helvetica', 72),
                      style='Logo.TLabel').pack(pady=20)

        ttk.Label(centerpiece, text="Join Us!", font=('Helvetica', 18, 'bold'),
                  style='Header.TLabel').pack(pady=10)

        # Action buttons with modern styling
        btn_register = ttk.Button(
            centerpiece, text="Create Account",
            command=self.handle_register,
            style='AccentLarge.TButton'
        )
        btn_register.pack(fill=X, pady=10, ipady=10)

        btn_login = ttk.Button(
            centerpiece, text="Already Registered? Login",
            command=self.show_login_screen,
            style='SecondaryLarge.TButton'
        )
        btn_login.pack(fill=X, pady=5, ipady=8)

        # Terms and conditions
        terms_frame = ttk.Frame(centerpiece)
        terms_frame.pack(fill=X, pady=20)
        ttk.Label(terms_frame, text="By registering, you agree to our",
                  font=('Helvetica', 8)).pack(side=LEFT)
        terms_link = ttk.Label(terms_frame, text="Terms & Conditions",
                               font=('Helvetica', 8, 'underline'), foreground='blue')
        terms_link.pack(side=LEFT)
        terms_link.bind("<Button-1>", lambda e: self.show_terms())
        copyright_label = ttk.Label(right_frame, text="© All rights reserved to @codewithsajjad",
                                    font=("Helvetica", 8), anchor="center")
        copyright_label.pack(side=BOTTOM, fill=X, pady=2)
        # Configure styles for this screen
        self.configure_register_styles()

    def create_form_field(self, parent, label_text, entry_name, show="", feedback_label=None, bg_color=None):
        frame = ttk.Frame(parent)
        frame.pack(fill=X, pady=10)

        # Use the provided background color or default to white
        label_bg = bg_color if bg_color else 'white'

        # Label with icon, set background to match parent
        label = tk.Label(frame, text=label_text, font=('Helvetica', 10), bg=label_bg)
        label.pack(side=LEFT, padx=(0, 10))

        # Entry field
        entry = ttk.Entry(frame, width=25, show=show, font=('Helvetica', 10))
        entry.pack(side=LEFT, expand=True, fill=X)
        setattr(self, entry_name, entry)

        # Feedback label if specified
        if feedback_label:
            feedback = tk.Label(frame, text="", font=('Helvetica', 8), bg=label_bg)
            feedback.pack(side=LEFT, padx=5)
            setattr(self, feedback_label, feedback)

    def configure_register_styles(self):
        """Configure all styles for the registration screen"""
        # Main background style
        self.style.configure('TFrame', background='#f5f7fa')

        # Left panel - light blue
        self.style.configure('LeftPanel.TFrame', background='#e6f2ff',
                             borderwidth=2, relief='groove')

        # Middle panel - light purple
        self.style.configure('MiddlePanel.TFrame', background='#f0e6ff',
                             borderwidth=2, relief='groove')

        # Right panel - gradient background would be ideal, but solid color for now
        self.style.configure('RightPanel.TFrame', background='#ffffff',
                             borderwidth=2, relief='groove')

        # Centerpiece in right panel
        self.style.configure('Centerpiece.TFrame', background='#ffffff')

        # Header styles
        self.style.configure('Header.TFrame', background='#4e73df')
        self.style.configure('Header.TLabel', background='#4e73df', foreground='white')
        self.style.configure('Subheader.TLabel', background='#4e73df', foreground='#e6f2ff')

        # Logo style
        self.style.configure('Logo.TLabel', background='#ffffff')

        # Button styles
        self.style.configure('AccentLarge.TButton',
                             background='#4e73df', foreground='white',
                             font=('Helvetica', 12, 'bold'), padding=10)
        self.style.map('AccentLarge.TButton',
                       background=[('active', '#3a56b4'), ('disabled', '#cccccc')])

        self.style.configure('SecondaryLarge.TButton',
                             background='#1cc88a', foreground='white',
                             font=('Helvetica', 10), padding=8)
        self.style.map('SecondaryLarge.TButton',
                       background=[('active', '#17a673'), ('disabled', '#cccccc')])

        # Entry field styles
        self.style.configure('TEntry', fieldbackground='white', padding=5)

    def show_main_app(self):
        self.clear_window()
        self.style.configure('Main.TFrame', background='#f8f9fc')
        self.style.configure('Sidebar.TFrame', background='#4e73df')
        self.style.configure('Sidebar.TLabel', background='#4e73df', foreground='white')
        self.style.configure('Sidebar.TButton', background='#4e73df', foreground='white', borderwidth=0, font=('Helvetica', 10, 'bold'))
        self.style.map('Sidebar.TButton', background=[('active', '#3a56b4')], foreground=[('active', 'white')])
        main_container = ttk.Frame(self.root, style='Main.TFrame')
        main_container.pack(fill=BOTH, expand=True)
        sidebar = ttk.Frame(main_container, width=220, style='Sidebar.TFrame')
        sidebar.pack(side=LEFT, fill=Y)
        user_frame = ttk.Frame(sidebar, style='Sidebar.TFrame')
        user_frame.pack(fill=X, pady=20, padx=10)
        # Fetch profile image path from DB
        user_info = DatabaseManager.execute_query(
            "SELECT full_name, profile_image FROM users WHERE id = ?",
            (self.current_user_id,),
            fetch=True
        )
        full_name = user_info[0][0] if user_info and user_info[0][0] else self.current_user
        profile_image_path = user_info[0][1] if user_info and user_info[0][1] else None
        try:
            if profile_image_path and os.path.exists(profile_image_path):
                user_img = Image.open(profile_image_path).resize((80, 80), Image.LANCZOS)
            else:
                user_img = Image.open(resource_path("user.png")).resize((80, 80), Image.LANCZOS)
            self.user_photo = ImageTk.PhotoImage(user_img)
            ttk.Label(user_frame, image=self.user_photo, style='Sidebar.TLabel').pack()
        except:
            ttk.Label(user_frame, text="👤", font=('Helvetica', 36), style='Sidebar.TLabel').pack()
        # Show full name if available
        ttk.Label(user_frame, text=full_name, font=('Helvetica', 12, 'bold'), style='Sidebar.TLabel').pack(pady=5)
        nav_frame = ttk.Frame(sidebar, style='Sidebar.TFrame')
        nav_frame.pack(fill=X, pady=20, padx=10)
        buttons = [
            ("📁 Upload Document", self.show_upload_screen),
            ("📚 My Documents", self.show_my_documents_screen),
            ("💬 Chat History", self.show_history_screen),
            ("⚙️ Settings", self.show_settings_screen),
            ("🚪 Logout", self.logout)
        ]
        for text, command in buttons:
            btn = ttk.Button(nav_frame, text=text, command=command, style='Sidebar.TButton')
            btn.pack(fill=X, pady=5, ipady=8)
        status_frame = ttk.Frame(sidebar, style='Sidebar.TFrame')
        status_frame.pack(side=BOTTOM, fill=X, pady=10)
        # Get real chat/doc counts
        doc_count = DatabaseManager.execute_query(
            "SELECT COUNT(*) FROM documents WHERE user_id = ?",
            (self.current_user_id,), fetch=True
        )[0][0]
        chat_count = DatabaseManager.execute_query(
            "SELECT COUNT(*) FROM chats WHERE user_id = ?",
            (self.current_user_id,), fetch=True
        )[0][0]
        ttk.Label(status_frame, text=f"Chats: {chat_count} | Docs: {doc_count}", style='Sidebar.TLabel').pack()
        self.content_area = ttk.Frame(main_container, style='Main.TFrame')
        self.content_area.pack(side=RIGHT, fill=BOTH, expand=True)
        welcome_frame = ttk.Frame(self.content_area, style='Main.TFrame')
        welcome_frame.pack(expand=True, fill=BOTH, pady=50)
        ttk.Label(welcome_frame, text="Welcome to Document QnA System", font=('Helvetica', 18, 'bold')).pack(pady=10)
        ttk.Label(welcome_frame, text="Upload a document to start asking questions or view your previous chats", font=('Helvetica', 12)).pack(pady=5)
        copyright_label = ttk.Label(self.content_area, text="© All rights reserved to @codewithsajjad",
                                    font=("Helvetica", 8), anchor="center")
        copyright_label.pack(side=BOTTOM, fill=X, pady=2)
        action_frame = ttk.Frame(welcome_frame)
        action_frame.pack(pady=20)
        ttk.Button(action_frame, text="Upload Document", command=self.show_upload_screen, style='Accent.TButton').pack(side=LEFT, padx=10)
        ttk.Button(action_frame, text="View History", command=self.show_history_screen, style='Secondary.TButton').pack(side=LEFT, padx=10)
        recent_docs = DatabaseManager.execute_query(
            "SELECT id, filename FROM documents WHERE user_id = ? ORDER BY uploaded_at DESC LIMIT 3",
            (self.current_user_id,), fetch=True
        )
        if recent_docs:
            ttk.Label(welcome_frame, text="Recent Documents:", font=('Helvetica', 12, 'bold')).pack(pady=10)
            for doc_id, filename in recent_docs:
                doc_frame = ttk.Frame(welcome_frame)
                doc_frame.pack(fill=X, pady=5, padx=50)
                ttk.Label(doc_frame, text=filename, font=('Helvetica', 10)).pack(side=LEFT)
                ttk.Button(doc_frame, text="Open", command=lambda id=doc_id, name=filename: self.load_existing_document(id, name), style='Secondary.TButton').pack(side=RIGHT)

    def show_upload_screen(self):
        self.clear_content_area()
        ttk.Label(self.content_area, text="Document Upload & Analysis", font=('Helvetica', 16, 'bold')).pack(pady=10)
        upload_frame = ttk.Frame(self.content_area)
        upload_frame.pack(pady=20, fill=X)

        ttk.Button(upload_frame, text="Browse Document", command=self.browse_document,style='Secondary.TButton').pack(side=LEFT, padx=5)

        # Reparent the file_label to the upload_frame
        self.file_label.pack_forget()
        self.file_label = ttk.Label(upload_frame, text="No file selected")
        self.file_label.pack(side=LEFT, padx=5)

        ttk.Button(upload_frame, text="Analyze Document", command=self.analyze_document, style='Accent.TButton').pack(
            side=LEFT, padx=5)
        # ... rest of the method ...
        recent_frame = ttk.Frame(self.content_area)
        recent_frame.pack(fill=BOTH, expand=True, pady=20)
        ttk.Label(recent_frame, text="Recent Documents", font=('Helvetica', 12, 'bold')).pack(anchor=W)
        recent_docs = DatabaseManager.execute_query(
            "SELECT id, filename FROM documents WHERE user_id = ? ORDER BY uploaded_at DESC LIMIT 5",
            (self.current_user_id,),
            fetch=True
        )
        if recent_docs:
            for doc_id, filename in recent_docs:
                doc_frame = ttk.Frame(recent_frame)
                doc_frame.pack(fill=X, pady=5)
                ttk.Label(doc_frame, text=filename).pack(side=LEFT, padx=5)
                ttk.Button(
                    doc_frame, text="Load",
                    command=lambda id=doc_id, name=filename: self.load_existing_document(id, name),
                    style='Secondary.TButton'
                ).pack(side=RIGHT, padx=5)
        else:
            ttk.Label(recent_frame, text="No recent documents").pack()
        self.upload_status = ttk.Label(self.content_area, text="")
        self.upload_status.pack(pady=10)
        copyright_label = ttk.Label(self.content_area, text="© All rights reserved to @codewithsajjad",
                                    font=("Helvetica", 8), anchor="center")
        copyright_label.pack(side=BOTTOM, fill=X, pady=2)

    def browse_document(self):
        filetypes = [
            ("All supported files", "*.pdf;*.docx;*.doc;*.txt;*.csv;*.xlsx"),
            ("PDF files", "*.pdf"),
            ("Word documents", "*.docx;*.doc"),
            ("Text files", "*.txt"),
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx")
        ]
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath:
            if hasattr(self, 'file_label') and self.file_label is not None and self.file_label.winfo_exists():
                self.file_label.config(text=os.path.basename(filepath))
            self.current_document = os.path.basename(filepath)
            self.current_document_path = filepath

    def analyze_document(self):
        filepath = getattr(self, 'current_document_path', None)
        if not filepath or self.file_label.cget("text") == "No file selected":
            SonnerToast(self.root, "Please select a document file first!",variant='warning', buttons=[{"text": "OK", "command": None}])
            return
        self.upload_status.config(text="Analyzing document...")
        self.root.update()
        self.analyze_document_threaded(filepath)

    def analyze_document_threaded(self, filepath):
        def worker():
            try:
                if hasattr(self.analyzer, 'current_doc') and self.analyzer.current_doc:
                    try:
                        self.analyzer.current_doc.close()
                    except:
                        pass
                    self.analyzer.current_doc = None
                self.analyzer.text_chunks = []
                self.analyzer.chunk_sources = []
                self.analyzer.documents = []
                self.analyzer.index = None
                success = self.analyzer.load_document(filepath, self.update_progress)
                if success:
                    # Save document to database for all file types
                    self.current_document = os.path.basename(filepath)
                    self.current_document_id = ChatManager.save_document(
                        self.current_user_id,
                        self.current_document,
                        filepath
                    )
                self.root.after(0, self.document_analysis_complete, success, filepath)
            except Exception as e:
                print(f"Error in document analysis worker: {e}")
                self.root.after(0, self.document_analysis_complete, False, filepath)

        self.show_progress_window()
        threading.Thread(target=worker, daemon=True).start()

    def load_existing_document(self, doc_id, filename):
        result = DatabaseManager.execute_query(
            "SELECT filepath FROM documents WHERE id = ?",
            (doc_id,),
            fetch=True
        )

        if not result:
            SonnerToast(self.root, "Document not found in database!",variant='error', buttons=[{"text": "OK", "command": None}])
            return

        filepath = result[0][0]
        if not os.path.exists(filepath):
            SonnerToast(self.root, "The document file was not found at the original location!",variant='error',
                        buttons=[{"text": "OK", "command": None}])
            return

        self.current_document = filename
        self.current_document_id = doc_id

        # Check if upload_status exists before configuring
        if hasattr(self, 'upload_status') and self.upload_status.winfo_exists():
            self.upload_status.config(text="Analyzing document...")

        self.root.update()
        self.analyze_document_threaded(filepath)


    def show_chat_screen(self):
        self.clear_content_area()
        header_frame = ttk.Frame(self.content_area)
        header_frame.pack(fill=X, pady=10, padx=20)

        # Always show the current document name here
        doc_label = ttk.Label(header_frame,
                              text=f"Document: {self.current_document}",
                              font=('Helvetica', 14, 'bold'))
        doc_label.pack(side=LEFT)

        action_frame = ttk.Frame(header_frame)
        action_frame.pack(side=RIGHT)
        # ... rest of the method ...
        ttk.Button(action_frame, text="📝 New Chat", command=self.new_chat, style='Small.TButton').pack(side=LEFT, padx=5)
        ttk.Button(action_frame, text="💾 Save Chat", command=self.save_current_chat, style='Small.TButton').pack(side=LEFT, padx=5)
        ttk.Button(action_frame, text="⬅️ Back", command=self.show_upload_screen, style='Small.TButton').pack(side=LEFT, padx=5)
        paned_window = ttk.PanedWindow(self.content_area, orient=HORIZONTAL)
        paned_window.pack(fill=BOTH, expand=True)
        chat_frame = ttk.Frame(paned_window, padding=10)
        paned_window.add(chat_frame, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        chat_frame.columnconfigure(0, weight=1)
        self.chat_text = scrolledtext.ScrolledText(chat_frame, wrap=WORD, font=('Helvetica', 11), padx=15, pady=15, bg='white', relief='flat', spacing1=5, spacing3=5)
        self.chat_text.grid(row=0, column=0, sticky='nsew')
        self.chat_text.tag_config("user", background="#e3f2fd", lmargin1=20, rmargin=20, borderwidth=0, relief='flat', spacing2=5)
        self.chat_text.tag_config("bot", background="#f5f5f5", lmargin1=20, rmargin=20, borderwidth=0, relief='flat', spacing2=5)
        self.chat_text.tag_config("bold", font=('Helvetica', 11, 'bold'))
        self.chat_text.tag_config("irrelevant", foreground="#999999", lmargin1=20, rmargin=20)
        self.chat_text.tag_config("highlight", background="#fffde7")
        self.chat_text.tag_config("ref_link", foreground="#4e73df", underline=1)
        self.chat_text.tag_config("source", foreground="#666666", font=('Helvetica', 9))
        self.chat_text.tag_config("timestamp", foreground="#999999", font=('Helvetica', 9))
        input_frame = ttk.Frame(chat_frame)
        input_frame.grid(row=1, column=0, sticky='ew', pady=(10, 0))
        input_frame.columnconfigure(0, weight=1)
        self.question_entry = ttk.Entry(input_frame, font=('Helvetica', 11))
        self.question_entry.grid(row=0, column=0, sticky='ew', padx=5, ipady=8)
        self.question_entry.bind("<Return>", lambda e: self.ask_question())
        ask_btn = ttk.Button(input_frame, text="Ask", command=self.ask_question, style='Accent.TButton')
        ask_btn.grid(row=0, column=1, padx=5, ipadx=15)
        ref_frame = ttk.Frame(paned_window, padding=10)
        paned_window.add(ref_frame, weight=2)
        ref_header = ttk.Frame(ref_frame)
        ref_header.pack(fill=X, pady=(0, 10))
        ttk.Label(ref_header, text="Document Reference", font=('Helvetica', 14, 'bold')).pack(side=LEFT)
        nav_frame = ttk.Frame(ref_header)
        nav_frame.pack(side=RIGHT)
        ttk.Button(nav_frame, text="-", width=3, command=lambda: self.change_zoom(-0.2), style='Small.TButton').pack(side=LEFT, padx=2)
        ttk.Button(nav_frame, text="+", width=3, command=lambda: self.change_zoom(0.2), style='Small.TButton').pack(side=LEFT, padx=2)
        self.prev_btn = ttk.Button(nav_frame, text="◄", width=3, command=lambda: self.change_ref_page(-1), style='Small.TButton')
        self.prev_btn.pack(side=LEFT, padx=2)
        self.next_btn = ttk.Button(nav_frame, text="►", width=3, command=lambda: self.change_ref_page(1), style='Small.TButton')
        self.next_btn.pack(side=LEFT, padx=2)
        self.ref_page_label = ttk.Label(nav_frame, text="Page: 0/0")
        self.ref_page_label.pack(side=LEFT, padx=10)
        pdf_frame = ttk.Frame(ref_frame)
        pdf_frame.pack(fill=BOTH, expand=True)
        self.ref_pdf_canvas = Canvas(pdf_frame, bg='white', highlightthickness=0, borderwidth=0)
        self.ref_pdf_canvas.pack(side=LEFT, fill=BOTH, expand=True, padx=5, pady=5)
        self.ref_info_text = scrolledtext.ScrolledText(ref_frame, wrap=WORD, font=('Helvetica', 10), height=8, bg='#f8f9fc', padx=10, pady=10)
        self.ref_info_text.pack(fill=X, pady=(10, 0))
        if self.current_chat:
            for msg in self.current_chat:
                self._display_message(msg['question'], msg['answer'], msg.get('sources', []), msg.get('is_relevant', True))
        self.update_ref_pdf_viewer()

    def change_zoom(self, delta):
        new_zoom = max(0.5, min(3.0, self.zoom_var.get() + delta))
        self.zoom_var.set(new_zoom)
        self.update_ref_pdf_viewer()

    def show_settings_screen(self):
        self.clear_content_area()

        # Main frame with improved layout
        main_frame = ttk.Frame(self.content_area, padding=20, style='Main.TFrame')
        main_frame.pack(fill=BOTH, expand=True)
        self.style.configure('Main.TFrame', background='#f8f9fc')

        # Title with decorative line
        title_frame = ttk.Frame(main_frame, style='Main.TFrame')
        title_frame.pack(fill=X, pady=(0, 20))
        ttk.Label(title_frame, text="User Settings", font=('Helvetica', 16, 'bold')).pack(side=LEFT)
        ttk.Separator(title_frame, orient='horizontal').pack(side=LEFT, fill=X, expand=True, padx=10)

        # Two-column layout for profile and form
        content_frame = ttk.Frame(main_frame, style='Main.TFrame')
        content_frame.pack(fill=BOTH, expand=True)

        # Left column - Profile picture
        left_frame = ttk.Frame(content_frame, width=200, style='Main.TFrame')
        left_frame.pack(side=LEFT, fill=Y, padx=(0, 20))

        # Profile picture with border
        profile_container = ttk.Frame(left_frame, relief='groove', borderwidth=2, style='Main.TFrame')
        profile_container.pack(pady=10)

        # Fetch profile image path from DB
        user_info = DatabaseManager.execute_query(
            "SELECT profile_image FROM users WHERE id = ?",
            (self.current_user_id,),
            fetch=True
        )
        profile_image_path = user_info[0][0] if user_info and user_info[0][0] else None
        try:
            if profile_image_path and os.path.exists(profile_image_path):
                with open(profile_image_path, "rb") as f:
                    profile_img = Image.open(io.BytesIO(f.read())).resize((150, 150), Image.LANCZOS)
                    profile_img.load()
            else:
                with open(resource_path("user.png"), "rb") as f:
                    profile_img = Image.open(io.BytesIO(f.read())).resize((150, 150), Image.LANCZOS)
                    profile_img.load()
            self.profile_img = ImageTk.PhotoImage(profile_img)
            ttk.Label(profile_container, image=self.profile_img).pack()
        except Exception:
            ttk.Label(profile_container, text="👤", font=('Helvetica', 72)).pack(padx=10, pady=10)
        ttk.Button(left_frame, text="Change Picture", command=self.change_profile_picture, style='Secondary.TButton').pack(pady=5)

        # Right column - User information form
        right_frame = ttk.Frame(content_frame, style='Main.TFrame')
        right_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        # Fetch ALL available user info from database
        user_info = DatabaseManager.execute_query(
            """SELECT username, full_name, email, contact, \
               last_login, created_at FROM users WHERE id = ?""",
            (self.current_user_id,),
            fetch=True
        )

        # Unpack with defaults for all fields
        if user_info:
            (username, full_name, email, contact,
             last_login, created_at) = user_info[0]
        else:
            username = full_name = email = contact = ""
            last_login = created_at = None

        # Form fields - always show all possible fields
        form_frame = ttk.Frame(right_frame, style='Main.TFrame')
        form_frame.pack(fill=X, pady=10)

        # Personal Information Section
        ttk.Label(form_frame, text="Personal Information",
                  font=('Helvetica', 12, 'bold')).grid(row=0, column=0, columnspan=2,
                                                       pady=10, sticky=W)

        # Create form fields with consistent styling
        fields = [
            ("👤 Username:", "settings_username_entry", username, False),
            ("📛 Full Name:", "settings_name_entry", full_name or "", True),
            ("📧 Email:", "settings_email_entry", email or "", True),
            ("📱 Contact:", "settings_contact_entry", contact or "", True),
        ]

        for i, (label, entry_name, default_value, editable) in enumerate(fields):
            ttk.Label(form_frame, text=label).grid(row=i + 1, column=0,
                                                   padx=5, pady=5, sticky=W)
            entry = ttk.Entry(form_frame, width=30)
            entry.grid(row=i + 1, column=1, padx=5, pady=5, sticky='ew')
            entry.insert(0, default_value)
            if not editable:
                entry.config(state='readonly')
            setattr(self, entry_name, entry)

        # Account Information Section
        ttk.Label(form_frame, text="Account Information",
                  font=('Helvetica', 12, 'bold')).grid(row=len(fields) + 1, column=0,
                                                       columnspan=2, pady=10, sticky=W)

        # Read-only account info
        ttk.Label(form_frame, text="🆔 User ID:").grid(row=len(fields) + 2, column=0,
                                                      padx=5, pady=5, sticky=W)
        ttk.Label(form_frame, text=str(self.current_user_id)).grid(row=len(fields) + 2,
                                                                   column=1,
                                                                   padx=5, pady=5,
                                                                   sticky='w')

        # Password change section
        ttk.Label(form_frame, text="Change Password",
                  font=('Helvetica', 12, 'bold')).grid(row=len(fields) + 5, column=0,
                                                       columnspan=2, pady=10, sticky=W)

        password_fields = [
            ("Current Password:", "current_pass_entry", "*"),
            ("New Password:", "new_pass_entry", "*"),
            ("Confirm New Password:", "confirm_pass_entry", "*")
        ]

        for i, (label, entry_name, show) in enumerate(password_fields):
            ttk.Label(form_frame, text=label).grid(row=len(fields) + 6 + i, column=0,
                                                   padx=5, pady=5, sticky=W)
            entry = ttk.Entry(form_frame, width=30, show=show)
            entry.grid(row=len(fields) + 6 + i, column=1, padx=5, pady=5, sticky='ew')
            setattr(self, entry_name, entry)

        # Save button with improved styling
        btn_frame = ttk.Frame(right_frame, style='Main.TFrame')
        btn_frame.pack(fill=X, pady=20)

        ttk.Button(btn_frame, text="Save Changes", command=self.save_settings,
                   style='Accent.TButton').pack(side=RIGHT, ipadx=20)

        # Configure grid weights for responsive layout
        form_frame.columnconfigure(1, weight=1)
        right_frame.columnconfigure(0, weight=1)
        copyright_label = ttk.Label(self.content_area, text="© All rights reserved to @codewithsajjad",
                                    font=("Helvetica", 8), anchor="center")
        copyright_label.pack(side=BOTTOM, fill=X, pady=2)
    def save_settings(self):
        full_name = self.settings_name_entry.get().strip()
        email = self.settings_email_entry.get().strip()
        contact = self.settings_contact_entry.get().strip()
        # Update user info in DB
        DatabaseManager.execute_query(
            "UPDATE users SET full_name = ?, email = ?, contact = ? WHERE id = ?",
            (full_name, email, contact, self.current_user_id)
        )
        SonnerToast(self.root, "Settings saved successfully!", variant='success',buttons=[{"text": "OK", "command": None}])

    def toggle_dark_mode(self):
        if self.dark_mode:
            # Switch to light mode
            self.style.theme_use('clam')
            self.style.configure('.', background='#f8f9fc', foreground='black')
            self.dark_mode = False
        else:
            # Switch to dark mode
            self.style.theme_use('alt')
            self.style.configure('.', background='#2d3748', foreground='white')
            self.dark_mode = True

        # Update all widgets
        for widget in self.root.winfo_children():
            widget.update()

    def update_ref_pdf_viewer(self, highlight=False):
        """Update the reference PDF viewer with improved functionality and DOCX/non-PDF support"""
        # --- PDF logic (unchanged) ---
        if self.analyzer.current_doc:
            # PDF logic as before
            if self.current_highlighted_source:
                page_num = self.current_highlighted_source.get('page', 1) - 1  # Convert to 0-based index
                if page_num < 0 or page_num >= len(self.analyzer.current_doc):
                    print(f"Invalid page number: {page_num + 1}")
                    page_num = 0
                should_highlight = highlight and self.current_highlighted_source.get('text', '')
            else:
                page_num = 0
                should_highlight = False
            zoom = self.zoom_var.get()
            if should_highlight:
                img = self.analyzer.highlight_reference(self.current_highlighted_source, zoom=zoom)
            else:
                img = self.analyzer.get_page_image(page_num, zoom=zoom)
            if img:
                self.ref_pdf_canvas.delete("all")
                self.ref_current_photo = ImageTk.PhotoImage(img)
                self.ref_pdf_canvas.create_image(0, 0, anchor=NW, image=self.ref_current_photo)
                self.ref_pdf_canvas.config(scrollregion=(0, 0, img.width, img.height))
                total_pages = len(self.analyzer.current_doc)
                self.ref_page_label.config(text=f"Page: {page_num + 1}/{total_pages}")
                self.prev_btn.config(state='normal' if page_num > 0 else 'disabled')
                self.next_btn.config(state='normal' if page_num < total_pages - 1 else 'disabled')
                self.ref_info_text.delete(1.0, END)
                if self.current_highlighted_source:
                    self.ref_info_text.insert(END, f"Document: {self.current_highlighted_source.get('filename', '')}\n",
                                              "bold")
                    self.ref_info_text.insert(END, f"Page: {page_num + 1}\n\n")
                    self.ref_info_text.insert(END, "Reference Text:\n", "bold")
                    self.ref_info_text.insert(END, self.current_highlighted_source.get('text', '') + "\n\n")
                    page_sources = [s for s in self.analyzer.chunk_sources if s['page'] == page_num + 1]
                    current_idx = next((i for i, s in enumerate(page_sources)
                                        if s.get('text', '') == self.current_highlighted_source.get('text', '')), -1)
                    if current_idx >= 0:
                        self.ref_info_text.insert(END, "Context:\n", "bold")
                        start_idx = max(0, current_idx - 2)
                        end_idx = min(len(page_sources), current_idx + 3)
                        for i in range(start_idx, end_idx):
                            if i == current_idx:
                                self.ref_info_text.insert(END, "> " + page_sources[i].get('text', '') + "\n",
                                                          "highlight")
                            else:
                                self.ref_info_text.insert(END, page_sources[i].get('text', '') + "\n")
                else:
                    self.ref_info_text.insert(END, f"Document: {self.current_document}\n", "bold")
                    self.ref_info_text.insert(END, f"Page: {page_num + 1}\n\n")
                    self.ref_info_text.insert(END, "Select a message to view its reference in the document")
            return
        # --- Non-PDF logic ---
        # If not a PDF, show chunk/paragraph preview in both Canvas and info box
        self.ref_pdf_canvas.delete("all")
        # Find current chunk index
        if self.current_highlighted_source:
            current_page = self.current_highlighted_source.get('page', 1) - 1
        else:
            current_page = 0
        total_chunks = len(self.analyzer.text_chunks)
        if current_page < 0:
            current_page = 0
        if current_page >= total_chunks:
            current_page = total_chunks - 1
        # Show chunk text in Canvas as well as info box
        if total_chunks > 0:
            chunk = self.analyzer.text_chunks[current_page]
            # Draw text preview in Canvas (wrap lines)
            text = chunk.get('text', '')
            lines = []
            max_line_length = 90
            for paragraph in text.split('\n'):
                while len(paragraph) > max_line_length:
                    split_idx = paragraph.rfind(' ', 0, max_line_length)
                    if split_idx == -1:
                        split_idx = max_line_length
                    lines.append(paragraph[:split_idx])
                    paragraph = paragraph[split_idx:].lstrip()
                lines.append(paragraph)
            y = 20
            for line in lines:
                self.ref_pdf_canvas.create_text(10, y, anchor='nw', text=line, font=('Helvetica', 12), fill='black')
                y += 22
            self.ref_pdf_canvas.config(scrollregion=(0, 0, 800, max(y, 400)))
        else:
            self.ref_pdf_canvas.create_text(300, 300, text="No text chunks available.", font=('Helvetica', 14))
        self.ref_page_label.config(text="Text Mode")
        # Show chunk text and context in ref_info_text
        self.ref_info_text.delete(1.0, END)
        if total_chunks > 0:
            self.ref_info_text.insert(END, f"Document: {self.current_document}\n", "bold")
            self.ref_info_text.insert(END, f"Chunk: {current_page + 1}/{total_chunks}\n\n")
            self.ref_info_text.insert(END, "Reference Text:\n", "bold")
            self.ref_info_text.insert(END, self.analyzer.text_chunks[current_page].get('text', '') + "\n\n")
            # Show some context
            start_idx = max(0, current_page - 2)
            end_idx = min(total_chunks, current_page + 3)
            self.ref_info_text.insert(END, "Context:\n", "bold")
            for i in range(start_idx, end_idx):
                if i == current_page:
                    self.ref_info_text.insert(END, "> " + self.analyzer.text_chunks[i].get('text', '') + "\n",
                                              "highlight")
                else:
                    self.ref_info_text.insert(END, self.analyzer.text_chunks[i].get('text', '') + "\n")
        else:
            self.ref_info_text.insert(END, f"Document: {self.current_document}\n", "bold")
            self.ref_info_text.insert(END, "No text chunks available.")
        # Update navigation buttons for non-PDF
        self.prev_btn.config(state='normal' if current_page > 0 else 'disabled')
        self.next_btn.config(state='normal' if current_page < total_chunks - 1 else 'disabled')

    def update_progress(self, progress):
        """Update progress bar from the queue"""
        self.progress_var.set(progress)
        self.progress_label.config(text=f"{progress}%")
        self.progress_window.update()

    def show_history_screen(self):
        self.clear_content_area()
        ttk.Label(self.content_area, text="Chat History", font=('Helvetica', 16, 'bold')).pack(pady=10)
        copyright_label = ttk.Label(self.content_area, text="© All rights reserved to @codewithsajjad",
                                    font=("Helvetica", 8), anchor="center")
        copyright_label.pack(side=BOTTOM, fill=X, pady=2)
        # Fetch all chat history for the user (no limit)
        history = ChatManager.load_chat_history(self.current_user_id, limit=1000)
        if not history:
            ttk.Label(self.content_area, text="No chat history found").pack(pady=20)
            return
        tree_frame = ttk.Frame(self.content_area)
        tree_frame.pack(fill=BOTH, expand=True, pady=10)
        columns = ("timestamp", "document", "preview")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="browse")
        tree.heading("timestamp", text="Date/Time")
        tree.heading("document", text="Document")
        tree.heading("preview", text="Preview")
        tree.column("timestamp", width=150)
        tree.column("document", width=200)
        tree.column("preview", width=400)
        scrollbar = ttk.Scrollbar(tree_frame, orient=VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=RIGHT, fill=Y)
        tree.pack(fill=BOTH, expand=True)
        for item in history:
            tree.insert("", "end", iid=item['id'], values=(item['timestamp'], item['document'], item['preview']))

        def on_load_chat():
            selected_item = tree.focus()
            if not selected_item:
                SonnerToast(self.root, "Please select a chat from the list!",variant='warning', buttons=[{"text": "OK", "command": None}])
                return
            chat_id = int(selected_item)  # Use iid as chat_id
            try:
                messages = ChatManager.load_chat_messages(chat_id)
                if messages:
                    self.current_chat = messages
                    self.current_chat_id = chat_id
                    result = DatabaseManager.execute_query(
                        "SELECT document_id FROM chats WHERE id = ?",
                        (chat_id,),
                        fetch=True
                    )
                    if result and result[0][0]:
                        doc_result = DatabaseManager.execute_query(
                            "SELECT filename, filepath FROM documents WHERE id = ?",
                            (result[0][0],),
                            fetch=True
                        )
                        if doc_result:
                            filename, filepath = doc_result[0]
                            self.current_document = filename
                            self.current_document_id = result[0][0]
                            if os.path.exists(filepath):
                                self.analyzer.load_document(filepath)
                                self.show_chat_screen()
                                for msg in messages:
                                    if msg.get('sources'):
                                        for src in msg['sources']:
                                            if src.get('sentence', ''):
                                                self.current_highlighted_source = src
                                                self.update_ref_pdf_viewer(highlight=True)
                                                break
                                        break
                            else:
                                SonnerToast(self.root, "The document file was not found at the original location!",variant="warning", buttons=[{"text": "OK", "command": None}])
                        else:
                            SonnerToast(self.root, "Document information not found!", variant='error',buttons=[{"text": "OK", "command": None}])
                    else:
                        self.current_document = None
                        self.current_document_id = None
                        self.show_chat_screen()
                        SonnerToast(self.root, "Loaded Successfully!", variant='success',
                                    buttons=[{"text": "ok", "command": None}])
                else:
                    SonnerToast(self.root, "Could not load the selected chat!", variant='error',buttons=[{"text": "OK", "command": None}])
            except Exception as e:
                print(f"Error loading chat: {e}")
                SonnerToast(self.root, "An error occurred while loading the chat!", buttons=[{"text": "OK", "command": None}])

        def on_delete_chat():
            selected_item = tree.focus()
            if not selected_item:
                SonnerToast(self.root, "Please select a chat to delete.",variant='warning', buttons=[{"text": "OK", "command": None}])
                return
            chat_id = int(selected_item)  # Use iid as chat_id
            def on_yes():
                DatabaseManager.execute_query("DELETE FROM chats WHERE id = ? AND user_id = ?",
                                              (chat_id, self.current_user_id))
                tree.delete(selected_item)
                SonnerToast(self.root, "Chat deleted.",variant='success', buttons=[{"text": "OK", "command": None}])
                self.show_history_screen()  # Refresh after delete
            def on_no():
                pass
            SonnerToast(self.root, "Are you sure you want to delete this chat?",variant='info', buttons=[{"text": "Yes", "command": on_yes}, {"text": "No", "command": on_no}])

        ttk.Button(self.content_area, text="Load Selected Chat", command=on_load_chat, style='Secondary.TButton').pack(
            pady=5)
        ttk.Button(self.content_area, text="Delete Selected Chat", command=on_delete_chat, style='Danger.TButton').pack(
            pady=5)

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def clear_content_area(self):
        for widget in self.content_area.winfo_children():
            widget.destroy()

    def handle_login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        if not username or not password:
            SonnerToast(self.root, "Please enter both username and password!",variant='warning', buttons=[{"text": "OK", "command": None}])
            return

        success, message = UserManager.authenticate_user(username, password)
        if success:
            # Get user ID
            result = DatabaseManager.execute_query(
                "SELECT id FROM users WHERE username = ?",
                (username,),
                fetch=True
            )

            if result:
                self.current_user = username
                self.current_user_id = result[0][0]
                self.show_main_app()
                SonnerToast(self.root, "Welcome Back!", variant='success', buttons=[{"text": "OK", "command": None}])
            else:
                SonnerToast(self.root, "User data not found!",variant='error', buttons=[{"text": "OK", "command": None}])
        else:
            SonnerToast(self.root, "Login Failed: " + message, variant='error',buttons=[{"text": "OK", "command": None}])

    def handle_register(self):
        # Get all fields
        name = self.reg_name_entry.get().strip()
        email = self.reg_email_entry.get().strip()
        contact = self.reg_contact_entry.get().strip()
        username = self.reg_username_entry.get().strip()
        password = self.reg_password_entry.get()
        confirm = self.reg_confirm_entry.get()

        # Validate all fields
        if not all([name, email, contact, username, password, confirm]):
            SonnerToast(self.root, "Please fill in all fields!",variant='warning', buttons=[{"text": "OK", "command": None}])
            return

        if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
            SonnerToast(self.root, "Please enter a valid email address!",variant='warning', buttons=[{"text": "OK", "command": None}])
            return

        if password != confirm:
            SonnerToast(self.root, "Passwords do not match!", buttons=[{"text": "OK", "command": None}])
            return

        # Check username availability one more time
        result = DatabaseManager.execute_query(
            "SELECT id FROM users WHERE username = ?",
            (username,),
            fetch=True
        )
        if result:
            SonnerToast(self.root, "Username is already taken!",variant='error', buttons=[{"text": "OK", "command": None}])
            return

        # Create new user with additional fields
        try:
            DatabaseManager.execute_query(
                """INSERT INTO users (username, password_hash, full_name, email, contact, created_at) 
                VALUES (?, ?, ?, ?, ?, ?)""",
                (username, UserManager.hash_password(password), name, email, contact,
                 datetime.now().isoformat())
            )  # Closing parenthesis added here

            # Create user chat directory
            os.makedirs(os.path.join(CHATS_DIR, username), exist_ok=True)

            SonnerToast(self.root, "Registration successful!",variant='success', buttons=[{"text": "OK", "command": None}])
            self.show_login_screen()
        except Exception as e:
            SonnerToast(self.root, "Registration failed: " + str(e), variant='error',buttons=[{"text": "OK", "command": None}])

    def check_username_availability(self, event=None):
        username = self.reg_username_entry.get().strip()
        if not username:
            self.username_status_label.config(text="", foreground="gray")
            return

        # Check in database
        result = DatabaseManager.execute_query(
            "SELECT id FROM users WHERE username = ?",
            (username,),
            fetch=True
        )

        if result:
            self.username_status_label.config(text="Username taken", foreground="red")
        else:
            self.username_status_label.config(text="Username available", foreground="green")

    def update_password_strength(self, event=None):
        password = self.reg_password_entry.get()
        if not password:
            self.password_strength.config(text="")
            return

        strength = 0
        # Length check
        if len(password) >= 8: strength += 1
        if len(password) >= 12: strength += 1
        # Complexity checks
        if re.search(r'[A-Z]', password): strength += 1
        if re.search(r'[a-z]', password): strength += 1
        if re.search(r'[0-9]', password): strength += 1
        if re.search(r'[^A-Za-z0-9]', password): strength += 1

        if strength <= 2:
            self.password_strength.config(text="Weak", foreground="red")
        elif strength <= 4:
            self.password_strength.config(text="Medium", foreground="orange")
        else:
            self.password_strength.config(text="Strong", foreground="green")

    def check_password_match(self, event=None):
        password = self.reg_password_entry.get()
        confirm = self.reg_confirm_entry.get()

        if not password or not confirm:
            self.password_match_label.config(text="")
            return

        if password == confirm:
            self.password_match_label.config(text="Passwords match", foreground="green")
        else:
            self.password_match_label.config(text="Passwords don't match", foreground="red")

    def show_progress_window(self):
        """Show progress window during document processing"""
        self.progress_window = Toplevel(self.root)
        self.progress_window.title("Processing Document")
        self.progress_window.geometry("300x150")
        self.progress_window.resizable(False, False)

        # Center the progress window
        self.progress_window.update_idletasks()
        width = self.progress_window.winfo_width()
        height = self.progress_window.winfo_height()
        x = (self.progress_window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.progress_window.winfo_screenheight() // 2) - (height // 2)
        self.progress_window.geometry(f'+{x}+{y}')

        ttk.Label(self.progress_window, text="Analyzing document...").pack(pady=10)

        self.progress_var = IntVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_window, orient=HORIZONTAL,
            length=200, mode='determinate', variable=self.progress_var
        )
        self.progress_bar.pack(pady=10)

        self.progress_label = ttk.Label(self.progress_window, text="0%")
        self.progress_label.pack()

        cancel_btn = ttk.Button(
            self.progress_window, text="Cancel",
            command=self.cancel_processing
        )
        cancel_btn.pack(pady=10)

        self.progress_window.grab_set()
        self.progress_window.protocol("WM_DELETE_WINDOW", self.cancel_processing)

        # Start progress updater
        self.update_progress_from_queue()

    def update_progress_from_queue(self):
        """Check the queue for progress updates"""
        try:
            while not self.analyzer.progress_queue.empty():
                progress = self.analyzer.progress_queue.get()
                self.progress_var.set(progress)
                self.progress_label.config(text=f"{progress}%")
                self.progress_window.update()
        except:
            pass

        if self.progress_window:
            self.root.after(100, self.update_progress_from_queue)

    def cancel_processing(self):
        """Cancel the current document processing"""
        self.analyzer.cancel_processing()
        if self.progress_window:
            self.progress_window.destroy()
            self.progress_window = None
        SonnerToast(self.root, "Document processing was cancelled!",variant='success', buttons=[{"text": "OK", "command": None}])

    def document_analysis_complete(self, success, filepath):
        # Handle completion of document analysis
        try:
            if self.progress_window:
                self.progress_window.destroy()
                self.progress_window = None
            if success:
                # Only update upload_status if it exists and is valid
                if hasattr(self, 'upload_status') and self.upload_status and self.upload_status.winfo_exists():
                    self.upload_status.config(text="Document analyzed successfully!")
                self.current_document = os.path.basename(filepath)
                # Always save to database after analysis
                existing = DatabaseManager.execute_query(
                    "SELECT id FROM documents WHERE user_id = ? AND filepath = ?",
                    (self.current_user_id, filepath),
                    fetch=True
                )
                if not existing:
                    self.current_document_id = ChatManager.save_document(
                        self.current_user_id,
                        self.current_document,
                        filepath
                    )
                else:
                    self.current_document_id = existing[0][0]
                self.current_chat = []
                self.current_chat_id = None
                self.show_chat_screen()
                self.current_highlighted_source = None
                self.update_ref_pdf_viewer()
                print("[DEBUG] Documents in DB for user:", DatabaseManager.execute_query(
                    "SELECT id, filename, filepath FROM documents WHERE user_id = ?", (self.current_user_id,),
                    fetch=True))
            else:
                if hasattr(self, 'upload_status') and self.upload_status and self.upload_status.winfo_exists():
                    self.upload_status.config(text="Error analyzing document")
                SonnerToast(self.root, "Failed to analyze the document!",variant='error', buttons=[{"text": "OK", "command": None}])
        except Exception as e:
            print(f"Error completing document analysis: {e}")
            if hasattr(self, 'upload_status') and self.upload_status and self.upload_status.winfo_exists():
                self.upload_status.config(text="Error analyzing document")
            SonnerToast(self.root, "An unexpected error occurred while analyzing the document!",variant='error', buttons=[{"text": "OK", "command": None}])

    def ask_question(self):
        """Process user question and generate answer with improved accuracy. Save chat to DB after each Q&A."""
        question = self.question_entry.get().strip()
        if not question:
            return
        self.question_entry.delete(0, END)
        self._display_message(question, is_user=True)
        result = self.analyzer.query_documents(question)
        if not isinstance(result, dict):
            result = {'answer': str(result), 'confidence': 0.0, 'source': None, 'highlighted_image': None}
        answer = result.get('answer', '')
        sources = [result.get('source')] if result.get('source') else []
        # Determine if answer is relevant (not a 'not relevant' message)
        is_relevant = not answer.startswith("This question does not appear to be related") and not answer.startswith(
            "Sorry, I encountered an error")
        if is_relevant:
            answer = answer.strip()
            if sources:
                confidence = self.calculate_confidence(answer, sources[0].get('sentence', ''))
                if confidence < 0.7:
                    answer = f"Based on the document, {answer}"
                elif confidence < 0.9:
                    answer = f"According to the document, {answer}"
            answer = self.format_answer(answer)
        self._display_message(answer, is_user=False, sources=sources, is_relevant=is_relevant)
        # Save to current chat (in-memory) only if relevant
        if is_relevant:
            self.current_chat.append({
                'question': question,
                'answer': answer,
                'sources': sources,
                'is_relevant': is_relevant,
                'timestamp': datetime.now().isoformat()
            })
            # Save chat to database after each Q&A
            if self.current_user_id and self.current_document_id:
                ChatManager.save_chat(self.current_user_id, self.current_document_id, self.current_chat)
                # Debug print: show all chats for this user
                print("[DEBUG] Chats in DB for user:", ChatManager.load_chat_history(self.current_user_id))
        if sources:
            self.current_highlighted_source = sources[0]
            self.update_ref_pdf_viewer(highlight=True)

    def _display_message(self, message, is_user=True, sources=None, is_relevant=True):
        """Display a message in the chat with improved formatting and interaction"""
        tag = "user" if is_user else ("bot" if is_relevant else "irrelevant")

        # Configure tags with improved styling
        self.chat_text.tag_config("ref_link", foreground="blue", underline=1)
        self.chat_text.tag_config("highlight", background="#fffde7")
        self.chat_text.tag_config("source", foreground="#666666", font=('Helvetica', 9))

        if is_user:
            self.chat_text.insert(END, "You: ", ("bold", tag))
            self.chat_text.insert(END, message + "\n\n", tag)
        else:
            self.chat_text.insert(END, "Assistant: ", ("bold", tag))
            self.chat_text.insert(END, message + "\n\n", tag)

            if sources and is_relevant:
                for source in sources:
                    # Use the page number from the source directly
                    page_num = source.get('page', 1)
                    ref_text = f"[View Reference: Page {page_num}]"
                    self.chat_text.insert(END, ref_text + "\n", "ref_link")

                    # Create unique tags for each reference link
                    ref_tag = f"ref_{len(self.chat_text.tag_names())}"
                    self.chat_text.tag_add(ref_tag, "end-2c linestart", "end-2c")

                    # Bind click event to the specific reference
                    self.chat_text.tag_bind(ref_tag, "<Button-1>",
                                            lambda e, s=source: self.show_reference(s))
                    self.chat_text.tag_bind(ref_tag, "<Enter>",
                                            lambda e: self.chat_text.config(cursor="hand2"))
                    self.chat_text.tag_bind(ref_tag, "<Leave>",
                                            lambda e: self.chat_text.config(cursor=""))

        self.chat_text.insert(END, "\n", tag)
        self.chat_text.see(END)
        self.update_nav_buttons()

    def update_nav_buttons(self):
        if not self.analyzer.current_doc:
            self.prev_btn.config(state='disabled')
            self.next_btn.config(state='disabled')
        else:
            current_page = self.current_highlighted_source['page'] - 1 if self.current_highlighted_source else 0
            self.prev_btn.config(state='normal' if current_page > 0 else 'disabled')
            self.next_btn.config(
                state='normal' if current_page < len(self.analyzer.current_doc) - 1 else 'disabled')

    def show_reference(self, source):
        try:
            if not self.analyzer.current_doc:
                SonnerToast(self.root, "No document is currently loaded!",variant='warning', buttons=[{"text": "OK", "command": None}])
                return
            if not source or not source.get('page'):
                SonnerToast(self.root, "Invalid reference source!", variant='error',buttons=[{"text": "OK", "command": None}])
                return
            self.current_highlighted_source = source.copy()
            page_num = source.get('page', 1) - 1
            if page_num < 0 or page_num >= len(self.analyzer.current_doc):
                SonnerToast(self.root, f"Invalid page number: {page_num + 1}", variant='error',buttons=[{"text": "OK", "command": None}])
                return
            print(f"[DEBUG] Jumping to page {page_num + 1} for highlight, text: {source.get('text', '')[:50]}")
            self.update_ref_pdf_viewer(highlight=True)
            self.update_nav_buttons()
        except Exception as e:
            print(f"Error showing reference: {e}")
            SonnerToast(self.root, "Could not display the reference!",variant='warning', buttons=[{"text": "OK", "command": None}])

    def load_selected_chat(self, tree):
        """Load a selected chat from history"""
        selected_item = tree.focus()
        if not selected_item:
            SonnerToast(self.root, "Please select a chat from the list!", variant='warning',buttons=[{"text": "OK", "command": None}])
            return

        item_data = tree.item(selected_item)
        chat_id = int(item_data['values'][0])  # First value is the ID

        try:
            # Load the chat messages
            messages = ChatManager.load_chat_messages(chat_id)

            if messages:
                self.current_chat = messages
                self.current_chat_id = chat_id

                # Get document info if available
                result = DatabaseManager.execute_query(
                    "SELECT document_id FROM chats WHERE id = ?",
                    (chat_id,),
                    fetch=True
                )

                if result and result[0][0]:
                    doc_result = DatabaseManager.execute_query(
                        "SELECT filename, filepath FROM documents WHERE id = ?",
                        (result[0][0],),
                        fetch=True
                    )

                    if doc_result:
                        filename, filepath = doc_result[0]
                        self.current_document = filename
                        self.current_document_id = result[0][0]

                        # Load the document if it exists
                        if os.path.exists(filepath):
                            self.analyzer.load_document(filepath)
                            self.show_chat_screen()

                            # Highlight the first answer's source if available
                            for msg in messages:
                                if msg.get('sources'):
                                    self.current_highlighted_source = msg['sources'][0]
                                    self.update_ref_pdf_viewer(highlight=True)
                                    break
                            SonnerToast(self.root, "Chat Loaded successfully!",variant='success',
                                        buttons=[{"text": "OK", "command": None}])
                        else:
                            SonnerToast(self.root, "The document file was not found at the original location!", variant='warning',buttons=[{"text": "OK", "command": None}])
                    else:
                        SonnerToast(self.root, "Document information not found!",variant='error', buttons=[{"text": "OK", "command": None}])
                else:
                    self.current_document = None
                    self.current_document_id = None
                    self.show_chat_screen()
            else:
                SonnerToast(self.root, "Could not load the selected chat!",variant='error', buttons=[{"text": "OK", "command": None}])
        except Exception as e:
            print(f"Error loading chat: {e}")
            SonnerToast(self.root, "An error occurred while loading the chat!",variant='warning', buttons=[{"text": "OK", "command": None}])

    def logout(self):
        # Save current chat if there is one
        if self.current_chat and (self.current_document_id or self.current_chat_id):
            if self.current_chat_id:
                # Update existing chat
                pass  # In a real app, you might want to update the existing chat
            else:
                # Create new chat record
                self.current_chat_id = ChatManager.save_chat(
                    self.current_user_id,
                    self.current_document_id,
                    self.current_chat
                )

        # Reset state
        self.current_user = None
        self.current_user_id = None
        self.analyzer = DocumentAnalyzer()
        self.current_document = None
        self.current_document_id = None
        self.current_chat = []
        self.current_chat_id = None
        self.current_highlighted_source = None

        # Show login screen
        self.show_login_screen()

    def calculate_confidence(self, answer, source_text):
        """Calculate confidence score for the answer"""
        # Calculate semantic similarity
        answer_embedding = embedding_model.encode([answer])
        source_embedding = embedding_model.encode([source_text])
        similarity = util.pytorch_cos_sim(answer_embedding, source_embedding).item()

        # Calculate word overlap
        answer_words = set(word_tokenize(answer.lower()))
        source_words = set(word_tokenize(source_text.lower()))
        overlap = len(answer_words.intersection(source_words)) / len(answer_words)

        # Combine scores
        confidence = (similarity + overlap) / 2
        return confidence

    def format_answer(self, answer):
        return self.analyzer.format_answer(answer)

    def change_ref_page(self, delta):
        """Change the current page in the reference viewer"""
        if not self.analyzer.current_doc:
            return

        if self.current_highlighted_source:
            current_page = self.current_highlighted_source['page'] - 1
        else:
            current_page = 0

        new_page = current_page + delta
        if 0 <= new_page < len(self.analyzer.current_doc):
            if self.current_highlighted_source:
                self.current_highlighted_source['page'] = new_page + 1
            else:
                # Create a new source for navigation
                self.current_highlighted_source = {
                    'page': new_page + 1,
                    'filename': self.current_document,
                    'sentence': ''
                }
            self.update_ref_pdf_viewer(highlight=False)
            self.update_nav_buttons()

    def show_my_documents_screen(self, filtered_docs=None):
        self.clear_content_area()

        # Header with search
        header_frame = ttk.Frame(self.content_area)
        header_frame.pack(fill=X, pady=10)

        ttk.Label(header_frame, text="My Documents", font=('Helvetica', 16, 'bold')).pack(side=LEFT)
        copyright_label = ttk.Label(self.content_area, text="© All rights reserved to @codewithsajjad",
                                    font=("Helvetica", 8), anchor="center")
        copyright_label.pack(side=BOTTOM, fill=X, pady=2)
        # Search box
        search_frame = ttk.Frame(header_frame)
        search_frame.pack(side=RIGHT)

        self.search_entry = ttk.Entry(search_frame, width=30)
        self.search_entry.pack(side=LEFT, padx=5)
        self.search_entry.bind('<KeyRelease>', self.filter_documents)

        ttk.Button(search_frame, text="🔍", width=3, command=self.filter_documents, style='Small.TButton').pack(
            side=LEFT)

        # Document grid view
        canvas = Canvas(self.content_area)
        scrollbar = ttk.Scrollbar(self.content_area, orient=VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

        # Get documents
        if filtered_docs is None:
            documents = DatabaseManager.execute_query(
                "SELECT filename, uploaded_at, filepath, id FROM documents WHERE user_id = ? ORDER BY uploaded_at DESC",
                (self.current_user_id,),
                fetch=True
            )
        else:
            documents = filtered_docs

        # Display as cards
        row, col = 0, 0
        for doc in documents:
            card_frame = ttk.Frame(scrollable_frame, padding=10, relief='raised', borderwidth=1)
            card_frame.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')

            # Document icon based on type
            ext = os.path.splitext(doc[0])[1].lower()
            icon = "📄"  # Default
            if ext == '.pdf':
                icon = "📕"
            elif ext in ('.docx', '.doc'):
                icon = "📘"
            elif ext == '.txt':
                icon = "📝"
            elif ext in ('.xlsx', '.csv'):
                icon = "📊"

            ttk.Label(card_frame, text=icon, font=('Helvetica', 24)).pack()
            ttk.Label(card_frame, text=doc[0], font=('Helvetica', 10), wraplength=150).pack()
            ttk.Label(card_frame, text=doc[1].split('T')[0], font=('Helvetica', 8)).pack()

            # Action buttons
            btn_frame = ttk.Frame(card_frame)
            btn_frame.pack(pady=5)

            ttk.Button(btn_frame, text="Open", width=6,
                       command=lambda id=doc[3], name=os.path.basename(doc[0]): self.load_existing_document(id, name),
                       style='Accent.TButton').pack(side=LEFT, padx=2)

            ttk.Button(btn_frame, text="Delete", width=6,
                       command=lambda id=doc[3], name=doc[0]: self.delete_document(id, name),
                       style='Danger.TButton').pack(side=LEFT, padx=2)

            col += 1
            if col > 2:  # 3 columns
                col = 0
                row += 1

        # Configure grid weights
        for i in range(3):
            scrollable_frame.columnconfigure(i, weight=1)

    def filter_documents(self, event=None):
        """Filter documents based on search text"""
        search_text = self.search_entry.get().lower()
        # Get all documents
        documents = DatabaseManager.execute_query(
            "SELECT filename, uploaded_at, filepath, id FROM documents WHERE user_id = ? ORDER BY uploaded_at DESC",
            (self.current_user_id,),
            fetch=True
        )
        # Filter documents
        filtered_docs = [doc for doc in documents if search_text in doc[0].lower()]
        # Clear existing cards
        for widget in self.content_area.winfo_children():
            if isinstance(widget, Canvas):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Frame):
                        child.destroy()
        # Display filtered documents
        self.show_my_documents_screen(filtered_docs)

    def change_profile_picture(self):
        filetypes = [("Image files", "*.png;*.jpg;*.jpeg;*.gif"), ("All files", "*.*")]
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath:
            ext = os.path.splitext(filepath)[1]
            new_profile_path = os.path.join(PROFILES_DIR, f"{self.current_user}{ext}")
            # Remove old profile image if exists
            user_info = DatabaseManager.execute_query(
                "SELECT profile_image FROM users WHERE id = ?",
                (self.current_user_id,),
                fetch=True
            )
            old_profile_path = user_info[0][0] if user_info and user_info[0][0] else None
            if old_profile_path and os.path.exists(old_profile_path):
                try:
                    os.remove(old_profile_path)
                except Exception:
                    pass
            shutil.copy(filepath, new_profile_path)
            DatabaseManager.execute_query(
                "UPDATE users SET profile_image = ? WHERE id = ?",
                (new_profile_path, self.current_user_id)
            )
            SonnerToast(self.root, "Profile picture updated!", variant='success',buttons=[{"text": "OK", "command": None}])
            # Immediately update the UI
            self.show_settings_screen()
            self.show_main_app()

    def new_chat(self):
        """Start a new chat session"""
        if self.current_chat:
            def on_yes():
                self.save_current_chat()
                self.current_chat = []
                self.current_chat_id = None
                self.current_highlighted_source = None
                self.show_chat_screen()
            def on_no():
                self.current_chat = []
                self.current_chat_id = None
                self.current_highlighted_source = None
                self.show_chat_screen()
            SonnerToast(self.root, "Do you want to save the current chat before starting a new one?",variant='info', buttons=[{"text": "Yes", "command": on_yes}, {"text": "No", "command": on_no}], )
        else:
            self.current_chat = []
            self.current_chat_id = None
            self.current_highlighted_source = None
            self.show_chat_screen()

    def save_current_chat(self):
        """Save the current chat to database"""
        if not self.current_chat:
            SonnerToast(self.root, "No chat to save!",variant='warning', buttons=[{"text": "OK", "command": None}])
            return
        if not self.current_document_id:
            SonnerToast(self.root, "No document selected!", variant='warning',buttons=[{"text": "OK", "command": None}])
            return
        try:
            if self.current_chat_id:
                # Update existing chat
                DatabaseManager.execute_query(
                    "DELETE FROM messages WHERE chat_id = ?",
                    (self.current_chat_id,)
                )
                DatabaseManager.execute_query(
                    "DELETE FROM sources WHERE message_id IN (SELECT id FROM messages WHERE chat_id = ?)",
                    (self.current_chat_id,)
                )
            else:
                # Create new chat
                DatabaseManager.execute_query(
                    "INSERT INTO chats (user_id, document_id, created_at) VALUES (?, ?, ?)",
                    (self.current_user_id, self.current_document_id, datetime.now().isoformat())
                )
                self.current_chat_id = DatabaseManager.execute_query(
                    "SELECT id FROM chats WHERE user_id = ? AND document_id = ? ORDER BY created_at DESC LIMIT 1",
                    (self.current_user_id, self.current_document_id),
                    fetch=True
                )[0][0]
            # Save messages
            for msg in self.current_chat:
                DatabaseManager.execute_query(
                    '''INSERT INTO messages 
                    (chat_id, question, answer, is_relevant, created_at) 
                    VALUES (?, ?, ?, ?, ?)''',
                    (self.current_chat_id, msg['question'], msg['answer'], int(msg['is_relevant']), msg['timestamp'])
                )
                message_id = DatabaseManager.execute_query(
                    "SELECT id FROM messages WHERE chat_id = ? ORDER BY created_at DESC LIMIT 1",
                    (self.current_chat_id,),
                    fetch=True
                )[0][0]
                # Save sources
                if msg.get('sources'):
                    for source in msg['sources']:
                        DatabaseManager.execute_query(
                            '''INSERT INTO sources 
                            (message_id, page_number, text, start_char, end_char) 
                            VALUES (?, ?, ?, ?, ?)''',
                            (
                                message_id,
                                source.get('page', 1),
                                source.get('sentence', ''),
                                source.get('char_span', (0, 0))[0],
                                source.get('char_span', (0, 0))[1]
                            )
                        )
            SonnerToast(self.root, "Chat saved successfully!", variant='success',buttons=[{"text": "OK", "command": None}])
        except Exception as e:
            SonnerToast(self.root, "Failed to save chat: " + str(e), variant='error',buttons=[{"text": "OK", "command": None}])

    def delete_document(self, doc_id, filename):
        # Remove from database
        DatabaseManager.execute_query(
            "DELETE FROM documents WHERE id = ? AND user_id = ?",
            (doc_id, self.current_user_id)
        )
        # Optionally, delete the file from disk
        # result = DatabaseManager.execute_query(
        #     "SELECT filepath FROM documents WHERE id = ?", (doc_id,), fetch=True
        # )
        # if result:
        #     filepath = result[0][0]
        #     if os.path.exists(filepath):
        #         os.remove(filepath)
        self.show_my_documents_screen()
        SonnerToast(self.root, f"Document '{filename}' deleted.",variant='success', buttons=[{"text": "OK", "command": None}])


def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


if __name__ == "__main__":
    root = Tk()
    try:
        root.iconbitmap(resource_path("logo.ico"))
    except Exception as e:
        print(f"Error setting window icon: {e}")
    app = DocumentQnAApp(root)
    root.mainloop()
