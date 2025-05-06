# Combined Interview Prep and Real-time Coaching Tool (with Milvus)

import os
import sys # Ensure sys is imported
# import getpass # No longer needed
import time
import threading
import queue
import traceback
from io import BytesIO
from typing import List, Dict, Any, Optional

# --- File 1 Imports ---
# import google.generativeai as genai # Not used
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer

# --- File 2 Imports ---
import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import mediapipe as mp
# import ipywidgets as widgets # Replaced with print for broader compatibility
# from IPython.display import display, clear_output # Replaced with print
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



# --- Milvus Imports ---
try:
    from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility
    MILVUS_AVAILABLE = True
except ImportError:
    print("[WARN] pymilvus library not found. Milvus integration will be disabled.") # Keep warning
    MILVUS_AVAILABLE = False
    # Define dummy classes/functions if library is missing to avoid NameErrors later
    class Collection: pass
    class utility:
        @staticmethod
        def has_collection(name): return False
    class connections:
        @staticmethod
        def connect(*args, **kwargs): raise ConnectionError("pymilvus not installed")
    class DataType: pass
    class FieldSchema: pass
    class CollectionSchema: pass



# --- Configuration & Setup ---

# -- File 1 Config --
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "resume_collection"
EMBEDDING_DIMENSION = 384 # For 'all-MiniLM-L6-v2'
INDEX_TYPE = "IVF_FLAT"
METRIC_TYPE = "L2" # Or "COSINE" or "IP"
INDEX_PARAMS = {"nlist": 1024}
FETCH_TRENDING_QUESTIONS = True
NUM_TRENDING_RESULTS = 3

# -- File 2 Config --
SAMPLE_RATE = 16000 # Hz
CHUNK_DURATION = 10 # seconds
FEEDBACK_INTERVAL = 12 # seconds (Note: Real-time feedback printout is disabled below)
WEBCAM_INDEX = 0 # Default webcam
MAX_FRAMES_FOR_METRICS = 30 * 5 # Store last 5 seconds
SESSION_TIMEOUT = 300 # seconds (5 minutes)


# Set environment variables
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# --- API Key Handling ---
# (getpass removed)
# print("Attempting to get Groq API Key from environment variable 'GROQ_API_KEY'...") # Commented out
groq_api_key = os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    print("[IMPORTANT] Groq API Key NOT FOUND in environment variable 'GROQ_API_KEY'.") # Keep important warnings
    print("            Please set the environment variable before running the script.")
    print("            Features requiring Groq (transcription, feedback, question generation) will be disabled.")
# else:
    # print("Groq API Key found in environment variable.") # Commented out

# --- Initialize Clients & Models ---
groq_client = None
llm = None
if groq_api_key:
    try:
        groq_client = Groq(api_key=groq_api_key)
        # Ensure the correct model name is used if changed previously
        llm = ChatGroq(temperature=0.2, model_name="meta-llama/llama-4-maverick-17b-128e-instruct", api_key=groq_api_key) # Adjusted model name based on previous context, verify if needed
        # print("Groq API client and Langchain LLM initialized successfully.") # Commented out
    except Exception as e:
        print(f"Error initializing Groq/Langchain: {e}") # Keep errors
        print(traceback.format_exc(), file=sys.stderr)
        groq_client = None
        llm = None
else:
    print("Groq API Key not set. Transcription, feedback, and question generation features will be unavailable.") # Keep important warnings

# Sentence Transformer Model
sentence_model = None
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    model_dim = sentence_model.get_sentence_embedding_dimension()
    if model_dim != EMBEDDING_DIMENSION:
         print(f"[WARN] SentenceTransformer model dimension ({model_dim}) does not match EMBEDDING_DIMENSION constant ({EMBEDDING_DIMENSION}). Adjust EMBEDDING_DIMENSION.") # Keep warnings
    # print(f"Sentence Transformer model loaded successfully (Dimension: {model_dim}).") # Commented out
except Exception as e:
    print(f"Error loading Sentence Transformer model: {e}. Embedding generation/Milvus storage disabled.") # Keep errors
    sentence_model = None
    MILVUS_AVAILABLE = False

# MediaPipe Pose
pose_detector = None
mp_drawing = None
mp_pose = None
pose_landmarks_list = []
try:
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    # print("MediaPipe Pose initialized successfully.") # Commented out
except Exception as e:
    print(f"Error initializing MediaPipe Pose: {e}. Body language analysis disabled.") # Keep errors
    print(traceback.format_exc(), file=sys.stderr)

# --- Milvus Connection Setup ---
resume_collection = None
milvus_connected = False

def connect_milvus():
    """Connects to the Milvus instance."""
    global MILVUS_HOST, MILVUS_PORT, milvus_connected
    if not MILVUS_AVAILABLE:
        # print("Milvus library not available. Skipping connection.") # Commented out
        return False
    try:
        # print(f"Attempting to connect to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...") # Commented out
        if "default" in connections.list_connections():
             connections.disconnect("default")
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT, timeout=10)
        # print(f"Successfully connected to Milvus.") # Commented out
        milvus_connected = True
        return True
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}") # Keep errors
        milvus_connected = False
        return False

def create_milvus_collection() -> Optional[Collection]:
     """Creates the Milvus collection if it doesn't exist, or loads it."""
     global COLLECTION_NAME, EMBEDDING_DIMENSION, milvus_connected, resume_collection
     if not MILVUS_AVAILABLE or not milvus_connected:
         # print("Milvus not available or not connected. Cannot create/load collection.") # Commented out
         return None
     try:
        has_coll = utility.has_collection(COLLECTION_NAME)
        if has_coll:
            # print(f"Collection '{COLLECTION_NAME}' already exists. Loading...") # Commented out
            collection = Collection(COLLECTION_NAME)
            if not collection.has_index():
                 print(f"[WARN] Collection '{COLLECTION_NAME}' exists but has no index. Creating index...") # Keep warnings
                 index_params_dict = {"index_type": INDEX_TYPE, "metric_type": METRIC_TYPE, "params": INDEX_PARAMS}
                 collection.create_index(field_name="embedding", index_params=index_params_dict)
                 # print("Index created.") # Commented out
            collection.load()
            # print(f"Collection '{COLLECTION_NAME}' loaded.") # Commented out
            resume_collection = collection
            return collection
        else:
            # print(f"Creating collection '{COLLECTION_NAME}'...") # Commented out
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="job_title", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="resume_text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION)
            ]
            schema = CollectionSchema(fields, description="Resume vector collection", enable_dynamic_field=False)
            collection = Collection(name=COLLECTION_NAME, schema=schema, using='default')
            # print("Collection object created.") # Commented out
            # print("Creating index on 'embedding' field...") # Commented out
            index_params_dict = {"index_type": INDEX_TYPE, "metric_type": METRIC_TYPE, "params": INDEX_PARAMS}
            collection.create_index(field_name="embedding", index_params=index_params_dict)
            # print("Index created.") # Commented out
            # print("Loading collection...") # Commented out
            collection.load()
            # print(f"Collection '{COLLECTION_NAME}' created and loaded.") # Commented out
            resume_collection = collection
            return collection
     except Exception as e:
         print(f"Error during Milvus collection creation/loading: {e}") # Keep errors
         print(traceback.format_exc(), file=sys.stderr)
         resume_collection = None
         return None

# --- Shared Resources ---
audio_queue = queue.Queue(maxsize=5)
transcript_queue = queue.Queue(maxsize=10)
metrics_queue = queue.Queue(maxsize=30)
stop_event = threading.Event()
MAIN_LOOP_RUNNING = True

# --- Phase 1 Functions (Resume Parsing & Question Generation) ---
def parse_resume(resume_path: str) -> str:
    """Parses text content from PDF, DOCX, or TXT files."""
    text = ""
    if not os.path.exists(resume_path):
        print(f"Error: Resume file not found at '{resume_path}'") # Keep errors
        return ""
    try:
        if resume_path.lower().endswith(".pdf"):
            with open(resume_path, 'rb') as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text: text += page_text + "\n"
        elif resume_path.lower().endswith(".docx"):
            document = Document(resume_path)
            for paragraph in document.paragraphs: text += paragraph.text + "\n"
        elif resume_path.lower().endswith(".txt"):
            with open(resume_path, 'r', encoding='utf-8', errors='ignore') as file: text = file.read()
        else:
            print(f"Unsupported file format: {os.path.splitext(resume_path)[1]}") # Keep errors
            return ""
        # print(f"Successfully parsed resume: {os.path.basename(resume_path)}") # Commented out
        text = '\n'.join([line.strip() for line in text.splitlines() if line.strip()])
        text = ' '.join(text.split())
        return text.strip()
    except Exception as e:
        print(f"Error parsing resume '{resume_path}': {e}") # Keep errors
        return ""

def generate_embedding(text: str) -> Optional[List[float]]:
    """Generates sentence embedding using the loaded model."""
    global sentence_model
    if sentence_model is None: return None
    if not text: return None
    try:
        embedding = sentence_model.encode(text)
        return embedding.tolist()
    except Exception as e: print(f"Error generating embedding: {e}"); return None # Keep errors

def store_in_milvus(collection: Optional[Collection], job_title: str, resume_text: str, embedding: Optional[List[float]]):
    """Stores resume data and embedding in the provided Milvus collection."""
    if collection is None or not MILVUS_AVAILABLE or not milvus_connected: return
    if embedding is None: print("Embedding not generated, skipping Milvus storage."); return # Keep info
    if not job_title or not resume_text: print("Missing job title or resume text, skipping Milvus storage."); return # Keep info
    if len(resume_text) > 65535: print("[WARN] Resume text exceeds max length (65535), truncating for Milvus storage."); resume_text = resume_text[:65535] # Keep warning
    data = [[job_title], [resume_text], [embedding]]
    try:
        # print(f"Attempting to insert data for '{job_title}' into Milvus...") # Commented out
        mr = collection.insert(data)
        collection.flush()
        # print(f"Resume for '{job_title}' stored in Milvus (IDs: {mr.primary_keys}).") # Commented out
    except Exception as e: print(f"Error storing data in Milvus: {e}"); print(traceback.format_exc(), file=sys.stderr) # Keep errors

def generate_interview_questions_groq(resume_text: str, job_title: str) -> str:
    """Generates interview questions using Groq's LLM via LangChain."""
    global llm
    if llm is None: return "Groq LLM not available for question generation." # Keep info
    if not resume_text or not job_title: return "Cannot generate questions without resume text and job title." # Keep info
    # Modified prompt to ask for 15 questions including coding - CHECK IF LLM MODEL SUPPORTS THIS WELL
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an AI interview coach. Based on the following resume text and the target job title, generate 15 diverse interview questions. Include a mix of technical, behavioral, coding and situational questions relevant to the role and the candidate's background. Keep questions concise and clear, formatted as a numbered list. Job Title: {job_title}\n\nResume Text (Excerpt):\n---\n{resume_text[:3000]}\n---\n\nGenerate 15 Questions (numbered list):\n1.\n2.\n...\n15."""), # Adjusted prompt
        ("human", "Generate interview questions based on the resume and job title.")])
    chain = prompt | llm | StrOutputParser()
    try:
        questions = chain.invoke({})
        return questions.strip() if questions else "Groq LLM returned an empty response."
    except Exception as e: print(f"Error generating questions with Groq LLM: {e}"); return f"Error during Groq question generation: {type(e).__name__}" # Keep errors

def fetch_trending_questions_web(search_query: str, num_results: int = 5) -> List[str]:
    """Fetches relevant links from DuckDuckGo HTML search results."""
    if not search_query: return []
    search_url = f"https://duckduckgo.com/html/?q={search_query.replace(' ', '+')}+interview+questions"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    links = []
    # print(f"Fetching related links for '{search_query}'...") # Commented out
    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('a', class_='result__a')
        count = 0; added_domains = set()
        for link in results:
            href = link.get('href')
            if href and ('interview' in href.lower() or 'question' in href.lower()):
                if href.startswith('/l/'):
                    try:
                        from urllib.parse import unquote, parse_qs
                        query_params = parse_qs(href.split('?')[-1])
                        target_url = query_params.get('uddg', [None])[0]
                        if target_url: href = unquote(target_url)
                    except Exception: pass
                domain = href.split('/')[2].replace('www.', '') if href.startswith('http') else None
                if domain and domain in added_domains: continue
                links.append(href)
                if domain: added_domains.add(domain)
                count += 1
                if count >= num_results: break
        # print(f"Found {len(links)} relevant links.") # Commented out
        return links
    except requests.exceptions.RequestException as e: print(f"Error fetching trending questions from web: {e}"); return [] # Keep errors
    except Exception as e: print(f"Error parsing web search results: {e}"); return [] # Keep errors

# --- Phase 2 Functions (Real-time Analysis & Feedback) ---
def transcribe_audio_chunk_groq(audio_data: np.ndarray, sr: int, client: Optional[Groq]) -> str:
    """Transcribes an audio chunk using Groq's Whisper API."""
    if not client: return "Error: Groq client not available."
    if audio_data is None or audio_data.size < int(sr * 0.1): return ""
    if not isinstance(audio_data, np.ndarray): return "Error: Invalid audio data format."
    buffer = None
    try:
        buffer = BytesIO()
        if audio_data.dtype != np.float32: audio_data = audio_data.astype(np.float32)
        audio_clipped = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
        sf.write(buffer, audio_int16, sr, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        temp_filename = "audio_chunk.wav"
        file_content = buffer.read()
        transcription = client.audio.transcriptions.create(file=(temp_filename, file_content), model="whisper-large-v3", response_format="verbose_json")

        if hasattr(transcription, 'text'): return transcription.text
        elif isinstance(transcription, dict) and 'text' in transcription: return transcription['text']
        else: print(f"[WARN] Unexpected transcription format: {type(transcription)}"); return "Transcription format unrecognized." # Keep warning
    except Exception as e:
        return f"Error during Groq transcription: {type(e).__name__}" # Keep simplified error
    finally:
        if buffer is not None and not buffer.closed: buffer.close()

def update_realtime_metrics(pose_results: Optional[Any]):
    """Analyzes MediaPipe pose results and puts metrics into the queue."""
    global pose_landmarks_list, mp_pose, pose_detector, metrics_queue, MAX_FRAMES_FOR_METRICS
    current_metrics = {"posture_shifts": 0, "head_nods_proxy": 0, "face_visible": False}
    if pose_detector is None or mp_pose is None:
        try: metrics_queue.put(current_metrics, block=False)
        except queue.Full:
            try: metrics_queue.get_nowait(); metrics_queue.put(current_metrics, block=False)
            except (queue.Empty, queue.Full): pass
        return
    try:
        landmarks = None
        if pose_results and pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            face_lmk_indices = [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.MOUTH_LEFT, mp_pose.PoseLandmark.MOUTH_RIGHT]
            visible_face_lmks = sum(1 for i in face_lmk_indices if landmarks[i].visibility > 0.5)
            current_metrics["face_visible"] = visible_face_lmks >= 3
        pose_landmarks_list.append(landmarks)
        if len(pose_landmarks_list) > MAX_FRAMES_FOR_METRICS: pose_landmarks_list.pop(0)
        if len(pose_landmarks_list) > 15 and landmarks is not None:
            last_frame_lmks = pose_landmarks_list[-1]
            prev_frame_lmks = pose_landmarks_list[-10]
            recent_lmks_for_nod = pose_landmarks_list[-15:]
            required_shift_indices = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]
            if last_frame_lmks and prev_frame_lmks and all(last_frame_lmks[i].visibility > 0.4 for i in required_shift_indices) and all(prev_frame_lmks[i].visibility > 0.4 for i in required_shift_indices):
                prev_shoulder_y = (prev_frame_lmks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + prev_frame_lmks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
                curr_shoulder_y = (last_frame_lmks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + last_frame_lmks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
                if abs(curr_shoulder_y - prev_shoulder_y) > 0.02: current_metrics["posture_shifts"] = 1
            nose_y_coords = [lm[mp_pose.PoseLandmark.NOSE].y for lm in recent_lmks_for_nod if lm and lm[mp_pose.PoseLandmark.NOSE].visibility > 0.5]
            if len(nose_y_coords) > 5 and np.std(nose_y_coords) > 0.006: current_metrics["head_nods_proxy"] = 1
    except (IndexError, KeyError, AttributeError): pass
    except Exception as e: print(f"[ERROR] Unexpected error in update_realtime_metrics: {e}") # Keep errors
    try: metrics_queue.put(current_metrics, block=False)
    except queue.Full:
        try: metrics_queue.get_nowait(); metrics_queue.put(current_metrics, block=False)
        except (queue.Empty, queue.Full): print("[ERROR] Metrics queue still full after discarding.", end='\r') # Keep errors

# --- LLM Feedback Functions (Simple AI Agents via LangChain) ---
def generate_bl_feedback_agent(metrics_str: str) -> str:
    """Generates brief, real-time body language feedback using the LLM."""
    global llm
    if not llm: return "LLM N/A (BL)"
    if not metrics_str or metrics_str.isspace(): return "(No significant movement detected)"
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are an interview coach providing VERY brief (10-15 words max), real-time feedback based on basic sensor data. Observations: ({metrics_str}). Give ONLY 1 concise positive note OR actionable tip (e.g., 'Good steady posture!', 'Try to keep your face centered.', 'Slightly restless, maybe take a breath.', 'Face clearly visible.', 'Engaged head movement.'). Be direct and encouraging."),
        ("human", "Brief feedback based on these metrics?")])
    chain = prompt | llm | StrOutputParser()
    try:
        feedback = chain.invoke({})
        return feedback.strip() if feedback else "(BL feedback error)"
    except Exception as e: return f"BL Feedback Error: {type(e).__name__}" # Keep errors

def generate_cs_feedback_agent(transcript_chunk: str) -> str:
    """Generates brief, real-time content/speech feedback using the LLM."""
    global llm
    if not llm: return "LLM N/A (CS)"
    if not transcript_chunk or transcript_chunk.isspace() or "error" in transcript_chunk.lower() or "unrecognized" in transcript_chunk.lower() or len(transcript_chunk) < 15: return "(Listening...)"
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are an interview coach reviewing a short speech segment. Provide ONLY 1 VERY concise (10-15 words max) comment on clarity, conciseness, confidence, filler words OR give brief positive reinforcement (e.g., 'Clear point!', 'Good energy!', 'Watch 'um's.', 'Slightly fast pace.', 'Sounding confident.'). Segment: \"{transcript_chunk[:150]}...\""),
        ("human", "Brief feedback on this speech segment?")])
    chain = prompt | llm | StrOutputParser()
    try:
        feedback = chain.invoke({})
        return feedback.strip() if feedback else "(CS feedback error)"
    except Exception as e: return f"Content Feedback Error: {type(e).__name__}" # Keep errors

# --- Thread Functions ---
def audio_capture_thread_func():
    """Captures audio from microphone and puts chunks into audio_queue."""
    global stop_event, MAIN_LOOP_RUNNING, audio_queue, SAMPLE_RATE, CHUNK_DURATION
    audio_buffer = []; last_chunk_time = time.time(); stream = None
    def audio_callback(indata, frames, time_info, status):
        nonlocal audio_buffer
        if status and status != sd.CallbackFlags.output_underflow: print(f"Audio Callback Status Warning: {status}", file=sys.stderr) # Keep warnings
        if stop_event.is_set(): raise sd.CallbackStop
        audio_buffer.append(indata.copy().astype(np.float32))
    # print("Audio thread started. Initializing audio stream...") # Commented out
    try:
        with sd.InputStream( callback=audio_callback, samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=int(SAMPLE_RATE * 0.2) ) as stream:
            # print(f"Audio stream active. Listening... (Sample Rate: {stream.samplerate} Hz)") # Commented out
            while not stop_event.is_set():
                current_time = time.time()
                if current_time - last_chunk_time >= CHUNK_DURATION and audio_buffer:
                    audio_chunk_np = np.concatenate(audio_buffer, axis=0); audio_buffer = []; last_chunk_time = current_time
                    if stop_event.is_set(): break
                    try: audio_queue.put((audio_chunk_np, SAMPLE_RATE), timeout=1.0)
                    except queue.Full:
                        try: audio_queue.get_nowait(); audio_queue.put((audio_chunk_np, SAMPLE_RATE), block=False)
                        except (queue.Empty, queue.Full): pass
                if stop_event.wait(timeout=0.1): break
            # print("Audio stream processing finished.") # Commented out
    except sd.PortAudioError as pae: print(f"[FATAL] PortAudio Error in audio thread: {pae}. Check microphone.", file=sys.stderr); stop_event.set(); MAIN_LOOP_RUNNING = False # Keep fatal errors
    except sd.CallbackStop: pass # print("Audio callback gracefully stopped.") # Commented out
    except Exception as e: print(f"[ERROR] Unexpected error in audio capture thread: {type(e).__name__} - {e}", file=sys.stderr); stop_event.set() # Keep errors
    # finally: print("Audio thread finished.") # Commented out

def transcription_thread_func():
    """Gets audio chunks, transcribes via Groq, puts text into transcript_queue."""
    global stop_event, groq_client, audio_queue, transcript_queue, SAMPLE_RATE
    # print(f"[{time.strftime('%H:%M:%S')}] Transcription thread started.") # Commented out
    while not stop_event.is_set():
        audio_data = None; sr = None
        try:
            try: audio_data, sr = audio_queue.get(timeout=0.5)
            except queue.Empty: time.sleep(0.05); continue
            if stop_event.is_set(): break
            if audio_data is not None and sr is not None and audio_data.size > SAMPLE_RATE * 0.1:
                transcript = transcribe_audio_chunk_groq(audio_data, sr, groq_client)
                if stop_event.is_set(): break
                if transcript and not stop_event.is_set():
                    try: transcript_queue.put(transcript, timeout=1.0)
                    except queue.Full:
                        try: transcript_queue.get_nowait(); transcript_queue.put(transcript, block=False)
                        except (queue.Empty, queue.Full): pass
        except Exception as e: print(f"[{time.strftime('%H:%M:%S')}] [Transcription Thread ERROR] {type(e).__name__}: {e}", file=sys.stderr); time.sleep(0.5) # Keep errors
        finally:
            if audio_data is not None:
                try: audio_queue.task_done()
                except ValueError: pass
                except Exception as td_e: print(f"[{time.strftime('%H:%M:%S')}] [ERROR] calling audio_queue.task_done(): {td_e}", file=sys.stderr) # Keep errors
    # print(f"[{time.strftime('%H:%M:%S')}] Transcription thread finished.") # Commented out

def feedback_thread_func():
    """Gets data, generates feedback using LLM agents, displays it, and generates final summary."""
    global stop_event, transcript_queue, metrics_queue, llm, mp_pose, FEEDBACK_INTERVAL
    # print("Feedback thread started.") # Commented out
    all_transcripts = []; all_metrics_history = []
    # current_bl_feedback = "(Initializing BL Feedback...)"; current_cs_feedback = "(Initializing CS Feedback...)" # Not needed if not printing RT
    last_feedback_update_time = time.time()

    # --- Comment out Real-time feedback printing ---
    # def print_realtime_feedback(bl_feedback, cs_feedback, latest_speech=""):
        # clear_line = "\r" + " " * 80 + "\r"
        # print(clear_line + "="*50)
        # print(f"--- Real-time Interview Feedback ({time.strftime('%H:%M:%S')}) ---")
        # print(f"Body Language : {bl_feedback}")
        # print(f"Speech/Content: {cs_feedback}")
        # if latest_speech: print(f"Latest Speech : '{latest_speech[:70]}...'")
        # print("="*50)
        # print("Say 'stop analysis' or press 'q' in video window to quit.", end='')
    # --- End comment out ---

    while not stop_event.is_set():
        new_metrics_data = []; new_transcript_data = []
        try:
            while not metrics_queue.empty(): new_metrics_data.append(metrics_queue.get_nowait()); metrics_queue.task_done()
        except (queue.Empty, ValueError): pass
        except Exception as e: print(f" Error getting metrics: {e}") # Keep errors
        try:
            while not transcript_queue.empty():
                item = transcript_queue.get_nowait(); transcript_queue.task_done()
                if item and isinstance(item, str) and "stop analysis" in item.lower().strip():
                    # print(f"\n[{time.strftime('%H:%M:%S')}] 'Stop analysis' voice command detected!") # Commented out
                    stop_event.set(); break
                if item and isinstance(item, str) and "error" not in item.lower() and "unrecognized" not in item.lower() and not item.isspace():
                    new_transcript_data.append(item)
            if stop_event.is_set(): break
        except (queue.Empty, ValueError): pass
        except Exception as e: print(f" Error getting transcript: {e}") # Keep errors

        if new_metrics_data: all_metrics_history.extend(new_metrics_data)
        if new_transcript_data: all_transcripts.extend(new_transcript_data)

        # --- No need to generate or print real-time feedback if only final summary is desired ---
        if stop_event.wait(timeout=0.2): break # Check stop event periodically

    # --- Final Summary ---
    # print(f"\n[{time.strftime('%H:%M:%S')}] Feedback thread: Stop detected. Preparing final summary...") # Commented out
    time.sleep(0.5) # Allow queues to settle
    try:
        print("\n" + "="*60 + "\n--- SESSION STOPPED - FINAL SUMMARY ---\n" + f"--- Timestamp: {time.strftime('%H:%M:%S')} ---\n" + "="*60 + "\n") # Keep this header

        final_transcripts = list(all_transcripts)
        while not transcript_queue.empty():
            try: item = transcript_queue.get_nowait(); final_transcripts.append(item); transcript_queue.task_done()
            except (queue.Empty, ValueError): break
        final_metrics = list(all_metrics_history)
        while not metrics_queue.empty():
            try: item = metrics_queue.get_nowait(); final_metrics.append(item); metrics_queue.task_done()
            except (queue.Empty, ValueError): break

        final_valid_transcripts = [t for t in final_transcripts if t and isinstance(t, str) and not t.isspace() and "error" not in t.lower() and "unrecognized" not in t.lower()]

        print("\n--- Content & Communication Summary ---") # Keep this header
        if final_valid_transcripts and llm:
            full_transcript_text = " ".join(final_valid_transcripts)
            safe_transcript_text = full_transcript_text.encode('utf-8', errors='replace').decode('utf-8')
            max_summary_chars = 15000
            if len(safe_transcript_text) > max_summary_chars:
                # print(f"(Transcript truncated. Original: {len(safe_transcript_text)} chars)") # Commented out
                safe_transcript_text = safe_transcript_text[:max_summary_chars] + "..."

            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an interview coach. Provide a concise summary of the candidate's communication during the interview based on the following transcript. Comment on strengths (e.g., clarity, confidence, articulation, relevance) and areas for improvement (e.g., filler words, rambling, clarity, pace, nervousness). Offer 1-2 specific, actionable suggestions. Be constructive and encouraging. Transcript: ```" + safe_transcript_text + "```"),
                ("human", "Provide overall communication feedback based on the transcript.")])
            summary_chain = summary_prompt | llm | StrOutputParser()
            try:
                summary_result = summary_chain.invoke({})
                # --- MODIFIED PRINT with Encoding Fix ---
                console_encoding = sys.stdout.encoding if sys.stdout.encoding else 'utf-8' # Get console encoding or default to utf-8
                printable_summary = summary_result.encode(console_encoding, errors='replace').decode(console_encoding)
                print(printable_summary)
                # --- END MODIFIED PRINT ---
            except Exception as e:
                print(f"Error generating content summary via LLM: {e}") # Keep errors
                print(traceback.format_exc(), file=sys.stderr) # Keep tracebacks for LLM errors

        elif not llm: print("LLM not available for content summary.") # Keep info
        else: print("No significant speech content detected for summary.") # Keep info

        # ** ADDED DEBUG PRINT for metrics list length **
        print(f"[DEBUG] Length of final_metrics list: {len(final_metrics)}")
        print("\n--- Body Language & Presence Summary ---") # Keep this header
        if final_metrics and llm and mp_pose and pose_detector:
            total_shifts = sum(m.get("posture_shifts", 0) for m in final_metrics if m)
            total_nods = sum(m.get("head_nods_proxy", 0) for m in final_metrics if m)
            face_visible_frames = sum(1 for m in final_metrics if m and m.get("face_visible", False))
            total_frames_processed = len(final_metrics)
            face_visible_percentage = (face_visible_frames / total_frames_processed * 100) if total_frames_processed > 0 else 0
            metrics_summary_str = f"Frames: {total_frames_processed}, Shifts: {total_shifts}, Nods: {total_nods}, Face Visible: {face_visible_percentage:.1f}%."
            # print(f"(Metrics Summary: {metrics_summary_str})") # Commented out
            bl_summary_prompt = ChatPromptTemplate.from_messages([
                ("system", f"You are an interview coach. Based ONLY on the aggregated body language metrics ({metrics_summary_str}), provide brief overall feedback on presence. Comment on posture stability (inferred from shifts), engagement (inferred from head movement/nods and face visibility). Suggest one potential area for awareness (e.g., maintaining consistent eye contact/face visibility, managing restless movements, using gestures). Keep it concise and constructive."),
                ("human", "Provide summary feedback on body language based on these aggregated metrics.")])
            bl_summary_chain = bl_summary_prompt | llm | StrOutputParser()
            try: print(bl_summary_chain.invoke({})) # Print BL summary directly
            except Exception as e: print(f"Error generating body language summary via LLM: {e}") # Keep errors
        elif not llm: print("LLM not available for body language summary.") # Keep info
        elif not mp_pose or not pose_detector: print("MediaPipe Pose was not initialized/active. No body language data collected.") # Keep info
        else: print("No body language data collected or processed for summary.") # Keep info
        print("\n" + "="*60 + "\n--- End of Summary ---\n" + "="*60) # Keep this footer
    except Exception as e: print(f" Error during final summary generation: {e}"); print(traceback.format_exc(), file=sys.stderr) # Keep errors
    # finally: print(f"[{time.strftime('%H:%M:%S')}] Feedback thread finished.") # Commented out



# --- Main Execution Logic ---
if __name__ == "__main__":
    # print("--- Combined Interview Prep & Real-time Coach (Milvus Enabled) ---") # Commented out
    # --- Initialize Milvus ---
    if MILVUS_AVAILABLE:
        if connect_milvus():
            resume_collection = create_milvus_collection()
            # if resume_collection: print("Milvus setup successful.") # Commented out
            # else: print("[WARN] Milvus connected, but failed to create/load collection. Milvus storage disabled."); milvus_connected = False # Keep warning only if needed
        # else: print("[WARN] Failed to connect to Milvus. Proceeding without Milvus integration."); MILVUS_AVAILABLE = False # Keep warning only if needed
    # else: print("Proceeding without Milvus integration (library unavailable or embedding model failed).") # Commented out

    # === Phase 1: Resume Input and Question Generation ===
    # print("\n--- Phase 1: Resume Analysis & Question Generation ---") # Commented out
    try:
        # print("Reading resume path from stdin...") # Commented out
        sys.stdout.flush()
        resume_file_path = sys.stdin.readline().strip()
        # print(f"Received resume path: {resume_file_path}") # Commented out
        sys.stdout.flush()
        if not resume_file_path or not os.path.isfile(resume_file_path):
             print(f"Error: Invalid resume file path received from stdin: '{resume_file_path}'") # Keep errors
             sys.exit(1) # Exit on critical input error

        # print("Reading job title from stdin...") # Commented out
        sys.stdout.flush()
        job_title = sys.stdin.readline().strip()
        # print(f"Received job title: {job_title}") # Commented out
        sys.stdout.flush()
        if not job_title:
             print("Error: Empty job title received from stdin.") # Keep errors
             sys.exit(1) # Exit on critical input error

    except Exception as stdin_err:
         print(f"Error reading initial inputs from stdin: {stdin_err}") # Keep errors
         print("Critical error: Cannot proceed without initial inputs from Streamlit.")
         sys.exit(1)


    # --- Continue with parsing and Phase 1 ---
    parsed_text = parse_resume(resume_file_path)
    generated_questions = ""; fetched_links = []
    if parsed_text:
        if MILVUS_AVAILABLE and milvus_connected and sentence_model and resume_collection:
             # print("\nGenerating resume embedding...") # Commented out
             vector_embedding = generate_embedding(parsed_text)
             if vector_embedding: store_in_milvus(resume_collection, job_title, parsed_text, vector_embedding)
             # else: print("Failed to generate embedding. Skipping Milvus storage.") # Commented out only if error printed in function
        # elif not sentence_model: print("\nSentence model unavailable, cannot generate embedding for Milvus.") # Commented out
        if llm:
            # print("\nGenerating tailored interview questions using Groq LLM...") # Commented out
            generated_questions = generate_interview_questions_groq(parsed_text, job_title)
            print("\n--- AI Generated Questions ---"); print(generated_questions); print("-" * 28) # KEEP THIS
        # else: print("\nSkipping AI question generation (Groq LLM not available).") # Commented out

        if FETCH_TRENDING_QUESTIONS:
             fetched_links = fetch_trending_questions_web(job_title, num_results=NUM_TRENDING_RESULTS)
             if fetched_links:
                 print("\n--- Related Online Resources ---") # KEEP THIS
                 for idx, link in enumerate(fetched_links, 1): print(f"{idx}. {link}") # KEEP THIS
                 print("-" * 28) # KEEP THIS
             # else: print("\nCould not fetch related online resources.") # Commented out
        # else: print("\nSkipping fetching of online resources.") # Commented out
    else: print("\nCould not parse resume. Skipping question generation and embedding.") # Keep info

    # === Phase 2: Real-time Interview Simulation ===
    # print("\n--- Phase 2: Real-time Interview Simulation ---") # Commented out
    # print(f"Session will automatically stop after {SESSION_TIMEOUT / 60:.1f} minutes.") # Commented out
    # print("Or say 'stop analysis' clearly.") # Commented out

    # --- Check prerequisites ---
    # print("\nChecking prerequisites for real-time analysis...") # Commented out

    # === BYPASS HARDWARE CHECKS ===
    # print("Skipping hardware checks as requested...") # Commented out
    mic_available = True
    cam_available = True
    # sys.stdout.flush() # Commented out
    # === END BYPASS ===

    # Check Groq/LLM and MediaPipe availability
    groq_available = groq_client is not None and llm is not None
    pose_available = pose_detector is not None

    # if not groq_available: print("[WARN] Groq client/LLM unavailable...") # Commented out (already handled)
    # if not pose_available: print("[WARN] MediaPipe Pose unavailable...") # Commented out (already handled)

    can_run_realtime = groq_available

    if not can_run_realtime:
        print("\n[ERROR] Cannot start real-time analysis due to missing non-hardware prerequisites (Groq/LLM). Check warnings above.") # Keep errors
        sys.exit(1)
    # else:
        # print("\nProceeding to real-time analysis phase (hardware checks bypassed).") # Commented out


    # --- REMOVED INTERACTIVE INPUT ---
    # print("Starting simulation automatically...") # Commented out
    sys.stdout.flush()
    # --- END REMOVAL ---


    # --- Start Background Threads ---
    # print("\nInitializing background threads...") # Commented out
    stop_event.clear(); MAIN_LOOP_RUNNING = True
    audio_thread = threading.Thread(target=audio_capture_thread_func, daemon=True, name="AudioCaptureThread")
    transcription_thread = threading.Thread(target=transcription_thread_func, daemon=True, name="TranscriptionThread")
    feedback_thread = threading.Thread(target=feedback_thread_func, daemon=True, name="FeedbackThread")
    audio_thread.start(); time.sleep(1.5) # Give audio stream time to init
    if not stop_event.is_set():
        transcription_thread.start(); feedback_thread.start()
    else:
        print("[ERROR] Audio thread failed during initialization. Cannot continue.") # Keep errors
        if audio_thread.is_alive(): audio_thread.join(timeout=1.0)
        sys.exit(1)

    # --- Webcam Initialization and Main Loop ---
    # print(f"Initializing webcam (Index: {WEBCAM_INDEX})...") # Commented out
    cap = None; main_start_time = time.time(); display_image = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened(): raise IOError(f"Cannot open webcam (Index {WEBCAM_INDEX}).") # Let error propagate
        # print("Webcam opened. Main analysis loop starting...") # Commented out
        # print("Displaying video feed. Press 'q' in the window to quit.") # Commented out

        while MAIN_LOOP_RUNNING and not stop_event.is_set():
            if time.time() - main_start_time > SESSION_TIMEOUT: stop_event.set(); break # print(f"\nSession timeout reached."); # Shortened
            ret_grab = cap.grab()
            if not ret_grab: time.sleep(0.05); continue # Slightly longer sleep if grab fails
            if stop_event.is_set(): break
            ret_retrieve, frame = cap.retrieve()
            if not ret_retrieve or frame is None: continue
            if stop_event.is_set(): break

            pose_results = None
            processed_image = frame
            if pose_detector and mp_pose:
                try:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); image_rgb.flags.writeable = False
                    pose_results = pose_detector.process(image_rgb)
                except Exception as pose_err:
                    print(f"[WARN] Error during MediaPipe processing: {pose_err}") # Keep warning
                    processed_image = frame

            update_realtime_metrics(pose_results)

            # --- OPENCV DISPLAY COMMENTED OUT ---
            time.sleep(0.01) # Prevent tight loop

    except (IOError, cv2.error) as e: print(f"[FATAL] Webcam Error: {e}."); stop_event.set() # Keep fatal errors
    except Exception as e: print(f"[FATAL] Unexpected error in main loop: {type(e).__name__} - {e}"); print(traceback.format_exc(), file=sys.stderr); stop_event.set() # Keep fatal errors
    finally:
        # === Phase 3: Cleanup ===
        # print("\n--- Phase 3: Cleanup ---") # Commented out
        stop_event.set(); MAIN_LOOP_RUNNING = False
        if cap is not None and cap.isOpened(): cap.release() # print("Releasing webcam..."); # Shortened
        # print("OpenCV window cleanup skipped.") # Commented out

        # print("Waiting for background threads to complete...") # Commented out
        thread_timeout_transcribe = CHUNK_DURATION + 15.0; thread_timeout_feedback = 20.0
        threads_to_join = [audio_thread, transcription_thread, feedback_thread]
        timeouts = [5.0, thread_timeout_transcribe, thread_timeout_feedback]
        for thread, timeout in zip(threads_to_join, timeouts):
             if thread.is_alive():
                 # print(f"- Waiting for {thread.name} (Timeout: {timeout}s)...") # Commented out
                 thread.join(timeout=timeout)
                 if thread.is_alive(): print(f"  [WARN] {thread.name} did not stop cleanly.") # Keep warning

        # print("Clearing final items from queues (if any)...") # Commented out
        for q in [audio_queue, transcript_queue, metrics_queue]:
            while not q.empty():
                try: q.get_nowait(); q.task_done()
                except (queue.Empty, ValueError): break
        if MILVUS_AVAILABLE and milvus_connected:
            try: connections.disconnect("default"); milvus_connected = False # print("Disconnecting from Milvus..."); # Shortened
            except Exception as e: print(f"Error disconnecting from Milvus: {e}") # Keep error

        # print("\nCleanup complete. Program terminated.") # Commented out
