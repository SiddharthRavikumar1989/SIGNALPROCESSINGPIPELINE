# Add these imports at the top
import base64
from PIL import Image
from pathlib import Path
import streamlit as st
import os
import random
import string
import numpy as np
import wave
import librosa
# Import other necessary libraries...
from silero_vad import get_speech_timestamps, read_audio, load_silero_vad
import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import ASRModel
# from googletrans import Translator # Removed as we use M2M100
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import requests
from transformers import pipeline
import time
from yt_dlp import YoutubeDL
from pydub import AudioSegment
from huggingface_hub import snapshot_download
from audiorecorder import audiorecorder
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import tempfile # Added for temporary file handling
#import torchaudio

# --- Instructions for Running in Docker ---
# (Keep instructions as before)
# ------------------------------------------

# --- Authentication Function (Reads AUTH_KEY from env via st.secrets) ---
def check_authentication():
    """Returns True if the user is authenticated, False otherwise."""
    # Try to get the correct password from Streamlit secrets.
    # In a Docker container without secrets.toml, Streamlit populates this
    # from the environment variable named ST_SECRETS_AUTH_KEY or just AUTH_KEY.
    try:
        correct_password = st.secrets["AUTH_KEY"]
    except KeyError:
        st.error("⚠️ AUTH_KEY secret not found.")
        st.info("Please ensure the AUTH_KEY environment variable is set when running the Docker container.")
        return False
    except Exception as e:
        st.error(f"Error accessing secrets: {e}")
        return False

    # Initialize session state for authentication
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    elif st.session_state["authenticated"]:
        return True # Already authenticated in this session

    # Show password input using a placeholder
    password_placeholder = st.empty()
    password_attempt = password_placeholder.text_input("Enter Access Key:", type="password", key="auth_password_input")

    # Check password on button click
    # Using a specific key for the login button
    if st.button("Login", key="auth_login_button"):
        if password_attempt == correct_password:
            st.session_state["authenticated"] = True
            password_placeholder.empty() # Clear the login form
            st.rerun() # Rerun to show the main app
        elif password_attempt:
            st.error("😕 Incorrect Access Key")
            st.session_state["authenticated"] = False

    return st.session_state["authenticated"]


import streamlit as st
from datetime import datetime, timedelta, date # Import required modules
import os # Keep os if needed elsewhere

# --- Authentication Function (Reads AUTH_KEY info from env via st.secrets) ---
def check_authentication_timed():
    """
    Checks user authentication using a key that expires after a set number of days.
    Reads configuration from Streamlit secrets.
    Returns True if the user is authenticated and the key is valid, False otherwise.
    """
    # --- 1. Read Configuration from Secrets ---
    try:
        # --- Option A: Using Structured Secret (Preferred for secrets.toml) ---
        auth_config = st.secrets["AUTH_KEY"]
        correct_password = auth_config["key"]
        start_date_str = auth_config["start_date"]
        # Ensure valid_days is treated as a string initially for robust parsing checks
        valid_days_str = str(auth_config.get("valid_days", 0)) # Default to 0 if missing

       
    except KeyError as e:
        st.error(f"⚠️ Missing authentication configuration in secrets: Required key '{e}' not found.")
        st.info("Ensure secrets include 'auth.key', 'auth.start_date', and 'auth.valid_days' (or equivalent env vars).")
        return False
    except Exception as e:
        st.error(f"Error accessing authentication secrets: {e}")
        return False

    # --- 2. Parse and Validate Configuration ---
    try:
        # Parse the start date (expecting YYYY-MM-DD)
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        # Parse the validity duration
        valid_days = int(valid_days_str)
        if valid_days < 0:
             st.error("⚠️ Invalid configuration: 'valid_days' cannot be negative.")
             return False # Indicate configuration error
    except ValueError:
        st.error("⚠️ Invalid configuration: Check 'start_date' format (must be YYYY-MM-DD) or 'valid_days' (must be an integer).")
        return False
    except Exception as e:
        st.error(f"Error parsing authentication configuration: {e}")
        return False

    # --- 3. Calculate Expiry Date ---
    # The key is valid from start_date up to (but not including) expiry_date
    expiry_date = start_date + timedelta(days=valid_days)
    today = date.today()

    # --- 4. Check Session State ---
    # Initialize or check existing authentication status
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["auth_message"] = "" # Store message to avoid re-showing on rerun
    elif st.session_state.get("authenticated", False):
        return True

    # --- 5. Display Login Form ---
    password_placeholder = st.empty()
    password_attempt = password_placeholder.text_input("Enter Access Key:", type="password", key="auth_timed_password_input") # Use a unique key

    # Display previous error message if authentication failed on last attempt
    if st.session_state.get("auth_message"):
        st.error(st.session_state["auth_message"])

    if st.button("Login", key="auth_timed_login_button"): # Use a unique key
        # --- 6. Perform Checks on Button Click ---
        is_password_correct = (password_attempt == correct_password)
        # Check if today is on or after the start date AND before the expiry date
        is_within_valid_period = (today >= start_date and today < expiry_date)

        if is_password_correct and is_within_valid_period:
            st.session_state["authenticated"] = True
            st.session_state["auth_message"] = "" # Clear any previous error message
            password_placeholder.empty() # Clear the login form
            st.rerun() # Rerun to show the main app
        elif is_password_correct and not is_within_valid_period:
            if today < start_date:
                message = f"😕 Access Key is not yet valid (Valid from {start_date_str})."
            else: # today >= expiry_date
                message = f"😕 Access Key has expired (Valid for {valid_days} days from {start_date_str})."
            st.session_state["auth_message"] = message
            st.session_state["authenticated"] = False
            st.rerun() # Rerun to show the error message consistently
        elif password_attempt: # Only show incorrect key if they actually entered something
            st.session_state["auth_message"] = "😕 Incorrect Access Key"
            st.session_state["authenticated"] = False
            st.rerun() # Rerun to show the error message consistently
        else:
            # No password entered, do nothing until button clicked again
            st.session_state["auth_message"] = "Please enter the access key." # Optional prompt
            st.rerun()


    return st.session_state["authenticated"] # Return current state

# --- Global Variables and Setup ---
CACHE_DIR = "./model_download" # Using a relative path is often better for Docker builds
TEMP_UPLOAD_DIR = "./temp_uploads" # Directory for uploaded snippets
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
Path(TEMP_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# Load HF API Key from secrets (populated by HF_API_KEY env var)
try:
    # Reads HF_API_KEY environment variable via st.secrets
    hf_api_key = st.secrets["HF_API_KEY"]
    os.environ["HF_API_KEY"] = hf_api_key # Set for libraries that read env var directly
except KeyError:
    st.warning("HF_API_KEY secret not found.")
    st.info("Please ensure the HF_API_KEY environment variable is set. Some features might be limited.")
    hf_api_key = None
except Exception as e:
     st.warning(f"Could not load HF_API_KEY from secrets: {e}. Some features might not work.")
     hf_api_key = None

# Define model identifiers/paths globally or pass them if preferred
QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_MODEL_PATH = os.path.join(CACHE_DIR, "Qwen2.5-1.5B-Instruct")
TRANS_MODEL_NAME = "facebook/m2m100_418M"
ASR_MODEL_REPO = "nvidia/stt_fa_fastconformer_hybrid_large"
ASR_MODEL_FILENAME = "stt_fa_fastconformer_hybrid_large.nemo"
ASR_MODEL_DIR = os.path.join(CACHE_DIR, "stt_fa_fastconformer_hybrid_large")
ASR_MODEL_PATH = os.path.join(ASR_MODEL_DIR, ASR_MODEL_FILENAME)

#fastconformer_model = ASRModel.from_pretrained("nvidia/stt_en_fastconformer_hybrid_large_pc")
#fastconformer_model.change_attention_model("rel_pos_local_attn", [128, 128])  # Switch to local attention
#fastconformer_model.change_subsampling_conv_chunking_factor(1)  # Auto select chunking factor
#vad_model = load_silero_vad()


# --- Model Loading ---
@st.cache_resource # Caches the entire function's return value (the tuple of models)
def load_models():
    """Loads all necessary models. Returns a tuple of models."""
    tokenizer, model, summarizer, trans_tokenizer, trans_model, asr_model = None, None, None, None, None, None
    models_loaded_successfully = False # Flag to track success

    try: # Outermost try block
        st.info("Starting AI model loading process...")

        # --- Load Qwen Tokenizer and Model ---
        try:
            tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL_PATH, device_map="auto", torch_dtype="auto", local_files_only=True
            )
            st.info("Qwen model loaded from local cache.")
        except Exception as e_qwen_local:
            st.info(f"Local Qwen model not found or error ({e_qwen_local}). Attempting download...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, cache_dir=CACHE_DIR)
                model = AutoModelForCausalLM.from_pretrained(
                    QWEN_MODEL_ID, device_map="auto", torch_dtype="auto", cache_dir=CACHE_DIR
                )
                # Attempt to save for next time
                try:
                    tokenizer.save_pretrained(QWEN_MODEL_PATH)
                    model.save_pretrained(QWEN_MODEL_PATH)
                    st.info("Qwen model downloaded and cached.")
                except Exception as e_qwen_save:
                    st.warning(f"Could not save downloaded Qwen model locally: {e_qwen_save}")
            except Exception as e_qwen_download:
                st.error(f"Failed to download Qwen model: {e_qwen_download}")
                raise RuntimeError("Qwen model loading failed.") from e_qwen_download # Propagate critical failure


        # --- Load Summarizer pipeline ---
        if model and tokenizer:
            summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
            st.info("Summarizer pipeline loaded.")
        else:
            raise RuntimeError("Cannot load summarizer without Qwen model/tokenizer.") # Should not happen if Qwen loaded

        # --- Load Translation Model ---
        try:
            trans_tokenizer = M2M100Tokenizer.from_pretrained(
                TRANS_MODEL_NAME, src_lang="fa", cache_dir=CACHE_DIR, local_files_only=True
            )
            trans_model = M2M100ForConditionalGeneration.from_pretrained(
                TRANS_MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True
            )
            st.info("Translation model loaded from local cache.")
        except Exception as e_trans_local:
            st.info(f"Local Translation model not found ({e_trans_local}). Attempting download...")
            try:
                trans_tokenizer = M2M100Tokenizer.from_pretrained(TRANS_MODEL_NAME, src_lang="fa", cache_dir=CACHE_DIR)
                trans_model = M2M100ForConditionalGeneration.from_pretrained(TRANS_MODEL_NAME, cache_dir=CACHE_DIR)
                st.info("Translation model downloaded and cached.")
            except Exception as e_trans_download:
                 st.error(f"Failed to download Translation model: {e_trans_download}")
                 # Decide if this is critical - maybe proceed without translation? For now, let's make it critical.
                 raise RuntimeError("Translation model loading failed.") from e_trans_download

        # --- Load ASR Model ---
        try:
            if not os.path.exists(ASR_MODEL_PATH):
                st.info("ASR model not found locally, attempting download...")
                snapshot_download(repo_id=ASR_MODEL_REPO, cache_dir=CACHE_DIR,
                                  allow_patterns=[ASR_MODEL_FILENAME], local_dir=ASR_MODEL_DIR,
                                  local_dir_use_symlinks=False) # Avoid symlinks in Docker
                if not os.path.exists(ASR_MODEL_PATH):
                     raise FileNotFoundError("ASR model download failed or file not found after download.")
                st.info("ASR model downloaded and cached.")
            else:
                 st.info("ASR model found in local cache.")
            # Restore the ASR model
            asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(restore_path=ASR_MODEL_PATH)
            st.info("ASR model loaded.")
        except Exception as e_asr:
             st.error(f"Error loading/downloading ASR model: {e_asr}")
             raise RuntimeError("ASR model loading failed.") from e_asr # Propagate critical failure

        # If all loads succeeded up to this point
        models_loaded_successfully = True
        st.success("All required AI models appear to be loaded.")

    except Exception as e: # Catch propagated critical errors from inner blocks
        st.error(f"A critical error occurred during the model loading sequence: {e}")
        models_loaded_successfully = False
        # No explicit return here, flow goes to finally, then returns the (partially) None tuple

    finally:
        # This block will ALWAYS execute after the try or except block finishes
        if models_loaded_successfully:
            st.info("Model loading process finished (Status: Success).")
        else:
            # This message appears if the except block was triggered or if loading wasn't fully successful
            st.warning("Model loading process finished (Status: Errors encountered). Check logs above. Some models may be unavailable.")

    return tokenizer, model, summarizer, trans_tokenizer, trans_model, asr_model


# --- Helper Functions (Keep as before, with minor additions if needed) ---
def generate_random_string(length=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

# --- (generate_summary, download_youtube_audio_as_wav, translate_with_m2m100, process_wav_files, correct_and_improve_sentence, preprocess_audio remain unchanged) ---
def generate_summary(text, summarizer_pipeline):
    # ... (no changes)
    if not summarizer_pipeline:
        st.warning("Summarizer model not loaded. Cannot generate summary.")
        return "Summary generation unavailable."
    max_input_length = 20000
    truncated_text = text[:max_input_length] + "..." if len(text) > max_input_length else text
    prompt = f"""
    Please provide a concise abstractive summary (around 5-7 key points or lines) and the topics  of the following text.
    Focus on the main ideas and essential information. Don't include any additional metadata or internal thinking. Structure should be pointers followed by the topic.

    Text: {truncated_text}

    Summary:
    """
    try:
        response = summarizer_pipeline(
            prompt, max_new_tokens=150, num_return_sequences=1, temperature=0.6,
            top_p=0.9, do_sample=True, pad_token_id=summarizer_pipeline.tokenizer.eos_token_id
        )
        generated_text = response[0]['generated_text']
        summary_marker = "Summary:"
        summary_start_index = generated_text.rfind(summary_marker)
        if summary_start_index != -1:
            summary = generated_text[summary_start_index + len(summary_marker):].strip()
        else:
            summary = generated_text.replace(prompt, "").strip()
        return summary if summary else "Summary could not be extracted."
    except Exception as e:
        st.error(f"Error during summary generation: {e}")
        return "Could not generate summary."

def download_youtube_audio_as_wav(url):
    # ... (no changes)
    random_filename = generate_random_string()
    download_dir = "./temp_audio_downloads"
    os.makedirs(download_dir, exist_ok=True)
    output_template = os.path.join(download_dir, random_filename)
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'wav',}],
        'outtmpl': output_template, 'quiet': True, 'noplaylist': True, 'nocheckcertificate': True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            st.info(f"Attempting to download audio from: {url}")
            info_dict = ydl.extract_info(url, download=True)
            downloaded_file_expected = f"{output_template}.wav"
            if os.path.exists(downloaded_file_expected):
                 st.success(f"Audio downloaded: {downloaded_file_expected}")
                 return os.path.abspath(downloaded_file_expected)
            else:
                 found_files = list(Path(download_dir).glob(f"{random_filename}*.wav"))
                 if found_files:
                     st.success(f"Audio downloaded (found match): {found_files[0]}")
                     return os.path.abspath(found_files[0])
                 else:
                     st.error(f"Audio file not found after download attempt in {download_dir}")
                     return None
    except Exception as e:
        st.error(f"Failed to download or process YouTube audio: {e}")
        return None

def translate_with_m2m100(persian_texts, tokenizer, model):
    # ... (no changes)
    if not tokenizer or not model:
        st.warning("Translation model not loaded. Cannot translate.")
        return ["Translation unavailable."] * len(persian_texts if isinstance(persian_texts, list) else [persian_texts])
    try:
        if isinstance(persian_texts, str):
            persian_texts = [persian_texts]
        if not persian_texts or all(not text for text in persian_texts):
             return ["No input text provided."] * len(persian_texts)

        inputs = tokenizer(persian_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["en"], max_length=512
        )
        return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error during translation: {e}")
        return ["Translation failed."] * len(persian_texts)

def translate_with_m2m100_simple(persian_text, trans_tokenizer, trans_model):
    # ... (no changes)
    if not persian_text: return "No text to translate."
    # Handle potential list input if needed, but primarily expect string here
    print("input-------")
    print(persian_text)
    text_to_translate = persian_text[0] if isinstance(persian_text, list) else persian_text

    translation_list = translate_with_m2m100(text_to_translate, trans_tokenizer, trans_model)
    raw_translation = translation_list[0] if translation_list else "Translation failed."

    # Check if translation was successful before attempting correction
    if raw_translation and "failed" not in raw_translation.lower() and "unavailable" not in raw_translation.lower() and "no input text" not in raw_translation.lower():
         corrected_sentence = correct_and_improve_sentence(raw_translation, correct_tokenizer, correct_model)
         return corrected_sentence
    elif not raw_translation:
        return "Translation resulted in empty output."
    else:
        return raw_translation 


def process_wav_files(wav_file_path):
    # ... (no changes)
    try:
        vad_model = load_silero_vad()
        print("after loading silero vad--")
        base_name = os.path.splitext(os.path.basename(wav_file_path))[0]
        output_folder = f"./audio_chunks/{base_name}_chunks"
        os.makedirs(output_folder, exist_ok=True)
        target_sr = 16000
        # Ensure VAD model is available (it should be if loaded earlier)
        if vad_model is None:
            st.error("VAD model not available for processing.")
            print("VAD model not available for processing.")
            return []
        wav = read_audio(wav_file_path, sampling_rate=target_sr)

        print("afetr reading audio")
        print("wav file path--")
        print(wav_file_path)
        speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=target_sr, return_seconds=True)
        print("speech timestamps--")
        print(speech_timestamps)
        if not speech_timestamps:
            st.warning("No speech detected by VAD.")
            print("No speech detected by VAD")
            return []
        audio = AudioSegment.from_file(wav_file_path)
        print("after loading audio segment from torch wav file path")
        chunk_paths = []
        min_duration_ms = 500
        for i, timestamp in enumerate(speech_timestamps):
            start_ms = int(timestamp['start'] * 1000)
            end_ms = int(timestamp['end'] * 1000)
            duration_ms = end_ms - start_ms
            if duration_ms < min_duration_ms: continue
            segment = audio[start_ms:end_ms]
            output_filename = os.path.join(output_folder, f"segment_{i}.wav")
            segment.export(output_filename, format="wav")
            chunk_paths.append(os.path.abspath(output_filename))
        return chunk_paths
    except Exception as e:
        st.error(f"Failed to process WAV file for VAD: {e}")
        return []

def correct_and_improve_sentence(input_sentence, tokenizer, model):
    """
    Corrects grammar and improves phrasing of an English sentence using an LLM,
    returning only the corrected sentence.
    """
    if not tokenizer or not model:
        # st.warning("Correction model not loaded. Cannot correct sentence.") # Keep commented out if desired
        return input_sentence # Return original if models aren't available

    if not input_sentence or not isinstance(input_sentence, str):
         return input_sentence # Return original for invalid input

    # Explicit instruction asking for *only* the corrected sentence
    instruction = (
        "Correct the grammar and improve the phrasing of the following English sentence. "
        "Make it clear and natural-sounding without changing the core meaning. "
        "Output ONLY the corrected sentence itself, without any additional explanation, notes, or introductory text.\n\n"
        f"Input Sentence: {input_sentence}\n\n"
        "Corrected Sentence:"
    )

    try:
        inputs = tokenizer(instruction, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,       # Max length for the *correction* itself
                num_return_sequences=1,
                temperature=0.2,          # Lower temperature for less creative deviation
                top_p=0.9,                # Keep some nucleus sampling
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                # Consider adding specific stop sequences if the model reliably uses them before explanations
                # eos_token_id=tokenizer.eos_token_id # Already implicitly handled by pad_token_id usually
            )

        corrected_sentence_full = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # --- Enhanced Extraction Logic ---
        marker = "Corrected Sentence:"
        marker_index = corrected_sentence_full.rfind(marker) # Use rfind for robustness

        corrected_sentence = "" # Initialize

        if marker_index != -1:
            # Extract text *after* the marker
            potential_output = corrected_sentence_full[marker_index + len(marker):].strip()

            # Find the first newline character AFTER the potential start of the sentence
            first_newline_index = potential_output.find('\n')

            if first_newline_index != -1:
                # If a newline exists, take everything *before* it as the likely correction
                # This effectively cuts off explanations starting on subsequent lines
                corrected_sentence = potential_output[:first_newline_index].strip()
            else:
                # If no newline, the entire extracted part is likely the correction
                corrected_sentence = potential_output
        else:
            # --- Fallback (if marker is missing, less reliable) ---
            # Try to remove the prompt part. This is a guess.
            prompt_text_to_remove = instruction.replace("Corrected Sentence:", "").strip()
            # Check if the full output *starts* with the prompt structure
            if corrected_sentence_full.strip().startswith(prompt_text_to_remove):
                 corrected_sentence = corrected_sentence_full.strip()[len(prompt_text_to_remove):].strip()
            else:
                 corrected_sentence = input_sentence # Default to original if fallback fails


        # --- Final Check and Return ---
        # Return the cleaned sentence only if it's non-empty and actually different from the original input
        if corrected_sentence and corrected_sentence != input_sentence:
            return corrected_sentence
        else:
            # If cleaning failed, resulted in empty string, or is same as input, return the original
            return input_sentence

    except Exception as e:
        # Use st.error only if Streamlit is available in this context
        # Otherwise, print or log the error
        print(f"Error during sentence correction: {e}") # Changed to print
        # st.error(f"Error during sentence correction: {e}")
        return input_sentence # Return original on error


def preprocess_audio(audio_path):
    # ... (no changes)
    try:
        # Ensure audio is loaded as mono and resampled to 16kHz
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        return audio
    except Exception as e:
        st.error(f"Error preprocessing audio {os.path.basename(audio_path)}: {e}")
        return None

def process_and_transcribe(wav_file_path, asr_model):
    """Transcribes a single WAV audio file."""
    if wav_file_path is not None and asr_model:
        print(f"Processing and transcribing: {wav_file_path}") # Debugging
        preprocessed_audio = preprocess_audio(wav_file_path)
        if preprocessed_audio is None:
            st.error(f"Preprocessing failed for {os.path.basename(wav_file_path)}")
            return None
        try:
            # NeMo expects a list of numpy arrays
            audio_signal = [np.asarray(preprocessed_audio)]
            # Use transcribe method which handles batching internally if needed
            transcript_list = asr_model.transcribe(audio=audio_signal, batch_size=1)
            print(f"ASR Raw Output for {os.path.basename(wav_file_path)}: {transcript_list}") # Debugging

            return transcript_list[0]
        except Exception as e:
            st.error(f"Error during ASR transcription for {os.path.basename(wav_file_path)}: {e}")
            return None
    elif not asr_model:
        st.warning("ASR model not loaded. Cannot transcribe.")
        return None
    else:
        st.error("Invalid audio file path provided for transcription.")
        return None

#from collections import deque
#import torch
#from silero_vad import VADIterator
#silero_model = load_silero_vad()
def process_and_transcribe_chunks(wav_file_path, asr_model, vad_threshold=0.4):
    """Transcribes audio using Silero VAD for speech-aware chunking"""
    from collections import deque
    import numpy as np
    import torch

    print(f"Applying silero vad to break down the chunks--")
    vad_model = VADIterator(silero_model, sampling_rate=16000)  # Use 8000 if needed
    print("vad model loaded--")
    print(vad_model)
    # Read and normalize audio
    audio, sr = read_audio(wav_file_path)
    audio = audio.numpy() if torch.is_tensor(audio) else audio
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio.astype(np.float32) / max_val
    else:
        audio = audio.astype(np.float32)

    # VAD processing
    speech_segments = []
    current_segment = []
    speech_buffer = deque(maxlen=3)  # Stability buffer

    print(f"Applying 32ms overlap to handle abrupt cutting--")
    print(f"Length of Audio chunk---{len(audio)}")
    for i in range(0, len(audio), 512):  # 32ms chunks at 16kHz
        chunk = audio[i:i+512]
        if len(chunk) < 512:
            chunk = np.pad(chunk, (0, 512 - len(chunk)), 'constant')
        print(f"chunk")
        print(f"Chunk length--") 
        print(type(chunk))
        print(chunk)
        print("sampling rate")
        print(sr)
        speech_prob = vad_model(chunk, sr)
        print("speech_probability")
        print(speech_prob)
        speech_prob_value = 0.0
        if speech_prob is None:
            speech_prob_value = 0.0
        elif isinstance(speech_prob, dict):
            speech_prob_value = speech_prob.get('start', 0.0)
            print(f"speech_prob_value{speech_prob_value}")
        else:
            speech_prob_value = speech_prob.item()
            print(f"speech prob value{speech_prob_value}")

        speech_buffer.append(speech_prob_value > vad_threshold)

        # Process buffer regardless of whether it's full to catch short speech
        if sum(speech_buffer) >= 1:
            current_segment.extend(chunk)
        elif current_segment:
            speech_segments.append(np.array(current_segment))
            current_segment = []

    # Append any remaining speech segment after the loop
    if current_segment:
        speech_segments.append(np.array(current_segment))

    print(f"Number of Speech Segments {len(speech_segments)}")

    # Transcribe segments
    print("Transcribing each speech segment--")
    transcripts = []
    # min_samples = int(0.5 * sr)
    for seg in speech_segments:
        # if len(seg) < min_samples:
        #     continue
        try:
            print("Inside transcribe loop")
            transcript = asr_model.transcribe(
                audio=[seg],
                # batch_size=1,
                return_hypotheses=False
            )
            transcripts.extend(transcript)
        except Exception as e:
            print(f"Transcription error: {e}")

    print(f"Overall transcripts generated: {len(transcripts)}")
    print(transcripts)
    # Flatten list of lists
    flat_transcripts = [item for sublist in transcripts for item in sublist]
    print("flat_transcripts")
    print(flat_transcripts)

    return " ".join(flat_transcripts)


#def read_audio(path, sampling_rate=16000):
#    """Modified from Silero's example to handle normalization"""
#    audio, sr = torchaudio.load(path)
#    if sr != sampling_rate:
#        audio = torchaudio.functional.resample(audio, sr, sampling_rate)
#    return audio.squeeze(), sampling_rate

    


def process_and_transcribe_long(wav_file_path, asr_model):
    """Transcribes a single WAV audio file."""
    if wav_file_path is not None and fastconformer_model:
        print(f"Processing and transcribing: {wav_file_path}") # Debugging
        preprocessed_audio = preprocess_audio(wav_file_path)
        if preprocessed_audio is None:
            st.error(f"Preprocessing failed for {os.path.basename(wav_file_path)}")
            return None
        try:
            # NeMo expects a list of numpy arrays
            audio_signal = [np.asarray(preprocessed_audio)]
            # Use transcribe method which handles batching internally if needed
            transcript_list =fastconformer_model.transcribe(audio=audio_signal, batch_size=1)
            print(f"ASR Raw Output for {os.path.basename(wav_file_path)}: {transcript_list}") # Debugging

            return transcript_list[0].text
        except Exception as e:
            st.error(f"Error during ASR transcription for {os.path.basename(wav_file_path)}: {e}")
            return None
    elif not asr_model:
        st.warning("ASR model not loaded. Cannot transcribe.")
        return None
    else:
        st.error("Invalid audio file path provided for transcription.")
        return None


def translate_to_en(persian_text, trans_tokenizer, trans_model, correct_tokenizer, correct_model):
    # ... (no changes)
    if not persian_text: return "No text to translate."
    # Handle potential list input if needed, but primarily expect string here
    text_to_translate = persian_text[0] if isinstance(persian_text, list) else persian_text

    translation_list = translate_with_m2m100(text_to_translate, trans_tokenizer, trans_model)
    raw_translation = translation_list[0] if translation_list else "Translation failed."

    # Check if translation was successful before attempting correction
    if raw_translation and "failed" not in raw_translation.lower() and "unavailable" not in raw_translation.lower() and "no input text" not in raw_translation.lower():
         corrected_sentence = correct_and_improve_sentence(raw_translation, correct_tokenizer, correct_model)
         return corrected_sentence
    elif not raw_translation:
        return "Translation resulted in empty output."
    else:
        return raw_translation # Return the failure message


# --- Updated set_page_style function ---
def set_page_style():
    """Sets the page style with a light blue, white, and green palette."""
    st.markdown("""
    <style>
        /* Define color variables for the light theme */
        :root {
            --primary-green: #4CAF50;    /* Main green */
            --secondary-green: #8BC34A;  /* Lighter green */
            --primary-blue: #64B5F6;     /* Light-medium blue */
            --light-blue-bg: #E3F2FD;    /* Very light blue background */
            --white: #FFFFFF;           /* White for cards and accents */
            --dark-text: #374151;       /* Dark gray for text */
            --medium-text: #6B7280;     /* Medium gray for secondary text */
            --border-color: #D1D5DB;    /* Light gray for borders */
            --hover-green: #388E3C;     /* Darker green for hover */
            --hover-blue: #42A5F5;      /* Slightly darker blue for hover */
        }

        /* Apply base styles */
        body, .stApp {
            background-color: var(--light-blue-bg) !important;
            color: var(--dark-text) !important;
            font-family: sans-serif; /* Optional: Set a clean font */
        }

        .main-content {
            padding: 2rem;
            background-color: var(--light-blue-bg); /* Ensure main content matches body */
        }

        /* Sidebar styles */
        .stSidebar > div:first-child {
            background-color: var(--white); /* White sidebar background */
            padding: 1.5rem;
            border-radius: 0 15px 15px 0;
            border-right: 1px solid var(--border-color); /* Add subtle border */
        }

        /* Sidebar button styles */
        .stSidebar .stButton > button {
            background-color: var(--primary-green) !important;
            color: var(--white) !important;
            border: 1px solid var(--primary-green) !important;
            border-radius: 8px !important;
            padding: 10px 15px !important;
            width: 100%;
            margin: 0.5rem 0;
            transition: background-color 0.3s, transform 0.2s;
            text-align: left;
            font-weight: bold; /* Make label clearer */
        }
        .stSidebar .stButton > button:hover {
            background-color: var(--hover-green) !important;
            border-color: var(--hover-green) !important;
            transform: translateX(3px);
        }
        .stSidebar .stButton > button:disabled {
            background-color: #BDBDBD !important; /* Grey out disabled buttons */
            color: #757575 !important;
            border-color: #BDBDBD !important;
            cursor: not-allowed;
        }


        /* General button styles (main content area) */
        /* Targets buttons NOT in the sidebar specifically */
        .stButton > button:not(.stSidebar .stButton > button) {
            background-color: var(--primary-blue);
            color: var(--white);
            border: 1px solid var(--primary-blue);
            border-radius: 8px;
            padding: 10px 20px; /* Adjust padding as needed */
            font-weight: bold; /* Make label clearer */
            transition: background-color 0.3s, border-color 0.3s;
        }
        .stButton > button:not(.stSidebar .stButton > button):hover {
            background-color: var(--hover-blue);
            border-color: var(--hover-blue);
        }
        .stButton > button:not(.stSidebar .stButton > button):disabled {
            background-color: #BDBDBD; /* Grey out disabled buttons */
            color: #757575;
            border-color: #BDBDBD;
            cursor: not-allowed;
        }


        /* Headings */
        h1, h2, h3 {
            color: var(--primary-blue); /* Use primary blue for headings */
        }
        h1 {
            border-bottom: 2px solid var(--primary-green); /* Add subtle underline to main title */
            padding-bottom: 0.5rem;
        }

        /* Input fields */
        .stTextInput input, .stTextArea textarea {
            background-color: var(--white);
            color: var(--dark-text);
            border: 1px solid var(--border-color);
            border-radius: 5px;
        }
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: var(--primary-blue); /* Highlight focus */
            box-shadow: 0 0 0 2px rgba(100, 181, 246, 0.3); /* Soft blue glow on focus */
        }


        /* File uploader */
        .stFileUploader label {
            color: var(--dark-text) !important; /* Ensure label is dark */
            font-weight: bold;
        }
        /* Style the box area of the file uploader */
         .stFileUploader > div > div {
             border: 2px dashed var(--primary-green);
             background-color: #F0FFF4; /* Very light green background */
             border-radius: 8px;
         }

        /* Card styles */
        .audio-card, .segment-card, .summary-card, .snippet-card {
            background-color: var(--white);
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05); /* Lighter shadow */
            border: 1px solid var(--border-color);
        }

        /* Section titles */
        .section-title {
            /* Use border instead of background for separation */
            border-left: 5px solid var(--primary-green);
            padding-left: 15px;
            margin: 1.5rem 0 1rem 0;
        }
        .section-title h3 { /* Style the h3 within the section title */
             margin-bottom: 0.2rem; /* Reduce space below h3 */
             color: var(--dark-text); /* Make section titles darker for readability */
        }

        /* Expander styles */
        .stExpander {
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background-color: var(--white);
            margin-bottom: 1rem;
        }
        .stExpander header {
            font-weight: bold;
            color: var(--dark-text); /* Dark text for expander header */
            padding: 0.75rem 1rem !important; /* Adjust padding */
        }
        .stExpander > div { /* Content inside expander */
             padding: 0 1rem 1rem 1rem; /* Adjust padding */
        }


        /* Audio player styles */
        .stAudio {
            width: 100%;
            margin: 0.5rem 0;
        }
        .stAudio audio {
            border-radius: 5px;
            border: 1px solid var(--border-color); /* Add subtle border */
            width: 100%; /* Ensure audio player takes full width */
        }

        /* Footer */
        .footer {
            margin-top: 3rem;
            padding: 1rem;
            text-align: center;
            color: var(--medium-text); /* Medium gray for footer */
            font-size: 0.9em;
            border-top: 1px solid var(--border-color);
        }

        /* Results box styles */
        .results-box {
             margin: 0.5rem 0;
             padding: 0.8rem 1rem;
             border-radius: 5px;
             border: 1px solid var(--border-color);
             line-height: 1.5; /* Improve readability */
        }
        .results-box b { /* Make labels bold */
             color: var(--dark-text);
        }
        .results-box p {
             margin-top: 0.3rem;
             color: var(--medium-text); /* Slightly lighter text for results */
        }

        /* Specific result box backgrounds */
        .persian-box {
            background-color: #E6FFFA; /* Very light teal/green */
            border-left: 3px solid var(--secondary-green); /* Accent border */
        }
        .english-box {
            background-color: #EFF6FF; /* Very light blue */
            border-left: 3px solid var(--primary-blue); /* Accent border */
        }

        /* --- Remove old wave animation --- */
        /*
        .audio-wave { ... }
        .wave-bar { ... }
        @keyframes wave { ... }
        */

    </style>
    """, unsafe_allow_html=True)


# ====== Main Application Logic ======
def main_app(tokenizer, model, summarizer, trans_tokenizer, trans_model, asr_model):
    set_page_style() # Apply the new styles

    # Simple check if *any* critical model failed (could be more granular)
    if not all([tokenizer, model, summarizer, trans_tokenizer, trans_model, asr_model]):
         st.error("One or more critical AI models failed to load during initialization. Application functionality will be severely limited. Please check the logs.")
         # Optionally display which models are missing
         missing = [name for name, obj in zip(["Tokenizer", "LLM", "Summarizer", "TransTokenizer", "TransModel", "ASRModel"],
                                              [tokenizer, model, summarizer, trans_tokenizer, trans_model, asr_model]) if obj is None]
         if missing:
             st.info(f"Missing/failed models: {', '.join(missing)}")
         st.stop() # Stop execution if critical models are missing

    # Initialize session state for app data (if not already done)
    if 'wav_file_path' not in st.session_state: st.session_state.wav_file_path = None # YouTube processed path
    if 'chunk_paths' not in st.session_state: st.session_state.chunk_paths = []
    if 'transcriptions' not in st.session_state: st.session_state.transcriptions = {} # Segment transcriptions
    if 'summary' not in st.session_state: st.session_state.summary = None
    if 'current_view' not in st.session_state: st.session_state.current_view = "audio" # View for YouTube results
    if 'segment_expanded_state' not in st.session_state: st.session_state.segment_expanded_state = {}

 
    


    # --- NEW Session State for Uploaded Snippet ---
    if 'uploaded_audio_path' not in st.session_state: st.session_state.uploaded_audio_path = None
    if 'snippet_analysis' not in st.session_state: st.session_state.snippet_analysis = {} # {'persian': ..., 'english': ...}

    

    # --- Sidebar for Actions (Related to YouTube processed audio) ---
    with st.sidebar:
        st.markdown("<h3 style='color: var(--dark-text); margin-bottom: 1rem;'>Actions (YouTube Audio)</h3>", unsafe_allow_html=True)

        # Disable sidebar buttons if no YouTube audio is processed
        no_youtube_audio = st.session_state.wav_file_path is None

        # Button to generate summary
        if st.button("📝 Generate Summary", key="summary_btn_sidebar", help="Generate a summary of the entire YouTube audio", use_container_width=True, disabled=no_youtube_audio):
            if st.session_state.wav_file_path:
                st.session_state.current_view = "summary"
                st.session_state.summary = None # Clear previous summary
                with st.spinner("Generating full summary... (Transcription > Translation > Correction > Summarization)"):
                    # Pass all required models explicitly
                    full_transcript_fa = process_and_transcribe(st.session_state.wav_file_path, asr_model)
                    if full_transcript_fa:
                        corrected_english_full = translate_to_en(full_transcript_fa, trans_tokenizer, trans_model, tokenizer, model)
                        if corrected_english_full and "failed" not in corrected_english_full.lower() and "unavailable" not in corrected_english_full.lower():
                            summary_text = generate_summary(corrected_english_full, summarizer)
                            st.session_state.summary = summary_text
                        else:
                             st.warning("Translation or Correction failed, cannot generate summary.")
                             st.session_state.summary = "Summary generation failed (Translation/Correction issue)."
                    else:
                        st.warning("Full transcription failed, cannot generate summary.")
                        st.session_state.summary = "Summary generation failed (Transcription issue)."
                st.rerun() # Rerun to show summary view
            # No need for else, button is disabled

        # Button to extract segments
        if st.button("🎚️ Extract & Analyze Segments", key="extract_btn_sidebar", help="Extract and analyze speech segments from YouTube audio", use_container_width=True, disabled=no_youtube_audio):
             if st.session_state.wav_file_path:
                 st.session_state.current_view = "segments"
                 with st.spinner("Extracting speech segments..."):
                     chunk_paths = process_wav_files(st.session_state.wav_file_path)
                     st.session_state.chunk_paths = chunk_paths
                     st.session_state.transcriptions = {} # Clear old analysis
                     st.session_state.segment_expanded_state = {}
                     if not chunk_paths:
                         st.warning("No speech segments found or error during extraction.")
                 st.rerun() # Rerun to show segment view
             # No need for else, button is disabled

        # Button to show source audio view
        if st.session_state.wav_file_path: # Only show if YT audio exists
             if st.button("🎧 Show YouTube Audio", key="show_audio_btn", use_container_width=True):
                  st.session_state.current_view = "audio"
                  st.rerun()


    # --- Main Content Area ---
    st.markdown('<h1>Audio AI Processor</h1>', unsafe_allow_html=True) # Uses H1 style from CSS

    # --- Input Section: YouTube URL ---
    st.markdown("<div class='section-title'><h3>📡 Process from YouTube</h3></div>", unsafe_allow_html=True)
    youtube_url = st.text_input("Enter YouTube URL:", key="youtube_url", placeholder="Paste YouTube video link here")
    # Use the general button style defined in CSS
    if st.button("Process YouTube Audio", key="download_btn"):
         if youtube_url:
             with st.spinner("Downloading and preparing YouTube audio..."):
                 wav_file_path = download_youtube_audio_as_wav(youtube_url)
                 if wav_file_path and os.path.exists(wav_file_path):
                     # Reset state for YouTube processing
                     st.session_state.wav_file_path = wav_file_path
                     st.session_state.chunk_paths = []
                     st.session_state.transcriptions = {}
                     st.session_state.summary = None
                     st.session_state.current_view = "audio"
                     st.session_state.segment_expanded_state = {}
                     # --- Reset snippet state when processing YouTube ---
                     st.session_state.uploaded_audio_path = None
                     st.session_state.snippet_analysis = {}
                     # ---------------------------------------------------
                     st.success("YouTube audio ready for processing!")
                     st.rerun()
                 else:
                     st.error("Failed to download or locate the audio file.")
         else:
             st.warning("Please enter a YouTube URL.")

    # --- Input Section: Audio Snippet Upload ---
    st.markdown("---") # Separator
    st.markdown("<div class='section-title'><h3>📤 Process Local Audio Snippet</h3></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload an audio snippet (WAV, MP3, etc.)",
        type=['wav', 'mp3', 'ogg', 'flac', 'm4a'], # Added more common types
        key="snippet_uploader"
    )

  
 

    # --- Input Section: Audio Recording ---
    # --- Input Section: Audio Recording ---
    st.markdown("---")
    st.markdown("<div class='section-title'><h3>🎤 Record Audio</h3></div>", unsafe_allow_html=True)

    # Initialize session state
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'recorded_audio_path' not in st.session_state:
        st.session_state.recorded_audio_path = None
    if 'recorded_analysis' not in st.session_state:
        st.session_state.recorded_analysis = {}

    # Start Recording Button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Recording", key="start_recording_btn"):
            st.session_state.recording = True
            st.session_state.recorded_audio_path = None
            st.session_state.recorded_analysis = {}

    with col2:
        if st.button("⏹️ Stop Recording", key="stop_recording_btn"):
            st.session_state.recording = False

    # Audio Recorder UI
    if st.session_state.recording:
        st.warning("Recording in progress... Speak now!")
        audio_rec = audiorecorder("", "Recording...", key="recorder")
    else:
        audio_rec = audiorecorder("", "", key="recorder")

    # When recording stops and audio is available
    if not st.session_state.recording and len(audio_rec) > 0 and not st.session_state.recorded_audio_path:
        random_filename = generate_random_string()
        recorded_path = os.path.join(TEMP_UPLOAD_DIR, f"recorded_{random_filename}.wav")
        audio_rec.export(recorded_path, format="wav")

        st.session_state.recorded_audio_path = recorded_path
        st.rerun()

    # Display recorded audio and transcription
    if st.session_state.recorded_audio_path and os.path.exists(st.session_state.recorded_audio_path):
        st.markdown("<h4>🎧 Recorded Audio</h4>", unsafe_allow_html=True)
        st.audio(st.session_state.recorded_audio_path, format="audio/wav")

        if st.button("📝 Transcribe and Translate Recorded Audio", key="transcribe_recorded_btn"):
            with st.spinner("Transcribing and translating..."):
                persian_transcript = process_and_transcribe(st.session_state.recorded_audio_path, asr_model)
                persian_transcript=''
                english_translation = translate_to_en(persian_transcript, trans_tokenizer, trans_model, tokenizer, model)

                st.session_state.recorded_analysis = {
                'persian': persian_transcript if persian_transcript else "Transcription failed.",
                'english': english_translation
                }
            st.rerun()
    
    # Display results if available
    if st.session_state.recorded_analysis:
        st.markdown("---")
        st.markdown("<h5>🧾 Analysis Results</h5>", unsafe_allow_html=True)

        persian_text = st.session_state.recorded_analysis['persian']
        english_text = st.session_state.recorded_analysis['english']

        st.markdown(f"""
        <div class='results-box persian-box'>
        <b>📝 Persian Transcription:</b>
        <p>{persian_text}</p>
        </div>
        <div class='results-box english-box'>
        <b>🌐 English Translation:</b>
        <p>{english_text}</p>
        </div>
        """, unsafe_allow_html=True)



    if uploaded_file is not None:
        # Save uploaded file temporarily and convert to WAV if needed
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(f"Uploaded: {file_details['FileName']} ({round(file_details['FileSize']/(1024*1024), 2)} MB)") # Show size in MB

        # Generate a unique path in the temp upload directory
        temp_filename_base = generate_random_string()
        temp_wav_path = os.path.join(TEMP_UPLOAD_DIR, f"{temp_filename_base}.wav")

        try:
            # Use a temporary file to handle the upload stream safely
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_original_path = tmp_file.name # Get path of the saved original file

            # Convert to WAV using pydub
            with st.spinner(f"Converting '{uploaded_file.name}' to WAV..."):
                audio = AudioSegment.from_file(temp_original_path)
                # Ensure mono and 16kHz for ASR model compatibility during export
                audio = audio.set_channels(1).set_frame_rate(16000)
                audio.export(temp_wav_path, format="wav")
            st.session_state.uploaded_audio_path = temp_wav_path
            # Clear previous snippet analysis results when a new file is uploaded
            st.session_state.snippet_analysis = {}

            # Clean up the original temporary file
            if os.path.exists(temp_original_path):
                 os.remove(temp_original_path)

            # Optionally, reset YouTube state if needed (or keep separate flows)
            # st.session_state.wav_file_path = None # Decide if uploading should clear YT state

            st.success(f"Snippet '{uploaded_file.name}' processed and ready (as WAV).")
            # No rerun needed here, display happens below based on session state check

        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
            st.session_state.uploaded_audio_path = None
            st.session_state.snippet_analysis = {}
            # Clean up potentially created files on error
            if 'temp_original_path' in locals() and os.path.exists(temp_original_path):
                 os.remove(temp_original_path)
            if os.path.exists(temp_wav_path):
                 os.remove(temp_wav_path)


    # --- Display Area for Uploaded Snippet and Analysis ---
    if st.session_state.uploaded_audio_path and os.path.exists(st.session_state.uploaded_audio_path):
        st.markdown("<div class='snippet-card'>", unsafe_allow_html=True) # Use card style
        st.markdown("<h4>Uploaded Audio Snippet</h4>", unsafe_allow_html=True)
        st.audio(st.session_state.uploaded_audio_path, format="audio/wav")

        # Button to trigger transcription and translation for the snippet
        # Use the general button style defined in CSS
        if st.button("Transcribe and translate uploaded audio", key="transcribe_snippet_btn"):
            with st.spinner("Analyzing uploaded snippet..."):
                snippet_persian = process_and_transcribe(st.session_state.uploaded_audio_path, asr_model)
                snippet_english = "Analysis failed."
                if snippet_persian:
                    snippet_english = translate_to_en(snippet_persian, trans_tokenizer, trans_model, tokenizer, model)

                st.session_state.snippet_analysis = {
                    'persian': snippet_persian if snippet_persian else "Transcription failed or no speech detected.",
                    'english': snippet_english
                }

        # Display snippet analysis results if available
        if st.session_state.get('snippet_analysis'):
            st.markdown("---") # Separator within the card
            st.markdown("<h5>Analysis Results:</h5>", unsafe_allow_html=True)
            persian_result = st.session_state.snippet_analysis.get('persian', 'N/A')
            english_result = st.session_state.snippet_analysis.get('english', 'N/A')

            # Use the updated results-box styles
            st.markdown(f"""
            <div class='results-box persian-box'>
                <b>📝 Persian Transcription:</b>
                <p>{persian_result}</p>
            </div>
            <div class='results-box english-box'>
                <b>🌐 English Translation:</b>
                <p>{english_result}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True) # Close snippet-card

    # --- Display Area (Based on current_view for YouTube results) ---
    st.markdown("---") # Separator before YouTube results

    if st.session_state.wav_file_path: # Only show YT results section if YT audio was processed
        if st.session_state.current_view == "audio":
            st.markdown("<div class='section-title'><h3>🎧 Source YouTube Audio</h3></div>", unsafe_allow_html=True)
            st.markdown("<div class='audio-card'>", unsafe_allow_html=True)
            if os.path.exists(st.session_state.wav_file_path):
                st.audio(st.session_state.wav_file_path, format="audio/wav")
            else:
                st.error("Source YouTube audio file not found. Please process again.")
                st.session_state.wav_file_path = None # Reset if file is missing
            st.markdown("</div>", unsafe_allow_html=True)

        elif st.session_state.current_view == "summary":
            st.markdown("<div class='section-title'><h3>📝 YouTube Audio Summary</h3></div>", unsafe_allow_html=True)
            st.markdown("<div class='summary-card'>", unsafe_allow_html=True)
            if st.session_state.summary:
                # Use st.markdown for potentially better formatting control if needed later
                st.markdown(f"<div style='white-space: pre-line; color: var(--medium-text);'>{st.session_state.summary}</div>", unsafe_allow_html=True)
            else:
                st.info("Summary is being generated or was not requested for the YouTube audio.")
            st.markdown("</div>", unsafe_allow_html=True)

        elif st.session_state.current_view == "segments":
            st.markdown(f"<div class='section-title'><h3>🎚️ YouTube Speech Segments ({len(st.session_state.chunk_paths)})</h3></div>", unsafe_allow_html=True)
            if not st.session_state.chunk_paths:
                st.warning("No segments extracted yet for the YouTube audio. Click 'Extract & Analyze Segments' in the sidebar.")
            else:
                num_columns = 2 # Number of columns for segments
                cols = st.columns(num_columns)
                for i, chunk_path in enumerate(st.session_state.chunk_paths):
                    col_index = i % num_columns
                    with cols[col_index]: # Place content in columns
                        if not os.path.exists(chunk_path):
                            st.warning(f"Segment {i+1} file not found at {chunk_path}. Skipping.")
                            continue

                        segment_key_base = f"segment_{i}"
                        # Get the desired state for this expander
                        is_expanded = st.session_state.segment_expanded_state.get(i, False)

                        # Use Expander within the column
                        with st.expander(f"Segment {i + 1}", expanded=is_expanded):
                            st.audio(chunk_path, format="audio/wav")
                            analysis_results = st.container() # Container for results inside expander

                            # Display existing analysis for this segment
                            if i in st.session_state.transcriptions:
                                with analysis_results:
                                    transcript_data = st.session_state.transcriptions[i]
                                    st.markdown(f"""
                                <div class='results-box persian-box'>
                                    <b>📝 Persian:</b>
                                    <p>{transcript_data.get('persian', 'N/A')}</p>
                                </div>
                                <div class='results-box english-box'>
                                    <b>🌐 English:</b>
                                    <p>{transcript_data.get('english', 'N/A')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                # Show analyze button only if not already analyzed
                                analyze_key = f"{segment_key_base}_analyze"
                                # Use general button style
                                if st.button(f"Analyze Segment {i + 1}", key=analyze_key):
                                    with st.spinner(f"Analyzing Segment {i + 1}..."):
                                        persian_transcript = process_and_transcribe(chunk_path, asr_model)
                                        print(persian_transcript)
                                        english_translation = "Analysis failed."
                                        if persian_transcript:
                                            english_translation = translate_to_en(persian_transcript, trans_tokenizer, trans_model, tokenizer, model)
                                        st.session_state.transcriptions[i] = {
                                            'persian': persian_transcript if persian_transcript else "Transcription failed.",
                                            'english': english_translation
                                        }
                                        # Keep expander open after analysis
                                        st.session_state.segment_expanded_state[i] = True
                                    st.rerun() # Rerun to show results in the expander

    #elif not st.session_state.uploaded_audio_path: # Show message if neither YT nor snippet is loaded
    #    st.info("Enter a YouTube URL or upload an audio snippet to begin.")


    # --- Footer ---
    st.markdown("""
    <div class="footer"> <p>Audio AI Processor v2.5 • Powered by Streamlit</p> </div>
    """, unsafe_allow_html=True)


# ====== App Entry Point ======
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Audio AI Processor", initial_sidebar_state="expanded")

    # Apply styles early, even for login page
    set_page_style()

    if check_authentication():
        # Load models only *after* successful authentication & only once per session
        if 'models_loaded' not in st.session_state:
            st.session_state.models_loaded = False # Initialize flag

        if not st.session_state.models_loaded:
            # Show spinner only during the actual loading phase
            with st.spinner("Loading AI models... This might take a few minutes on first run or download."):
                models_tuple = load_models() # Call the updated load_models function
            st.session_state.models = models_tuple # Store result (even if None)
            st.session_state.models_loaded = True # Mark loading attempt as complete
            st.rerun() # Rerun to proceed with loaded (or None) models

        # Retrieve potentially loaded models from session state
        models_tuple = st.session_state.get('models', (None,) * 6)

        # Check if the first model (tokenizer) is valid as a basic check for success
        if models_tuple and models_tuple[0] is not None:
            # Unpack the tuple and run the main application, passing all models
            main_app(
                tokenizer=models_tuple[0],
                model=models_tuple[1],
                summarizer=models_tuple[2],
 
                trans_tokenizer=models_tuple[3],
                trans_model=models_tuple[4],
                asr_model=models_tuple[5]
            )
 
        elif st.session_state.models_loaded: # If loading was attempted but failed
             # Error message is already shown in load_models or main_app checks
             st.warning("Proceeding with limited functionality or stopping due to model loading errors.")
             
    else:
        st.info("Please enter the access key to unlock the application.")


