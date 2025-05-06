import streamlit as st
import subprocess
import os
import tempfile
import threading
import queue
import time
import sys
import re # Import regular expressions for parsing

# --- Configuration ---
# Ensure this points to your main Python script (e.g., Main.py or your_script.py)
script_path = r"C:\Users\Supriya Gali\Desktop\Sp\Main.py" # Or "your_script.py" - ADJUST IF NEEDED
# --- End Configuration ---

def enqueue_output(pipe, q, pipe_name="stdout"):
    """Reads output from a pipe and puts it into a queue."""
    try:
        for line in iter(pipe.readline, ''):
            q.put((pipe_name, line))
    except ValueError:
        q.put((pipe_name, f"Error: {pipe_name} pipe closed unexpectedly.\n"))
    except Exception as e:
        q.put((pipe_name, f"Error reading {pipe_name} output: {type(e).__name__} - {e}\n"))
    finally:
        try:
            if pipe and not pipe.closed: pipe.close()
        except Exception: pass

def extract_section(output_text, start_marker, end_marker=None):
    """Extracts text between start_marker and end_marker (or end of string)."""
    try:
        start_index = output_text.index(start_marker) + len(start_marker)
        if end_marker:
            # Find the *next* occurrence of end_marker after start_index
            try:
                 end_index = output_text.index(end_marker, start_index)
                 # Go back one newline from the end marker if possible for cleaner split
                 prev_newline = output_text.rfind('\n', start_index, end_index)
                 if prev_newline != -1:
                      end_index = prev_newline
            except ValueError: # end_marker not found after start_marker
                 end_index = len(output_text)
        else:
            end_index = len(output_text)

        section = output_text[start_index:end_index].strip()
        # Optional: Remove leading/trailing hyphens/equals used as separators
        section = re.sub(r'^[-=]+\n?', '', section)
        section = re.sub(r'\n?[-=]+$', '', section)
        return section.strip()
    except ValueError: # start_marker not found
        return None
    except Exception as e:
        # st.warning(f"Error extracting section '{start_marker}': {e}") # Optional debug
        return None


st.set_page_config(layout="wide")
st.title("Run Interview Prep Script")

st.write(f"This app will run the Python script: `{script_path}`")
st.error("Ensure your script has getpass removed and uses environment variables for API keys.")
st.warning(f"Ensure all dependencies for `{script_path}` are installed in the active environment.")
st.warning("The real-time video window from the script will NOT appear here.")

# --- Inputs ---
uploaded_file = st.file_uploader("1. Upload Resume File (PDF, DOCX, TXT)", type=['pdf', 'docx', 'txt'])
job_title = st.text_input("2. Enter Target Job Title")

# --- Initialize Session State ---
if 'process' not in st.session_state: st.session_state.process = None
if 'output' not in st.session_state: st.session_state.output = ""
if 'running' not in st.session_state: st.session_state.running = False
if 'output_queue' not in st.session_state: st.session_state.output_queue = queue.Queue()
if 'read_thread_stdout' not in st.session_state: st.session_state.read_thread_stdout = None
if 'read_thread_stderr' not in st.session_state: st.session_state.read_thread_stderr = None
if 'resume_file_path_for_cleanup' not in st.session_state: st.session_state.resume_file_path_for_cleanup = None
# State for extracted sections
if 'extracted_questions' not in st.session_state: st.session_state.extracted_questions = None
if 'extracted_links' not in st.session_state: st.session_state.extracted_links = None
if 'extracted_comm_summary' not in st.session_state: st.session_state.extracted_comm_summary = None
if 'extracted_bl_summary' not in st.session_state: st.session_state.extracted_bl_summary = None


# --- Execution Control ---
start_button = st.button("Start Analysis", disabled=st.session_state.running, key="start_btn")

if start_button and uploaded_file is not None and job_title:
    if not os.path.exists(script_path):
        st.error(f"Script file not found at: {script_path}.")
    else:
        # Reset state for new run
        st.session_state.output = ""
        st.session_state.extracted_questions = None
        st.session_state.extracted_links = None
        st.session_state.extracted_comm_summary = None
        st.session_state.extracted_bl_summary = None
        st.session_state.running = True

        try:
            file_suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                resume_file_path = tmp_file.name
                st.session_state['resume_file_path_for_cleanup'] = resume_file_path
        except Exception as e:
            st.error(f"Error creating temporary file for resume: {e}")
            st.session_state.running = False # Ensure running is false if temp file fails
            st.stop()

        st.info(f"Running script: {script_path}")
        st.info(f"Inputs -> Resume: {uploaded_file.name}, Job Title: {job_title}")
        st.session_state.output = f"Starting script...\nResume Temp Path: {resume_file_path}\nJob Title: {job_title}\n" + "="*20 + "\n"

        try:
            env = os.environ.copy()
            st.session_state.process = subprocess.Popen(
                ["python", "-u", script_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace',
                bufsize=1, env=env)

            try:
                if st.session_state.process.stdin:
                    st.session_state.process.stdin.write(f"{resume_file_path}\n")
                    st.session_state.process.stdin.write(f"{job_title}\n")
                    st.session_state.process.stdin.flush()
            except (OSError, BrokenPipeError, AttributeError, ValueError) as e:
                 st.warning(f"Could not write initial input to script's stdin: {e}.")

            st.session_state.output_queue = queue.Queue()
            st.session_state.read_thread_stdout = threading.Thread(target=enqueue_output, args=(st.session_state.process.stdout, st.session_state.output_queue, "stdout"), daemon=True)
            st.session_state.read_thread_stderr = threading.Thread(target=enqueue_output, args=(st.session_state.process.stderr, st.session_state.output_queue, "stderr"), daemon=True)
            st.session_state.read_thread_stdout.start()
            st.session_state.read_thread_stderr.start()
            st.rerun()

        except Exception as e:
            st.error(f"Failed to start script process: {e}")
            st.session_state.running = False
            st.session_state.process = None
            if st.session_state.resume_file_path_for_cleanup and os.path.exists(st.session_state.resume_file_path_for_cleanup):
                try: os.unlink(st.session_state.resume_file_path_for_cleanup); st.session_state.resume_file_path_for_cleanup = None
                except Exception as unlink_e: st.warning(f"Could not delete temporary file on start error: {unlink_e}")
            st.rerun()

elif start_button and uploaded_file is None: st.warning("Please upload a resume file.")
elif start_button and not job_title: st.warning("Please enter the target job title.")

# --- Output Display Area ---
st.subheader("Full Script Output Log")
output_placeholder = st.empty()

if st.session_state.running:
    process_return_code = st.session_state.process.poll() if st.session_state.process else None
    process_running = process_return_code is None and st.session_state.process is not None

    while not st.session_state.output_queue.empty():
        try:
            pipe_name, line = st.session_state.output_queue.get_nowait()
            st.session_state.output += f"[STDERR] {line}" if pipe_name == "stderr" else line
        except queue.Empty: break
        except Exception as q_e: st.session_state.output += f"\nError reading from output queue: {q_e}\n"; break

    output_placeholder.text_area("Log", value=st.session_state.output, height=400, key="output_area_running")

    if process_running:
        st.info("Script is running...")
        if st.button("Stop Script", key="stop_btn"):
            st.warning("Attempting to stop script...")
            # ... (Stop button logic remains the same) ...
            try:
                if st.session_state.process:
                    st.session_state.process.terminate()
                    try: st.session_state.process.wait(timeout=2)
                    except subprocess.TimeoutExpired: st.warning("Process did not terminate gracefully, trying kill..."); st.session_state.process.kill()
            except ProcessLookupError: st.info("Process already finished.")
            except Exception as stop_e: st.error(f"Error stopping script process: {stop_e}")
            st.session_state.running = False
            st.rerun()
        time.sleep(1.0)
        st.rerun()
    else:
        # --- Script Finished ---
        st.info(f"Script has finished (Return Code: {process_return_code}). Reading final output...")
        st.session_state.running = False

        # Ensure reading threads finish
        if st.session_state.read_thread_stdout and st.session_state.read_thread_stdout.is_alive(): st.session_state.read_thread_stdout.join(timeout=1.0)
        if st.session_state.read_thread_stderr and st.session_state.read_thread_stderr.is_alive(): st.session_state.read_thread_stderr.join(timeout=1.0)

        # Read final output
        while not st.session_state.output_queue.empty():
             try:
                 pipe_name, line = st.session_state.output_queue.get_nowait()
                 st.session_state.output += f"[STDERR] {line}" if pipe_name == "stderr" else line
             except queue.Empty: break
             except Exception as q_e: st.session_state.output += f"\nError reading final output from queue: {q_e}\n"; break

        # Display final full output
        output_placeholder.text_area("Log", value=st.session_state.output, height=400, key="output_area_final")

        # --- PARSE AND EXTRACT SECTIONS ---
        final_output_text = st.session_state.output
        st.session_state.extracted_questions = extract_section(final_output_text, "--- AI Generated Questions ---", "----------------------------")
        st.session_state.extracted_links = extract_section(final_output_text, "--- Related Online Resources ---", "----------------------------")
        # For summaries, parse from specific headers until the next known header or end marker
        st.session_state.extracted_comm_summary = extract_section(final_output_text, "--- Content & Communication Summary ---", "--- Body Language & Presence Summary ---")
        st.session_state.extracted_bl_summary = extract_section(final_output_text, "--- Body Language & Presence Summary ---", "--- End of Summary ---") # Look for end marker


        # Clean up temp file
        if st.session_state.resume_file_path_for_cleanup and os.path.exists(st.session_state.resume_file_path_for_cleanup):
             try: os.unlink(st.session_state.resume_file_path_for_cleanup); st.session_state.resume_file_path_for_cleanup = None
             except Exception as e: st.warning(f"Could not delete temporary file: {e}")

        # Reset process/thread states
        st.session_state.process = None
        st.session_state.read_thread_stdout = None
        st.session_state.read_thread_stderr = None
        st.rerun() # Rerun to display extracted sections

elif st.session_state.output: # Display previous run's full output if not currently running
     output_placeholder.text_area("Log (Previous Run)", value=st.session_state.output, height=400, key="output_area_stale")

# --- Display Extracted Sections (only after script finishes) ---
if not st.session_state.running and st.session_state.output:
    st.divider()
    st.subheader("Extracted Key Information")

    if st.session_state.extracted_questions:
        with st.expander("AI Generated Questions", expanded=True):
            st.markdown(st.session_state.extracted_questions)
    else:
        st.info("AI Generated Questions section not found in the output.")

    if st.session_state.extracted_links:
        with st.expander("Related Online Resources", expanded=True):
            # Simple split by line, assuming links are one per line
            links_list = [link for link in st.session_state.extracted_links.splitlines() if link.strip()]
            for link in links_list:
                 st.markdown(f"* {link.strip()}") # Display as markdown list
    # else:
    #     st.info("Related Online Resources section not found in the output.") # Optional message

    if st.session_state.extracted_comm_summary:
        with st.expander("Content & Communication Summary", expanded=True):
            st.markdown(st.session_state.extracted_comm_summary)
    # else:
    #     st.info("Content & Communication Summary section not found in the output.") # Optional message

    if st.session_state.extracted_bl_summary:
        with st.expander("Body Language & Presence Summary", expanded=True):
            # Remove the "(Metrics Summary: ...)" line if present using regex
            bl_summary_cleaned = re.sub(r'\n?\(Metrics Summary:.*?\)\n?', '', st.session_state.extracted_bl_summary, flags=re.DOTALL).strip()
            st.markdown(bl_summary_cleaned)
    # else:
    #     st.info("Body Language & Presence Summary section not found in the output.") # Optional message
