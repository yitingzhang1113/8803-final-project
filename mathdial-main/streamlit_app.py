from typing import Optional
import json
import os
from pathlib import Path
import re
from dotenv import load_dotenv, find_dotenv, dotenv_values
import streamlit as st

from interactivetutoring.history import History
from interactivetutoring.message import Message
from interactivetutoring.roles import Roles
from interactivetutoring.teachers_hf import HFFinetunedTeacher
from interactivetutoring.students_hf import HFStudent

load_dotenv()

st.set_page_config(page_title="MathDial Interactive Tutor", page_icon="🧮", layout="wide")
# Load .env from this app folder (mathdial-main) and also from current working directory if present
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv()
st.title("🧮 MathDial Interactive Tutor (HF)")
# Model label updates dynamically below after we resolve model id

# Sidebar: modes (Pro is placeholder)
st.sidebar.header("Settings")
mode = st.sidebar.radio("Mode", ["Basic", "Pro"], index=0)

# Prefer Streamlit secrets, then env var; only if both absent do we show sidebar input for manual override
hf_token_secret = None
try:
    hf_token_secret = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", None)  # type: ignore[attr-defined]
except Exception:
    hf_token_secret = None
hf_token_env = os.getenv("HUGGINGFACEHUB_API_TOKEN")

hf_token: Optional[str] = None
if hf_token_secret or hf_token_env:
    hf_token = (hf_token_secret or hf_token_env)
    masked = hf_token[:6] + "…" + hf_token[-4:] if hf_token and len(hf_token) > 10 else "(set)"
    st.sidebar.success(f"Using HF token from secrets/.env: {masked}")
    # Provide a small note about how to override
    st.sidebar.caption("To temporarily override, clear your .env/Secrets and use the input below (will appear when no token found).")
else:
    hf_token_input = st.sidebar.text_input(
        "HF token (optional)",
        type="password",
        help="Used for Hugging Face Inference API; leave empty to try anonymous.",
    )
    hf_token = hf_token_input or None

# Allow overriding model via env or sidebar
default_model = os.getenv("HF_MODEL_ID") or os.getenv("MODEL_ID") or "eth-nlped/MathDial-SFT-Qwen2.5-1.5B-Instruct"
fallback_model = os.getenv("HF_FALLBACK_MODEL_ID") or "mistralai/Mistral-7B-Instruct-v0.3"
fallback_models_env = os.getenv("HF_FALLBACK_MODELS")
fallback_models = [m.strip() for m in (fallback_models_env.split(",") if fallback_models_env else []) if m.strip()]
if fallback_model and fallback_model not in fallback_models:
    fallback_models.append(fallback_model)
model_id = st.sidebar.text_input("Model ID", value=default_model, help="Hugging Face model repo id to use for inference.")
fallback_models_input = st.sidebar.text_input(
    "Fallback models (comma-separated)",
    value=", ".join(fallback_models) if fallback_models else "mistralai/Mistral-7B-Instruct-v0.3, microsoft/Phi-3.5-mini-instruct, google/gemma-2-2b-it",
    help="If the primary model fails on serverless inference, the app will try these in order.")
fallback_models = [m.strip() for m in fallback_models_input.split(",") if m.strip()]
# Prompting mode is forced to guided: use problem/ground truth and metadata
prompt_mode = "guided"
st.caption(f"Model: {model_id}")

# Surface a subtle hint if token missing
if not hf_token:
    st.caption("No HF token detected from sidebar/secrets/.env — anonymous calls may be rate-limited or blocked.")
else:
    # Masked preview to confirm token loaded (first 6 + last 4)
    masked = hf_token[:6] + "…" + hf_token[-4:] if len(hf_token) > 10 else "(set)"
    st.caption(f"HF token loaded: {masked}")

@st.cache_resource
def get_teacher(token: Optional[str], model: str, fb_models: list):
    # Cache per (token, model, fallbacks) key
    return HFFinetunedTeacher(model_name=model, hf_token=token or None, fallback_models=fb_models)

@st.cache_resource
def get_student(token: Optional[str], model: str, fb_models: list):
    return HFStudent(model_name=model, hf_token=token or None, fallback_models=fb_models)

if mode == "Pro":
    st.subheader("Pro mode")
    st.info("Coming soon. This will include advanced analytics and export.")
    st.stop()

"""Basic mode: minimal flow — input problem, teacher greets, you reply, teacher responds via HF model."""

# Persist and show last error if any (to survive reruns)
if "last_error" in st.session_state and st.session_state.last_error:
    st.error(f"Last error: {st.session_state.last_error}")
    if st.button("Clear error"):
        st.session_state.last_error = None
        st.rerun()

# Gentle nudge when no token provided
if not hf_token:
    st.caption("Note: Anonymous HF Inference calls can be rate-limited. Add your HF token in the sidebar if generation fails.")

problem = st.text_area(
    "Math problem",
    value=st.session_state.get("problem", ""),
    height=120,
    placeholder="Paste or type the math word problem here…",
)
st.session_state.problem = problem

teacher = get_teacher(hf_token, model_id, fallback_models)
student_agent = get_student(hf_token, model_id, fallback_models)

# Optional: load dataset samples to guide prompt
@st.cache_resource
def load_samples(path: str):
    # Load samples from a JSONL file
    # Each line should be a valid JSON object
    # Returns a list of samples
    try:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    samples.append(obj)
                except Exception:
                    continue
        return samples
    except FileNotFoundError:
        return []

samples = load_samples(str(BASE_DIR / "data" / "example.jsonl"))
if not samples:
    st.sidebar.warning("No dataset samples found in data/example.jsonl. You can still type your own problem.")
sample_options = ["None"] + [f"{s.get('qid')} — {s.get('question','')[:50]}" for s in samples[:50]]
selected = st.sidebar.selectbox("Dataset sample (optional)", sample_options, index=0)
use_meta = st.sidebar.checkbox("Use dataset profile/confusion in prompt", value=True)
auto_apply = st.sidebar.checkbox("Auto-apply on selection", value=True)
auto_first_reply = st.sidebar.checkbox("Auto-generate first teacher reply", value=True)

def build_history_from_session_dialogue() -> History:
    h = History()
    for t in st.session_state.dialogue:
        if t["role"] == "teacher":
            h.add_message(Message(Roles.TEACHER, t["content"]))
        else:
            h.add_message(Message(Roles.STUDENT, t["content"]))
    return h

def apply_sample_record(rec: dict, generate_first_reply: bool):
    # Fill problem and metadata
    st.session_state.problem = rec.get("question", "")
    st.session_state.dataset_ground_truth = rec.get("ground_truth")
    st.session_state.dataset_student_profile = rec.get("student_profile")
    st.session_state.dataset_teacher_confusion = rec.get("teacher_described_confusion")

    # Seed conversation: teacher greeting + optional student's incorrect solution
    # Teacher greeting (guided mode)
    greet = "Hi, could you please walk me through your solution?"
    st.session_state.dialogue = [
        {"role": "teacher", "content": greet}
    ]
    sis = rec.get("student_incorrect_solution")
    if sis:
        st.session_state.dialogue.append({"role": "student", "content": sis})
    st.session_state.dataset_student_incorrect_solution = sis or ""

    if generate_first_reply:
        # Auto-simulate alternating Teacher/Student until student reaches the correct answer or max rounds
        simulate_until_correct(max_pairs=6)
    st.session_state["last_selected_sample"] = rec.get("qid")
    st.rerun()

def extract_final_answer(ground_truth: str) -> Optional[str]:
    if not ground_truth:
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", ground_truth)
    if not nums:
        return None
    return nums[-1]

def student_reached_correct(student_text: str, ground_truth: str) -> bool:
    ans = extract_final_answer(ground_truth)
    if not ans:
        return False
    pattern = rf"(?<!\d){re.escape(ans)}(?!\d)"
    if re.search(pattern, student_text):
        return True
    return False

def simulate_until_correct(max_pairs: int = 6):
    for _ in range(int(max_pairs)):
        history = build_history_from_session_dialogue()
        last_role = st.session_state.dialogue[-1]["role"] if st.session_state.dialogue else "student"
        if last_role != "teacher":
            with st.spinner("Teacher is thinking…"):
                try:
                    t_reply = teacher.response(
                        history,
                        st.session_state.problem or "",
                        st.session_state.get("dataset_ground_truth") or "",
                        student_profile=(st.session_state.get("dataset_student_profile") if use_meta else None),
                        teacher_confusion=(st.session_state.get("dataset_teacher_confusion") if use_meta else None),
                    )
                except Exception as e:
                    st.session_state.last_error = repr(e)
                    t_reply = "(generation failed)"
            st.session_state.dialogue.append({"role": "teacher", "content": t_reply})

        history = build_history_from_session_dialogue()
        with st.spinner("Student is replying…"):
            try:
                s_reply = student_agent.response(
                    history,
                    st.session_state.problem or "",
                    st.session_state.get("dataset_student_incorrect_solution") or "",
                )
            except Exception as e:
                st.session_state.last_error = repr(e)
                s_reply = "(generation failed)"
        st.session_state.dialogue.append({"role": "student", "content": s_reply})

        if student_reached_correct(s_reply, st.session_state.get("dataset_ground_truth") or ""):
            break

def parse_mathdial_conversation(conv: str):
    # Parse MathDial conversation string like "Teacher: ...|EOM|Student: ...|EOM|..."
    # into a list of {role, content} turns
    turns = []
    for chunk in conv.split("|EOM|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith("Teacher:"):
            content = chunk[len("Teacher:"):].strip()
            turns.append({"role": "teacher", "content": content})
        elif chunk.startswith("Student:"):
            content = chunk[len("Student:"):].strip()
            turns.append({"role": "student", "content": content})
        else:
            # Fallback: treat as teacher by default
            turns.append({"role": "teacher", "content": chunk})
    return turns

if selected != "None":
    idx = sample_options.index(selected) - 1
    if 0 <= idx < len(samples):
        s = samples[idx]
        # Auto-apply when selection changes
        if auto_apply and st.session_state.get("last_selected_sample") != s.get("qid"):
            apply_sample_record(s, auto_first_reply)
        # Manual apply button as fallback
        if st.sidebar.button("Use this sample"):
            apply_sample_record(s, auto_first_reply)
        # Replay the exact conversation from dataset (no generation)
        if st.sidebar.button("Replay sample conversation"):
            convo = s.get("conversation") or ""
            parsed = parse_mathdial_conversation(convo)
            if parsed:
                st.session_state.dialogue = parsed
                st.session_state["last_selected_sample"] = s.get("qid")
                st.rerun()

# Show dataset context if loaded
if st.session_state.get("dataset_ground_truth") or st.session_state.get("dataset_student_profile") or st.session_state.get("dataset_teacher_confusion"):
    with st.expander("Dataset context", expanded=False):
        if st.session_state.get("dataset_student_profile"):
            st.markdown("**Student profile**")
            st.write(st.session_state.get("dataset_student_profile"))
        if st.session_state.get("dataset_teacher_confusion"):
            st.markdown("**Teacher-described confusion**")
            st.write(st.session_state.get("dataset_teacher_confusion"))
        if st.session_state.get("dataset_ground_truth"):
            show_gt = st.checkbox("Show ground truth (reference)", value=False)
            if show_gt:
                st.markdown("**Ground truth**")
                st.code(st.session_state.get("dataset_ground_truth"), language="markdown")
        if st.button("Clear dataset context"):
            for k in [
                "dataset_ground_truth",
                "dataset_student_profile",
                "dataset_teacher_confusion",
            ]:
                if k in st.session_state:
                    st.session_state.pop(k)
            st.rerun()

# Initialize dialogue state
if "dialogue" not in st.session_state or not st.session_state.get("dialogue"):
    st.session_state.dialogue = [
        {"role": "teacher", "content": "Hi, could you please walk me through your solution?"}
    ]

st.markdown("### Conversation")
for turn in st.session_state.dialogue:
    role = "assistant" if turn["role"] == "teacher" else "user"
    with st.chat_message(role):
        st.markdown(turn["content"]) 

student_input = st.chat_input("Your response as the student…")
if student_input:
    st.session_state.dialogue.append({"role": "student", "content": student_input})

    # Build History from dialogue
    history = build_history_from_session_dialogue()

    with st.spinner("Teacher is thinking…"):
        try:
            reply = teacher.response(
                history,
                st.session_state.problem or "",
                st.session_state.get("dataset_ground_truth") or "",
                student_profile=(st.session_state.get("dataset_student_profile") if use_meta else None),
                teacher_confusion=(st.session_state.get("dataset_teacher_confusion") if use_meta else None),
            )
        except Exception as e:
            # keep the message visible after rerun
            st.session_state.last_error = repr(e)
            st.error(f"Teacher generation failed: {e!r}")
            reply = "(generation failed)"

    st.session_state.dialogue.append({"role": "teacher", "content": reply})
    st.rerun()

# Allow generating a teacher reply without typing, helpful after loading a sample
if st.button("Generate teacher reply"):
    history = build_history_from_session_dialogue()
    with st.spinner("Teacher is thinking…"):
        try:
            reply = teacher.response(
                history,
                st.session_state.problem or "",
                st.session_state.get("dataset_ground_truth") or "",
                student_profile=(st.session_state.get("dataset_student_profile") if use_meta else None),
                teacher_confusion=(st.session_state.get("dataset_teacher_confusion") if use_meta else None),
            )
        except Exception as e:
            st.session_state.last_error = repr(e)
            st.error(f"Teacher generation failed: {e!r}")
            reply = "(generation failed)"
    st.session_state.dialogue.append({"role": "teacher", "content": reply})
    st.rerun()

# Manual multi-round generator removed per request; auto-simulation occurs on sample apply
col1, col2 = st.columns(2)
with col1:
    if st.button("Reset conversation", use_container_width=True):
        st.session_state.dialogue = [
            {"role": "teacher", "content": "Hi, could you please walk me through your solution?"}
        ]
        st.rerun()
with col2:
    st.caption("Tip: For faster dev loop, install Watchdog (optional). No API keys required.")

# Export JSONL removed per request

# Optional diagnostics to help explain empty-response issues
with st.sidebar.expander("Diagnostics", expanded=False):
    if st.button("Run HF connectivity check"):
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            # Whoami (if token provided)
            if hf_token:
                try:
                    who = api.whoami(token=hf_token)
                    st.success(f"Authenticated as: {who.get('name') or who.get('email') or 'unknown'}")
                except Exception as e:
                    st.warning(f"Token invalid or insufficient: {e}")
            else:
                st.info("No token provided; anonymous calls may be rate-limited or blocked.")

            # Check primary model availability
            def probe_model(mid: str):
                try:
                    info = api.model_info(mid, token=hf_token if hf_token else None)
                    gated = getattr(info, "gated", False)
                    private = getattr(info, "private", False)
                    st.write(f"- {mid}: ok (gated={gated}, private={private})")
                    return True
                except Exception as e:
                    st.write(f"- {mid}: not accessible ({e})")
                    return False

            st.write("Model access:")
            primary_ok = probe_model(model_id)
            for m in fallback_models:
                probe_model(m)

            # Quick sanity inference to a small public model
            try:
                from huggingface_hub import InferenceClient
                test_model = fallback_models[0] if fallback_models else model_id
                tc = InferenceClient(model=test_model, token=hf_token if hf_token else None)
                out = tc.text_generation("Hello", max_new_tokens=8, stream=False, return_full_text=False)
                st.success(f"Test inference ok on {test_model}: '{out[:80]}'")
            except Exception as e:
                st.error(f"Test inference failed: {e}")
        except Exception as e:
            st.error(f"Diagnostics failed: {e}")

# .env visibility helper
with st.sidebar.expander(".env status", expanded=False):
    try:
        env_path = find_dotenv()
        st.write(f"find_dotenv(): {env_path or '(not found)'}")
        if env_path:
            vals = dotenv_values(env_path)
            present_keys = [k for k in ["HUGGINGFACEHUB_API_TOKEN", "HF_MODEL_ID", "HF_FALLBACK_MODELS"] if k in vals]
            if present_keys:
                st.write("Keys present:", ", ".join(present_keys))
                # Mask token if present
                tok = vals.get("HUGGINGFACEHUB_API_TOKEN")
                if tok:
                    masked = tok[:6] + "…" + tok[-4:] if len(tok) > 10 else "(set)"
                    st.write(f"HUGGINGFACEHUB_API_TOKEN: {masked}")
            else:
                st.write("No expected keys found in .env (expected HUGGINGFACEHUB_API_TOKEN/HF_MODEL_ID/HF_FALLBACK_MODELS)")
        else:
            st.write("No .env discovered by find_dotenv(). Ensure it is placed at mathdial-main/.env")
    except Exception as e:
        st.write(f".env inspection failed: {e}")
