#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from typing import Dict, Iterable, List, Optional, Tuple
import importlib
import os
import time
from typing import Any

# ---------------------------
# Optional symbolic engine
# ---------------------------
try:  # pragma: no cover
    sp = importlib.import_module("sympy")
except Exception:  # pragma: no cover
    sp = None

HINT_TYPES = {"directive", "procedural", "conceptual"}
REF_TURN_RE = re.compile(r"\b(step|turn|line)\s*(\d+)\b", re.I)

# capture simple expressions for symbolic compare
_EXPR_RE = re.compile(
    r"([A-Za-z]\w*\s*=\s*[-+*/()\w\.]+|[-+]?\d+(?:\.\d+)?\s*[+\-*/]\s*[-+]?\d+(?:\.\d+)?)"
)


# ---------------------------
# IO helpers
# ---------------------------
def _read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _index_dependencies(deps_path: str) -> Dict[str, List[dict]]:
    idx: Dict[str, List[dict]] = {}
    for d in _read_jsonl(deps_path):
        key = str(d.get("qid"))
        if not key:
            # tolerate "id" as key
            key = str(d.get("id", ""))
        if not key:
            continue
        idx.setdefault(key, []).append(d)
    return idx


# ---------------------------
# Dialogue parsing
# ---------------------------
def _parse_conversation(raw: str) -> List[Tuple[str, str]]:
    """Split conversation by |EOM|; infer role by prefix Teacher:/Student:; default Student."""
    parts = [p.strip() for p in raw.split("|EOM|") if p.strip()]
    out: List[Tuple[str, str]] = []
    for p in parts:
        role = "Student"
        text = p
        if ":" in p:
            head, tail = p.split(":", 1)
            r = head.strip().lower()
            role = "Teacher" if r.startswith("teacher") else "Student"
            text = tail.strip()
            # trim inline "(...)" meta after role, e.g. "Teacher: (smiles) ..."
            if text.startswith("(") and ")" in text[:20]:
                text = text.split(")", 1)[1].strip()
        out.append((role, text))
    return out


def _dialogue_upto_student_index_only_students(
    pairs: List[Tuple[str, str]], t_ord: int
) -> str:
    """Return transcript up to student index t_ord, including teachers' lines for context."""
    s_count = -1
    collected: List[str] = []
    for role, txt in pairs:
        if role == "Student":
            s_count += 1
            collected.append(f"Student: {txt}")
            if s_count == t_ord:
                break
        else:
            collected.append(f"Teacher: {txt}")
    return "\n".join(collected)


# ---------------------------
# Prompt & JSON parsing
# ---------------------------
def _build_prompt(
    dialogue_text: str,
    student_text: str,
    t_ord: int,
    a_op: str,
    a_dep: Optional[int],
    a_conf: float,
) -> str:
    return (
        "You are an expert math tutor specialized in diagnosing misconceptions.\n"
        "Return ONLY a valid JSON object. Never include markdown fences.\n"
        "Your job is NOT to answer the math, but to:\n"
        "- Identify the student's thinking state or misconception\n"
        "- Decide which earlier student turn this depends on (if any)\n"
        "- Provide exactly ONE teacher-style hint\n"
        "- Classify the hint into {conceptual, procedural, directive}\n"
        "- Comment on symbolic writing (notation, units, parentheses, signs)\n"
        "- Set a reasonable confidence (0~1)\n\n"
        "Definitions:\n"
        "- conceptual: clarifies underlying principle (units, meaning of operations)\n"
        "- procedural: suggests next algorithmic step\n"
        "- directive: direct instruction (e.g., “Rewrite ... as ...”)\n\n"
        "Rules:\n"
        "- Do NOT reveal final numeric answers\n"
        "- Keep 'thought' ≤ 60 words; 'dependency_reason' ≤ 40 words\n"
        "- 'ref_turn' must be an EARLIER student turn or null\n"
        "- Choose hint type based on actual need; avoid always procedural\n\n"
        f"[Dialogue up to student turn #{t_ord}]\n{dialogue_text}\n\n"
        f"[Current student turn #{t_ord}]\n\"{student_text}\"\n\n"
        f"Operation from dependency extractor: {a_op}\n"
        f"Candidate dependency turn: {a_dep}\n"
        f"Extractor confidence: {a_conf}\n\n"
        "Output JSON:\n"
        "{\n"
        ' "thought": "...(<=60w)",\n'
        ' "dependency_reason": "...(<=40w)",\n'
        ' "ref_turn": null,\n'
        ' "hint_type": "conceptual|procedural|directive",\n'
        ' "symbolic_comment": "...",\n'
        ' "confidence": 0.7\n'
        "}\n"
    )


def _try_parse_json(text: str) -> Optional[Dict]:
    # direct
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # strip code fences
    fenced = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text.strip())
    if fenced != text:
        try:
            obj = json.loads(fenced)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # extract first {...}
    if "{" in text and "}" in text:
        s = text.find("{")
        e = text.rfind("}")
        if e > s:
            snippet = text[s : e + 1]
            try:
                obj = json.loads(snippet)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    return None


def _sanitize_and_validate(text: str) -> Tuple[Dict, bool]:
    obj = _try_parse_json(text)
    if obj is None:
        return {}, False
    # normalize
    obj.setdefault("thought", "")
    obj.setdefault("dependency_reason", "")
    obj.setdefault("ref_turn", None)
    obj.setdefault("hint_type", "procedural")
    obj.setdefault("symbolic_comment", "")
    obj.setdefault("confidence", 0.5)
    # guard hint type
    if obj["hint_type"] not in HINT_TYPES:
        obj["hint_type"] = "procedural"
    # refuse lazy placeholders
    lazy_vals = {"...", "(<=60w)", "(=>60w)"}
    if obj.get("symbolic_comment") in lazy_vals:
        obj["symbolic_comment"] = "Clarify symbolic notation, units, and grouping."
    # extract referenced turn from free text if any
    m = REF_TURN_RE.search(obj.get("dependency_reason", ""))
    if m:
        try:
            obj["ref_turn_candidate"] = int(m.group(2))
        except Exception:
            pass
    return obj, True


# ---------------------------
# Symbolic critic
# ---------------------------
def _extract_exprs(text: str) -> List[str]:
    if not text:
        return []
    return [m.group(0) for m in _EXPR_RE.finditer(text)]


def _symbolic_compare(e1: str, e2: str) -> Optional[bool]:
    if not e1 or not e2:
        return None
    if sp is None:
        # basic whitespace-insensitive compare
        return re.sub(r"\s+", "", e1) == re.sub(r"\s+", "", e2)
    try:
        def rhs(s: str):
            return s.split("=", 1)[1] if "=" in s else s
        v = sp.simplify(sp.sympify(rhs(e1)) - sp.sympify(rhs(e2)))
        return bool(v == 0)
    except Exception:
        return None


# ---------------------------
# Providers
# ---------------------------
def _call_model(prompt: str, provider: str = "noop", **kwargs) -> str:
    if provider == "noop":
        return json.dumps(
            {
                "thought": "Reflect on the algebraic step and next action.",
                "dependency_reason": "Builds on a prior substitution step.",
                "ref_turn": None,
                "hint_type": "procedural",
                "symbolic_comment": "Check signs and parentheses when combining like terms.",
                "confidence": 0.7,
            }
        )

    if provider == "hf":
        model = kwargs.get("hf_model")
        temperature = float(kwargs.get("hf_temp", 0.2))
        max_new_tokens = int(kwargs.get("hf_max_new_tokens", 256))
        _ = int(kwargs.get("hf_timeout", 60))
        token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise RuntimeError("HUGGINGFACEHUB_API_TOKEN is not set in environment.")
        try:
            from huggingface_hub import InferenceClient
        except Exception as e:
            raise RuntimeError(
                "huggingface_hub not installed. Please `pip install huggingface_hub`."
            ) from e
        client = InferenceClient(model=model, token=token)
        # prefer chat API
        try:
            chat = client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert math tutor. Return only a VALID JSON object.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
            )
            msg = chat.choices[0].message
            if isinstance(msg, dict):
                return msg.get("content", "{}")
            return getattr(msg, "content", "{}")
        except Exception:
            generated = client.text_generation(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                repetition_penalty=1.1,
                do_sample=True,
                return_full_text=False,
            )
            return generated

    if provider == "openai":
        model = kwargs.get("openai_model", "gpt-4o-mini")
        base_url = kwargs.get("openai_base") or os.environ.get("OPENAI_BASE_URL")
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("openai not installed. Please `pip install openai`.") from e
        client_args: Dict[str, Any] = {}
        if base_url:
            client_args["base_url"] = base_url
        client = OpenAI(**client_args)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a math tutor. Return only a VALID JSON object."},
                {"role": "user", "content": prompt},
            ],
            temperature=float(kwargs.get("openai_temp", 0.2)),
            max_tokens=int(kwargs.get("openai_max_tokens", 256)),
        )
        return resp.choices[0].message.content or "{}"

    if provider == "together":
        # Requires: TOGETHER_API_KEY
        model = kwargs.get("together_model", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        try:
            import together
        except Exception as e:
            raise RuntimeError("together not installed. Please `pip install together`.") from e
        together.api_key = os.environ.get("TOGETHER_API_KEY")
        if not together.api_key:
            raise RuntimeError("TOGETHER_API_KEY is not set in environment.")
        resp = together.Completions.create(
            model=model,
            prompt=prompt,
            temperature=float(kwargs.get("together_temp", 0.2)),
            max_tokens=int(kwargs.get("together_max_tokens", 256)),
        )
        return resp.get("output", {}).get("text", "{}")

    if provider == "ollama":
        # Requires: local Ollama server (default http://localhost:11434)
        import json as _json
        import requests

        model = kwargs.get("ollama_model", "qwen2.5:3b-instruct")
        url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        endpoint = url + "/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": float(kwargs.get("ollama_temp", 0.2)),
                "num_predict": int(kwargs.get("ollama_max_new_tokens", 256)),
            },
            "stream": False,
        }
        r = requests.post(
            endpoint,
            data=_json.dumps(payload),
            timeout=int(kwargs.get("ollama_timeout", 60)),
        )
        r.raise_for_status()
        data = r.json()
        return data.get("response", "{}")

    # fallback
    return json.dumps(
        {
            "thought": "",
            "dependency_reason": "",
            "ref_turn": None,
            "hint_type": "procedural",
            "symbolic_comment": "",
            "confidence": 0.6,
        }
    )


# ---------------------------
# Core pipeline
# ---------------------------
def run(
    data_path: str,
    deps_path: str,
    out_path: str,
    min_conf: float,
    provider: str,
    max_samples: Optional[int],
    sleep_ms: int,
    prov_opts: Dict[str, Any],
    strict_filter: bool,
    fallback_conf: float,
):
    deps_index = _index_dependencies(deps_path)
    written = 0

    with open(out_path, "w", encoding="utf-8") as w:
        for ex in _read_jsonl(data_path):
            qid = str(ex.get("qid") or ex.get("id") or "")
            if not qid:
                continue

            raw_conv = ex.get("conversation", "")
            if not isinstance(raw_conv, str) or not raw_conv.strip():
                continue

            dep_list = deps_index.get(qid, [])
            # candidate turns: operation != other or high-conf
            cand_turns = sorted(
                {
                    d.get("turn_id")
                    for d in dep_list
                    if d.get("turn_id") is not None
                    and (d.get("operation") != "other" or float(d.get("confidence", 0.0)) >= min_conf)
                }
            )
            if not cand_turns:
                continue

            pairs = _parse_conversation(raw_conv)
            rec_by_turn = {d["turn_id"]: d for d in dep_list if "turn_id" in d}

            for t_ord in cand_turns:
                # transcript up to student turn
                dialogue_text = _dialogue_upto_student_index_only_students(pairs, t_ord)

                # find current student text
                s_count = -1
                student_text = ""
                for role, txt in pairs:
                    if role == "Student":
                        s_count += 1
                        if s_count == t_ord:
                            student_text = txt
                            break

                r = rec_by_turn.get(
                    t_ord, {"operation": "other", "depends_on": None, "confidence": 0.0}
                )
                a_op = r.get("operation", "other")
                a_dep = r.get("depends_on")
                try:
                    a_c = float(r.get("confidence", 0.0))
                except Exception:
                    a_c = 0.0

                # symbolic critic (light)
                dep_text = None
                if isinstance(a_dep, int):
                    cnt = -1
                    for role, txt in pairs:
                        if role == "Student":
                            cnt += 1
                            if cnt == a_dep:
                                dep_text = txt
                                break

                exprs_now = _extract_exprs(student_text)
                exprs_prev = _extract_exprs(dep_text or "")
                sym_ok: Optional[bool] = None
                sym_comment = "Check parentheses and units when writing expressions."
                if exprs_now and exprs_prev:
                    comp = _symbolic_compare(exprs_now[0], exprs_prev[-1])
                    sym_ok = comp
                    if comp is False:
                        sym_comment = "Your symbolic form differs; check signs or grouping."

                # detect looping/stuck patterns to bias directive
                loop_flag = any(
                    kw in student_text.lower()
                    for kw in ["again", "still", "stuck", "don't know", "dont know"]
                )

                # single call prompt
                prompt = _build_prompt(dialogue_text, student_text, t_ord, a_op, a_dep, a_c)
                raw = _call_model(prompt, provider=provider, **prov_opts)
                obj, ok = _sanitize_and_validate(raw)
                if not ok:
                    raw2 = _call_model(
                        prompt + "\nReturn ONLY a valid JSON object.",
                        provider=provider,
                        **prov_opts,
                    )
                    obj, ok2 = _sanitize_and_validate(raw2)
                    if not ok2:
                        obj = {
                            "thought": "",
                            "dependency_reason": "",
                            "ref_turn": None,
                            "hint_type": "procedural",
                            "symbolic_comment": "",
                            "confidence": max(0.0, min(1.0, float(fallback_conf))),
                            "__fallback__": True,
                        }

                # coordinator: hint_type variety + confidence fusion
                hint_type = obj.get("hint_type", "procedural")
                if hint_type not in HINT_TYPES:
                    hint_type = "procedural"

                # simple heuristics to diversify hint types
                st_low = student_text.lower()
                if "unit" in st_low or "meaning" in st_low or "why" in st_low:
                    hint_type = "conceptual"
                if loop_flag:
                    hint_type = "directive"
                if a_op == "add" and "unit" in st_low:
                    hint_type = "conceptual"

                # confidence fusion
                if obj.get("__fallback__"):
                    conf = max(0.0, min(1.0, float(fallback_conf)))
                else:
                    conf = max(0.0, min(1.0, float(a_c)))

                if sym_ok is False:
                    if hint_type == "directive":
                        hint_type = "procedural"
                    conf = max(0.0, conf - 0.05)
                if a_c < min_conf:
                    conf = min(conf, 0.5)

                # small type-based scaling
                if hint_type == "conceptual":
                    conf = min(1.0, conf + 0.05)
                elif hint_type == "directive":
                    conf = max(0.0, conf - 0.03)

                # enforce earlier ref_turn
                ref_turn = obj.get("ref_turn")
                if isinstance(ref_turn, int) and ref_turn >= t_ord:
                    ref_turn = a_dep if (isinstance(a_dep, int) and a_dep < t_ord) else None
                if ref_turn is None and isinstance(a_dep, int) and a_dep < t_ord:
                    ref_turn = a_dep

                # assemble row
                row = {
                    "qid": qid,
                    "turn_id": t_ord,
                    "thought": obj.get("thought", ""),
                    "dependency_reason": obj.get("dependency_reason", ""),
                    "ref_turn": ref_turn,
                    "hint_type": hint_type,
                    "symbolic_comment": sym_comment or obj.get("symbolic_comment", ""),
                    "confidence": round(conf, 2),
                }

                # optional quality gate
                if strict_filter:
                    if not (
                        a_c >= min_conf
                        and row["dependency_reason"].strip()
                        and row["hint_type"] in HINT_TYPES
                    ):
                        continue

                # write
                if row["hint_type"] in HINT_TYPES:
                    w.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1

                    if sleep_ms > 0:
                        time.sleep(sleep_ms / 1000.0)
                    if max_samples and written >= max_samples:
                        print(f"Wrote {written} teacher signals -> {out_path}")
                        return

    print(f"Wrote {written} teacher signals -> {out_path}")


# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--deps", default="mathdial-main/research_ext/results/dependencies.jsonl")
    ap.add_argument("--out", default="mathdial-main/research_ext/results/teacher_signals.jsonl")
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--provider", choices=["noop", "hf", "openai", "together", "ollama"], default="noop")

    # HF provider options
    ap.add_argument("--hf-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--hf-temp", type=float, default=0.2)
    ap.add_argument("--hf-max-new-tokens", type=int, default=256)
    ap.add_argument("--hf-timeout", type=int, default=60)

    # OpenAI provider options (works with real OpenAI or OpenAI-compatible base_url like Ollama/openrouter)
    ap.add_argument("--openai-model", default="gpt-4o-mini")
    ap.add_argument("--openai-base", default="")
    ap.add_argument("--openai-temp", type=float, default=0.2)
    ap.add_argument("--openai-max-tokens", type=int, default=256)

    # Together
    ap.add_argument("--together-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    ap.add_argument("--together-temp", type=float, default=0.2)
    ap.add_argument("--together-max-tokens", type=int, default=256)

    # Ollama
    ap.add_argument("--ollama-model", default="qwen2.5:3b-instruct")
    ap.add_argument("--ollama-temp", type=float, default=0.2)
    ap.add_argument("--ollama-max-new-tokens", type=int, default=256)
    ap.add_argument("--ollama-timeout", type=int, default=60)

    # Rate limiting & misc
    ap.add_argument("--sleep-ms", type=int, default=0, help="Sleep between provider calls in milliseconds")
    ap.add_argument("--max-samples", type=int, default=0, help="Stop after writing this many samples (0 = no limit)")
    ap.add_argument("--strict-filter", action="store_true", help="Filter out low-quality signals at write time")
    ap.add_argument("--fallback-conf", type=float, default=0.3, help="Confidence to use when JSON parsing fails twice")

    args = ap.parse_args()

    prov_opts: Dict[str, Any] = {}
    if args.provider == "hf":
        prov_opts.update(
            {
                "hf_model": args.hf_model,
                "hf_temp": args.hf_temp,
                "hf_max_new_tokens": args.hf_max_new_tokens,
                "hf_timeout": args.hf_timeout,
            }
        )
    elif args.provider == "openai":
        prov_opts.update(
            {
                "openai_model": args.openai_model,
                "openai_base": args.openai_base,
                "openai_temp": args.openai_temp,
                "openai_max_tokens": args.openai_max_tokens,
            }
        )
    elif args.provider == "together":
        prov_opts.update(
            {
                "together_model": args.together_model,
                "together_temp": args.together_temp,
                "together_max_tokens": args.together_max_tokens,
            }
        )
    elif args.provider == "ollama":
        prov_opts.update(
            {
                "ollama_model": args.ollama_model,
                "ollama_temp": args.ollama_temp,
                "ollama_max_new_tokens": args.ollama_max_new_tokens,
                "ollama_timeout": args.ollama_timeout,
            }
        )

    run(
        args.data,
        args.deps,
        args.out,
        args.min_conf,
        args.provider,
        args.max_samples or None,
        args.sleep_ms,
        prov_opts,
        args.strict_filter,
        args.fallback_conf,
    )


if __name__ == "__main__":
    main()
