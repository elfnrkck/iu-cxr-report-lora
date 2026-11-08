"""Lossless sentence-packed CLIP windows, adapted from the study's r4_token_plan."""
from dataclasses import dataclass
import hashlib
import html
import json
import re


@dataclass(frozen=True)
class ReportPlan:
    text: str
    input_ids: list
    content_lengths: list
    content_tokens: int
    forced_sentence_splits: int

    @property
    def windows(self):
        return len(self.input_ids)

    @property
    def token_ids_sha256(self):
        return hashlib.sha256(json.dumps(self.input_ids, sort_keys=True).encode()).hexdigest()


def normalize_report(report):
    """Match the study's explicit HTML unescape policy; no clinical rewriting."""
    if not isinstance(report, str) or not report.strip():
        raise ValueError('Supply a nonempty English report.')
    text = report
    for _ in range(2):
        changed = html.unescape(text)
        if changed == text:
            break
        text = changed
    return text


def plan_report(report, tokenizer, max_windows=16):
    text = normalize_report(report)
    if (tokenizer.bos_token_id, tokenizer.eos_token_id) != (49406, 49407):
        raise ValueError('Expected the pinned SD 1.5 CLIP tokenizer.')
    full = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
    sentences = [tokenizer.encode(s, add_special_tokens=False, truncation=False, verbose=False)
                 for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if [token for sentence in sentences for token in sentence] != full:
        raise ValueError('Sentence packing changed report tokens; generation was stopped.')
    chunks, current, forced = [], [], 0
    for sentence in sentences:
        if len(sentence) > 75:
            if current:
                chunks.append(current)
                current = []
            forced += 1
            chunks.extend(sentence[i:i+75] for i in range(0, len(sentence), 75))
        elif len(current) + len(sentence) <= 75:
            current.extend(sentence)
        else:
            if current:
                chunks.append(current)
            current = list(sentence)
    if current:
        chunks.append(current)
    if not chunks:
        chunks = [[]]
    if max_windows < 1 or len(chunks) > max_windows:
        raise ValueError(f'Report needs {len(chunks)} windows; limit is {max_windows}. No tokens were silently truncated.')
    ids = [[49406] + chunk + [49407] * (76-len(chunk)) for chunk in chunks]
    recovered = [v for row, chunk in zip(ids, chunks) for v in row[1:1+len(chunk)]]
    if recovered != full or any(len(row) != 77 for row in ids):
        raise ValueError('Lossless CLIP packing failed.')
    return ReportPlan(text, ids, [len(c) for c in chunks], len(full), forced)


def empty_windows(number):
    if not isinstance(number, int) or number < 1:
        raise ValueError('At least one conditioning window is required.')
    return [[49406] + [49407] * 76 for _ in range(number)]

