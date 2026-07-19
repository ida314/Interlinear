"""Sentence boundary detection over aligned words.

Whisper's own segment boundaries are artifacts of its 30-second decoding window and VAD, not
linguistic sentence ends, and its punctuation is unreliable in exactly the languages we care
most about. Getting sentence ends right is what makes the output feel deliberate rather than
chopped, so it is a first-class stage with its own signals and provenance.

Three independent signals are fused:

1. **Terminal punctuation** from ASR — strong when present, but never trusted alone.
2. **A textual boundary model** (wtpsplit/SaT) — robust to missing punctuation and casing,
   which is precisely the ASR failure mode, and covers CJK where there is no whitespace.
3. **The acoustic pause** between words — free from forced alignment, and genuinely
   informative. Pure-text segmenters throw this away; we have it, so we use it.

Ambiguous boundaries — and only those — are escalated to an LLM arbiter. Everything here
operates on *indices into the word array*. No component may rewrite text, because every word
is bound to a forced-alignment timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from app.models.timeline import BoundarySource, RenderParams, Word

# Terminal punctuation across the scripts we support. CJK full stops are distinct codepoints
# from the ASCII period. Semicolons are deliberately absent: they join clauses rather than
# ending utterances, and Whisper's F1 on them is near zero anyway.
TERMINAL_PUNCT = ".!?…。！？։۔؟"
CLOSING_PUNCT = "\"')]}»”’"

# Whisper's own punctuation measurably outperforms dedicated punctuation-restoration models
# in these languages, because it has prosodic access — it hears the rising intonation of a
# question where a text-only model sees only words. So a terminal mark here is strong
# evidence, not a weak prior.
STRONG_PUNCT_LANGS = {"en", "es", "pt", "fr", "de", "ru", "it", "nl", "pl", "ca"}

# Whisper frequently omits terminal punctuation altogether in these scripts. Note this is a
# *recall* problem: when a mark does appear it is still trustworthy. Noisy-OR fusion handles
# that asymmetry naturally, since an absent signal contributes nothing rather than arguing
# against a boundary. The practical consequence is that these languages genuinely need the
# textual model — punctuation alone will run sentences together.
LOW_PUNCT_RECALL_LANGS = {"zh", "ja", "th", "ko", "lo", "my", "km"}

# Abbreviations whose trailing period is not a sentence end. The textual model handles this
# properly; this list only backstops the no-model fallback path.
_ABBREVIATIONS = {
    # Latin script
    "mr", "mrs", "ms", "dr", "prof", "sr", "sra", "srta", "st", "vs", "etc", "approx",
    "no", "vol", "fig", "al", "ca", "cf", "eg", "ie", "jr", "inc", "ltd", "co",
    "abbr", "art", "cap", "pág", "núm", "bzw", "ggf", "usw", "bspw", "z.b", "u.a",
    # Cyrillic. Russian writes many of these mid-sentence and they end in a period, so
    # without this list every "в 1995 г." becomes a spurious sentence boundary.
    "г", "гг", "в", "вв", "т.д", "т.е", "т.п", "т.к", "др", "пр", "ул", "д", "кв",
    "стр", "с", "рис", "табл", "см", "напр", "акад", "проф", "им", "руб", "коп",
    "тыс", "млн", "млрд", "обл", "р", "оз", "изд", "ред", "сокр", "букв",
}

# Languages written without inter-word spacing; each aligned Word is one character.
NO_SPACE_LANGS = {"zh", "ja", "th", "lo", "my", "km"}


@dataclass
class BoundaryCandidate:
    """A potential sentence end *after* `word_index`."""

    word_index: int
    score: float = 0.0
    sat_prob: float | None = None
    pause_after: float = 0.0
    has_terminal_punct: bool = False
    source: BoundarySource = "fused"


class BoundaryDetector(Protocol):
    """Produces a textual boundary probability per word position."""

    def probabilities(self, words: Sequence[Word], lang: str) -> list[float]: ...


# `(words, candidate_indices) -> set of indices the LLM confirms as real sentence ends`
Arbiter = Callable[[Sequence[Word], list[int], str], set[int]]


def word_join(lang: str) -> str:
    return "" if lang in NO_SPACE_LANGS else " "


def has_terminal_punct(text: str) -> bool:
    stripped = text.rstrip(CLOSING_PUNCT).rstrip()
    if not stripped or stripped[-1] not in TERMINAL_PUNCT:
        return False
    if stripped[-1] == ".":
        token = stripped[:-1].split()[-1] if stripped[:-1].split() else ""
        if token.lower().strip("([{") in _ABBREVIATIONS:
            return False
        # A single initial ("J.") is not a sentence end.
        if len(token) == 1 and token.isalpha():
            return False
    return True


class PunctuationPauseDetector:
    """Zero-dependency fallback: punctuation plus pause only.

    Used when the textual model is unavailable (a laptop without the GPU extras) and as the
    baseline that SaT must beat during evaluation.
    """

    def probabilities(self, words: Sequence[Word], lang: str) -> list[float]:
        return [1.0 if has_terminal_punct(w.text) else 0.0 for w in words]


class SaTDetector:
    """wtpsplit / "Segment any Text". Imported lazily so this module stays importable
    without the GPU extras installed."""

    def __init__(self, model: str = "sat-3l-sm", device: str = "cuda") -> None:
        self.model_name = model
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from wtpsplit import SaT  # noqa: PLC0415 — optional heavy dependency

            self._model = SaT(self.model_name)
            if self.device.startswith("cuda"):
                self._model.half().to(self.device)
        return self._model

    def probabilities(self, words: Sequence[Word], lang: str) -> list[float]:
        model = self._load()
        sep = word_join(lang)
        text = sep.join(w.text for w in words)

        char_probs = model.predict_proba(text)
        # Map character-level probabilities back onto word positions: the probability of a
        # boundary after word i is the value at that word's final character.
        out: list[float] = []
        cursor = 0
        for i, w in enumerate(words):
            cursor += len(w.text)
            idx = min(cursor - 1, len(char_probs) - 1)
            out.append(float(char_probs[idx]) if idx >= 0 else 0.0)
            if i < len(words) - 1:
                cursor += len(sep)
        return out


def punct_confidence(lang: str) -> float:
    """How much a terminal mark is worth, given how reliably Whisper produces it."""
    if lang in STRONG_PUNCT_LANGS:
        return 0.95
    return 0.85  # still strong when present, everywhere — see LOW_PUNCT_RECALL_LANGS


def score_boundaries(
    words: Sequence[Word],
    lang: str,
    params: RenderParams,
    detector: BoundaryDetector | None = None,
) -> list[BoundaryCandidate]:
    """Fuse textual, orthographic and acoustic evidence into one score per word position.

    Combined as a noisy-OR rather than a weighted average, because these signals are
    *corroborating*, not competing. A pause should be able to lift an uncertain textual
    prediction over the line, but must never drag a confident one below it — and a missing
    signal should count as "no information", not as evidence against a boundary. A weighted
    average gets both of those wrong: it lets a silent gap veto an unambiguous full stop.
    """
    detector = detector or PunctuationPauseDetector()
    text_probs = detector.probabilities(words, lang)
    is_fallback = isinstance(detector, PunctuationPauseDetector)
    punct_weight = punct_confidence(lang)

    candidates: list[BoundaryCandidate] = []
    for i, word in enumerate(words):
        punct = has_terminal_punct(word.text)
        pause = (words[i + 1].start - word.end) if i + 1 < len(words) else float("inf")
        pause_norm = min(1.0, pause / params.pause_saturation) if params.pause_saturation else 0.0

        # The fallback detector reports punctuation only, so folding it in again would
        # double-count the same evidence.
        text_score = 0.0 if is_fallback else (text_probs[i] if i < len(text_probs) else 0.0)
        punct_score = punct_weight if punct else 0.0
        pause_score = params.pause_weight * pause_norm

        score = 1.0 - (1.0 - text_score) * (1.0 - punct_score) * (1.0 - pause_score)
        candidates.append(
            BoundaryCandidate(
                word_index=i,
                score=min(1.0, score),
                sat_prob=None if is_fallback else text_score,
                pause_after=0.0 if pause == float("inf") else pause,
                has_terminal_punct=punct,
                source="punctuation" if (is_fallback and punct) else "fused",
            )
        )

    if candidates:
        last = candidates[-1]
        last.score = 1.0
        last.source = "eof"
    return candidates


def ambiguous_indices(
    candidates: Sequence[BoundaryCandidate], gray_zone: tuple[float, float]
) -> list[int]:
    """Boundaries the cheap signals could not settle. Only these are worth an LLM call."""
    lo, hi = gray_zone
    return [c.word_index for c in candidates if lo <= c.score < hi and c.source != "eof"]


def apply_arbitration(
    words: Sequence[Word],
    candidates: list[BoundaryCandidate],
    params: RenderParams,
    arbiter: Arbiter,
    lang: str,
) -> list[BoundaryCandidate]:
    """Let an LLM settle the ambiguous boundaries, in place.

    Pushed above/below the threshold rather than set to absolutes, so a later threshold
    change still behaves sensibly, and the original fused score stays visible in the
    provenance record.
    """
    unsure = ambiguous_indices(candidates, params.llm_gray_zone)
    if not unsure:
        return candidates

    confirmed = arbiter(words, unsure, lang)
    by_index = {c.word_index: c for c in candidates}
    for idx in unsure:
        c = by_index[idx]
        agreed = idx in confirmed
        c.score = max(c.score, params.sat_threshold + 0.1) if agreed else min(
            c.score, params.sat_threshold - 0.1
        )
        c.source = "llm"
    return candidates
