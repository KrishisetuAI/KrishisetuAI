"""Translation stub (prompt 6).

Renders the VERIFIED structured action text into Hindi/Hinglish for a language
tag. This is a rule-based/placeholder translator for now (Bhashini/Gemini hooks
later); it only transforms already-verified fields — the input must be a parsed,
schema-valid :class:`~krishisethu.renderer.structured_output.Advisory` (or its
verified action text). It never invents content: an unknown English phrase is
kept as-is and flagged rather than silently guessed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from krishisethu.renderer.structured_output import Advisory

#: Language tags the stub understands.
LANG_EN = "en"
LANG_HI = "hi"

#: en -> hi phrase map for the known Tier-1 action phrases (seeded from the
#: deterministic rule engine + the advisory action list). Unknown phrases are
#: passed through untranslated and flagged in ``untranslated``.
_PHRASES_HI: dict[str, str] = {
    "Suspend irrigation": "सिंचाई रोक दें",
    "Clear furrow outlets": "पानी के निकास (फ़रो) के रास्ते साफ़ करें",
    "Delay foliar spray 72h": "पत्ती छिड़काव 72 घंटे के लिए टालें",
    "Avoid field traffic": "खेत में आवाजाही से बचें",
    "Proceed with routine schedule": "नियमित कार्यक्रम के अनुसार जारी रखें",
    "Maintain irrigation to avoid moisture stress": "नमी तनाव से बचने के लिए सिंचाई जारी रखें",
    "Consult the KVK agronomist before any chemical application.": (
        "किसी भी रासायनिक उपयोग से पहले कृषि विज्ञान केंद्र (KVK) कृषि विशेषज्ञ से परामर्श लें।"
    ),
}


def translate_text(text: str, lang: str = LANG_HI) -> tuple[str, bool]:
    """Translate one verified action text; returns ``(translated, untranslated?)``.

    ``lang="en"`` is a pass-through. For Hindi the stub looks the phrase up in
    the seed map; an unknown phrase is returned verbatim with ``untranslated=True``
    (a real translator — Bhashini — replaces this later; the stub never guesses).
    """
    if lang == LANG_EN or not text.strip():
        return text, False
    translated = _PHRASES_HI.get(text.strip())
    if translated is not None:
        return translated, False
    return text, True  # untranslated, flagged


@dataclass(frozen=True)
class LocalizedAdvisory:
    """A verified advisory rendered into a language tag.

    Only the already-verified action texts are transformed; evidence markers,
    input-class flags and the confidence breakdown are carried over unchanged
    (translation must never alter a source marker).
    """

    language: str
    actions: tuple[str, ...] = ()
    input_classes: tuple[str | None, ...] = ()
    untranslated: tuple[int, ...] = ()  # indices kept verbatim
    confidence_breakdown: object | None = None


def localize(advisory: Advisory, lang: str = LANG_HI) -> LocalizedAdvisory:
    """Render a *verified* advisory's action texts into ``lang``."""
    texts: list[str] = []
    input_classes: list[str | None] = []
    untranslated: list[int] = []
    for i, action in enumerate(advisory.actions):
        translated, missed = translate_text(action.text, lang)
        texts.append(translated)
        input_classes.append(action.input_class)
        if missed:
            untranslated.append(i)
    return LocalizedAdvisory(
        language=lang,
        actions=tuple(texts),
        input_classes=tuple(input_classes),
        untranslated=tuple(untranslated),
        confidence_breakdown=advisory.confidence_breakdown,
    )
