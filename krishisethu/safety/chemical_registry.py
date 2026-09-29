"""Tier-2-backed registry of active ingredients + numeric-dosage detector.

Gate-2 material. The non-negotiable invariant: KrishiSetu never emits a specific
numeric dosage or a specific active ingredient as a recommendation — it
recommends an input CLASS only. This module is the single source of truth for
*what counts as an active ingredient* and *what counts as a dosage quantity*, so
the chemical gate is a pure term-level check rather than a regex re-invented in
the UI. The terms are a curated, Tier-2-reviewed seed (plan Sec 8); extending the
registry is a reviewable fact change, not an ad hoc filter edit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ChemicalEntry:
    """One active ingredient backed by its region-verified input class.

    ``match_pattern`` is the case-insensitive word-boundary pattern used to spot
    the ingredient name (and its aliases) in advisory text.
    """

    name: str
    input_class: str
    aliases: tuple[str, ...] = ()

    @property
    def match_pattern(self) -> str:
        tokens = [self.name, *self.aliases]
        return rf"\b(?:{'|'.join(re.escape(t) for t in tokens)})\b"


#: Registry of active ingredients and their input classes (Tier-2 reviewed seed).
#:
#: The first eight are the prompt-specified seed (chlorpyrifos, imidacloprid,
#: glyphosate, carbendazim, cypermethrin, mancozeb, deltamethrin, neem-oil). The
#: remainder are common Haryana / West-UP actives added for defense-in-depth so a
#: regional ingredient is not invisible to the chemical gate. Extending the
#: registry is a reviewable fact change, not an ad hoc filter edit.
CHEMICALS: tuple[ChemicalEntry, ...] = (
    ChemicalEntry("chlorpyrifos", "organophosphate insecticide", aliases=("chlorpyriphos",)),
    ChemicalEntry("imidacloprid", "systemic neonicotinoid insecticide"),
    ChemicalEntry("glyphosate", "non-selective systemic herbicide"),
    ChemicalEntry("carbendazim", "systemic benzimidazole fungicide"),
    ChemicalEntry("cypermethrin", "synthetic pyrethroid insecticide"),
    ChemicalEntry("mancozeb", "contact dithiocarbamate fungicide"),
    ChemicalEntry("deltamethrin", "synthetic pyrethroid insecticide"),
    ChemicalEntry("neem-oil", "botanical (neem) insecticide / acaricide", aliases=("neem oil",)),
    # defense-in-depth regional actives
    ChemicalEntry("quinalphos", "organophosphate insecticide"),
    ChemicalEntry("profenofos", "organophosphate insecticide"),
    ChemicalEntry("acephate", "organophosphate insecticide"),
    ChemicalEntry("thiamethoxam", "neonicotinoid insecticide"),
    ChemicalEntry("lambda-cyhalothrin", "synthetic pyrethroid insecticide", aliases=("lambda cyhalothrin",)),
    ChemicalEntry("fipronil", "phenylpyrazole insecticide"),
    ChemicalEntry("indoxacarb", "oxadiazine insecticide"),
    ChemicalEntry("atrazine", "triazine herbicide"),
)

#: Combined ingredient matcher (built lazily, case-insensitive, word-boundary).
_CHEM_RE = re.compile(
    "|".join(c.match_pattern for c in CHEMICALS), re.IGNORECASE
)

#: Numeric dosage quantity + unit. Longest units first so ``ml/l`` matches before
#: ``ml`` or ``l``; ``per acre`` / ``per ha`` handle rate-style wording. ``%`` is
#: included because concentrations like ``0.3%`` or ``2%`` ARE chemical dosages
#: (e.g. "mancozeb at 0.3%"). Standalone agronomic percentages (e.g. "20% gap")
#: will also match but that is SAFE: Gate 2 will escalate rather than deliver a
#: dosage. The renderer post-parse validation also catches these.
#:
#: The separator after the quantity tolerates ``2.5 ml``, ``2.5ml`` and
#: ``2.5 ml/l`` (optional spaces and/or slash), and letter-initial units use a
#: ``(?<![A-Za-z])`` guard instead of ``\\b`` so a unit glued to a digit
#: (``2.5ml``, ``5g``) is caught — a digit-to-letter transition has no ``\\b``.
_UNIT = (
    r"(?P<unit>"
    r"ml\s*/\s*(?:ha|acre|hectare|liter|litre|l)"
    r"|g\s*/\s*(?:ha|acre|hectare|liter|litre|l)"
    r"|kg\s*/\s*(?:ha|acre|hectare)"
    r"|per\s+(?:hectare|acre|ha|litre|liter|l)"
    r"|(?<![A-Za-z])mls?(?![A-Za-z])"
    r"|(?<![A-Za-z])litres?(?![A-Za-z])"
    r"|(?<![A-Za-z])liters?(?![A-Za-z])"
    r"|(?<![A-Za-z])ccs?(?![A-Za-z])"
    r"|(?<![A-Za-z])l(?![A-Za-z])"
    r"|(?<![A-Za-z])g(?:rams?|ms?)?(?![A-Za-z])"
    r"|(?<![A-Za-z])kgs?(?![A-Za-z])"
    r"|(?<![A-Za-z])mg(?![A-Za-z])"
    r"|(?<![A-Za-z])ppm(?![A-Za-z])"
    r"|(?<![A-Za-z])(?:teaspoons?|tsp|spoons?)(?![A-Za-z])"
    r"|%"
    r")"
)
DOSAGE_RE = re.compile(
    rf"(?P<qty>\d+(?:[.,]\d+)?)\s*/?\s*{_UNIT}", re.IGNORECASE
)

#: Dose-intent phrasing (no numeric needed): how-much / per-unit / strength /
#: concentration / dilution request wording. A chemical mentioned with any of
#: these intent words escalates even without a numeric quantity. Unit words are
#: included so a WORD-form dosage ("fifty grams of chlorpyrifos", "two hundred
#: ml of profenofos") still trips the pair through the intent leg.
DOSE_INTENT_RE = re.compile(
    r"\b(?:dose|dosage|dosing|rate|quantity|amount|how\s+much|much|strength|"
    r"concentration|dilution|per\s+litre|per\s+liter|per\s+hectare|per\s+acre|per\s+ha|"
    r"grams?|gms?|millilit(?:er|re)s?|ml|lit(?:er|re)s?|kilograms?|kilogramme|"
    r"cc|ppm|percent|per\s+cent)\b",
    re.IGNORECASE,
)


def find_chemicals(text: str) -> list[ChemicalEntry]:
    """Every active ingredient mentioned in ``text`` (deduped, in registry order)."""
    found: dict[str, ChemicalEntry] = {}
    for m in _CHEM_RE.finditer(str(text)):
        token = m.group(0).lower()
        for entry in CHEMICALS:
            if token in {entry.name.lower(), *(a.lower() for a in entry.aliases)}:
                found.setdefault(entry.name, entry)
                break
    return list(found.values())


def find_dosages(text: str) -> list[str]:
    """Every numeric-dosage token in ``text`` (e.g. ``2.5 ml/l``)."""
    return [m.group(0).strip() for m in DOSAGE_RE.finditer(str(text))]


def has_numeric_dosage(text: str) -> bool:
    """True when ``text`` contains a numeric dosage quantity."""
    return bool(DOSAGE_RE.search(str(text)))


def has_dose_intent(text: str) -> bool:
    """True when ``text`` requests/mentions a dose, rate or per-unit amount."""
    return bool(DOSE_INTENT_RE.search(str(text)))


def has_dosage_pair(text: str) -> bool:
    """True when an active ingredient is paired with a dosage expression OR when
    ANY numeric/word dosage or dose-intent phrasing is present.

    The invariant: NO advisory containing a dosage passes Gate 2, regardless of
    whether the chemical name is in the registry or unknown. This function returns
    True if ANY of the following hold:
      (a) A registered chemical is paired with a numeric dosage or dose intent
      (b) ANY numeric dosage is present (has_numeric_dosage)
      (c) ANY dose-seeking intent is detected (has_dose_intent)
    """
    t = str(text)
    chemicals_present = bool(find_chemicals(t))
    numeric_dosage = has_numeric_dosage(t)
    dose_intent = has_dose_intent(t)
    return (chemicals_present and (numeric_dosage or dose_intent)) or numeric_dosage or dose_intent


def strip_dosages(text: str) -> str:
    """Remove every numeric dosage token from ``text``.

    Used to produce the input-class-only substitute; the ingredient name itself is
    neutralized separately by the gate (replaced with its input class).
    """
    return DOSAGE_RE.sub("", str(text)).strip()
