"""
KrishiSetu AI - Complete Architecture Demonstration
Demonstrating Tier 1 (Rules), Tier 3 (RAG), Safety Backbone, and Tier 4 (Renderer)
"""
from pathlib import Path
from krishisethu.domain import Crop, Season
from krishisethu.rules import weather_to_action, ratoon_plan
from krishisethu.safety.chemical_registry import has_dosage_pair, strip_dosages
from krishisethu.rag import ChromaLocal, PlotContext, Retriever
from krishisethu.rag.embedder import OllamaEmbedder, StubEmbedder
from krishisethu.renderer import translate_text, LANG_HI

CHROMA_PATH = Path("./data/chroma").resolve()

print("=" * 70)
print("1. DEMO: SAFETY BACKBONE (CHEMICAL DOSAGE DETECTION & STRIPPING)")
print("=" * 70)
malicious_input = "Farmer should spray 2.5 ml/L of chlorpyrifos immediately for shoot borer."
print(f"Malicious Input   : {malicious_input}")
print(f"Lethal Pair Found : {has_dosage_pair(malicious_input)}")
print(f"Sanitized Advice  : {strip_dosages(malicious_input)}")
print("[Safe-by-Construction: Active ingredient + dosage triggers human KVK escalation]")
print()

print("=" * 70)
print("2. DEMO: TIER 1 DETERMINISTIC RULE ENGINE (SUB-MILLISECOND CPU MATH)")
print("=" * 70)
# Sub-case A: Weather Matrix (positional arguments: crop, stage, alert, drainage)
actions = weather_to_action(
    Crop.SUGARCANE,
    Season.GRAND_GROWTH,
    "heavy_rain_warning",
    "M"
)
print("Sohna Sugarcane (Heavy Rain Warning on Moderate Drainage):")
for a in actions:
    print(f"  [Prio {a.priority}] {a.action:<25} | Trace: {a.trace}")

# Sub-case B: Ratoon Economic Threshold (Sohna Benchmark)
plan = ratoon_plan(stage="grand_growth", gap_fraction=0.45, ratoon_number=3, viability=0.50)
print("\nRatoon Field Decision (3rd ratoon, 45% gap, 0.50 viability):")
print(f"  Decision   : {plan.gap.decision}")
print(f"  Rotate?    : {plan.rotation.rotate}")
print(f"  Recommend  : {plan.rotation.recommended_crop}")
print(f"  Reason     : {plan.rotation.reason}")
print()

print("=" * 70)
print("3. DEMO: TIER 3 LOCAL VECTOR RAG (CHROMADB + 1024-d EMBEDDINGS)")
print("=" * 70)
store = ChromaLocal(str(CHROMA_PATH))
try:
    embedder = OllamaEmbedder()
    print("Using live OllamaEmbedder (bge-m3:567m, 1024-d)")
except Exception:
    embedder = StubEmbedder(seed=0)
    print("Using hermetic StubEmbedder")

retriever = Retriever(embedder, store)
ctx = PlotContext(crop="sugarcane", district="sohna", season="kharif")
retrieval = retriever.retrieve("sugarcane ratoon gap fill guidance", ctx, k=2)

print(f"S_RAG Max Cosine Score: {retrieval.s_rag:.4f}")
for chunk in retrieval.chunks:
    sid, idx, title = chunk.provenance()
    print(f"  [Score: {chunk.score:.3f}] {title} ({sid}::chunk_{idx})")
print()

print("=" * 70)
print("4. DEMO: TIER 4 LOCALIZATION & PARAPHRASER (PROMPT 6)")
print("=" * 70)
test_action = "Suspend irrigation"
hindi_text, missed = translate_text(test_action, LANG_HI)
print(f"Verified Action (EN) : {test_action}")
print(f"Devanagari (Hindi)   : {hindi_text}")
print(f"Translation Missed?  : {missed} (Zero guess-work / only verified agronomic dictionary)")
print("=" * 70)