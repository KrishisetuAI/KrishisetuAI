"""KrishiSetu AI — safe-by-construction 4-tier hybrid decision & evidence engine.

Tiers:
  Tier1 krishisethu.rules       deterministic rule engine (pure Python)
  Tier2 krishisethu.knowledge   canonical knowledge (YAML+Markdown, exact lookup)
  Tier3 krishisethu.rag         domain-governed vector RAG (ChromaDB, top-k<=5)
  Tier4 krishisethu.renderer    LLM paraphraser (Ollama) only

Plus: config, domain, confidence, safety, pipeline, api.
"""

__version__ = "0.1.0"
