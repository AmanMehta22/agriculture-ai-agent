"""
POP Bank_Kshitij Data Loader
Lazy-loads agricultural knowledge from POP Bank PDFs by state and crop on-demand
"""

import json
from pathlib import Path
from typing import List, Tuple
from tools import _normalize_text, _read_pdf_pages, _ocr_pdf_pages


class POPBankLoader:
    def __init__(self, pop_bank_path: str = "POP Bank_Kshitij"):
        self.pop_bank_path = Path(pop_bank_path)
        self.knowledge_base = {}
        self.pdf_index = {}  # {state: {crop: pdf_path}}
        self.cache_file = Path("pop_bank_cache.json")
        self.index_file = Path("pop_bank_index.json")
        self._loaded_states = set()
        self._initialized = False
        self._state_lookup = {}
        self._state_aliases = {}

    def _normalize_key(self, value: str) -> str:
        return "".join(ch for ch in (value or "").lower() if ch.isalnum())

    def _build_state_lookup(self):
        self._state_lookup = {
            self._normalize_key(state_name): state_name
            for state_name in self.pdf_index.keys()
        }
        self._state_aliases = {}

        alias_map = {
            "Andra Pradesh": ["Andhra Pradesh"],
            "Chattisgarh": ["Chhattisgarh"],
        }

        for state_name in self.pdf_index.keys():
            aliases = [state_name]
            aliases.extend(alias_map.get(state_name, []))
            self._state_aliases[state_name] = aliases
            for alias in aliases:
                self._state_lookup[self._normalize_key(alias)] = state_name

    def resolve_state_name(self, state: str) -> str:
        if not state:
            return ""

        if state in self.pdf_index:
            return state

        normalized = self._normalize_key(state)
        return self._state_lookup.get(normalized, state)

    def get_state_aliases(self) -> dict:
        self.initialize()
        return self._state_aliases

    def _build_pdf_index(self):
        """Build index of PDF files without extracting content"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.pdf_index = json.load(f)
                    self._build_state_lookup()
                    return
            except Exception:
                pass

        print("Building POP Bank PDF index...")
        
        if not self.pop_bank_path.exists():
            print(f"Warning: POP Bank path not found: {self.pop_bank_path}")
            return

        for state_dir in sorted(self.pop_bank_path.iterdir()):
            if not state_dir.is_dir():
                continue
            
            state_name = state_dir.name
            state_pdfs = {}
            
            for pdf_file in sorted(state_dir.glob("*.pdf")):
                crop_name = pdf_file.stem.replace("POP - ", "").replace("POP-", "")
                state_pdfs[crop_name] = str(pdf_file)
            
            if state_pdfs:
                self.pdf_index[state_name] = state_pdfs

        self._build_state_lookup()

        # Save index for future use
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.pdf_index, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        print(f"✓ Index built: {len(self.pdf_index)} states")

    def initialize(self):
        """Initialize the loader - builds index but doesn't extract data yet"""
        if self._initialized:
            return
        
        self._build_pdf_index()
        self._initialized = True

    def _load_state_data(self, state: str):
        """Load data for a specific state from cache or extract from PDFs"""
        state = self.resolve_state_name(state)

        if state in self._loaded_states:
            return
        
        # Try to load from cache
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    if state in cache:
                        self.knowledge_base[state] = cache[state]
                        self._loaded_states.add(state)
                        return
            except Exception:
                pass

        # Extract from PDFs if not in cache
        if state not in self.pdf_index:
            return
        
        print(f"Loading {state}...")
        state_data = {}
        
        for crop_name, pdf_path in self.pdf_index[state].items():
            try:
                pages = _read_pdf_pages(pdf_path)
                
                if not pages or all(not p.get("text", "").strip() for p in pages):
                    pages = _ocr_pdf_pages(pdf_path)
                
                full_text = " ".join([
                    _normalize_text(page.get("text", ""))
                    for page in pages
                    if page.get("text", "").strip()
                ])
                
                if full_text and len(full_text) > 50:
                    state_data[crop_name] = full_text
            
            except Exception as e:
                print(f"  ⚠ Error loading {crop_name}: {str(e)[:40]}")
        
        if state_data:
            self.knowledge_base[state] = state_data
            self._loaded_states.add(state)

    def get_state_crops(self) -> dict:
        """Returns {state: [list of crops]} from index only (no extraction)"""
        self.initialize()
        return {state: list(crops.keys()) for state, crops in self.pdf_index.items()}

    def get_crop_info(self, crop_name: str, state: str = None) -> str:
        """Get knowledge about a specific crop"""
        self.initialize()
        
        if state:
            state = self.resolve_state_name(state)
            self._load_state_data(state)
            if state in self.knowledge_base and crop_name in self.knowledge_base[state]:
                return self.knowledge_base[state][crop_name][:3000]
        else:
            # Search across all states
            for s in self.pdf_index.keys():
                self._load_state_data(s)
                if s in self.knowledge_base and crop_name in self.knowledge_base[s]:
                    return self.knowledge_base[s][crop_name][:3000]
        
        return ""

    def get_state_info(self, state: str) -> str:
        """Get all crop information for a specific state"""
        self.initialize()
        state = self.resolve_state_name(state)
        self._load_state_data(state)
        
        if state not in self.knowledge_base:
            return ""
        
        all_text = []
        for crop_name, text in list(self.knowledge_base[state].items())[:5]:
            all_text.append(f"[{crop_name}] {text[:500]}")
        
        return "\n".join(all_text)

    def build_summary(self) -> str:
        """Build a summary of available knowledge"""
        self.initialize()
        
        if not self.pdf_index:
            return "POP Bank Knowledge Base: No data available"
        
        summary_lines = ["POP Bank_Kshitij Knowledge Base:"]
        total_crops = sum(len(crops) for crops in self.pdf_index.values())
        
        for state in sorted(self.pdf_index.keys()):
            crops = self.pdf_index[state]
            crop_count = len(crops)
            crop_list = ", ".join(sorted(crops.keys())[:3])
            if len(crops) > 3:
                crop_list += f", +{len(crops) - 3} more"
            summary_lines.append(f"  • {state}: {crop_count} crops")
        
        summary_lines.append(f"\n  Ready: {len(self.pdf_index)} states, {total_crops} crop profiles")
        
        return "\n".join(summary_lines)

    def _tokenize_query(self, query: str) -> List[str]:
        """Extract keywords from query."""
        words = query.lower().split()
        return [w.strip("?.!,;") for w in words if len(w) > 2]

    def retrieve_from_dataset(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """
        Search cached dataset for relevant content.
        Returns: [(state, crop_name, confidence_score), ...]
        """
        self.initialize()
        
        query_tokens = set(self._tokenize_query(query))
        if not query_tokens:
            return []

        results = []

        for state in self.pdf_index.keys():
            self._load_state_data(state)
            if state not in self.knowledge_base:
                continue

            for crop_name, content in self.knowledge_base[state].items():
                content_tokens = set(self._tokenize_query(content[:500]))
                
                matches = len(query_tokens & content_tokens)
                if matches == 0:
                    continue
                
                confidence = min(1.0, matches / len(query_tokens))
                results.append((state, crop_name, confidence, content[:1000]))

        results.sort(key=lambda x: x[2], reverse=True)
        return [(state, crop, conf) for state, crop, conf, _ in results[:top_k]]

    def get_best_dataset_answer(self, query: str, min_confidence: float = 0.3) -> Tuple[str, float]:
        """
        Get best answer from dataset. Returns (answer_text, confidence).
        """
        results = self.retrieve_from_dataset(query, top_k=1)
        
        if not results:
            return "", 0.0

        state, crop_name, confidence = results[0]
        if confidence < min_confidence:
            return "", confidence

        self._load_state_data(state)
        if state in self.knowledge_base and crop_name in self.knowledge_base[state]:
            content = self.knowledge_base[state][crop_name]
            answer = f"[{state} - {crop_name}]\n\n{content[:2000]}"
            return answer, confidence

        return "", 0.0


# Convenience function
_loader_instance = None

def get_loader() -> POPBankLoader:
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = POPBankLoader()
    return _loader_instance


