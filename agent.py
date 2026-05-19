import subprocess
from datetime import datetime
import json
from pathlib import Path

import ollama

from config import AVAILABLE_MODELS, MODEL_SELECTION, SYSTEM_PROMPT
from memory import Memory
from pop_bank_loader import get_loader


def _available_ollama_models():
    try:
        proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=8)
    except Exception:
        return []

    if proc.returncode != 0:
        return []

    models = []
    for line in proc.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models


def _select_model(preferred_model: str):
    if preferred_model not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{preferred_model}' is not in the approved shortlist.")

    installed_models = _available_ollama_models()
    if installed_models and preferred_model not in installed_models:
        for model_name in AVAILABLE_MODELS:
            if model_name in installed_models:
                return model_name

    return preferred_model


class AIAgent:
    def __init__(self, model_name: str = MODEL_SELECTION):
        self.memory = Memory()
        self.model_name = _select_model(model_name)
        self.interaction_log = Path("reports") / "interaction_log.jsonl"
        
        # Load POP Bank knowledge base (lazy-loaded, index only)
        print("Initializing POP Bank_Kshitij loader...")
        self.pop_loader = get_loader()
        self.pop_loader.initialize()
        self.state_crops = self.pop_loader.get_state_crops()
        self.state_aliases = self.pop_loader.get_state_aliases()
        print(f"✓ POP Bank ready: {len(self.state_crops)} states indexed")
        self.kb_summary = self.pop_loader.build_summary()

    def _normalize_key(self, value: str) -> str:
        return "".join(ch for ch in (value or "").lower() if ch.isalnum())

    def _find_state_mentions(self, user_input: str):
        user_normalized = self._normalize_key(user_input)
        matches = []

        seen = set()
        for state in sorted(self.state_crops.keys(), key=len, reverse=True):
            aliases = self.state_aliases.get(state, [state])
            for alias in sorted(aliases, key=len, reverse=True):
                if self._normalize_key(alias) in user_normalized and state not in seen:
                    matches.append(state)
                    seen.add(state)
                    break

        return matches

    def _build_direct_pop_bank_answer(self, user_input: str) -> str:
        user_lower = user_input.lower()
        state_matches = self._find_state_mentions(user_input)

        if state_matches:
            state = state_matches[0]
            crops = self.state_crops.get(state, [])
            if not crops:
                return ""

            crop_count = len(crops)
            sample_count = min(12, crop_count)
            crop_sample = ", ".join(crops[:sample_count])
            remaining = crop_count - sample_count

            if any(keyword in user_lower for keyword in ("how many", "count", "number of", "number", "total")):
                return f"{state}: {crop_count} crop profiles are indexed in the POP Bank knowledge base."

            if remaining > 0:
                return (
                    f"{state}: {crop_count} crop profiles are indexed. "
                    f"Examples: {crop_sample}. +{remaining} more."
                )

            return f"{state}: {crop_count} crop profiles are indexed. Crops: {crop_sample}."

        return ""

    def _classify_query(self, user_input: str) -> str:
        user_lower = user_input.lower()

        if self._is_lookup_query(user_input):
            return "pop_lookup"

        if any(word in user_lower for word in ("summarize", "summary", "brief", "list", "compare")):
            return "summarization"

        if any(word in user_lower for word in ("why", "how", "analyze", "reason", "explain", "should i", "what if")):
            return "reasoning"

        return "general"

    def _is_lookup_query(self, user_input: str) -> bool:
        user_lower = user_input.lower()
        lookup_markers = (
            "what crops",
            "which crops",
            "how many",
            "count",
            "number of",
            "list",
            "show",
            "crops in",
            "crop profiles",
            "available crops",
        )
        non_lookup_markers = (
            "summarize",
            "summary",
            "brief",
            "compare",
            "why",
            "how",
            "analyze",
            "reason",
            "explain",
            "should i",
            "what if",
        )

        if any(marker in user_lower for marker in non_lookup_markers):
            return False

        if self._find_state_mentions(user_input):
            return any(marker in user_lower for marker in lookup_markers)

        return False

    def _select_runtime_model(self, route: str) -> str:
        if route == "summarization":
            for candidate in ("phi4-mini:3.8b", "llama3:latest"):
                if candidate in AVAILABLE_MODELS:
                    return _select_model(candidate)

        if route == "reasoning":
            for candidate in ("qwen3:4b", "qwen2.5:1.5b", "llama3:latest"):
                if candidate in AVAILABLE_MODELS:
                    return _select_model(candidate)

        return self.model_name

    def _log_interaction(self, *, question: str, route: str, model: str, response: str, started_at: datetime):
        elapsed = (datetime.now() - started_at).total_seconds()
        accuracy = 1.0 if route == "pop_lookup" else (0.9 if response and not self._looks_like_not_found(response) else 0.0)
        record = {
            "timestamp": started_at.isoformat(timespec="seconds"),
            "question": question,
            "route": route,
            "model": model,
            "response_time": round(elapsed, 3),
            "accuracy": round(accuracy, 2),
        }

        self.interaction_log.parent.mkdir(parents=True, exist_ok=True)
        with self.interaction_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _extract_relevant_knowledge(self, user_input: str) -> str:
        """Extract relevant crop and state information from user input"""
        user_lower = user_input.lower()
        relevant_knowledge = []
        
        # Check for state mentions
        for state in self._find_state_mentions(user_input):
                state_info = self.pop_loader.get_state_info(state)
                if state_info:
                    relevant_knowledge.append(f"[{state} Information]\n{state_info[:2000]}")
                    break
        
        # Check for crop mentions
        for state_crops_list in self.state_crops.values():
            for crop in state_crops_list:
                if crop.lower() in user_lower:
                    crop_info = self.pop_loader.get_crop_info(crop)
                    if crop_info:
                        relevant_knowledge.append(f"[{crop} Information]\n{crop_info[:2000]}")
        
        if relevant_knowledge:
            return "\n\n---\n\n".join(relevant_knowledge)
        
        return ""

    def build_messages(self, user_input):
        system_message = SYSTEM_PROMPT
        
        # Add relevant knowledge from POP Bank
        relevant_knowledge = self._extract_relevant_knowledge(user_input)
        if relevant_knowledge:
            system_message += f"\n\n[POP Bank Knowledge Base]\n{relevant_knowledge}"
        
        messages = [{"role": "system", "content": system_message}]
        messages.extend(self.memory.get_all())
        return messages

    def chat(self, user_input):
        started_at = datetime.now()
        self.memory.add("user", user_input)
        route = self._classify_query(user_input)

        # Try direct POP Bank lookup first
        direct_answer = self._build_direct_pop_bank_answer(user_input) if route == "pop_lookup" else ""
        if direct_answer:
            self.memory.add("assistant", direct_answer)
            self._log_interaction(
                question=user_input,
                route=route,
                model="POP Bank direct answer",
                response=direct_answer,
                started_at=started_at,
            )
            return direct_answer

        # Try dataset retrieval for all other queries
        dataset_answer, confidence = self.pop_loader.get_best_dataset_answer(user_input, min_confidence=0.2)
        if dataset_answer and confidence >= 0.3:
            self.memory.add("assistant", dataset_answer)
            self._log_interaction(
                question=user_input,
                route="dataset_retrieval",
                model="POP Bank semantic search",
                response=dataset_answer,
                started_at=started_at,
            )
            return dataset_answer

        # Fall back to LLM only if dataset confidence is low
        runtime_model = self._select_runtime_model(route)

        try:
            result = ollama.chat(
                model=runtime_model,
                messages=self.build_messages(user_input),
            )
            response = result["message"]["content"]
        except Exception as exc:
            response = (
                "Data not found: the selected Ollama model is not available or the Ollama service is not ready. "
                f"Details: {exc}"
            )

        self.memory.add("assistant", response)
        self._log_interaction(
            question=user_input,
            route=route,
            model=runtime_model,
            response=response,
            started_at=started_at,
        )
        return response

    def research_brief(self):
        return f"Model: {self.model_name}\nMode: POP Bank_Kshitij Knowledge Base (loaded)\n\n{self.kb_summary}"

    def _looks_like_not_found(self, answer: str):
        normalized = (answer or "").strip().lower()
        if not normalized:
            return True

        not_found_markers = (
            "i don't know",
            "i do not know",
            "not found",
            "no data",
            "cannot find",
            "can't find",
            "no relevant",
            "not available",
            "unable to",
        )
        return any(marker in normalized for marker in not_found_markers)

    def generate_report(self, user_input):
        direct_answer = self._build_direct_pop_bank_answer(user_input) if self._classify_query(user_input) == "pop_lookup" else ""
        if direct_answer:
            return (
                f"# AI Agent Response\n\n"
                f"## Question\n{user_input}\n\n"
                f"## Answer\n{direct_answer}\n\n"
                f"## Mode\nPOP Bank direct answer\n"
            )

        answer = self.chat(user_input)
        if self._looks_like_not_found(answer):
            return (
                f"# Data Not Found Report\n\n"
                f"## Question\n{user_input}\n\n"
                f"## Status\nData not found or not confidently available from the model response.\n\n"
                f"## Response\n{answer}\n\n"
                f"## Model\n{self.model_name}\n"
            )

        return (
            f"# AI Agent Response\n\n"
            f"## Question\n{user_input}\n\n"
            f"## Answer\n{answer}\n\n"
            f"## Model\n{self.model_name}\n"
        )