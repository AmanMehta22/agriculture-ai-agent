#!/usr/bin/env python3
"""Rebuild POP Bank PDF index and regenerate extracted text cache."""

import json
from pop_bank_loader import get_loader


def main():
    loader = get_loader()
    print("Initializing and building index...")
    loader.initialize()

    states = list(loader.pdf_index.keys())
    print(f"Found {len(states)} states in index")

    total_loaded = 0
    for s in states:
        print(f"Loading state: {s}...")
        try:
            loader._load_state_data(s)
            if s in loader.knowledge_base:
                total_loaded += len(loader.knowledge_base[s])
        except Exception as e:
            print(f"  ⚠ Error loading {s}: {e}")

    # Write cache file
    try:
        with open(loader.cache_file, 'w', encoding='utf-8') as f:
            json.dump(loader.knowledge_base, f, indent=2, ensure_ascii=False)
        print(f"Wrote cache to {loader.cache_file} with {total_loaded} crop entries")
    except Exception as e:
        print("Failed to write cache:", e)

    print("Done.")


if __name__ == '__main__':
    main()
