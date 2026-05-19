import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import argparse
from datetime import datetime
from pathlib import Path

from agent import AIAgent
from config import AVAILABLE_MODELS, DEFAULT_MODEL


def parse_args():
    parser = argparse.ArgumentParser(description="Direct agricultural AI chat agent")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=AVAILABLE_MODELS,
        help="Approved Ollama model to use",
    )
    parser.add_argument(
        "--ask",
        default=None,
        help="Ask one question and exit instead of starting the interactive loop",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    agent = AIAgent(model_name=args.model)
    session_saved = False

    def save_session_memory():
        nonlocal session_saved
        if session_saved:
            return

        markdown_path, json_path = agent.memory.save()
        print(f"Session memory saved to {markdown_path}")
        print(f"Session memory JSON saved to {json_path}")
        session_saved = True

    print("Direct AI Agent Started (type 'exit' to stop)\n")
    print(agent.research_brief() + "\n")
    print("Ask for crop advice, comparisons, summaries, drafting help, or step-by-step reasoning.\n")

    if args.ask:
        print(agent.generate_report(args.ask))
        save_session_memory()
        return

    try:
        while True:
            user_input = input("You: ")

            normalized = user_input.strip()
            if not normalized:
                continue

            lowered = normalized.lower()

            if lowered in {"exit", "quit"}:
                break

            if lowered in {"help", "/help"}:
                print("Commands: help, brief, report <question>, reset, exit")
                continue

            if lowered in {"brief", "/brief"}:
                print(agent.research_brief())
                continue

            if lowered in {"reset", "/reset"}:
                agent.memory.clear()
                print("Conversation history cleared.")
                continue

            if lowered.startswith("report "):
                question = normalized.split(" ", 1)[1].strip()
                if not question:
                    print("Usage: report <question>")
                    continue

                report = agent.generate_report(question)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_dir = Path("reports")
                report_dir.mkdir(parents=True, exist_ok=True)
                report_path = report_dir / f"agent_report_{timestamp}.md"
                report_path.write_text(report, encoding="utf-8")
                print(report)
                print(f"Report saved to {report_path}")
                continue

            response = agent.chat(normalized)
            print("Agent:", response)
    finally:
        save_session_memory()


if __name__ == "__main__":
    main()