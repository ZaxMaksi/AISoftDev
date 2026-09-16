import json
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from app import llm

def run_comparison(config_name: str):
    print(f"\n>>> Running configuration: {config_name}")
    print(f">>> Model: {os.getenv('LLM_MODEL')}, Temp: {os.getenv('LLM_TEMPERATURE', 'default')}")
    
    requests_path = Path(__file__).parent / "requests.json"
    results_path = Path(__file__).parent / f"results_{config_name}.json"
    context_path = Path(__file__).parent.parent / "context.md"
    
    with open(requests_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    context = context_path.read_text(encoding="utf-8")
    results = []
    
    for req in data["звернення"]:
        print(f"Processing: {req['вид']}...")
        try:
            res = llm.ask(req["текст"], context)
            results.append({
                "вид": req["вид"],
                "запит": req["текст"],
                "відповідь": res["answer"],
                "час": res["elapsed"],
                "модель": res["model"]
            })
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "вид": req["вид"],
                "запит": req["текст"],
                "помилка": str(e)
            })
            
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Done. Results saved to {results_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_comparison(sys.argv[1])
    else:
        print("Usage: python run_compare.py <config_name>")
