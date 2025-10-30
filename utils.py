import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

def load_prompt(prompt_name: str):

    prompt_path = BASE_DIR / "prompts" / f"{prompt_name}.txt"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Không tìm thấy tệp prompt tại: {str(prompt_path)}.\n"
            f"Hãy chắc chắn rằng tệp '{prompt_name}.txt' tồn tại trong thư mục 'prompts'."
        )

def build_prompt(input_data: dict, prompt_name: str):

    prompt_template = load_prompt(prompt_name)
    input_json = json.dumps(input_data, indent=2, ensure_ascii=False)
    return prompt_template.replace("{input_json}", input_json)
