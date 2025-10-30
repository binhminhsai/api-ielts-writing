import os
import json
from utils import build_prompt 
from dotenv import load_dotenv
import openai
from openai import OpenAI

load_dotenv()

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except TypeError:
    print("Lỗi: Không tìm thấy OPENAI_API_KEY. Vui lòng kiểm tra lại file .env của bạn.")
    exit() 

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o mini")

def run_test():
    """
    Hàm này chuẩn bị dữ liệu, gọi API OpenAI và xử lý kết quả trả về.
    """

    test_input = {
        "question": "The line graph shows the percentages of Australian export with four countries. The graph below shows the percentage of Australian exports to 4 countries from 1990 to 2012.",
        "url": "https://engnovatewebsitestorage.blob.core.windows.net/ielts-writing-task-1-images/f94aab25c631c4ad",
        "topic": "Line Graph",
        "level": "Band 6.5"
    }

    prompt = build_prompt(test_input)
    print("==== FULL PROMPT ====\n")
    print(prompt)
    print("\n==== ĐANG GỌI OPENAI ====\n")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an IELTS Writing assistant. "
                "Your job is to extract input, execute through the prompt, "
                "and output a structured JSON response exactly as requested. "
                "Do not include explanations or markdown formatting — only pure JSON."
            )
        },
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
        )

        output_text = response.choices[0].message.content.strip()

        try:
            result = json.loads(output_text)
            print("==== KẾT QUẢ JSON ĐÃ ĐƯỢC PHÂN TÍCH ====\n")

            print(json.dumps(result, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print("==== KẾT QUẢ DẠNG VĂN BẢN THÔ (Lỗi phân tích JSON) ====\n")
            print(output_text)

    except openai.APIError as e:
        print(f"Đã xảy ra lỗi với API của OpenAI: {e}")
    except Exception as e:
        print(f"Đã xảy ra một lỗi không mong muốn: {e}")

if __name__ == "__main__":
    run_test()