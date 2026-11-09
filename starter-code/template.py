"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

LLM: Groq qua API tương thích OpenAI (OPENAI_API_KEY, OPENAI_BASE_URL trong .env ở thư mục gốc lab).
"""

import json
import os
import re
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from tools import TOOL_DEFINITIONS, TOOL_MAP

MODEL_ID = "openai/gpt-oss-120b"
TEMPERATURE = 0.0
RETRY_TEMPERATURE = 0.3
ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")

# Lưu ý: prompt viết dạng văn xuôi, KHÔNG dùng khung mẫu kiểu "Observation: <...>" hay ngoặc nhọn trong ví dụ.
# Khung mẫu đó từng làm Gemini trả MALFORMED_RESPONSE ở lượt sau khi có kết quả tool.
SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup. Trả lời bằng tiếng Việt.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Cách làm việc:
- Mỗi lượt, viết một dòng Thought ngắn gọn về việc cần làm tiếp theo.
- Nếu cần dữ liệu, viết một dòng Action gồm một object JSON có khóa "name" (tên tool) và "args" (tham số). Sau đó dừng lại và chờ kết quả tool.
- Khi đã đủ dữ liệu, viết dòng Final Answer với câu trả lời hoàn chỉnh cho khách hàng.

Quy tắc:
- Chỉ viết một Action mỗi lượt.
- Không tự bịa kết quả tool.
- Với giá vé, truyền max_price là số nguyên VND.
- Nếu tool trả về lỗi 2 lần liên tiếp, hãy đưa Final Answer báo lỗi cho khách hàng.
- Final Answer phải nêu số liệu cụ thể (mã chuyến bay, giá vé, nhiệt độ).
"""

# Nhắc định dạng ở cuối mỗi lượt: model hay bỏ qua khung ReAct và trả lời tự do nếu không được nhắc.
FORMAT_REMINDER = (
    "Hãy viết tiếp đúng định dạng: một dòng Thought, rồi MỘT dòng Action (JSON) "
    "hoặc dòng Final Answer. Không tự viết Observation hay kết quả tool."
)

BASELINE_PROMPT = """Bạn là trợ lý hỗ trợ khách hàng Vingroup. Trả lời ngắn gọn bằng tiếng Việt.
Bạn KHÔNG có công cụ tra cứu dữ liệu thời gian thực. Nếu câu hỏi cần dữ liệu cụ thể (chuyến bay, giá vé, thời tiết),
hãy nói rõ bạn không tra cứu được và không được bịa số liệu."""


def _get_client() -> OpenAI:
    load_dotenv(ENV_PATH)
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise RuntimeError("Thiếu OPENAI_API_KEY. Hãy khai báo trong file .env ở thư mục gốc lab.")
    return OpenAI(api_key=api_key, base_url=base_url)


def _generate(client: OpenAI, system_prompt: str, user_text: str, stop=None, max_retries: int = 6) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    for attempt in range(max_retries + 1):
        # temperature=0 có thể lặp lại đúng output lỗi, nên các lần thử lại dùng nhiệt độ nhỏ > 0.
        temperature = TEMPERATURE if attempt == 0 else RETRY_TEMPERATURE
        try:
            response = client.chat.completions.create(model=MODEL_ID, messages=messages, temperature=temperature)
        except RateLimitError:
            # Hết quota/rate limit: chờ rồi thử lại thay vì crash.
            if attempt == max_retries:
                raise
            time.sleep(min(10 * (attempt + 1), 60))
            continue

        text = (response.choices[0].message.content or "").strip()
        if text:
            # Cắt tại "Observation:" bằng code để LLM không tự bịa kết quả tool.
            if stop:
                for marker in stop:
                    idx = text.find(marker)
                    if idx != -1:
                        text = text[:idx].rstrip()
            return text
        # Response rỗng: chờ lâu hơn rồi thử lại.
        time.sleep(3 * (attempt + 1))
    raise RuntimeError("LLM trả về response rỗng sau nhiều lần thử.")


def parse_price(value) -> int:
    """Chuyển giá từ LLM (số, '2 triệu', '1.5 triệu', '500k', '2.000.000') thành VND nguyên."""
    if isinstance(value, bool):
        raise ValueError(f"Không đọc được giá: {value}")
    if isinstance(value, (int, float)):
        return int(value)

    s = str(value).lower().replace("vnđ", "").replace("vnd", "").replace("đ", "").strip()
    multiplier = 1
    if "triệu" in s or s.endswith("tr"):
        multiplier = 1_000_000
        s = s.replace("triệu", "").replace("tr", "").strip()
    elif "nghìn" in s or s.endswith("k"):
        multiplier = 1_000
        s = s.replace("nghìn", "").replace("k", "").strip()

    if multiplier > 1:
        return int(round(float(s.replace(",", ".")) * multiplier))

    digits = s.replace(".", "").replace(",", "")
    if not digits.isdigit():
        raise ValueError(f"Không đọc được giá: {value}")
    return int(digits)


def _call_tool(name: str, args: dict):
    """Thực thi tool trong TOOL_MAP. Mọi lỗi được trả về dạng {"error": ...} thay vì làm crash agent."""
    key = name.strip().lower()
    if key not in TOOL_MAP:
        return {"error": f"Tool không tồn tại: {name}. Các tool hợp lệ: {list(TOOL_MAP)}"}

    args = dict(args)
    if "max_price" in args:
        try:
            args["max_price"] = parse_price(args["max_price"])
        except ValueError as e:
            return {"error": str(e)}

    try:
        return TOOL_MAP[key](**args)
    except TypeError as e:
        return {"error": f"Sai tham số cho {key}: {e}"}
    except Exception as e:  # lỗi bất ngờ từ tool
        return {"error": f"Tool {key} lỗi: {e}"}


def _format_observation(obs) -> str:
    """
    Đưa kết quả tool về dạng văn bản dễ đọc trước khi gửi lại cho LLM.
    Gửi nguyên JSON làm Gemini trả MALFORMED_RESPONSE ở lượt kế tiếp (đã kiểm chứng trong quá trình làm lab).
    """
    if isinstance(obs, dict) and "error" in obs:
        return f"Lỗi: {obs['error']}"
    if isinstance(obs, list):
        if not obs:
            return "Không tìm thấy chuyến bay phù hợp."
        lines = []
        for f in obs:
            price = f"{int(f.get('price_vnd', 0)):,}".replace(",", ".")
            lines.append(
                f"{f.get('flight_number')} ({f.get('airline')}): {f.get('origin')} -> {f.get('destination')}, "
                f"khởi hành {f.get('departure_time')}, giá {price} VND"
            )
        return "Các chuyến bay tìm được:\n" + "\n".join(lines)
    if isinstance(obs, dict) and "temperature_c" in obs:
        return (
            f"Thời tiết {obs.get('city')}: {obs.get('temperature_c')}°C, {obs.get('condition')}, "
            f"độ ẩm {obs.get('humidity_pct')}%. Gợi ý trang phục: {obs.get('recommendation')}"
        )
    return str(obs)


def _parse_action(raw: str):
    """Trả về (name, args, error). error != None nếu JSON không hợp lệ."""
    try:
        data, _ = json.JSONDecoder().raw_decode(raw)
    except json.JSONDecodeError as e:
        return None, None, f"Invalid JSON format: {e.msg}"
    if not isinstance(data, dict) or "name" not in data:
        return None, None, "Invalid JSON format: thiếu trường 'name'"
    args = data.get("args") or {}
    if not isinstance(args, dict):
        return None, None, "Invalid JSON format: 'args' phải là object"
    return str(data["name"]), args, None


def _extract_thought(text: str) -> str:
    m = re.search(r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)", text, re.S)
    return m.group(1).strip() if m else ""


def _extract_action(text: str):
    """Trả về phần sau 'Action:' (bắt đầu từ '{' nếu có), hoặc None nếu không có Action."""
    m = re.search(r"Action:\s*", text)
    if not m:
        return None
    raw = text[m.end():]
    brace = raw.find("{")
    return raw[brace:] if brace != -1 else raw.strip()


def _extract_final(text: str):
    m = re.search(r"Final Answer:\s*(.*)", text, re.S)
    return m.group(1).strip() if m else None


def _result(status, answer, iterations, trace, tool_calls):
    return {
        "status": status,
        "answer": answer,
        "iterations": iterations,
        "trace": trace,
        "tool_calls": tool_calls,
    }


class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""

    def __init__(self, client=None):
        self._client = client

    def query(self, user_input: str) -> dict:
        # TODO: Trả về câu trả lời tĩnh hoặc gọi LLM 1 lượt (không dùng tool)
        if self._client is None:
            self._client = _get_client()
        text = _generate(self._client, BASELINE_PROMPT, user_input)
        return {"status": "success", "answer": text, "tool_calls": []}


class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""

    def __init__(self, max_iterations: int = 5, client=None):
        self.max_iterations = max_iterations
        self.trace = []
        self._client = client

    def run(self, user_input: str) -> dict:
        if self._client is None:
            self._client = _get_client()

        # TODO 1: Khởi tạo mảng lưu lịch sử conversation / traces
        self.trace = []
        # Toàn bộ lịch sử (câu hỏi, Thought/Action, Observation) gửi như MỘT message user.
        history = [f"Câu hỏi của khách: {user_input}"]
        system_prompt = SYSTEM_PROMPT.format(
            tools=json.dumps(TOOL_DEFINITIONS, ensure_ascii=False, indent=2)
        )
        tool_calls = []
        error_streak = 0

        # TODO 2: Thiết lập vòng lặp while iteration < self.max_iterations
        iteration = 0
        while iteration < self.max_iterations:
            # Mỗi lượt là một lần gọi LLM; cắt tại "Observation:" để LLM không tự bịa kết quả tool.
            try:
                user_text = "\n".join(history) + "\n\n" + FORMAT_REMINDER
                text = _generate(self._client, system_prompt, user_text, stop=["Observation:"])
            except RuntimeError as e:
                # LLM vẫn trả response rỗng sau khi thử lại: kết thúc có kiểm soát thay vì crash.
                return _result("llm_error", f"Hệ thống đang gặp sự cố: {e}", iteration, self.trace, tool_calls)
            history.append(text)
            iteration += 1

            # TODO 3: Phân tích Thought / Action từ Agent
            thought = _extract_thought(text)
            action_raw = _extract_action(text)
            final_answer = _extract_final(text)

            if action_raw is None and final_answer is not None:
                self.trace.append({"step": iteration, "thought": thought, "final_answer": final_answer})
                return _result("completed", final_answer, iteration, self.trace, tool_calls)

            # TODO 4: Thực thi Tool trong TOOL_MAP nếu có Action
            if action_raw is None:
                obs = {"error": "Không tìm thấy Action hoặc Final Answer. Hãy trả lời đúng định dạng."}
            else:
                name, args, parse_error = _parse_action(action_raw)
                if parse_error:
                    obs = {"error": parse_error}
                else:
                    tool_calls.append(name)
                    obs = _call_tool(name, args)

            is_error = isinstance(obs, dict) and "error" in obs
            obs_text = _format_observation(obs)

            # TODO 5: Ghi lại Observation và lặp lại cho tới khi ra Final Answer
            self.trace.append({
                "step": iteration,
                "thought": thought,
                "action": action_raw,
                "observation": obs_text,
            })
            error_streak = error_streak + 1 if is_error else 0

            feedback = f"Observation: {obs_text}"
            if error_streak >= 2:
                feedback += "\nLỗi đã xảy ra 2 lần liên tiếp. Hãy đưa Final Answer báo lỗi cho khách hàng."
            history.append(feedback)

        # Milestone 4: safeguard chống lặp vô tận
        return _result(
            "max_iterations_reached",
            "Không thể hoàn thành trong số bước tối đa.",
            iteration,
            self.trace,
            tool_calls,
        )


def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(json.dumps(chatbot.query(user_query), indent=2, ensure_ascii=False))

    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
