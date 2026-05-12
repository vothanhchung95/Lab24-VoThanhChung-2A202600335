"""Phase C: Llama Guard 3 output safety check via Groq API."""

import os
import sys
from dataclasses import dataclass
from typing import Optional

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLAMA_GUARD_MODEL = "llama-guard-3-8b"

UNSAFE_KEYWORDS = [
    "vũ khí", "chất nổ", "ma túy", "khủng bố", "hack", "phá khóa", "khai thác lỗ hổng",
    "weapon", "explosive", "drug", "terrorism", "malware", "synthesize dangerous",
    "how to make a bomb", "tổng hợp chất",
]


@dataclass
class GuardResult:
    safe: bool
    category: Optional[str]
    method: str
    raw_response: str = ""


def _keyword_fallback(text: str) -> GuardResult:
    t = text.lower()
    for kw in UNSAFE_KEYWORDS:
        if kw in t:
            return GuardResult(safe=False, category="harmful_content", method="keyword_fallback")
    return GuardResult(safe=True, category=None, method="keyword_fallback")


def check_output(response_text: str) -> GuardResult:
    """Check if RAG response is safe. Falls back to keyword heuristic if no GROQ_API_KEY."""
    if not GROQ_API_KEY:
        return _keyword_fallback(response_text)

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": response_text}
                ]
            }
        ]

        completion = client.chat.completions.create(
            model=LLAMA_GUARD_MODEL,
            messages=messages,
            max_tokens=100,
            temperature=0.0,
        )
        raw = completion.choices[0].message.content.strip().lower()

        if raw.startswith("safe"):
            return GuardResult(safe=True, category=None, method="llama_guard", raw_response=raw)
        elif raw.startswith("unsafe"):
            lines = raw.split("\n")
            category = lines[1].strip() if len(lines) > 1 else "unknown"
            return GuardResult(safe=False, category=category, method="llama_guard", raw_response=raw)
        else:
            return _keyword_fallback(response_text)

    except Exception as e:
        print(f"[LlamaGuard] API error ({type(e).__name__}: {e}), using keyword fallback")
        return _keyword_fallback(response_text)


def _safe_print(text: str) -> None:
    """Print text, replacing unencodable characters to avoid UnicodeEncodeError."""
    encoded = text.encode(sys.stdout.encoding or "utf-8", errors="replace")
    sys.stdout.buffer.write(encoded + b"\n")
    sys.stdout.buffer.flush()


def test_guard():
    samples = [
        ("Nhân viên được nghỉ phép 12 ngày mỗi năm theo chính sách công ty.", True),
        ("Mật khẩu phải thay đổi mỗi 90 ngày theo chính sách IT.", True),
        ("Để hack vào hệ thống bạn cần dùng công cụ khai thác lỗ hổng bảo mật.", False),
        ("Hướng dẫn tổng hợp chất nổ từ các thành phần thông dụng.", False),
    ]
    passed = 0
    for text, expected_safe in samples:
        result = check_output(text)
        ok = result.safe == expected_safe
        if ok:
            passed += 1
        status = '[PASS]' if ok else '[FAIL]'
        _safe_print(f"{status} safe={result.safe} ({result.method}) | {text[:60]}")
        if not result.safe:
            _safe_print(f"   category: {result.category}")
    _safe_print(f"\nPassed: {passed}/{len(samples)}")


if __name__ == "__main__":
    test_guard()
