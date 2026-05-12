"""Phase C: PII redaction using Presidio + Vietnamese custom patterns."""

import re


def _try_presidio():
    try:
        import presidio_analyzer  # noqa
        import presidio_anonymizer  # noqa
        return True
    except ImportError:
        return False


def _build_engine():
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
    from presidio_anonymizer import AnonymizerEngine

    analyzer = AnalyzerEngine()

    # Vietnamese CCCD: exactly 12 digits
    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[Pattern(name="VN_CCCD", regex=r"\b\d{12}\b", score=0.9)],
        context=["cccd", "căn cước", "chứng minh", "định danh"],
    )

    # Vietnamese phone: 0[3-9]XXXXXXXX or +84/84[3-9]XXXXXXXX
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern(name="VN_PHONE", regex=r"\b(0[3-9]\d{8}|\+?84[3-9]\d{8})\b", score=0.85)],
        context=["điện thoại", "số điện thoại", "liên hệ", "phone", "sdt"],
    )

    # Vietnamese tax code: 10 digits (MST)
    tax_recognizer = PatternRecognizer(
        supported_entity="VN_TAX_CODE",
        patterns=[Pattern(name="VN_TAX_CODE", regex=r"\b\d{10}\b", score=0.7)],
        context=["mã số thuế", "mst", "tax code", "mã số"],
    )

    for r in [cccd_recognizer, phone_recognizer, tax_recognizer]:
        analyzer.registry.add_recognizer(r)

    return analyzer, AnonymizerEngine()


_ANALYZER, _ANONYMIZER = None, None
_PRESIDIO_AVAILABLE = _try_presidio()


def _get_engines():
    global _ANALYZER, _ANONYMIZER
    if _ANALYZER is None and _PRESIDIO_AVAILABLE:
        _ANALYZER, _ANONYMIZER = _build_engine()
    return _ANALYZER, _ANONYMIZER


def _regex_fallback(text: str) -> tuple[str, list[str]]:
    """Fallback redaction using pure regex when Presidio not available."""
    entities = []
    # CCCD: 12 digits
    if re.search(r'\b\d{12}\b', text):
        text = re.sub(r'\b\d{12}\b', '<VN_CCCD>', text)
        entities.append("VN_CCCD")
    # Phone
    if re.search(r'\b(0[3-9]\d{8}|\+?84[3-9]\d{8})\b', text):
        text = re.sub(r'\b(0[3-9]\d{8}|\+?84[3-9]\d{8})\b', '<VN_PHONE>', text)
        entities.append("VN_PHONE")
    # Email
    if re.search(r'\b[\w.+-]+@[\w-]+\.\w{2,}\b', text):
        text = re.sub(r'\b[\w.+-]+@[\w-]+\.\w{2,}\b', '<EMAIL_ADDRESS>', text)
        entities.append("EMAIL_ADDRESS")
    return text, entities


def redact_pii(text: str, language: str = "en") -> tuple[str, list[str]]:
    """Redact PII from text. Returns (redacted_text, list_of_entity_types)."""
    if not _PRESIDIO_AVAILABLE:
        return _regex_fallback(text)

    analyzer, anonymizer = _get_engines()
    if analyzer is None:
        return _regex_fallback(text)

    results = analyzer.analyze(text=text, language=language)
    vn_results = analyzer.analyze(text=text, language="en",
                                   entities=["VN_CCCD", "VN_PHONE", "VN_TAX_CODE"])
    all_results = results + [r for r in vn_results if r.entity_type.startswith("VN_")]
    seen, unique = set(), []
    for r in sorted(all_results, key=lambda x: x.start):
        key = (r.start, r.end)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    if not unique:
        return text, []
    redacted = anonymizer.anonymize(text=text, analyzer_results=unique)
    return redacted.text, list({r.entity_type for r in unique})


def test_redaction():
    samples = [
        ("Tên: Nguyễn Văn A, CCCD: 034567890123, SĐT: 0912345678", ["VN_CCCD", "VN_PHONE"]),
        ("Email: john@example.com, MST: 0106769437", ["EMAIL_ADDRESS"]),
        ("Liên hệ qua số +84987654321 hoặc john.doe@company.vn", ["VN_PHONE", "EMAIL_ADDRESS"]),
        ("Họ tên: Trần Thị B - Không có PII", []),
    ]
    passed = 0
    for text, expected_entities in samples:
        redacted, entities = redact_pii(text)
        print(f"IN:  {text}")
        print(f"OUT: {redacted}")
        print(f"ENT: {entities}\n")
        if all(e in entities for e in expected_entities):
            passed += 1
    print(f"Passed: {passed}/{len(samples)}")


if __name__ == "__main__":
    test_redaction()
