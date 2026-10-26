"""Checks DecimaPIIRecognizer inside Presidio's AnalyzerEngine and AnonymizerEngine.

    uv run -p 3.12 --no-project --with decima-ai --with presidio-analyzer --with presidio-anonymizer --with "preshed<3.0.11" \
        --with "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" \
        --with pytest pytest -q test_presidio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry  # noqa: E402
from presidio_analyzer.nlp_engine import NlpEngineProvider  # noqa: E402
from presidio_anonymizer import AnonymizerEngine  # noqa: E402

from decima_presidio import DecimaPIIRecognizer  # noqa: E402

AR = "اسمي محمود الشمري ورقم جوالي ٠٥٥١٢٣٤٥٦٧ وبريدي m.alshammari@example.com، الآيبان SA03 8000 0000 6080 1016 7519"
FA = "سلام، من سارا احمدی هستم. کد ملی من ۰۰۱۲۳۴۵۶۷۸ و شماره موبایلم ۰۹۱۲۳۴۵۶۷۸۹ است."


@pytest.fixture(scope="module")
def nlp():
    return NlpEngineProvider(nlp_configuration={"nlp_engine_name": "spacy",
                                                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]}).create_engine()


@pytest.fixture(scope="module")
def decima_only(nlp):
    registry = RecognizerRegistry(recognizers=[DecimaPIIRecognizer(threads=2)])
    return AnalyzerEngine(nlp_engine=nlp, registry=registry)


def test_entities_in_presidio_names(decima_only):
    got = {(r.entity_type, AR[r.start:r.end]) for r in decima_only.analyze(AR, language="en")}
    assert ("PHONE_NUMBER", "٠٥٥١٢٣٤٥٦٧") in got
    assert ("EMAIL_ADDRESS", "m.alshammari@example.com") in got
    assert ("IBAN_CODE", "SA03 8000 0000 6080 1016 7519") in got
    assert any(e == "PERSON" and "الشمري" in v for e, v in got)
    fa = {(r.entity_type, FA[r.start:r.end]) for r in decima_only.analyze(FA, language="en")}
    assert ("NATIONAL_ID", "۰۰۱۲۳۴۵۶۷۸") in fa and ("PHONE_NUMBER", "۰۹۱۲۳۴۵۶۷۸۹") in fa


def test_entity_filter_and_anonymizer(decima_only):
    res = decima_only.analyze(FA, language="en", entities=["PHONE_NUMBER", "NATIONAL_ID"])
    assert {r.entity_type for r in res} == {"PHONE_NUMBER", "NATIONAL_ID"}
    out = AnonymizerEngine().anonymize(FA, res).text
    assert "<NATIONAL_ID>" in out and "<PHONE_NUMBER>" in out and "۰۹۱۲" not in out and "سارا" in out


def test_alongside_presidio_pattern_recognizers(nlp):
    analyzer = AnalyzerEngine(nlp_engine=nlp)
    analyzer.registry.remove_recognizer("SpacyRecognizer")   # English spaCy NER tags junk spans on Arabic-script text
    analyzer.registry.add_recognizer(DecimaPIIRecognizer(threads=2))
    res = analyzer.analyze(FA, language="en")
    assert any(r.recognition_metadata.get("recognizer_name") == "DecimaPIIRecognizer" and r.entity_type == "NATIONAL_ID" for r in res)
    out = AnonymizerEngine().anonymize(FA, res).text
    assert out == "سلام، من <PERSON> هستم. کد ملی من <NATIONAL_ID> و شماره موبایلم <PHONE_NUMBER> است."
