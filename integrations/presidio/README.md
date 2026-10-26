# decima-pii as a Presidio recognizer

If you already run [Presidio](https://github.com/microsoft/presidio) (directly, or through LiteLLM, LangChain or
another tool built on it), you can add [decima-pii](https://huggingface.co/amyrmahdy/decima-pii) as one more
recognizer. It finds personal data in Arabic, Persian, Urdu, Turkish and English, including numbers in Persian and
Arabic-Indic digits, and reports it under Presidio's entity names, so your anonymizer settings keep working.

```bash
pip install decima-ai presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_sm      # Presidio needs a spaCy model to start; decima-pii does not use it
```

```python
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from decima_presidio import DecimaPIIRecognizer           # this folder's decima_presidio.py

nlp = NlpEngineProvider(nlp_configuration={"nlp_engine_name": "spacy",
                                           "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]}).create_engine()
analyzer = AnalyzerEngine(nlp_engine=nlp)
analyzer.registry.remove_recognizer("SpacyRecognizer")   # see below
analyzer.registry.add_recognizer(DecimaPIIRecognizer())

text = "سلام، من سارا احمدی هستم. کد ملی من ۰۰۱۲۳۴۵۶۷۸ و شماره موبایلم ۰۹۱۲۳۴۵۶۷۸۹ است."
results = analyzer.analyze(text, language="en")
print(AnonymizerEngine().anonymize(text, results).text)
# سلام، من <PERSON> هستم. کد ملی من <NATIONAL_ID> و شماره موبایلم <PHONE_NUMBER> است.
```

The same text with Presidio's defaults only:

```
defaults:                سلام، من سارا احمدی هستم. <ORGANIZATION> و شماره موبایلم <PERSON>.
defaults without spaCy:  سلام، من سارا احمدی هستم. کد ملی من ۰۰۱۲۳۴۵۶۷۸ و شماره موبایلم <PHONE_NUMBER> است.
```

**Why remove `SpacyRecognizer`:** the English spaCy model tags whole phrases of Arabic-script text as `ORGANIZATION`
or `PERSON` ("کد ملی من ۰۰۱۲۳۴۵۶۷۸" became an organisation above). Those spans are longer than decima-pii's and win
when the anonymizer merges overlapping results. Presidio's pattern recognizers (email, credit card, IBAN, IP, ...)
can stay. For English-only traffic you can keep spaCy too.

`language="en"` is only Presidio's routing key here: decima-pii reads all five languages whatever the key. If you
route text under other keys (`ar`, `fa`), register one recognizer per key: `DecimaPIIRecognizer(supported_language="fa")`.

## Entity names

| decima-pii | Presidio |
|---|---|
| PERSON_NAME | PERSON |
| PHONE | PHONE_NUMBER |
| EMAIL | EMAIL_ADDRESS |
| NATIONAL_ID | NATIONAL_ID (any country; Presidio's own are per country, e.g. US_SSN) |
| IBAN | IBAN_CODE |
| CARD | CREDIT_CARD |
| PASSPORT | PASSPORT (any country) |
| ADDRESS | LOCATION |
| DATE_OF_BIRTH | DATE_TIME |
| IP_ADDRESS | IP_ADDRESS |

Change it with `DecimaPIIRecognizer(entity_map={...})`. Other options: `model` (Hub repo or local directory),
`threads`, `score` (decima-pii returns spans, not probabilities, so every span gets this score; default 0.85).

## Test

[`test_presidio.py`](test_presidio.py) runs the analyzer and anonymizer on Arabic and Persian text. Tested with
presidio-analyzer and presidio-anonymizer 2.2.365, Python 3.12 (spaCy 3.8 fails to tokenize on Python 3.14, which is
why the command pins 3.12):

```bash
uv run -p 3.12 --no-project --with decima-ai --with presidio-analyzer --with presidio-anonymizer --with "preshed<3.0.11" \
    --with "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" \
    --with pytest pytest -q test_presidio.py
```

decima-pii is a preview model; its measured numbers and limits are on the
[model card](https://huggingface.co/amyrmahdy/decima-pii).
