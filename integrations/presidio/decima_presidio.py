"""decima-pii as a Presidio recognizer: Arabic, Persian, Urdu, Turkish and English personal data in Presidio's
entity names, next to (or instead of) Presidio's own recognizers.

    from presidio_analyzer import AnalyzerEngine
    from decima_presidio import DecimaPIIRecognizer

    analyzer = AnalyzerEngine()
    analyzer.registry.add_recognizer(DecimaPIIRecognizer())
    analyzer.analyze("اسمي محمود الشمري ورقم جوالي ٠٥٥١٢٣٤٥٦٧", language="en")

The recognizer does not use Presidio's NLP artifacts (spaCy); it runs decima-pii on the raw text, so it works on any
of its five languages whatever `language` the request is routed under (see README.md).
"""

from __future__ import annotations

from presidio_analyzer import AnalysisExplanation, EntityRecognizer, RecognizerResult

from decima.pii import DEFAULT_ONNX, DEFAULT_REPO, DecimaPII

# decima-pii label -> Presidio entity name (Presidio's built-in names where one fits)
ENTITY_MAP = {
    "PERSON_NAME": "PERSON",
    "PHONE": "PHONE_NUMBER",
    "EMAIL": "EMAIL_ADDRESS",
    "NATIONAL_ID": "NATIONAL_ID",       # Presidio has only country-specific IDs (US_SSN, IN_AADHAAR, ...); this one is any country
    "IBAN": "IBAN_CODE",
    "CARD": "CREDIT_CARD",
    "PASSPORT": "PASSPORT",             # likewise: any country (Presidio: US_PASSPORT, UK_PASSPORT, ...)
    "ADDRESS": "LOCATION",
    "DATE_OF_BIRTH": "DATE_TIME",
    "IP_ADDRESS": "IP_ADDRESS",
}


class DecimaPIIRecognizer(EntityRecognizer):
    """Presidio recognizer backed by decima-pii (ONNX, CPU).

    model / onnx / threads: as DecimaPII.from_pretrained (default amyrmahdy/decima-pii, int8)
    entity_map:  override the label -> Presidio entity mapping
    score:       the score given to every span (the tagger returns spans, not probabilities; default 0.85)
    supported_language: the Presidio language key this instance answers for; register one instance per key you route
                 text under (the model itself reads all five languages)
    """

    def __init__(self, model: str | None = None, onnx: str | None = None, threads: int = 4, entity_map: dict[str, str] | None = None,
                 score: float = 0.85, supported_language: str = "en", name: str = "DecimaPIIRecognizer", pii: DecimaPII | None = None):
        self.entity_map = dict(entity_map or ENTITY_MAP)
        self.score = score
        self._pii, self._kw = pii, dict(name_or_path=model or DEFAULT_REPO, onnx=onnx or DEFAULT_ONNX, threads=threads)
        super().__init__(supported_entities=sorted(set(self.entity_map.values())), name=name,
                         supported_language=supported_language, context=[])

    def load(self) -> None:
        if self._pii is None:
            self._pii = DecimaPII.from_pretrained(**self._kw)

    def analyze(self, text: str, entities: list[str], nlp_artifacts=None) -> list[RecognizerResult]:
        self.load()
        want = set(entities) if entities else None
        out = []
        for s in self._pii.find(text):
            ent = self.entity_map.get(s["label"])
            if ent is None or (want is not None and ent not in want):
                continue
            expl = AnalysisExplanation(recognizer=self.name, original_score=self.score, textual_explanation=f"decima-pii {s['label']}")
            out.append(RecognizerResult(entity_type=ent, start=s["start"], end=s["end"], score=self.score, analysis_explanation=expl,
                                        recognition_metadata={RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                                                              RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                                                              "decima_label": s["label"]}))
        return out
