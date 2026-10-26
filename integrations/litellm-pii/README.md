# decima-pii in front of LiteLLM

LiteLLM's built-in PII masking runs on Presidio, whose default recognizers are English-first: on Persian and Arabic
text they missed close to half of the values written in Persian or Arabic-Indic digits in our tests, and the English
spaCy model tags whole phrases as names and organisations ([example](../presidio)). This folder adds [decima-pii](https://huggingface.co/amyrmahdy/decima-pii)
as a LiteLLM custom guardrail instead. It masks names, phones, emails, national IDs, IBANs, cards, passports,
addresses, dates of birth and IP addresses in Arabic, Persian, Urdu, Turkish and English before the request leaves
the proxy. It runs on CPU inside the proxy process, with no extra service and no API key.

Tested with LiteLLM 1.106.2 and decima-ai 2.4.0.

## Run it

```bash
pip install decima-ai "litellm[proxy]"
python echo_server.py --port 9000          # a stand-in model that replies with what it received
LITELLM_MASTER_KEY=<your key> litellm --config config.yaml
```

`decima_pii_guardrail.py` must sit next to [`config.yaml`](config.yaml). The model (148 MB, int8) downloads from the
Hub when the proxy starts.

```bash
curl http://localhost:4000/v1/chat/completions -H "Authorization: Bearer <your key>" -H "Content-Type: application/json" \
  -d '{"model": "my-chat-model", "messages": [
        {"role": "user", "content": "سلام، من سارا احمدی هستم. کد ملی من ۰۰۱۲۳۴۵۶۷۸ و شماره موبایلم ۰۹۱۲۳۴۵۶۷۸۹ است."},
        {"role": "user", "content": "اسمي محمود الشمري ورقم جوالي ٠٥٥١٢٣٤٥٦٧ وبريدي m.alshammari@example.com، رقم الآيبان SA03 8000 0000 6080 1016 7519"}]}'
```

What the model received (from the echo server):

```
user: سلام، من [PERSON_NAME] هستم. کد ملی من [NATIONAL_ID] و شماره موبایلم [PHONE] است.
user: اسمي [PERSON_NAME] ورقم جوالي [PHONE] وبريدي [EMAIL]، رقم الآيبان [IBAN]
```

Streaming requests are masked the same way. Replace the `my-chat-model` entry with your real model.

## Options

Keys under the guardrail's `litellm_params` go to `DecimaPIIGuardrail`:

| key | default | |
|---|---|---|
| `mode` | | `pre_call` masks the request; `post_call` masks the model's reply (use two entries for both) |
| `default_on` | | `true`: every request; `false`: only requests that pass `"guardrails": ["decima-pii"]` |
| `entities` | all ten | e.g. `[PHONE, EMAIL, NATIONAL_ID, IBAN, CARD, PASSPORT]` to leave names and addresses alone |
| `mask_format` | `[{label}]` | `{label}` and `{text}` are filled in |
| `pii_action` | `mask` | `block` refuses the request with HTTP 400 and the labels found, e.g. `{"IBAN": 1}` |
| `model` | `amyrmahdy/decima-pii` | Hub repo or a local directory |
| `threads` | 4 | ONNX Runtime threads per call |

The guardrail uses LiteLLM's unified `apply_guardrail` hook, so the same class masks requests on
`/v1/chat/completions`, `/v1/responses` and `/v1/messages` (we checked what the backend received on all three;
`echo_server.py` itself only answers chat completions).

## Cost and limits

- About 13 ms per message on 6 CPU threads (model card). In our proxy test, three short messages added about 40 ms on
  2 threads of a busy machine.
- It masks, it does not restore: the model sees `[PHONE]`, and so does the reply. If you need the values back in the
  reply, keep a mapping yourself (`mask_format: "[{label}]"` plus `DecimaPII.find` in your app).
- It is a preview model: it misses some values and flags some that are not personal data. One example we hit: in
  "ممكن تساعدني أكتب رسالة شكر لمديري؟" ("help me write a thank-you note to my manager") it masks "مديري" (my manager)
  as a name. Read the [model card](https://huggingface.co/amyrmahdy/decima-pii) for measured numbers and limits.
