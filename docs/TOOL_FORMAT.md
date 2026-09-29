# Tool-Use Token Formatı — `anthropic-tools-v1`

Qwen3.5-4B'yi Go + tool-use için eğitirken kullanılan token formatı.

> **Tek kaynak:** [`src/gotooltrain/templates/anthropic-tools-v1.jinja`](../src/gotooltrain/templates/anthropic-tools-v1.jinja)
> Python kodu bu şablonu **yüklemez, kopyalamaz, yeniden yazmaz.** Eğitim ve çıkarım aynı
> şablonu kullanır; bu yüzden format iki yerde birden tanımlı olmaz.

## Neden bu yol

| Alternatif | Neden seçilmedi |
|---|---|
| Python'da string birleştirerek render | Eğitim ve çıkarım ayrışır; sapma sessiz kalır ve hata değil kalite kaybı olarak görünür |
| Karakter ofsetiyle loss maskesi | Tokenizer'ın gerçekte kullandığı kod yolunu test etmeyiz; `{% generation %}` zaten bunu yapıyor |
| Qwen'in stok şablonu | `{% generation %}` içermiyor → `return_assistant_tokens_mask` boş maske verir; ayrıca Anthropic tool blokları değil |

## Format

Anthropic'in `tag + JSON gövde` konvansiyonu, Qwen'in native turn çerçevesi üzerinde:

```
<|im_start|>system
You are a Go engineer.

<available_tools>
{tool json, her satırda bir tane}
</available_tools><|im_end|>
<|im_start|>user
Add a test for the parser.<|im_end|>
<|im_start|>assistant
Let me read the package first.
<tool_use>{"id": "call_1", "name": "read_file", "input": {"path": "parser/parser.go"}}</tool_use>
<|im_end|>
<|im_start|>user
<tool_result>{"tool_use_id": "call_1", "content": "package parser", "is_error": false}</tool_result>
<|im_end|>
```

Kesin baytlar: [`tests/golden/anthropic-tools-v1.txt`](../tests/golden/anthropic-tools-v1.txt)

### Boşluk tuzağı

transformers şablonları `trim_blocks=True` ile derler; bu, her block tag'inden sonraki
newline'ı siler. `{%- ... -%}` ile karıştırıldığında `<|im_start|>system` gövdesine
yapışıyor ve tool kataloğu tek satıra çöküyordu. Bu yüzden **her newline açıkça string
ifadesi olarak üretilir** (`{{- '<|im_start|>system\n' -}}`). Şablondaki çıplak
newline'ları "basitleştirmeyin" — golden test baytları kilitler.

## Loss maskesi

Loss **yalnızca asistan içeriğine** uygulanır. Bu, şablondaki `{% generation %}`
bloğundan gelir ve tokenizer maskeyi kendisi üretir
(`apply_chat_template(..., return_assistant_tokens_mask=True)`).

| Segment | Trainable | Maskelenmezse ne olur |
|---|---|---|
| `<available_tools>` kataloğu | ❌ | model tool'ları uydurur |
| `user` metni | ❌ | model kullanıcı cümlesini tekrarlar |
| `<tool_result>` bloğu | ❌ | **model tool çıktısı uydurur** — bu formatın var oluş sebebi |
| asistan metni / `<tool_use>` | ✅ | — |
| `<think>…</think>` | ✅ | `reasoning_content` üzerinden ayrıştırılır |

Maske `zip(..., strict=True)` ile denetlenir; `assert_mask_sane` alan uzunluğu farkını ve
boş maskeyi yakalar.

## Doğrulama kuralları (`gotooltrain.normalize`)

Validasyon render'dan **önce** çalışır: milyonlarca kayıtlık bir veri setinde hatalı tek
kayıt, eğitimin ortasında ortaya çıkarsa saatler kaybolur.

| Kural | Koruduğu hata |
|---|---|
| Araç kataloğu isme göre sıralı | Aynı kayıt iki kez farklı prompt üretir → tekrarlanabilirlik bozulur |
| Sistem mesajı yalnızca ilk sırada | Çift render / belirsiz prompt |
| Boşluktan ibaret asistan turn'ü reddedilir | Supervise edilmeyen boş turn |
| Tool call id var, benzersiz | Sonuç eşleşemez |
| Her sonuç önceki bir çağrıyı yanıtlar | Model "uydurma" tool sonucu üretmeyi öğrenir |
| Her çağrının sonucu var | Model tool çıktısını okumayı öğrenmez (yarım döngü) |
| Paralel sonuçlar tek turn'de gruplanır | Şablon tek `user` turn'ü bekler |
| Katalogdaki dışı araç reddedilir | Model görmediği bir tool çağırır |
| Güvenilmeyen metinde tag kaçışı | Dosya içeriği `</tool_result>` ile bloğu kapatıp sahte sınır üretir |

Kaçış varsayılan olarak **reddetme**, opt-in olarak `sanitize_reserved_tags=True`.
Asistan çıktısı (tool argümanları) **kaçırılmaz** — hedefi bozmak modele bozuk kod
yazmayı öğretirdi.

## Truncation tuzağı

`max_length` sağdan keser; uzun bir kayıtta **tüm asistan turn'leri** kesilip geriye
supervise edilecek token kalmayabilir. Bu örnekler varsayılan olarak **hata verir**
(`on_empty="error"`), istenirse atlanır (`on_empty="skip"`).

## Kullanım

```python
from transformers import AutoTokenizer
from gotooltrain import (
    install_template,
    normalize_conversation,
    render_example,
    save_template_artifacts,
)

tokenizer = install_template(AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B"))

conv = normalize_conversation(messages, tools)  # OpenAI biçimi kabul edilir
example = render_example(tokenizer, conv)  # input_ids + labels + assistant_mask
save_template_artifacts(tokenizer, "out/")  # chat_template.jinja doğrulanır
```

Girdi hem **OpenAI** (`role:"tool"` + `tool_call_id`) hem **kanonik** (`results` listesi)
biçimini kabul eder; xLAM / ToolACE / BFCL dışa aktarımları dönüştürülmeden kullanılabilir.

## Komutlar

```bash
pip install -e ".[dev]"

pytest                       # 73 test, coverage eşiği %90
ruff check . && ruff format --check .
mypy                         # strict
python -m gotooltrain.regenerate_golden   # formatı bilinçli değiştirdiysen
```

`regenerate_golden` sonrası diff, token formatındaki değişikliğin **inceleme artefaktıdır**:
format değişikliği eğitim verisi değişikliğidir.

## Sunum tarafı

`save_template_artifacts`, transformers'ın ayrı bir `chat_template.jinja` dosyası
yazdığını doğrular ve içeriğin kaynakla birebir aynı olduğunu kontrol eder. Bu dosya
kaybolursa sunucu sessizce taban modelin formatına düşer — ki bu bir yetenek kaybı
gibi görünür, paketleme hatası gibi değil.
