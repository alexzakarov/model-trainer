# TRL / eğitim entegrasyonu

`gotooltrain` veriyi **token id + label** olarak dışa aktarır; eğitim döngüsünün geri
kalanı TRL'e aittir. Bu dosya ikisini birbirine bağlar.

## 1. Veri önce doğrulanır (tokenizasyondan ayrı)

Doğrulama tokenizasyondan çok ucuzdur ve hataları GPU saatleri harcamadan yakalar.

```python
from gotooltrain import read_jsonl, validate_records

report = validate_records(read_jsonl("raw/train.jsonl"))
print(report.summary())
for skipped in report.skipped[:20]:
    print(skipped.index, skipped.reason)

# Sıfır düşen kayıt bekleniyorsa strict modu kullan: sessiz veri kaybı olmasın.
report = validate_records(read_jsonl("raw/train.jsonl"), strict=True)
```

`strict=True` ilk hatada durur. Büyük ve heterojen veri setlerinde önce `strict=False`
ile hata dağılımını görün, kök nedeni düzeltin, sonra `strict=True` ile kilitleyin.

## 2. Tokenize edilmiş veri dışa aktarılır

```python
from transformers import AutoTokenizer
from gotooltrain import (
    export_jsonl,
    install_template,
    load_template_source,
    normalize_conversation,
    read_jsonl,
)

tokenizer = install_template(
    AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B"), load_template_source()
)
conversations = [
    normalize_conversation(r["messages"], r.get("tools")) for r in read_jsonl("raw/train.jsonl")
]

export_jsonl("data/sft.jsonl", conversations, tokenizer, max_length=8192, on_empty="skip")
```

`export_jsonl` **token id** yazar, metin değil. Böylece eğitim doğrulanmış token'ları
tüketir ve yarıda kalan bir koşu, değişmiş bir şablonla sessizce yeniden render olmaz.

## 3. Eğitim (TRL)

`gotooltrain` `input_ids` / `attention_mask` / `labels` üretir; `trl` bunları olduğu gibi
kabul eder. Loss zaten `labels == -100` olan yerlerde sayılır.

```bash
trl sft \
  --model_name_or_path Qwen/Qwen3.5-4B \
  --dataset_name text \
  --dataset_kwargs '{"data_files":{"train":"data/sft.jsonl"}}' \
  --max_seq_length 8192 \
  --packing false \
  --output_dir runs/go-sft-1
```

Dikkat edilecekler:

- **`--packing false`** önerilir. Packing, `labels` dizisini örnekler arasında
  kaydırdığı için bizim boundary-based maskemizle birleşmez.
- **Çoklu eğitim (multi-turn) örneklerde `--assistant_only_loss`** kullanmayın;
  maske zaten `labels` içinde. Kullanırsanız ikinci kez maske uygulanır.
- **Learning rate**: tam fine-tune için ~1e-5 – 2e-5. LoRA'nın 2e-4 değeri tam
  fine-tune için gereğinden büyüktür ve pretrained yeteneği bozar.

## 4. Şablonu modelle birlikte dağıt

```python
from gotooltrain import save_template_artifacts

save_template_artifacts(tokenizer, "runs/go-sft-1")
```

Bu, `chat_template.jinja` dosyasının yazıldığını ve içeriğin kaynakla birebir aynı
olduğunu doğrular. Serving tarafında (vLLM / llama.cpp) aynı şablon kullanılmalıdır;
aksi halde eğitimde öğrenilen tool formatı ile sunumdaki format ayrışır.

## 5. Değerlendirme (Aşama 0)

Eğitime başlamadan sabitlenir:

- **Tool use:** BFCL v4 (single-turn, multi-turn, parallel, hallucination, relevance)
- **Agentic:** τ²-bench
- **Go:** Go-UT-Bench held-out + **execution-based** skorlama (üretilen test gerçekten
  `go test`'ten geçiyor mu)
- **Unutma takibi:** genel kod benchmark'ları (HumanEval+, LiveCodeBench)

Qwen3.5-4B zaten native tool calling ile geliyor; kazanç sıfırdan değil, üstüne çıkmak.

## Bilinen tuzaklar

| Tuzak | Sonuç | Önlem |
|---|---|---|
| Truncation tüm asistan turn'ünü keser | Supervise edilmeyen örnek | `on_empty="error"` varsayılan |
| `chat_template.jinja` dağıtılmıyor | Sunumda sessiz format kayması | `save_template_artifacts` doğrular |
| Loss, tool sonuçlarına sızıyor | Model tool çıktısı uydurur | `{% generation %}` + golden test |
| Uzun CoT, tool-call token'larını bastırıyor | Tool use zayıflar | BalanceSFT (ACL 2026 Findings) |
| Dar/düşük çeşitlilikte Go korpusu | Genel yetenek kaybı | Genel kod + akıl yürütme verisi karıştır, replay yap |
