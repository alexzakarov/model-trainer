# model-training

`Qwen3.5-4B` taban modelini **Golang uzmanı + Anthropic tool-blocks formatında tool use**
yeteneğine sahip bir modele tam fine-tune etmek için çalışma alanı.

## Durum

| Aşama | Durum |
|---|---|
| 1 — Token formatı, chat template, loss maskesi, validasyon | ✅ Tamamlandı |
| 1b — Görsel desteği (multimodal korunduğu için) | ✅ Tamamlandı |
| 2a — Go tool kataloğu (9 araç, RTK politikası, çıktı bütçeleri) | ✅ Tamamlandı |
| 2b — Execution harness çekirdeği (argv, status ayrımı, canary, paralel, pass@k) | ✅ Tamamlandı |
| 2c — Docker konteyner havuzu (izolasyon, gerçek konteynerle doğrulandı) | ✅ Tamamlandı |
| 3 — Eval orkestrasyonu (3 aşama, pass@N, store, resume, judge) | ✅ Tamamlandı |
| 4 — Veri üretimi: eval trajektörilerini madencleme + Go task üretimi | ✅ Tamamlandı |
| 4b — Go korpusu (Go-UT-Bench, lisans süzgeci, ölçüm) | ✅ Tamamlandı |
| 5 — Tam fine-tune (32K, thinking, vision dahil) — döngü gerçek modelle doğrulandı | ✅ Tamamlandı |
| 6 — Execution reward + DPO (tercih verisi + kayıp + döngü) | ✅ Tamamlandı |
| 7 — Gerçek 4B checkpoint ile GPU koşusu | ⏳ Donanım gerektiriyor — [Colab defteri](notebooks/gotooltrain_colab.ipynb) hazır |

Kalite kapıları: **1015 test (Windows, Docker'suz) / 1035 (Docker ve rtk ile)**,
coverage %100 (satır + dal), ruff temiz, mypy strict temiz (29 modül).

> Atlanan testler sessiz değil, gerekçeleri skip metninde yazılı: 20'si
> `test_sandbox_real.py`'de "no usable Docker daemon", 2'si `test_catalogue_real.py`'de
> "rtk grep Windows'ta busybox çağırıyor". Yani toplam, hangi isteğe bağlı
> bağımlılığın kurulu olduğuna bağlıdır — kapı tek bir sayı değil, **yeşil** demektir.

## Belgeler

- [`PIPELINE.md`](PIPELINE.md) — standart pipeline, faz sırası, alınan kararlar
- [`docs/TOOL_FORMAT.md`](docs/TOOL_FORMAT.md) — token formatı spec'i ve validasyon kuralları
- [`docs/EVAL.md`](docs/EVAL.md) — paralel eval altyapısı, idempotency, RTK politikası
- [`docs/INTEGRATION.md`](docs/INTEGRATION.md) — TRL entegrasyonu ve bilinen tuzaklar
- [`docs/COLAB.md`](docs/COLAB.md) — Colab'da 4B koşusu, periyodik Hub yüklemesi
- [`docs/BASELINE_RESEARCH.md`](docs/BASELINE_RESEARCH.md) — 4 GB VRAM'a sığan modeller araştırması

## Paket

`gotooltrain` — formatın ve eval altyapısının Python katmanı. Render mantığı **burada değil**;
token formatının tek kaynağı Jinja şablonudur.

```
src/gotooltrain/
  templates/anthropic-tools-v1.jinja   ← TEK KAYNAK (token formatı)
  schema.py                            kanonik mesaj modeli + ImageSpec
  normalize.py                         validasyon politikası
  template.py                          şablon yükleme + asistan maskesi
  dataset.py                           JSONL okuma/yazma, korpus doğrulama
  gotools.py                           9 Go aracı, RTK politikası, çıktı bütçeleri
  gorun.py                             argv güvenliği, status ayrımı, canary, paralel, pass@k
  sandbox.py                           konteyner havuzu, izolasyon, recycle
  evalstore.py                         dayanıklı, içerik-adresli eval deposu
  evalrun.py                           ajan döngüsü, judge, judge kuyruğu
  judge.py                             HttpJudge + interaktif session judge
  generator.py                         değerlendirilen model için OpenAI-uyumlu adaptör
  vision.py                            görsel token düzeni ve maskeleme
  collator.py                          32K batch collator'ı (vision dahil)
  corpus.py                            ölçülen ajan korpusu ve eksik veri hedefleri
  sftdata.py                           değerlendirmeden eğitim verisi madencleme
  gocorpus.py                          Go DAPT korpusu: içe aktarma, lisans, ölçüm
  tasks.py                             Go paketlerinden ölçülebilir görev üretimi
  train.py                             tam fine-tune planı ve döngüsü
  dpo.py                               execution reward üzerinde tercih döngüsü
  reward.py                            execution ödülü ve tercih çiftleri
  hub.py                               periyodik checkpoint yayını (HF Hub)
  build_notebook.py                    Colab defterinin üreticisi
  errors.py                            tipli hatalar
```

## Komutlar

```bash
# Değerlendirme: ajan döngüsünü koştur, judge kuyruğunu yaz
gotooltrain-eval queue --store runs/main --tasks tasks.json \
  --model alexzakkarov/qwen3.5-golang --revision rev-a --dataset-version holdout-1 \
  --model-url http://localhost:8000/v1 --out queue.jsonl \
  --sandbox docker --fixture repo/ --workers 8

# Judge verdict'larını al, skorlu raporu yaz
gotooltrain-eval judge --store runs/main --tasks tasks.json \
  --model alexzakkarov/qwen3.5-golang --revision rev-a --dataset-version holdout-1 \
  --run-id run-001 --queue queue.jsonl --verdicts verdicts.jsonl

# Eğitim verisi: yalnızca judge'ın başarılı bulduğu trajektöriler
gotooltrain-data mine --store runs/main --tasks tasks.json \
  --model alexzakkarov/qwen3.5-golang --revision rev-a --dataset-version holdout-1 \
  --seed 7 --out sft.jsonl --report mine-report.json

# Tercih verisi: aynı görevin geçen ve kalan örneklerinden çift
gotooltrain-data preferences --store runs/main --tasks tasks.json \
  --model alexzakkarov/qwen3.5-golang --revision rev-a --dataset-version holdout-1 \
  --n-samples 4 --out pairs.jsonl --report pref-report.json

# Go DAPT korpusu: indir, lisans süzgecinden geçir, ölç
gotooltrain-data go-pairs --download-to data/go-ut-bench \
  --out data/go-dapt.jsonl --report data/go-dapt-report.json

# Korpus gerçekten katalog öğretiyor mu? (yetersizse exit 1)
gotooltrain-data measure --corpus corpus.jsonl

# Go reposundan görev üret; testi zaten geçen paketler elenir
gotooltrain-data tasks --repository repo/ --repository-name acme/parser \
  --out tasks.jsonl

# SFT: madenlenmiş korpustan tam fine-tune
gotooltrain-train sft --model Qwen/Qwen3.5-4B --output runs/sft \
  --dataset sft.jsonl --epochs 3

# DPO: execution reward çiftleriyle tercih döngüsü
gotooltrain-train dpo --model runs/sft --output runs/dpo --pairs pairs.jsonl
```

## Colab'da 4B koşusu

Depoyu `origin`'den çeken, kurup **4B tam fine-tune'u başlatan** ve checkpoint'ı
**her N optimizer adımında** Hugging Face Hub'a gönderen defter:
[`notebooks/gotooltrain_colab.ipynb`](notebooks/gotooltrain_colab.ipynb).
Ayrıntılar ve maliyet notları: [`docs/COLAB.md`](docs/COLAB.md).

```bash
# Periyodik yayın: token model yüklenmeden ÖNCE çözülür, eksikse koşu başlamaz
HF_TOKEN=... gotooltrain-train sft \
  --model Qwen/Qwen3.5-4B --output runs/colab-sft --dataset data/sft.jsonl \
  --optimizer adafactor --gradient-checkpointing --max-length 8192 \
  --hub-repo-id alexzakkarov/qwen3.5-golang --hub-push-every 25

# Takvimi hiçbir şey yüklemeden prova et
gotooltrain-train sft ... --hub-repo-id a/b --hub-dry-run
```

Sandbox imajı **RTK içeren** `gotooltrain/go-sandbox:0.1.0` olmalıdır; stok `golang`
imajında rtk yoktur ve katalogdaki beş araç her çağrıda harness hatası verir
(gerekçe: [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)).

## Kalite kapıları

```bash
pip install -e ".[dev]"

make check        # lint + format + typecheck + coverage
make coverage     # coverage run -m pytest && coverage report  (eşik %100)
```

> Coverage bilinçli olarak `pytest-cov` eklentisi yerine `coverage run` üzerinden
> sürülüyor. Eklentinin paralel veri dosyası davranışı ortama göre değişiyor ve
> pytest geçici dizininde başlatılan bir alt süreç bu config'i bulamadığı için
> branch'siz veri yazıp koşuyu
> *"Can't combine statement coverage data with branch data"* ile düşürüyor.

Dördü de yeşil: **1015 test geçti (22 atlandı, gerekçeli), coverage %100, ruff temiz,
mypy strict temiz.**

`train()` ve `train_dpo()` gerçek mimariyle çalışır: testler `Qwen3.5-4B`
config'inden küçültülmüş ama **yapısı aynı** bir checkpoint kurar, tam bir
optimizer adımı atar ve checkpoint + plan yazar. Döngü hiçbir yerde koşmadan
"çalışıyor" denemezdi.

## Temel kurallar

1. **Şablon tek kaynaktır.** Render'ı Python'da yeniden yazma; eğitim ve çıkarım
   ayrışır ve sapma hata değil kalite kaybı olarak görünür.
2. **Loss yalnızca asistan içeriğine.** Katalog ve `<tool_result>` maskeye girerse model
   tool çıktısı uydurmayı öğrenir — bu formatın var oluş sebebi.
3. **Format değişikliği eğitim verisi değişikliğidir.** `python -m gotooltrain.regenerate_golden`
   sonrası diff incelenmelidir.
4. **Doğrulama render'dan önce çalışır.** `validate_records(..., strict=True)` milyonlarca
   kayıtta tek bir hatalı kaydın eğitimi sabote etmesini engeller.
5. **Hiçbir noktada fallback yok.** Şablon yoksa, sürüm uyuşmazsa, RTK yoksa, maske boşsa:
   hata. Sessizce bozulma yok.
6. **Idempotency anahtarı bir doğrulama iddiasıdır.** `FINGERPRINT_FIELDS` sonucu değiştirebilecek
   her girdiyi içerir; aynı anahtara farklı içerik yazılmaya çalışılırsa hata verir. Böylece
   "birisi `model_revision`'ı anahtara koymayı unutmuş" gibi hatalar yakalanır.
7. **Modelden gelen hiçbir metin shell'e girmez.** Tool çağrıları argv listesi olarak kurulur;
   hiçbir kod yolu model dizgisini komut satırına birleştirmez.
8. **`harness_error` ile `model_error`/`tool_error` ayrıdır.** Altyapı hatası asla model
   performansı olarak sayılmaz; paralel koşuda bu ayrım olmazsa skor şişer.
9. **Sessizce hiçbir yere yayınlama.** `hub.py` yayınlamayı bir *program* olarak
   modeller: takvim eğitimden önce hesaplanır, token model yüklenmeden çözülür, her
   push bir Hub commit'idir ve yayınlanan klasör `hub_push.json` ile kendi kökenini
   taşır. `--hub-dry-run` bunu hiçbir şey göndermeden prova eder.
