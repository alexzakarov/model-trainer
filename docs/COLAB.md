# Colab — 4B tam fine-tune, periyodik Hub yüklemesi

Bu pipeline'ın sunucu tarafında koşan hâli. Defter
[`notebooks/gotooltrain_colab.ipynb`](../notebooks/gotooltrain_colab.ipynb) kendi
kodunu `origin`'den çeker, paketi kurar, 4B modeli tam fine-tune eder ve checkpoint'ı
**her N optimizer adımında** Hugging Face Hub'a gönderir.

> **Neden Colab?** Projenin 7. aşaması — gerçek 4B checkpoint ile GPU koşusu —
> tek başına bekliyordu. Colab'un iki avantajı var: A100 40 GB ücretli olarak
> geliyor, ve **maliyeti üst sınırlaması olan tek makine**. İkincisi bu iş için
> önemli: 4B tam fine-tune'un maliyeti tahmin edilebilir değil, süresi de değil.

---

## Ön koşullar

| Gereken | Neden |
|---|---|
| Depoda en az bir commit, `origin`'de | Defter `git clone` ile çekiyor. **Şu an `origin` boş** (`No commits yet on main`), ilk çalıştırma 3. hücrede durur. |
| Colab runtime: **A100 40 GB** | 4B tam FT + AdamW ≈ 18 GB (ağırlık + gradyan). T4'de (15 GB) matematiksel olarak sığmaz. |
| Hub reposu + **yazma yetkili** token | `--hub-repo-id` verdiyseniz token'ın o repo'ya yazma yetkisi olmalı. |
| Colab secret `HF_TOKEN` | Token'ı hücreye yazmak, onu notebook çıktısına ve git geçmişine sızdırır. |

VRAM ön kontrolü defterin 2. hücresindedir ve yetersizse **adını ve gerekeni
söyleyip durur** — sessizce OOM'a düşmez.

## Akış

| Hücre | Ne yapar | Neden orada |
|---|---|---|
| 1 | Parametreler (model, repo, bağlam, `PUSH_EVERY`) | Tek yeri değiştirmen gereken yer |
| 2 | **VRAM ön kontrolü** | 40 dakika sonra OOM öğrenmektense başta öğrenmek |
| 3 | `git clone` + `pip install -e ".[dev,train]"` | Colab imajı değişir; sürümler burada sabitlenir |
| 4 | Token (Colab secret ya da mock) | `HF_TOKEN` yoksa **başlamadan** hata |
| 5 | **Kalite kapısı**: `pytest` | Kırık ağaca kanat takmadan önce 888 test (~1 dk) |
| 6 | Go korpusunu indir, lisans süz, ölç | `terraform` (BUSL-1.1) ve `go-ethereum` (LGPL-3.0) varsayılan reddedilir |
| 7 | Veriyi token'la, bağlam dışını **sayarak** at | Sessiz düşen kayıt ile küçük korpus ayırt edilemez |
| 8 | **Maske doğrulaması** (`assert_mask_sane`) | Maske bozuksa model araç çıktısı uydurmayı öğrenir ve kayıp normal görünür |
| 9 | `gotooltrain-train sft` + `--hub-*` | Asıl koşu |
| 10 | Yayının okunması | "Push ettim" yetmez; *ne* gittiği okunmalı |

## Periyodik yayınlama

```bash
gotooltrain-train sft \
  --model Qwen/Qwen3.5-4B --output runs/colab-sft \
  --dataset data/sft.jsonl \
  --optimizer adafactor --gradient-checkpointing \
  --hub-repo-id alexzakarov/qwen3.5-4b-go \
  --hub-push-every 25
```

| Bayrak | Anlamı |
|---|---|
| `--hub-repo-id` | Hedef Hub model deposu. Verilmezse hiçbir şey yayınlanmaz (hata değil, bir mod). |
| `--hub-push-every N` | **Optimizer adımı** sayar, batch değil. 8 kademe biriktirmeli olduğu için "her batch'te" demek accumulation ayarına bağımlı olurdu. |
| `--hub-token-env` | Token'ın okunacağı değişken (varsayılan `HF_TOKEN`). |
| `--hub-private` | Repo yoksa gizli oluşturur. |
| `--hub-dry-run` | Takvimi ve defter yazımını prova eder, **hiçbir şey yüklemez**. Log'da `DRY RUN` yazar. |

Dört kural, isimleriyle:

1. **Takvim eğitimden önce belli.** `push_steps(total, every)` saf aritmetiktir:
   son adım **her zaman** dahildir (bir partial accumulation window'unun bittiği
   yerdeki adım, yani koşunun bittiği haliyle durduğu ağırlıklar, planlanan bir
   adıma denk gelmeyebilir) ve ara adımlarla çakışmaz. Test, bu iki yazımın
   **aynı yordam** olduğunu bir ızgarada doğrular.
2. **Token isteğe bağlı değildir ve koşu ortasında keşfedilmez.** Pusher, model
   yüklenmeden **önce** kurulur; `HF_TOKEN` yoksa hata orada verir. Yazma yetkisi
   eksikse hata ilk push'ta verilir — 700 adım sonra değil.
3. **Yarım bir bayrak reddedilir.** `--hub-push-every` tek başına "yanlışlıkla
   yazdım" değildir; **"sildiğim sekmede ağırlıkların gitmediğini fark ettim"**
   olabilir. Repo adı olmadan reddedilir.
4. **Yayınlanan klasör kendini anlatır.** Her push'tan önce checkpoint dizinine
   `hub_push.json` yazılır: adım, toplam adım ve **tüm hiperparametreler**. Token
   oraya yazılmaz — sadece değişkenin adı. Hub'daki bir checkpoint'i açtığınızda
   hangi ayarlarla üretildiğini, silinmiş bir run dizinine bakmadan bilirsiniz.

Her push bir Hub commit'idir; commit mesajı `gotooltrain step 25/400` biçimindedir.
Yani Hub geçmişi bir training eğrisi gibi okunur.

### Push bir **checkpoint**, tam **resume** değil

Optimizer durumu saklanmaz. Kaldığınız yerden devam etmek `resume_from` ile
ağırlıkları yükleyip **yeniden eğitim** demektir. Bu bilinçli bir sınır: optimizer
state'i 4B'de birkaç GB'dır ve her push'ta taşımak her şeyi yavaşlatırdı.

## Veri: bu koşu ne ölçer

Colab koşusu **Go-UT-Bench'in "bu dosya için birim testi yaz" görevleriyle** SFT
yapar. Bu:

- ✅ token formatını gerçek `Qwen3.5-4B` tokenizer'ıyla üretir,
- ✅ kaybı yalnızca asistan token'larına uygular,
- ✅ 4B tam fine-tune'u gerçekten bir optimizer adımı attırır,
- ✅ periyodik yayın takvimini çalıştırır.

Ama ❌ **Go yeteneğini ölçmez.** Korpus tek turlu, araçsız ve sekiz depodan birine
ağırlıklı; `gotooltrain-data measure` onu "yetersiz" diye bildirecek ve **bu doğru
cevaptır** (bkz. [`../PIPELINE.md`](../PIPELINE.md) karar 11). Yetenek ölçümü
[`EVAL.md`](EVAL.md)'deki üç aşamalı eval'i gerektirir: konteyner havuzu, `rtk`,
`--network none` izolasyonu ve *gerçek* Go depoları. Colab'da bunların hiçbiri yok.

Bölme yolu: değerlendirme WSL'de ya da bir Docker host'unda koşar, veri
`ResultStore`'a düşer, **checkpoint'ler Colab'da** üretilir. Depolarda değil, iki
tarafın sözleşmesi budur.

## Not defterini yeniden üretmek

Defter elle yazılmaz — elle yazılan bir notebook, incelenemeyen 400 satırlık JSON
blob'dur ve bu projenin temel ilkesi (değişiklik incelenebilir olmalı) bunun
karşısındadır.

```bash
python -m gotooltrain.build_notebook      # notebooks/gotooltrain_colab.ipynb
```

`tests/test_build_notebook.py` şunları doğrular: JSON olarak parse ediliyor, her
kod hücresi geçerli Python, hücreler doğru sırada, pahalı kontroller (VRAM, kalite
kapısı, maske) eğitim hücresinden **önce** geliyor, ve hücrelerde gerçek bir token
sızmıyor. Notebook ruff kapsamı dışında (`extend-exclude`); üreten modül kapının
içinde.

## Bilinen sınırlar

| Sınır | Sonuç |
|---|---|
| Colab oturumu ~12 saat, bağlantı kopabilir | Yayın takvimi bunun için var. Kalan en fazla `PUSH_EVERY` adım. |
| `target` dizini oturumla birlikte silinir | Tekrar çalıştırmak baştan indirir. `PUSH_EVERY`'yi küçültün ya da Drive'a bağlayın. |
| Disk ~78 GB (Colab) | 4B bf16 ≈ 8,7 GB + Adafactor durumu + optimizer geçici dosyaları. `MAX_RECORDS` ile sınırlayın. |
| Tam FT, lora değil | 4B'yi tek kartta tutmanın tek yolu Adafactor + gradient checkpointing. LoRA'ya düşmek **bu projenin kararı değil**, ayrı bir karar olurdu. |
