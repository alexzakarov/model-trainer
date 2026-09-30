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
| Depoda en az bir commit, `origin`'de | Defter `git clone` ile çekiyor. `main` artık `95b1aa3` üzerinde. |
| Colab runtime: **A100 40 GB** | 4B tam FT + AdamW ≈ 18 GB (ağırlık + gradyan). T4'de (15 GB) matematiksel olarak sığmaz. |
| Hub deposu: [`alexzakkarov/qwen3.5-golang`](https://huggingface.co/alexzakkarov/qwen3.5-golang) | Hedef. **Şu an boş** (dosya yok, model card yok) — ilk push onu doldurur. |
| **O depoya yazma yetkili** token | Token'ın sahibi ile depo sahibi aynı hesap olmalı. `alexzakkarov` (HF) ile `alexzakarov` (GitHub) farklı yazımlardır; karışıklık bu iki harfin sessizce yer değiştirmesiyle başlar. |
| Colab secret `HF_TOKEN` | Token'ı hücreye yazmak, onu notebook çıktısına ve git geçmişine sızdırır. |

Hedef depo adı kodda **tek yerde** duruyor:
[`build_notebook.py`](../src/gotooltrain/build_notebook.py) →
`DEFAULT_HF_REPO_ID`. Not defteri, `--help` metni ve bu belge ondan türer;
`test_the_publication_target_appears_exactly_once` bunu zorlar. Depo adı iki
yerden biri güncellenirse koşu hâlâ çalışır ama checkpoint **kimsenin bakmadığı
bir yere** düşer — bir yayın yolunun en kötü kısmi güncellemesi odur.

VRAM ön kontrolü defterin 2. hücresindedir ve yetersizse **adını ve gerekeni
söyleyip durur** — sessizce OOM'a düşmez.

## Akış

| Hücre | Ne yapar | Neden orada |
|---|---|---|
| 1 | Parametreler (model, repo, bağlam, `PUSH_EVERY`) | Tek yeri değiştirmen gereken yer |
| 2 | **VRAM ön kontrolü** | 40 dakika sonra OOM öğrenmektense başta öğrenmek |
| 3 | `git clone` + `pip install -e ".[dev,train]"` + **çekirdek `sys.path`'i** | Colab imajı değişir; sürümler burada sabitlenir |
| 4 | **Go + rtk**, digest'lerle doğrulanmış | Colab'da ikisi de yok — aşağıya bak |
| 5 | Token (Colab secret ya da mock) | `HF_TOKEN` yoksa **başlamadan** hata |
| 6 | **Kalite kapısı**: `pytest -q -x -rs` | Kırık ağaca kanat takmadan önce ~2 dk test |
| 7 | Go korpusunu indir, lisans süz, ölç | `terraform` (BUSL-1.1) ve `go-ethereum` (LGPL-3.0) varsayılan reddedilir |
| 8 | Veriyi token'la, bağlam dışını **sayarak** at | Sessiz düşen kayıt ile küçük korpus ayırt edilemez |
| 9 | **Maske doğrulaması** (`assert_mask_sane`) | Maske bozuksa model araç çıktısı uydurmayı öğrenir ve kayıp normal görünür |
| 10 | `gotooltrain-train sft` + `--hub-*` | Asıl koşu |
| 11 | Yayının okunması | "Push ettim" yetmez; *ne* gittiği okunmalı |

## Neden defter Go ve rtk kuruyor (4. hücre)

Colab imajında ikisinin de **yoktur**. Bu bir eksik değil, kapının kırılma
biçimidir:

- Katalogdaki komutların çoğu **rtk üzerinden** çalışır: `rtk go build`,
  `rtk go test`, `rtk read`, `rtk grep`. `go_mod_tidy` ve `go_doc` ise düz `go`.
- Paketin kendi testleri **gerçek** komutları koşar. Go yokken üç test kırılır
  (`the Go toolchain is required to verify a task but is not installed`), rtk yokken
  `go_test` `rtk not found` ile `HARNESS_ERROR` döner.
- Her iki durumda da hata **kırık paket** gibi görünür. Oysa kırık olan ortamdır —
  ve bu ayrımı yanlış oklamak, ücretli bir makinede tam bir tur demektir.

Kurulum üç kurala uyuyor:

- **Sabit + doğrulanmış.** `go1.23.6.linux-amd64.tar.gz` go.dev'in kendi
  `?mode=json` dizinindeki sha256'sıyla karşılaştırılır. rtk'nin sürümü ve
  digest'i ise `docker/go-sandbox/Dockerfile`'ın sabitlediğiyle **aynen aynıdır** —
  yani defter ile sandbox imajı kanıtlanabilir biçimde aynı binary'yi koşar. Bu
  titizlik değil: katalogun çıktı formatı rtk'nin çıktı formatıdır, farklı bir
  build üzerinde ölçmek eval harness'ın ölçeceği formatı ölçmemek demektir.
- **rtk çift kaynaklı doğrulanır.** Gömülü sabit *ve* yayıncının kendi
  `checksums.txt`'i. Sabit "neyi inceledim"i söyler, indirme "yayıncı hâlâ öyle
  diyor"u kanıtlar. Uyuşmazsa kurulum iptal edilir. Dockerfile'ın yaptığı da bu.
- **`latest` yok, `GOTOOLCHAIN=local` var.** Go sürümü sürüklenirse iki koşu
  arasındaki fark kimseye atfedilemez. `GOTOOLCHAIN=local` ise `go.mod`'daki
  `go 1.23` direktifinin ağdan toolchain indirmesini engeller.

`apt-get install golang-go` **kullanılmıyor**: dağıtım paketi farklı bir sürüm ve
`GOTOOLCHAIN=local` ile testlerin ürettiği modülleri reddedebilirdi.

İkisi de `os.environ["PATH"]` üzerinden kuruluyor (shell export'u değil), böylece
**hem kalite kapısı hem eğitim koşusu** aynı toolchain'i görüyor. Shell'e
kurulup sürece aktarılmayan bir Go, kapıdan geçer ve eğitimde patlar.

Mimari kontrolü de var: sabitlenmiş build'ler x86_64, Colab'ın T4/L4/A100
runtime'ları da öyle. Başka bir mimaride hücre adıyla duruyor.

## Hücre 3'ün `sys.path` satırı neden var

`pip install -e` yalnızca **yeni** süreçlerde görünür. `.pth` ve
editable-finder çengelleri yorumlayıcı **başlangıcında** kurulur; Colab çekirdeği
ise bu hücreden çok önce başlamıştır.

Bunu tahmin etmedim, ölçtüm: pip'i çalıştıran yorumlayıcı, kendi kurduğu paketi
import edemiyor. Asimetri şu:

| | gotooltrain görür mü |
|---|---|
| Alt süreç (kalite kapısı, eğitim koşusu) | ✅ evet — kurulumdan sonra doğar |
| Colab çekirdeği | ❌ hayır — kurulumdan önce doğdu |

Sonuç tam olarak bildirilen hataydır: 6. hücre (kapı) yeşil, 8. hücre
`ModuleNotFoundError`. Kurulum **hücre 3'te**, hata **hücre 8'de** — arada
dört hücre.

Bu yüzden 3. hücre kaynak dizini çekirdeğin `sys.path`'ine de ekler ve
paketi hemen import ederek **doğrular**. Gizli bir düşüş değil: kurulum zaten
yapıldı, bu yalnızca aynı kurulumu *bu* sürece görünür kılıyor. Konsol
betikleri (`gotooltrain-train`) alt süreçlerde çalışır ve zaten yolunu bulur.

Doğrulama iki şeyi birden kontrol eder: import gerçekten çalışıyor ve cevap
veren kop **klonun kendisi**. `sys.path`'te başka bir `gotooltrain` varsa
(sık görülen tuzak: daha önce kurulmuş bir sürüm) defter onu sessizce eğitirdi;
şimdi adıyla duruyor.
## Periyodik yayınlama

```bash
gotooltrain-train sft \
  --model Qwen/Qwen3.5-4B --output runs/colab-sft \
  --dataset data/sft.jsonl \
  --optimizer adafactor --gradient-checkpointing \
  --hub-repo-id alexzakkarov/qwen3.5-golang \
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

## Bellek: 4B tam fine-tune neden 80 GB'ı dolduruyor

Ölçülmüş değil, hesaplanmış — `Qwen/Qwen3.5-4B` config'i okundu:

| | |
|---|---|
| hidden_size / katman | 2560 / 32 |
| **vocab_size** | **248.320** |
| katman türleri | 32'nin **24'ü** `linear_attention` (Gated DeltaNet), 8'i `full_attention` |
| DeltaNet durumu | `[32 v_heads, 128, 128]` = **token başına 1 MB** |

Kalıcı bellek (bf16, Adafactor, batch 1): ağırlık 8,66 GB + gradyan 8,66 GB +
adafactor ~0,05 GB = **17,4 GB**. Sorun bu değil.

| Bağlam | DeltaNet durumu (checkpointing'siz) | Loss logits (bf16 / fp32) |
|---|---|---|
| 2048 | 49 GB | 0,95 / 1,89 GB |
| 4096 | 98 GB | 1,89 / 3,79 GB |
| 8192 | **196 GB** | 3,79 / 7,58 GB |

İki terim de bağlamla doğrusal büyüyor; 248 binlik sözlük yüzünden **loss
terimi** de olağanüstü pahalı. 4B model 80 GB kartta 8K'da OOM vermesi
beklenen bir şey, sürpriz değil.

## Asıl hata: `--gradient-checkpointing` hiç çalışmıyordu

Transformers'ın sarmalayıcısının koşulu:

    if self.gradient_checkpointing and self.training:

`from_pretrained` modeli **eval modunda** döndürüyor. `train()` hiç
`model.train()` çağırmadığı için bayrak plana yazılıyor, log'a
"gradient checkpointing on" diye basılıyor, ve **hiç tetiklenmiyordu**.

Kurulu sarmalayıcıya karşı ölçtüm:

    layer.gradient_checkpointing = True
    model.training              = False
    eval() sonrasi   -> checkpointing çağrısı: 0
    train() sonrasi  -> checkpointing çağrısı: 1

Etkisi: 8K'da ~196 GB yerine ~1,3 GB saklanan aktivasyon. Yani
`--gradient-checkpointing` bayrağı, belki de projedeki **en pahalı tek satır**
hataydı — sessiz, plana yazılan ve hiç iş yapmayan.

Düzeltildi: `enter_training_mode(model)` modeli train moduna alır ve kaç
modülün geçtiğini döndürür, ki log "ne istendi" değil "ne oldu" desin.

## Bağlam 4096'ya indirildi

Bu korpusun gerçek uzunluk dağılımı (ölçüldü, ilk 400 kayıt):

    < 2048 :  46 (%11)     < 4096 : 168 (%42)     < 8192 : 290 (%72)
    medyan 4848    p90 17669    en uzun 78011

4096 hem sığdırıyor hem korpusun yarısını bırakıyor. Not defteri 8192'de
denendi ve OOM verdi; 4096 ölçülmüş bütçeye göre seçildi. 8192'yi denemek
isteyen önce `DRY_RUN=True` ile tek adımı denemeli.

## İki kurulum notu

* **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** ayarlanıyor. Parçalanmayı
  azaltır; bir bütçe aşımını **çözmez**. Etkisi çalışma zamanında görülür.
* **Füş DeltaNet çekirdekleri yok.** `flash-linear-attention` ve
  `causal_conv1d` kurulu değilse transformers referans PyTorch uygulamasına
  düşüyor ve *"correct but much slower"* diyor — yani doğru ama daha yavaş ve
  daha bellekli. Bunları kurmak hem hızı hem belleği iyileştirir, ama Colab'daki
  torch sürümüyle uyumu **doğrulanmamıştır**, bu yüzden deftere sessizce
  eklenmedi. Denemek istersen 4. hücreden sonra:

      pip install flash-linear-attention causal_conv1d

  kurulum başarısız olursa koşu yine çalışır, yalnızca yavaş olur.
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
| Disk ~78 GB (Colab) | 4B bf16 ≈ 8,7 GB + Adafactor durumu + Go (~1,5 GB) + optimizer geçici dosyaları. `MAX_RECORDS` ile sınırlayın. |
| Tam FT, LoRA değil | 4B'yi tek kartta tutmanın tek yolu Adafactor + gradient checkpointing. LoRA'ya düşmek **bu projenin kararı değil**, ayrı bir karar olurdu. |
| Model card yok | `upload_folder` kart yazmaz. Depo boş olduğu için ilk push'tan sonra elle bir `README.md` gerekiyor; jenerik bir "uploaded to the Hub" kartı, kartın yokluğundan iyidir ama bilgi taşımaz. |
| Docker kurulu değil | 20 `test_sandbox_real` testi atlanır. Colab'da Docker yok; eval harness'ının konteyner izolasyonu Colab'da **test edilemez**. Değerlendirme için ayrı bir Docker host'u gerekir. |
