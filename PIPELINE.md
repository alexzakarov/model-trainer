# Go + Tool-Use Eğitim Pipeline'ı — Standart Yöntem

Araştırma tarihi: 29 Eylül 2026. Hedef: `Qwen/Qwen3.5-4B` taban modelini
**Golang uzmanı + Anthropic tool-blocks formatında tool use** yeteneğine sahip modele
tam fine-tune (full fine-tune) etmek.

**Kısa cevap: Evet, bu alanda standart ve olgun bir pipeline var.** Aşağıdaki yapı,
literatürde ve üretimde tekrar tekrar kullanılan parçalardan oluşuyor; sıfırdan
icat etmemize gerek yok.

---

## 1. Standart var mı? — Evet, dört kanıt

| Kanıt | Kaynak | Bize ne verir |
|---|---|---|
| **Go-UT-Bench** | arXiv 2511.10868 | Go için gerçek bir fine-tuning veri seti: 10 permissively-licensed Go repos'undan (kubernetes, tidb, Go stdlib, terraform, prometheus, kserve) **5.264 `{code, unit test}`** çifti. Fine-tune edilmiş modeller görevlerin **%75+'ında** base'inden iyi. Ayrıca "1K–10K örnek" bandının bu tür adaptasyon için doğru ölçek olduğunu da kanıtlıyor. |
| **BFCL v4** | Berkeley Function Calling Leaderboard | Tool use için **fiili standart eval**. Single-turn, multi-turn, parallel, relevance, hallucination, AST, executable kategorileri. |
| **xLAM-function-calling-60k** + ToolACE + Glaive-fc-v2 + Hermes-fc-v1 | Salesforce / çeşitli | Tool-call SFT için standart veri setleri. |
| **BalanceSFT** | ACL 2026 Findings, 18094–18112 | Function calling SFT'nin en güncel iki sorununu çözen yöntem: (a) **dengesiz eğitim sinyalleri** — uzun CoT tokenları function-call tokenlarını bastırıyor, (b) **zor veri yeniden örnekleme**. 7B modelde function-calling'de GPT-5'i geçiyor. |

> **Kritik bulgu** — arXiv 2606.00135 (*On Effectiveness and Efficiency of Agentic
> Tool-calling and RL Training*): "**Küçük system-prompt değişiklikleri, RL fine-tuning'in
> etkisini geçecek kazançlar üretebilir.**" Yani en pahalı şeyi (eğitim) yapmadan önce
> system prompt ve tool şema render'ını doğru kurmak, büyük kazanç sağlar. Pipeline'ın
> en ucuz, en yüksek getirili adımı budur.
>
> Aynı çalışmanın ikinci bulgusu: multi-turn eğitim verisinin **varlığı** değil,
> **kalitesi/hizalanması** darboğaz. Yani az ama iyi ajan trajektörisi > çok ama dağınık.

---

## 2. Pipeline (6 aşama)

### Aşama 0 — Baseline ve eval sabitle *(eğitimden önce)*
Hiçbir eğitim yapmadan Qwen3.5-4B'ün sayılarını al. Qwen3.5 native tool calling ile
geldiği için **başlangıç çoktan güçlü**; bizim kazancımız sıfırdan değil, üstüne çıkmak.

- **Tool use:** BFCL v4 tüm alt-kategorileri + τ²-bench (agentic multi-turn).
- **Go:** Go-UT-Bench'in **held-out** bölümü + **execution-based** skorlama
  (üretilen testin gerçekten `go test`'ten geçmesi). LLM-judge'dan üstündür.
- **Unutma kontrolü:** genel kod benchmark'ları (HumanEval+/LiveCodeBench) baseline'ı.
- Tüm sayılar `model-training/eval/` altında, tek formatta (JSON) sabitlenir.

### Aşama 1 — Chat template ve token formatı *(en kritik teknik adım)*
Anthropic tool-blocks formatı **chat template'de** kodlanır; model JSON schema'yı
doğal olarak görmez, sadece template'in render ettiği şeyi görür (HF chat templating
dokümanı bunu açıkça söylüyor). Bu yüzden:

- Tokenizer'a `<tools>`, `<tool_use>`, `<tool_result>` bloklarını render eden
  `chat_template` yazılır ve **`tokenizer_config.json` ile birlikte** dağıtılır.
- **Eğitimde kayıp (loss) sadece asistan token'lerine uygulanır**; system prompt,
  tool tanımları ve `tool_result` girdileri maskelenir. Aksi halde model tool
  sonuçlarını da üretmeye çalışır.
- JSON-schema tool tanımlarını Anthropic bloklarına çeviren bir dönüştürücü yazılır
  (OpenAI `tools` formatı → Anthropic), böylece BFCL/xLAM verisi doğrudan kullanılabilir.
- **Eğitim ve çıkarım aynı render'ı kullanmalı** — tek bir fonksiyon, iki yerde de.
- Doğrulama: şablonu değiştirmek model yeteneğini değiştirir; format hatası sessizce
  kaliteyi öldürür. Bunu erken yakalamak için Aşama 0'ın prompt-only kolu kullanılır.

### Aşama 2 — Veri
**Tool use (standart setler):**
- `Salesforce/xlam-function-calling-60k` — doğrulanmış çağrılar, geniş araç çeşitliliği
- ToolACE, Glaive-function-calling-v2, Hermes-function-calling-v1
- **Sentetik üretim:** NVIDIA NeMo "Data Designer" tarzı, şema-kısıtlı üretim.
  Referans sonuç: 500 sentetik örnek, ince ayarlanmamış base modelde function-name
  doğruluğunu %1.4 → **%93** çıkarmış. Yani az ve temiz sentetik veri çok işe yarar.

**Golang:**
- Go-UT-Bench (5.264 çift) — hazır, lisans temiz, commit hash'leriyle tekrarlanabilir
- Gerçek Go repoları (lisans filtreli), stdlib + `pkg.go.dev` dokümanı
- PR/commit kaynaklı görev türetme (issue → patch)

**Karışım ve denge (BalanceSFT dersi):**
- Uzun CoT verisi, function-call token'larını **bastırıyor**. Loss'u dengeleyen bir
  mekanizma kurulmalı (kendi ağırlıklandırmamız veya SSB).
- **Multi-turn ajan trajektörileri:** Go geliştirme harness'ında
  (`read_file`, `write_file`, `grep`, `go_build`, `go_test`, `go_mod`) gerçekten
  çalıştırılarak üretilmiş örnekler. Kalitesi kritik.
- **Unutmayı önleme:** genel kod + akıl yürütme verisi karıştırılır, replay yapılır.
  Dar/kötü bir korpus genel yeteneği bozar (literatürde tekrarlanan uyarı).

### Aşama 3 — Eğitim (tam fine-tune)
- **Stack:** TRL (`trl sft` + YAML config) veya LlamaFactory; dağıtım için
  DeepSpeed ZeRO-3 veya FSDP + Accelerate.
- bf16, gradient checkpointing, sequence packing, cosine LR, warmup.
- LR: tam FT için ~1e-5 – 2e-5 (LoRA'nın 2e-4'ünün çok altında).
- Seq uzunluğu: Qwen3.5 hibrit DeltaNet mimarisi uzun bağlamda verimli; 4K–8K ile başla.
- **Önerilen sıra:** önce SFT (format + tool use), sonra Go DAPT/continued-pretraining.
  Tek aşamalı karışım da mümkün; ancak format öğrenimi (tool-blocks) kararlılık
  kazanmadan domain verisi eklemek hataları pekiştirir.
- **Risk:** Qwen3.5-4B zaten native multimodal + tool calling ile geliyor; tam FT
  bunları bozabilir. Önlem: düşük LR, replay karışımı, erken ve sık eval.

### Aşama 4 — Pekiştirme (opsiyonel, sonra)
- **DPO/ORPO** tercih çiftleriyle.
- **GRPO + execution reward:** Go için doğal ödül fonksiyonu var —
  ürettiği testin `go test`'ten geçip geçmesi, derlenip derlenmediği. Bu, alan-özgü
  olmanın en değerli yanı: *doğrulanabilir* bir reward var.

### Aşama 5 — Değerlendirme ve iterasyon
- BFCL v4 + τ²-bench (tool use)
- Go-UT-Bench held-out + gerçek `go test` ile execution skoru (Go)
- Genel benchmark'lar (unutma takibi)
- Hata analizi → zor veri geri besleme döngüsü (BalanceSFT'in HDR mantığı)

---

## 3. Sıralama gerekçesi (maliyet artan)

1. **System prompt + tool render** — bedava, RL etkisi kadar kazanç (2606.00135)
2. **Baseline + eval** — onsuz ilerlemek ölçülemez
3. **Chat template + loss mask** — sessiz kalite katili, ucuz
4. **Az ve temiz tool SFT** — 500 örnek bile %93 function-name doğruluğu
5. **Go domain verisi** — asıl alan uzmanlığının kaynağı
6. **Tam FT** — en pahalı adım, ancak format ve veri oturduktan sonra
7. **RL/DPO** — en pahalı, en son

---

## 4. Kaynaklar

- Go-UT-Bench: arXiv 2511.10868 (Pipalani vd., 14 Kas 2025; rev. 29 Mayıs 2026)
- Agentic tool-calling & RL: arXiv 2606.00135
- BalanceSFT: ACL 2026 Findings, s. 18094–18112
- LoGos (alan-uzmanlığı + genel CoT karışımı + GRPO deseni): arXiv 2601.16447
  ⚠️ Dikkat: buradaki "Go" **Go tahta oyunu**, Golang değil. Yöntem deseni
  (yapılandırılmış alan verisi + uzun CoT karışımı → RL) ilgili, veri seti değil.
- Qwen2.5-Coder teknik raporu: arXiv 2409.12186
- HF chat templating / tool use: `transformers` dokümanı
- NVIDIA NeMo: Tool-Calling Fine-Tuning with Synthetic Data
- BFCL: Berkeley Function Calling Leaderboard

---

## 5. Alınan kararlar

| # | Karar | Gerekçe / sonuç |
|---|---|---|
| 1 | Go değerlendirmesi: **execution-based + LLM-judge ikilisi** | Execution tek başına "derleniyor/geçiyor" der; idiomatik kaliteyi yakalamaz. İkisi birlikte iki farklı eksik ölçer. |
| 2 | Tool listesi: **dar çekirdek, 9 araç** | `read_file`, `write_file`, `edit_file`, `grep`, `go_build`, `go_test`, `go_doc`, `go_mod_tidy`, `rtk_recall`. `gofmt`/`go_vet` dışarıda: `go_build` zaten vet'in bir kısmını çalıştırıyor, katalog maliyeti hak etmiyor. `rtk_recall` sonradan eklendi çünkü RTK sıkıştırılmış çıktının hash'ini bırakıyor ve model onu okumak için bir araca ihtiyaç duyuyor (`CATALOG_VERSION` `go-tools-v2`). |
| 3 | **Multimodal korunuyor**, vision encoder dahil tam eğitim | Model görsel girdi de alabilir. Bedeli: Aşama 1'e görsel desteği eklenmeli (aşağıda). |
| 4 | **SFT önce, sonra Go DAPT** | Format/tool öğrenilmeden domain verisi eklemek hataları pekiştirir. |
| 5 | **Bağlam 32K** | Repo düzeyinde çalışır. Bedeli: 4.33B full FT'te ciddi hesap (ZeRO-3/FSDP, çoklu GPU); 8K'nın ~4 katı. |
| 6 | Tool çıktısı: **RTK + kırpma** | Aşağıdaki "RTK kararı" bölümü. |
| 7 | Vision encoder: **hepsi eğitilir** + görsel veri karışımı | Görsel yeteneği bozmamak için küçük bir görsel replay karışımı şart. |
| 8 | **Thinking (CoT) eğitilir** | Kodda çok adımlı hata ayıklama kalitesi. Bedeli: CoT, tool-call token'larını bastırır → sinyal dengeleme zorunlu (BalanceSFT). |
| 9 | Go harness: **Linux / WSL** | Go toolchain ve RTK'in doğal ortamı; `go test` çıktısı platformlar arası tutarlı. |
| 10 | Judge: **güçlü, baştan** | Space Bunny otomatik judge **olamaz** (aşaıdaki gerekçe). |
| 11 | Go DAPT korpusu: **önce ölçülür** | Uydurma hedef yok; korpus bulunup lisans/boyut dağılımı görüldükten sonra karar verilecek. |
| 12 | Eval **paralel** çalışır | Seri değil, varsayılan. Paralellik bir performans tercihi değil, izolasyon kararı: modelin ürettiği kodun patlama yarıçapını büyütür. |
| 13 | İzolasyon: **Docker konteyner havuzu** | Görev başına konteyner değil, ısıtılmış işçi havuzu; ağ kapalı, salt-okunur kök, bellek kapağı. Tamper sıfır değil, kabul edilebilir. |
| 14 | Örnekleme: **görev başına N örnek (pass@N)** | Skor gürültüsünü ölçer ve küçük farkları güvenilir kıyaslanabilir kılar. Maliyet N kat. |
| 15 | Eğitim verisi: **yalnızca judge'ın başarılı bulduğu trajektöriler** | Bir trajektöri "modelin ne yaptığının" kaydıdır; başarısız olduğu görevde yaptığı şey düzeltilmek istenen şeydir. Üzerinde eğitim, hatayı öğretir. Ölçüt `min_score=1.0` varsayılanı; harness hatası, truncation, yarım kalan trajektöri, tool çağrısı olmayan sohbet ve eşleşmeyen tool sonucu da reddedilir. Her ret **adlandırılır ve sayılır** — sessizce düşen bir trajektöri, ölçülemeyen bir koşudur. |
| 16 | Görev üretimi: **testi zaten geçen paketler görev sayılmaz** | Base modelin çözdüğü bir görev hiçbir şey ölçmez ve her skoru şişirir. Paket başına bir `go test` maliyeti, anlamlı bir korpusun bedelidir. |
| 17 | Go korpusu: **lisans iddiası denetlenir, varsayılmaz** | Go-UT-Bench "10 permisif lisanslı repo" diyor. Doğrulandığında ikisi değil: `hashicorp/terraform` BUSL-1.1 (kaynak-açık, açık kaynak değil), `ethereum/go-ethereum` LGPL-3.0 (copyleft). İkisi varsayılan olarak **reddedilir**; kabul edilmeleri bilinçli bir karar olmalı. Bilinmeyen bir repo da reddedilir, böylece yenilenmiş veri seti on birinci repoyu sessizce sokamaz. |
| 18 | Go korpusu: **yayınlanan split'ler örtüşüyor** | Makale 5.264 çift ve 3.684/790/790 bölünme diyor. Yayınlanan dosyalar 4.211 + 790 + 1.053 = **6.054** kayıt, 5.258 tekil hash: **796 tekrar**. Aynı içerik hem train hem test'te ise bu set *held-out* değerlendirme olarak kullanılamaz; DAPT içinse tekilleştirilir ve kaç tanesinin atıldığı rapora yazılır. |
| 19 | Tercih öğrenmesi: **execution reward, DPO kaybı kendi yazıldı** | Go'da ödül doğrulanabilir: üretilen test derlenip geçiyor ya da geçmiyor. Bu alanın en değerli yanı. Ödül **görevin bildirdiği doğrulama aracından** gelir, "son çalıştırılan komuttan" değil — yoksa testi kalan ama `read_file` ile biten bir örnek başarılı görünür. Harness hatası **puanlanmaz**, 0 değildir: 0 model hakkında bir iddiadır, ölü konteyner ise altyapı. DPO kaybı TRL'den değil buradan gelir; TRL torch ve transformers'ı kendi sınırlarıyla ikinci kez pinlerdi, ve çiftin iki üyesinin **aynı** token kümesi üzerinden puanlanması gerekliliği (prompt uzunluğu ödülü etkilemesin diye) kütüphane çağrısının arkasında kalırdı. |
| 20 | 7. aşama **Colab'da**, checkpoint **periyodik Hub'a** | Sunucu tarafında tek kart yok; Colab'un A100 40 GB'ı ücretli ama **maliyeti üst sınırlaması olan tek makine** — 4B tam fine-tune'un süresi değil maliyeti tahmin edilemez. Yayın takvimi buna bağlı: sekme kapanırsa kaybedilen şey `PUSH_EVERY` adımla sınırlıdır. Takvim **eğitimden önce** hesaplanır (`push_steps` saf aritmetik), token **model yüklenmeden** çözülür (eksikse koşu başlamaz), son adım her zaman dahildir (partial accumulation window'unun bittiği yerdeki ağırlıklar planlanan bir adıma denk gelmeyebilir), ve her push `hub_push.json` ile kendi kökenini taşır. Yarım bayrak reddedilir: `--hub-push-every` tek başına "sildiğim sekmede ağırlıklar gitmedi" olabilir. Bu bir **checkpoint**, tam **resume** değildir: optimizer durumu saklanmaz. |

### Aşama 0: paralel eval harness tasarımı

**Üç ayrı aşama, her biri kendi başına resumable.** Aşamaları ayırmanın sebebi:
30 dakikalık judge turu çökerse üretimi tekrar yapmak zorunda kalmamak.

| Aşama | Doğası | Paralel ekseni | Kalıcı çıktı |
|---|---|---|---|
| 1 — üretim | ağ-bağımlı | istek eşzamanlılığı | shard/model çıktıları |
| 2 — execution | CPU-bağımlı | işçi havuzu | shard/test sonuçları |
| 3 — judge | ağ-bağımlı | istek eşzamanlılığı | shard/puanlar |

- **Shard-per-task**: tek büyük dosya değil, görev başına JSONL. Koşu ortasında çökerse
  kayıp olmaz; `resume` tamamlanan görev kimliklerini atlayarak devam eder.
- **Execution izolasyonu**: her işçi ayrı `GOCACHE`/`GOMODCACHE`/`GOTMPDIR`, ayrı modül
  dizini, `-count=1` (test cache'ini atla), sert `-timeout`, bellek kapağı, **ağ kapalı**.
  Bağımlılıklar önceden `vendor` edilmiş olmalı — ağ kapalıyken `go mod download`
  çalışmaz. `-p 1` ile `go test` iç paralelliğini kapatmak genelde dış paralellikten
  daha iyi sonuç verir.
- **`harness_error` ile `model_fail` ayrı alanlarda.** Paralel koşuda harness hataları
  çoğalır; hepsi "başarısız" sayılırsa skor şişer ve gerçek performans düşüşü görünmez.
- **Canary ön koşusu**: paralel fan-out başlamadan önce bilinen-geçen ve bilinen-kalan iki
  örnek koşturulur. Harness ayrım yapamıyorsa koşu **başlamaz** — kırık bir harness
  %100 başarı döndürür ve bunu fark etmek haftalar sürer.
- **Deterministik toplama**: paralel sonuçlar görev kimliğine göre sıralanarak birleştirilir.
  Aksi halde aynı girdi iki koşuda farklı rapor üretir ve gelişme gürültüden ayırt edilemez.
- **Kaynak ayrımı**: üretim/judge ağ-bağımlı, execution CPU-bağımlı; iki aşamanın
  eşzamanlılık sınırları ayrı ayarlanır.

### RTK kararı

RTK (Rust Token Killer, Apache 2.0) CLI çıktısını ajan bağlamına girmeden önce
sıkıştıran bir proxy katmanı. Go desteği: `go test` %80-90 ("failures only"),
`go build` %75 ("errors only"), `grep` %70, `read` %60-80. Listelenmeyen komutlar
(`go_doc`, `go_mod_tidy`) değişmeden geçer.

**Kabul gerekçesi — eğitim/sunum tutarlılığı.** RTK'nın arkasında çalışan bir ajan
`✓ 184 passed · 0 failed` görür. Sıkıştırılmamış `go test` çıktısıyla eğitilirsek model
sunumda tanımadığı bir formatla karşılaşır. RTK bir sıkıştırma tercihi değil, **format
kararıdır.**

**Tuzak — yasak kombinasyon.** `rtk smart <file>` dosyayı yalnızca imzaları içeren 2
satırlık özete indirger (%85 kazanç). `read_file` için kullanılırsa modeli **sadece
imzaları gördüğü kodu düzenlemeye** teşvik eder. Kural: `read_file` → `rtk read` veya
düz çıktı; **`rtk smart` yasak.**

**İki katmanlı politika.**
1. RTK: `go_test` → `rtk go test`, `go_build` → `rtk go build`, `grep` → `rtk grep`, `read_file` → `rtk read`
2. RTK sonrası hâlâ bütçe aşımı → **kırpma + görünür işaret** (`... (N satır daha)`)

**Sürüm sabitlenmeli** (`v0.50.0`) ve dataset metadata'sına yazılmalı: RTK'nın çıktı
biçimi eğitim verisinin bir parçasıdır. Veri üretiminde RTK yoksa harness **hata
verir**, sessizce sıkıştırmaz.

### Space Bunny neden otomatik judge olamaz

Space Bunny (`opencode/space-bunny-free`) bu OpenCode oturumunu çalıştıran modeldir;
batch inference için sunulabilir bir uç noktası yoktur, sürüm sabitlenemez, oturum
bağlamı sürüklenebilir. Judge metriğinde gürültünün en pahalı olduğu yer tam da
tekrarlanabilirliktir.

**Değerli olduğu yerler:** rubric tasarımı ve uzlaşmazlıkların adjudication'ı (küçük
örneklem, kararlar denetlenebilir kaydedilir).

**Açık engel (bloklayıcı):** otomatik judge için ya sabitlenmiş bir API modeli
(kredi/anahtar gerekir) ya da yerel güçlü bir model servisi (vLLM, GPU gerekir) seçilmeli.

### Aşama 0 öncesi yapılacak ek iş: görsel desteği — **tamamlandı**

Multimodal korunduğu için şablonda ve data layer'da görsel girdi desteği gerekiyordu:

- `user` turn'leri görsel öğesi taşıyabilmeli (Qwen: `<|vision_start|><|vision_pad|>…<|vision_end|>`) — **yapıldı** (`ImageSpec`, şablon)
- data layer görsel referanslarını doğrulamalı (URI/denetim, sayı, sınır) — **yapıldı** (`normalize._normalize_images`, `_check_image_ref`)
- `<|vision_pad|>` token'larının maskeye nasıl gireceği netleşmeli (görsel token'lar hedef olmamalı) — **yapıldı** (`vision.mask_vision_labels`)

Ek olarak, token formatı kendini tanımladığı için piksellerin sayımı
**çapraz doğrulanır** (`vision.process_images`): template'in bastığı vision token
sayısı ile gerçek image processor'ın patch sayısı eşleşmezse batch kurulmaz.
Ayrıntı ve gerekçe: [EVAL.md](docs/EVAL.md).

Bunu Aşama 0'dan **önce** yapmak gerekiyordu; ilk görsel veri geldiğinde token
formatının ikinci kez değişmesi ve eğitim verisinin yeniden üretilmesi engellendi.
Doğrulama gerçek `Qwen/Qwen3.5-4B` tokenizer + image processor ile yapılıyor
(`tests/test_vision_real.py`).

## 6. Proje kuralı: hiçbir noktada fallback yok

Hataya düşsün. Sessizce bozulma, düşük kalite, uyarı, "devam" yok.

| Durum | Davranış |
|---|---|
| Tokenizer'da chat template yok | `TemplateError` — diskten sessiz yükleme yapılmaz |
| Şablon sürüm damgası farklı/eksik | `TemplateError` — eğitim/sunum ayrışması |
| `chat_template.jinja` kaybolmuş | `save_template_artifacts` doğrular ve reddeder |
| RTK yok (veri üretimi) | Hata — sıkıştırılmamış çıktı üretilmez |
| Truncation tüm asistan turn'ünü kesti | `DatasetError` (varsayılan `on_empty="error"`) |
| Maskede sızıntı / boş maske | `TemplateError` |
| Bozuk kayıt | `validate_records(strict=True)` ilk hatada durur |
| Tool sonucu kaçırılabilir tag içeriyor | `ReservedTagError` |

Açık seçenek olan iki yer operatörün bilinçli tercihidir, otomatik düşüş değildir:
`sanitize_reserved_tags=True` ve `on_empty="skip"`. Üçüncüsü: `--hub-dry-run` —
yayınlama takvimini hiçbir şey göndermeden prova eder; seçilmesi için policy'de
adıyla istenir, bir hata durumunda seçilmez.
