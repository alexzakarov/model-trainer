# Eval Altyapısı

Aşama 0'ın paralel execution-based değerlendirmesi. Mimari kararlar
[PIPELINE.md](../PIPELINE.md) içinde; burada uygulama ve kullanım.

## Üç katman, üç farklı sorumluluk

| Modül | Sorumluluk |
|---|---|
| [`evalstore.py`](../src/gotooltrain/evalstore.py) | Dayanıklılık + idempotency |
| [`gotools.py`](../src/gotooltrain/gotools.py) | Hangi araçlar, nasıl çalışır, çıktı bütçesi |
| [`workspace.py`](../src/gotooltrain/workspace.py) | Komutsuz araçlar: yol güvenliği, kesin düzenleme |
| [`gorun.py`](../src/gotooltrain/gorun.py) | argv güvenliği, status ayrımı, canary, paralel, pass@k |
| [`sandbox.py`](../src/gotooltrain/sandbox.py) | Konteyner izolasyonu, havuz, recycle |
| [`evalrun.py`](../src/gotooltrain/evalrun.py) | Agent döngüsü, tur indeksli resume, judge |

## Dayanıklılık ve idempotency

İki ayrı garanti, birbirine karıştırılmaz:

**Dayanıklılık.** Sonuçlar append-only. Koşu, tek büyük dosya değil, görev başına shard
+ event log'dur. Yazımlar atomiktir (temp + `os.replace`), log fsync'lenir. Çökme
anında en kötü halde o an yürüyen deneme kaybolur; `resume` tamamlananları diskten
okuyarak atlar. **Yırtık son satır hata verir, sessizce atlanmaz** — yazıcının çöktüğünü
gösterir.

```python
store = ResultStore("evalstore")
store.start_run(run_id, {"model_revision": "rev-a", "n_samples": 4})
store.append_event(run_id, {"event": "completed", "stage": "execute", "key": key})
store.completed_keys(run_id, stage="execute")  # resume için
```

**İdempotency.** Her deneme, sonucu değiştirebilecek *tüm* girdilerden türetilen bir
içerik anahtarıyla adreslenir: `task_id`, `sample_index`, `stage` (tur indeksini de
taşır, ör. `generate:t2`), `model_id`, **`model_revision`**, `format_version`,
`template_sha`, `harness_version`, `dataset_version`, `seed`, `decode_params`,
`tool_catalog_sha`, `image_digest`.

Anahtar **eksik-konfigürasyon denetimi** görevi de görür: aynı anahtara farklı içerik
yazılmaya çalışılırsa `IdempotencyViolationError` verir. Bu, "birisi `model_revision`'ı
anahtara koymayı unuttu" hatasını yakalar — aksi halde yeniden eğitilmiş bir model
eski sonuçları sessizce yeniden kullanır ve bir gerileme iyileşme gibi görünür.

```python
store.put(key, payload)  # yaz-bir-kez
store.put(key, same_payload)  # aynı içerik: no-op
store.put(key, other_payload)  # IdempotencyViolationError
```

**Dürüst sınır:** nedensel-LM örneklemesi bit-aynı tekrarlanamaz (batching sayısal
sonucu değiştirir). Bu yüzden garanti ikiye ayrılır:

- **Anahtarlı memoizasyon** — bir anahtar her zaman saklanan artifact'a çözer, asla yeni
  örnekleme yapmaz.
- **Byte-eksik doğrulama** — determinizmin sağlanabildiği aşamalarda
  (`verify_idempotency`): execution ve judge girdileri.

## Tool politikası

| Araç | Çalıştırma | Komut | Bütçe |
|---|---|---|---|
| `read_file` | RTK_READ | `rtk read {path}` | 20 000 |
| `write_file` | RAW | — (workspace mutasyonu) | 20 000 |
| `edit_file` | RAW | — (workspace mutasyonu) | 4 000 |
| `grep` | RTK | `rtk grep {pattern} {path}` | 8 000 |
| `go_build` | RTK | `rtk go build {pkg}` | 8 000 |
| `go_test` | RTK | `rtk go test {pkg}` | 12 000 |
| `go_doc` | RAW | `go doc {symbol}` | 6 000 |
| `go_mod_tidy` | RAW | `go mod tidy` | 4 000 |

`RAW` **"RTK kuralı yok, düz komut çalıştır"** demektir, "komut yok" değil.
`go_doc` ve `go_mod_tidy` bir zamanlar boş template ile sevk edildi ve hiçbir şey
yapmadan başarıyı raporluyordu; artık gerçek argv üretiyorlar.

**Yasak:** `rtk smart` hiçbir aracı arkasında duramaz. `rtk smart` dosyayı yalnızca
imzaları içeren 2 satırlık özete indirger; `read_file` için kullanılırsa modeli **sadece
imzaları gördüğü kodu düzenlemeye** teşvik eder. `assert_no_forbidden_smart()` bunu
denetler ve ihlali `AssertionError` ile durdurur.

`catalog_sha()` katalogun kimliğidir: açıklama, şema veya bütçe değişirse değişir, çünkü
hepsi tool seçimini etkiler.

## Komutsuz araçlar: sessiz no-op yasak

`write_file` ve `edit_file` kabuk karşılığı yoktur; workspace'i doğrudan
değiştirirler ([`workspace.py`](../src/gotooltrain/workspace.py)). Bunları boş
komut olarak modellemek **her çağrıyı hiçbir şey yapmadan başarılı raporlar** —
modelin düzenlemesinin uygulandığını öğrenirken dosya hiç değişmemiş olur ve sonraki
tur, modelin var olduğunu sandığı bir dosya üzerine akıl yürütür.

İki ayrı güvence:

- `assert_catalog_is_executable()` — katalog, "komut var **ya da** workspace mutasyonu
  olarak beyan edilmiş" kuralını denetler. Yeni bir araç bu halde eklenemez.
- `assert_harness_covers_catalog()` — beyan edilen her araç için gerçek bir handler
  olup olmadığını doğrular. `run_evaluation()` bunu çağırır, yani **bir araç
  çalıştırılamaz hâle gelirse koşu başlamaz.**

`edit_file` benzersizlik şartını **uygular**: `old_string` birden çok yerde geçerse
ilkini sessizce değiştirmez, hata döndürür. Sessizce yanlış fonksiyonu düzenlemek,
modelin neden başarısız olduğunu hiç öğrenememesi demektir.

Yol güvenliği çözülmüş yol üzerinden denetlenir (`resolve_in_workspace`): `..`
segmentleri, mutlak yol **ve** dışarı işaret eden bir symlink dışarı çıkamaz.

Satır sonları `\n`'ye normalize edilir; CRLF üreten bir model, `gofmt`'in yeniden
yazacağı ve `go build`'in farklı okuyacağı bir dosya üretirdi.

## Çıktı bütçeleri

Bütçe aşan çıktı **baş ve son** ile kesilir ve araya görünür bir işaret girer:

```
... (+318 more lines truncated)
```

Sondan kesmek yanlış olurdu: derleme ve test hataları en sonda durur. Sessiz kesim
modele "kısa çıktı = kısa koşu" yanlışını öğretirdi.

## Status ayrımı

| Status | Anlamı | Modele görünür mü |
|---|---|---|
| `OK` | komut çalıştı, başarılı | evet |
| `TOOL_ERROR` | komut çalıştı, başarısız (testler kırmızı) | **evet** |
| `MODEL_ERROR` | modelin isteği geçersiz (bilinmeyen araç, hatalı argüman) | hayır |
| `HARNESS_ERROR` | altyapı: konteyner, timeout, eksik binary | hayır |

Kırmızı test paketi **modelin görmesi gereken meşru bir sonuçtur**; çöken konteyner
altyapı hatasıdır. İkisini birleştirmek kırık bir harness'ı zayıf bir model gibi
gösterir — ve paralel koşuda bu sayı filyo büyüklüğüyle ölçeklenir.

## Canary

Paralel fan-out başlamadan önce bilinen-geçen ve bilinen-kalan iki örnek koşturulur:

```python
run_canary(
    executor,
    [
        CanaryCase("passing", req_ok, should_fail=False),
        CanaryCase("failing", req_bad, should_fail=True),
    ],
)
```

Ayrım yapamayan bir executor her şeyi başarılı döndürür ve **her koşuda %100** raporlar.
Bunu fark etmek haftalar sürer; bu yüzden canary geçmeden koşu başlamaz.

## Paralel sürüş ve resume

```python
results = run_many(
    requests,
    executor,
    max_parallel=8,
    should_run=lambda r: r.task_id not in done,  # resume
    on_result=persist,  # her sonuç diske yazılır
)
```

Sonuçlar **giriş sırasında** döner: paralel tamamlanma sırası rapora sızmasın diye.
`summarise()` görev bazlı gruplar, `pass@k`'yi çıkarır ve `task_id`'ye göre sıralar —
aksi halde gerçek gelişme gürültüden ayırt edilemez.

`pass@k` çarpımsız tahmin edicidir: `1 - C(n-c, k) / C(n, k)`, `k > n` ikisinde
de doygunlaşır.

## İzolasyon: konteyner havuzu

Model ürettiği kod bu yüzden host'ta değil, konteynerde koşar
([`sandbox.py`](../src/gotooltrain/sandbox.py)).

Görev başına konteyner açmak paralel faydadan pahalı olduğu için **ısıtılmış bir
havuz** kullanılır ve aşağıdaki garantiler sağlanır:

| Garanti | Uygulama |
|---|---|
| Ağ kapalı | `--network none`; `SandboxSpec(network=True)` yapıcıda **reddedilir** |
| Salt-okunur kök | `--read-only`; yalnızca workspace ve Go cache'leri tmpfs |
| Bellek/CPU kapağı | `--memory`, `--cpus`; alt sınırlar yapıcıda doğrulanır |
| Görev izolasyonu | Her görevde workspace fixture'dan **yeniden oluşturulur** (yerinde temizlik değil, sil-yap) |
| Ölü işçi recycle | Konteyner öldüyse yok edilir, yerine yenisi başlatılır, hata `HARNESS_ERROR` olarak raporlanır |

Workspace'in sil-yap ile sıfırlanması bilinçli: yerinde temizlik bir görevin
bıraktığı dosyayı hayatta tutabilir ve iki görev sessizce birbirini kirletir.

```python
spec = SandboxSpec(image="golang:1.23-bookworm", memory_mb=3072, cpus=2)
pool = ContainerPool(backend=DockerBackend(), spec=spec, size=8)
executor = ContainerExecutor(pool, fixture=Path("fixtures/go-ut-bench/go-0001"))
```

Havuz mantığı (rezervasyon, recycle, workspace sıfırlama, hata sınıflandırma)
`ContainerBackend` protokolünün arkasında olduğu için **Docker daemon olmadan**
test edilir. `container_summary()` izolasyon ayarlarını run manifestine yazar.

## Güvenlik: shell yok

Modelden gelen argümanlar güvenilmezdir. Komutlar **argv listesi** olarak kurulur ve
shell kullanılmadan çalıştırılır; hiçbir kod yolu model dizgisini komut satırına
birleştirmez. `build_argv` testleri bunu açıkça doğrular: `; rm -rf /` biçiminde bir
değer tek bir argv girdisi olarak kalır.

## Agent döngüsü: tur indeksli orkestrasyon

Değerlendirme üç bağımsız geçiş değil, bir **trajectory**'dir
([`evalrun.py`](../src/gotooltrain/evalrun.py)). Bir Go ajanı dosya okur, düzenler,
testleri çalıştırır, hatayı okur ve yeniden düzenler; yalnızca ilk turu değerlendirmek
modeli kimsenin istemeyeceği bir şey üzerinde puanlamak olurdu.

**Tool sonuçları modele geri gider.** Generator, `<tool_result>` blokları dahil
konuşma geçmişini alır — template'in bastığı biçimle. Alternatif (bir kez üret, ne
geldiyse çalıştır) tek turlu veri üretir ve modeli sonuçları tahmin etmeye eğitir.

**Turnar başına ayrı anahtar.** Her `(task, sample, turn)` üçlüsü kendi
fingerprint'ini alır (`stage` alanına `generate:t2` gibi yazılır), böylece çökme 4.
turda olursa 4. turdan devam eder. `go_test`'in 0. turdaki ve 3. turdaki çalıştırması
**farklı denemelerdir**; çakışırsa ikincisi önbelleklenmiş ilk sanılır.

**Tur içinde sıralı, örnekler arası paralel.** Bir örnek içindeki turlar nedensel
bağımlıdır ve örtüşemez; bağımsız örnekler eşzamanlı koşar. Turlar arası kesişen
paralellik, modelin henüz görmediği bir workspace durumuna karşı düzenleme yapardı.

**Dürüst puanlama.** Tool çağırmayan bir örnek başarılı sayılmaz: "hiçbir şey
yapmadı" bir başarısızlıktır, sessizce rapordan düşürülemez. Rapor görev listesinden
türetilir, sonuç listesinden değil — aksi halde ortalamalar şişer. Turn bütçesini
tüketen bir ajan `truncated` olarak ayrıca raporlanır.

Harness hataları **asla yargılanmaz**: kırık altyapı, modelin regresyonu gibi görünürdü.
Modele gösterilen içerik de bunu açıkça söyler (`[infrastructure error] ...`); boş
`<tool_result>`, çöken konteyner için "komut çıktı vermedi" yanlışını öğretirdi.

```python
report = run_evaluation(
    RunConfig(
        model_id="...",
        model_revision="rev-a",
        dataset_version="holdout-1",
        seed=7,
        decode_params={...},
        n_samples=4,
        max_turns=8,
    ),
    tasks,
    store,
    generator,
    executor,
    judge=judge,
    workspace="workspaces/go-0001",
    max_parallel=8,
    k=2,
)
report.to_record()
```

## Multimodal: token yerleşimi ile piksellerin sözleşmesi

Token formatı **kendini tanımlar**: bir `ImageSpec` `grid_thw` ve `merge_size`
taşır, yani `<|vision_pad|>` sayısı görüntü işleyici hiç çalışmadan **bilinir**.
Bu, mümkün olan şeyi mümkün kılar.

`vision.py` bu sözleşmeyi denetler. Vision encoder, `<|vision_pad|>` konumu başına
**bir satır** piksel tüketir ve indeksle eşleştirir. Eğer işleyici farklı bir patch
sayısı üretirse sonraki her görüntü yanlış hizalanır: eğitim "çalışır", loss düşer,
model **yanlış piksellere** bakmayı öğrenir. Hiçbir şey çökmez. Bu yüzden sayım
kıyaslanır ve uyuşmazlık batch kurulmadan **sert hata** verir.

Kritik ayrım: `merged_patches()` bilerek `ImageSpec.vision_tokens`'ın kopyası
değildir. **Beyan edilen yerleşim** ile **işleyicinin çıktısı** iki bağımsız
kaynaktır; aynı fonksiyonu iki tarafta çağırmak denetimi anlamsızlaştırırdı.

Üç kural:

- **Görsel token'ları asla denetlenmez.** Vision token'ları girdidir, çıktı değil.
  Etiketlenmeleri modele görüntü patch'i ürettirirdi.
- **Padding id'si vision pad id'si olamaz.** Değilse her dolgu konumu boş uzay
  gibi okunur.
- **Ref veri katmanında doğrulanır** — mutlak yol, sürücü harfi, `..` ve ters
  eğik çizgi reddedilir. `load_image` ayrıca çözülmüş yol üzerinden containment
  tekrar kontrol eder (symlink kaçışı ancak burada yakalanabilir).

Doğrulama iki katmanlıdır. Sahte işleyiciler mantığı kanıtlar; `test_vision_real.py`
**gerçek `Qwen/Qwen3.5-4B` tokenizer ve image processor**'ına bağlanır. Üç bağımsız
yazılmış parçanın (şablon, tokenizer, vision tower) birbirine uyduğunu doğrulayan
tek yer orasıdır.

```python
batch = process_images(processor, images, specs)  # grid uyuşmazlığı → DatasetError
total = check_batch_vision_layout(rows, vision_pad_id, pad_id=pad_id, expected=batch.total_patches)
```

## Judge: sabitlenmiş, keşfedilmez

Skorlama kararı **execution + güçlü LLM judge** ikilisiydi. Bu ikisi farklı şeyler
ölçer ve karşılaştırılamaz değildir; bu yüzden mod **sonucun parçası**, metadata
değil.

**Sabitlenmiş, keşfedilmez.** Sessizce "şu an erişilebilen modele" düşen bir judge,
günler arası skorları karşılaştırılamaz hale getirir: geçen haftanın 0.62'si ile bu
haftanın 0.62'si farklı modellerden gelmiş olabilir. Judge kimliği run manifestine
yazılır, rapora yazılır ve **farklı judge ile resume reddedilir** — iki modelin
verdict'leri tek sayıda harmanlanmaz. Daha keskin bir koruma: store'daki bir verdict
yalnızca onu yazan judge tarafından yeniden kullanılabilir, aksi halde session
notu, sunulan bir judge adıyla raporlanırdı.

**Bozuk verdict, sıfır değil hatadır.** Judge düz metin döndürürse, `choices`
içermezse, erişilemezse veya yapılandırılmamışsa `HarnessError` verilir. Sessizce
0.0 yazmak, judge kesintisini modelin başarısızlığına çevirirdi. Aynı şekilde
`{"score": "pass"}` ya da `{"reason": "iyi görünüyor"}` gibi verdict'lar da reddedilir:
puanlamayan bir judge sessizce puanlamamış olmamalıdır.

Judge, herhangi bir OpenAI-uyumlu `/chat/completions` ucuyla çalışır (vLLM, SGLang,
llama.cpp, TGI, hosted API). Bu bilinçli: judge güçlü olmak zorunda ve tek bir
satıcıya bağlı kılmak, döngünün en değiştirilemez bileşenini o satıcının fiyatına
ve erişilebilirliğine esir ederdi.

### Session judge: maliyet sıfır olan yol

Sunucu ucu yoksa veya maliyet kabul edilmiyorsa, judge **bu oturumun kendisi**
olur: eval trajectory'leri bir queue'ya yazar, oturumda puanlanır, verdict'ler
store'a geri yazılır. Kod yolu tam olarak gerçektir — sahte bir judge değil.

`judge_queue()` queue'u **store'dan türetir**: generation ve execution artifact'ları
kalıcı olduğu için trajectory yeniden kurulur, hiçbir şey yeniden çalıştırılmaz.
Bu yüzden çökmüş ya da resume edilmiş bir koşu, gerçekten çalıştırılmış
trajectory'leri puanlar.

Bedeli saklanmaz, **raporda yazılır**: `scoring: "session_judge"` ve
`reproducible: false`. Aynı trajectory sonraki bir oturumda farklı puan alabilir;
böyle bir koşuyu sunulan bir judge koşusuyla aynı ölçüm gibi karşılaştırmak yanlış
olur. `execution_only` koşular da `reproducible: true` döner — çünkü tek
tekrarlanamayan bileşen judge'dır; execution sonuçları store'da byte-eksik saklanır.

**Üç ölçüm, üç ayrı şey.** `execution_only` (yalnız exit code), `session_judge`
(bu oturum) ve `judge` (sunulan model). Her biri kendi etiketiyle raporlanır; ikisi
birleştirilmez.

### Judge'sız koşu ve payda kısalması

`judge=None` ile koşu reddedilmez ama işaretlenir: `scoring: "execution_only"`.

Bir görevin örneklerinden biri harness hatasına kurban giderse, görev yine
`n_samples` kadar örnek raporlar. Kaybolan örnek **silinmez**, başarısız sayılır —
aksi halde yarısı hiç var olmayan bir görev 1.0 pass@1 ile raporlanırdı. Tamamen
yargılanamayan görevler `unjudged_tasks` ile ayrıca sayılır.

Prompt, judge'ı kanıta bağlar: *"çıktı basmayan komut kendi başına geçer değildir,
exit code'a ve tanılara bak"*. Trajectory özetlenmez, ham komut çıktısıyla gösterilir;
"3 komut çalıştı, hepsi OK" diyen bir judge, gerçek bir düzeltmeyi yorum
biçimlendirmesinden ayırt edemez.

## Komut satırı

```bash
gotooltrain-eval queue \
  --store .evalstore --tasks holdout.jsonl \
  --model alexzakkarov/qwen3.5-golang --revision step2000 --dataset-version go-ut-bench-1 \
  --model-url http://localhost:8000/v1 \
  --sandbox docker --image golang:1.23-bookworm --workers 8 \
  --out .evalstore/queue.jsonl

# oturumda queue'yu oku, verdict'leri yaz, sonra:
gotooltrain-eval judge \
  --store .evalstore --tasks holdout.jsonl \
  --model alexzakkarov/qwen3.5-golang --revision step2000 --dataset-version go-ut-bench-1 \
  --run-id run-1 --queue .evalstore/queue.jsonl --verdicts verdicts.jsonl
```

`judge` komutu **hiçbir şey çalıştırmaz ve örnek üretmez**. Bir trajectory eksikse
o koşu tamamlanmamıştır; queue'yu tatmin etmek için yeniden örnek üretmek, verdict
dosyasının hiç bahsetmediği bir şeyi puanlamak olurdu.

### Fan-out öncesi iki kontrol

`queue` iki şeyi baştan reddeder:

1. **Eksik binary.** Katalogdaki komutların (`rtk`, `go`) PATH'te olup olmadığı
   doğrudan denetlenir. *Canary bunu yakalayamaz*: eksik binary her komutu
   "başarısız" yapar, ki must-fail probu tam olarak bunu bekler. Denetlenmezse
   her örnekte `HARNESS_ERROR` olarak görünür, tam koşu harcandıktan sonra, ve
   bozuk bir harness gibi okunur.
2. **Canary.** Toolchain'in başarıyı başarısızlıktan ayırt edebildiği kanıtlanır.
   Ayrım yapamayan bir executor her şeyi başarılı döndürür ve **her koşuda %100**
   raporlar.

`--sandbox local` host'ta çalıştırmak içindir ve `--allow-local-execution` olmadan
reddedilir: modelin ürettiği kodun host'ta koşması, konteyner havuzunun var oluş
sebebidir. Yerel çalıştırmada fixture her örneğin kendi workspace'ine kopyalanır —
**her örnek ayrı dizin alır**, çünkü örnekler eşzamanlı koşar ve paylaşılan bir
dizin birbirlerinin düzenlemelerini görürdü. Workspace kökü de CWD'ye düşmez:
`run_evaluation` ve `ContainerPool` geçici dizin kullanır, çünkü göreli bir
varsayılan per-sample dizinlerini sürecin bulunduğu yere — pratikte kaynak ağacına
— yazardı.

## Şimdiye kadar

Orkestrasyon, agent döngüsü, komutsuz araç işlemleri, vision katmanı, judge
(üç mod), generator, CLI ve dayanıklı store yazıldı: **548 test, %100 statement +
branch coverage**, ruff ve strict mypy temiz. Hem Windows'ta hem WSL'de
doğrulanıyor — kurulum ve o ortamların bulduğu gerçek hatalar için
[ENVIRONMENT.md](ENVIRONMENT.md).

Sırada: `rtk recall` kararı (bkz. ENVIRONMENT.md), veri üretim scriptleri ve Go
korpus ölçümü.
