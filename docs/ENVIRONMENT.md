# Geliştirme ve Değerlendirme Ortamı

İki ortam var ve **ikisi de yeşil olmalı**. Windows'ta geliştiriliyor, WSL'de
değerlendirme koşuyor; bu ayrım tesadüf değil, aşağıdaki bulguların sebebi.

## Kurulum (WSL)

```bash
# rtk v0.50.0 — musl build (statik, glibc'da da çalışır), checksum doğrulanır
# go 1.23.6  — go.dev checksum listesinden doğrulanır
# uv          — CPU torch için ayrıca: uv pip install --index-url \
#                 https://download.pytorch.org/whl/cpu torch torchvision
export PATH="$HOME/.local/bin:$HOME/.local/go/bin:$PATH"
```

Kurulum betikleri `temp/opencode` altında; her indirme yayıncının checksum'ıyla
doğrulanır. Doğrulamasız kurulan bir araç, koşunun sonunda anlaşılır.

### transformers 5.x şart

`Qwen/Qwen3.5-4B` config'i `model_type: qwen3_5` bildiriyor ve **transformers
4.x bunu reddediyor**: tokenizer yükleniyor, config yüklenmiyor. Yani tokenizer'ın
 ötesindeki hiçbir şey çalışmıyor. Doğrulanan sürüm **5.17.0**.

Tavanın kaldırılması token formatını değiştirmedi — aynı sohbet iki sürümde
**byte-byte aynı** işlendi:

| | transformers 4.57.6 | transformers 5.17.0 |
|---|---|---|
| render sha256 | `803899e78a5642e2…` | `803899e78a5642e2…` |
| karakter | 621 | 621 |
| token | 176 | 176 |
| assistant token | 42 | 42 |

Golden dosya geçerli, eğitim verisinin yeniden üretilmesi gerekmiyor.

### numpy < 2.5 (geliştirme aracı kısıtı)

numpy 2.5'in paketlediği `.pyi` dosyaları PEP 695 `type` sözdizimi kullanıyor.
Bu proje Python 3.10'u hedeflediği için mypy bunu **ayrıştıramıyor** ve
torch'un stub zinciri (`torch → torch._C → numpy`) üzerinden gelen bir hata
veriyor. Kaynak kodla ilgisi yok; `follow_imports = "skip"` bir `import *`
yeniden dışa aktarımının ayrıştırılmasını durdurmuyor. Bu yüzden `dev` extra'sında
`numpy<2.5` sabitlendi. Hiçbir yerde numpy tipi annotasyonu yok; sabit sadece
kapının makineden bağımsız olmasını sağlıyor.

### Windows Python'u paylaşılan

Bu depodaki Windows ortamı **paylaşılan** `C:\...\Python310`. `transformers`
5.17'ye yükseltildiğinde aynı interpreter'daki `vllm`, `trl` ve `unsloth-zoo`
paketleri kendi sınırlarıyla çelişti (pip bunları uyarı olarak bildiriyor).
Bu projeyi etkilemez — eğitim/eval ortamı WSL'deki ayrı venv
(`/root/venvs/gotooltrain`) — ama diğer projeler için bir venv ayrılması gerekir.

## Docker worker imajı

Değerlendirme konteynerleri `golang:1.23-bookworm` üzerine kurulur — ama o imajda
**rtk yok**. Katalogdaki dokuz aracın beşi (`go_build`, `go_test`, `grep`,
`read_file` ve `rtk_recall`'ın zinciri) rtk üzerinden çalışır; stok imajda her
`go_test` çağrısı "rtk is not installed" ile, yani **modelin hatasıymış gibi**,
harness error döner. Bu yüzden kendi imajımız var:

```bash
docker build -t gotooltrain/go-sandbox:0.1.0 -f docker/go-sandbox/Dockerfile .
docker run --rm --network none gotooltrain/go-sandbox:0.1.0 selfcheck
```

İmaj rtk'yı sürüm sabitleyip, hem Dockerfile'a gömülü sha256 ile hem de yayıncının
`checksums.txt` dosyasıyla doğrular; ikisi uyuşmazsa build **başarısız** olur.
`selfcheck` imajın içinde `go test`'in gerçekten çalışabildiğini, `GOCACHE` ve
`GOMODCACHE` yollarının pool'un tmpfs mount'larıyla çakıştığını build anında
doğrular.

İki canlı-Docker hatası yalnızca gerçek konteynerle bulundu, sahte backend'li
testlerle görünmedi:

| Bulgu | Sonuç |
|---|---|
| Docker `--tmpfs` mount'larını varsayılan olarak `noexec` yapıyor | `go test` derlediği binary'yi `/tmp` altında `exec` edemiyordu: `fork/exec ...x.test: permission denied`. Bu bir *başarısız görev* gibi görünüyordu. Mount'lara `exec` eklendi. |
| Ölü konteynerde `docker exec` çıkış kodu 1 verir | `build failed` ile **aynı**. Pool bunu araç hatası sayıp modele yazıyordu; worker'ları ölmüş bir koşu "kesin ve yanlış" bir sayı raporluyordu. Artık `docker inspect` ile gerçek durum soruluyor ve `HARNESS_ERROR` olarak raporlanıyor. |
| Workspace `rmtree` + `copytree` ile sıfırlanıyordu | Workspace bir bind mount hedefi; dizinin **inode**'u silinince mount kopuyordu. Linux'ta konteyner `/workspace`'i artık exec edilemiyor (`current working directory is outside of container mount namespace root`), mount silme öncesi içeriği göstermeye devam ediyordu. Windows bunu gizliyor, çünkü mount'ları her erişimde yol ile çözüyor. Sıfırlama artık dizini **boşaltıyor**, yeniden yaratmıyor. |
| WSL'de bir dizini bind-mount etmek, aynı sürücüdeki `getcwd()`'i geçersiz kılıyor | `docker run -v` tek başına yeterli; silme gerekmiyor. Koşudan sonra `/mnt/f` üzerinde cwd'si olan her süreçte `os.getcwd()` ENOENT veriyor. Uzun ömürlü bir harness bir sonraki subprocess'inde düşerdi. `DockerBackend` artık her `docker` çağrısına açık bir `cwd` (sistem geçici dizini) veriyor, sürecin cwd'sini miras almıyor. Test ayrıca pool kökünü `/mnt/<sürücü>` üzerinde, repo'nun **dışında** tutuyor. |


## Kapı

Her iki ortamda da aynı komut, aynı sonuç:

```bash
pip install -e ".[dev]"
make check
```

```
pytest + coverage : 1040 passed, 22 skipped (Docker'suz Windows) / 1060 (hepsi kurulu)
coverage          : %100 statement + branch (3723 satır, 1020 dal)
ruff check        : temiz
ruff format       : 72 dosya hazır
mypy              : 29 dosyada hatasız
```

Atlananlar iki kaynaktan gelir ve **gerekçeleri skip metninde yazılıdır**:

| Kaynak | Adet | Gerekçe |
|---|---|---|
| `test_sandbox_real.py` | 20 | `no usable Docker daemon on this host` |
| `test_catalogue_real.py` | 2 | `rtk grep` Windows'ta `busybox` çağırıyor, orada yok; WSL'de koşar |

Yani "888 test" gibi tek bir sabit sayı yerine **kapının yeşil olması** ölçüt:
hangi isteğe bağlı bağımlılık (Docker, WSL'de `rtk`) kuruluysa atlanan küçülür,
kural aynıdır.

`make check`, ilişkili komutları tek tek çağırır (`$(MAKE) coverage` yerine).
Özyinelemeli make, Windows'ta `make`'in kabuğuna bırakılan ortamda kullanılabilir
olmasına bağlı ve orada busybox shim'ine takılıyor; kapı, bozulan şeyin kendisi
olmamalı. Makefile ayrıca WSL'e özgü `cd /tmp && cd $(CURDIR)` önekini (`REENTER`)
platforma göre uygular; DrvFs etkisi yalnızca WSL'de var.

## Bu ortamların bulduğu gerçek hatalar

Bunların hiçbiri birim testle yakalanamazdı; hepsi ya gerçek bir komut ya da temiz
bir ortamdı.

| Bulgu | Neden önemli |
|---|---|
| `rtk grep` özyinelemiyordu | `-r` olmadan `rtk grep "x" .` → `grep: .: Is a directory`, exit 2. Katalogdaki komut **her dizin çağrısında** başarısızdı ve 500 test bunu göremedi. Şimdi `rtk grep -r` ve `tests/test_catalogue_real.py` her komutu gerçekten çalıştırıyor. |
| Pillow hiç beyan edilmemişti | `vision.py` çalışma zamanında `PIL` import ediyor; Windows'ta transformers'ın yan tesadüfü olarak gelmişti. Temiz WSL'de `ModuleNotFoundError`. Artık `dependencies`'te. |
| `jinja2` de beyan edilmemişti | Aynı sınıf hata, şablon testlerinde. `dev` extra'sına eklendi. |
| `torch` beyan edilmemişti | `AutoProcessor` modül seviyesinde torch import eden video-processor sınıfları çözüyor; **görüntü** işleyicisi bile yüklenemiyordu. `train` extra'sı + testler `importorskip` ile net atlıyor. |
| Varsayılan workspace CWD'ye yazıyordu | `run_evaluation` ve `ContainerPool` göreli yol varsayıyordu; testler kaynak ağacına `a-s0/`, `go-0001-s0..3/`, `canary/`, `sandbox-workspaces/` bırakıyordu. Artık ikisi de geçici dizin altında. |
| ruff sürümü sabitlenmemişti | Windows'ta 0.14.5, WSL'de 0.16.9. Yeni sürüm markdown içindeki Python bloklarını da biçimlendiriyor; aynı ağaç iki yerde farklı "uydu". Artık `ruff==0.16.9`. |
| Satır sonları platforma göre değişiyordu | CRLF/LF ayrımı `ruff format`'ın kararını değiştiriyordu. `line-ending = "lf"` + `.gitattributes`. |
| mypy numpy'ı çözümleyemiyordu | Pillow'ın stub'ları numpy import ediyor, numpy'ın stub'ları PEP 695 `type` kullanıyor (3.12+). Proje 3.10 destekliyor. `PIL.*` ve `numpy.*` üçüncü taraf muafiyetine alındı. |

Ortam farklarının kendisi bir sınıf hatadır: `>=` ile sabitlenmemiş bir araç iki
geliştiricide iki farklı davranır ve kapı hangisinin "doğru" olduğunu asla
söylemez.

## Sırada ne var

`rtk go test` çıktısı sıkıştırılmış hâlde gelir ve elenen kısmın hash'ini
bırakır: `[full output: rtk recall 66dbba90085b]`. Modele "tam çıktı şurada"
diyor, ama `rtk recall` araç listesinde yok — yani model çağıracak bir şey
bulamıyor. Kompakt çıktı zaten `TestCount` + `parser_test.go:7: Count = 2, want 3`
gibi aksiyona dönüştürülebilir bilgiyi taşıyor, ama bu bir karar: ya `rtk_recall`
aracı kataloğa eklenir (format değişir, `CATALOG_VERSION` artar, veri yeniden
üretilir) ya da araç açıklaması "tam çıktı bu araçlarla alınamaz" der.
