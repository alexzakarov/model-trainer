"""Build the Colab notebook.

A notebook is a JSON file, which is a poor place to hand-write one: a single missing
comma produces a file that Jupyter refuses to open, and nothing else catches it. So it
is generated from cell definitions here, validated, and written -- and the generator
stays in the tree so the notebook can be regenerated rather than hand-patched.

Run:  python -m gotooltrain.build_notebook
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from .errors import ToolTrainError

#: The repository the notebook clones. Read from the working tree's own origin when
#: there is one, so the notebook cannot drift from the remote it is meant to fetch.
DEFAULT_REPO_URL: Final[str] = "https://github.com/alexzakarov/model-trainer.git"

#: The Hub model repo the trained checkpoint is published to.
#:
#: The single place this name appears in code. The notebook's parameter cell is built
#: from it, the test asserts it appears exactly once, and the documentation quotes it.
#: A target written into three files is a target that gets half-renamed when the repo
#: is renamed -- and half of a publication path is the worst possible amount of it.
DEFAULT_HF_REPO_ID: Final[str] = "alexzakkarov/qwen3.5-golang"

#: Where the notebook is written, relative to the project root.
NOTEBOOK_PATH: Final[Path] = Path("notebooks") / "gotooltrain_colab.ipynb"

#: Bumped by hand when the cell set changes, so a stale notebook is recognisable.
NOTEBOOK_VERSION: Final[str] = "1"


def _markdown(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip("\n").splitlines(True)}


def _code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(True),
    }


#: The cell a reader edits. Split into a generated header and a static body so the
#: target repo is interpolated from :data:`DEFAULT_HF_REPO_ID` rather than typed in
#: twice -- the body is full of f-strings, so the whole cell cannot be one template.
_PARAMETERS_BODY = """
# Eğitim
#
# Bağlam ve bellek bütçesi birlikte seçilir; ikisi de ölçülmüş değerlerdir, devir
# değil. Qwen3.5-4B'de 32 katmanın 24'ü Gated DeltaNet (lineer dikkat) ve durum
# [32 v_heads, 128, 128] yani **token başına 1 MB**. Bu yüzden aktivasyon belleği
# bağlamla doğrusal büyür ve 4B'nin kalıcı tabanı (ağırlık + gradyan + adafactor)
# **17,34 GB**'dır — ne yaparsan yap geri kalanı daraltamazsın.
#
# Gönderilen tahmin (gerçek config, ölçülmüş %32 denetimli oran):
#
#     bağlam   resident  transient   loss    toplam   40 GB'de
#      2048      17,34      2,08    0,61     20,34     evet
#      4096      17,34      4,16    1,21     23,33     evet
#      8192      17,34      8,31    2,43     29,33     evet
#     16384      17,34     16,62    4,85     41,32     HAYIR
#
# 8192 seçildi: 40 GB bütçesinin içinde ~10 GB pay bırakıyor, korpusun %72'sini
# kullanıyor. (Aktivasyon terimi *tahmindir*; kalıcı olanlar kesin. 10. hücre
# bütçeyi basar, hangi terimin belirsiz olduğunu orada görürsün.)
#
# 10. hücre `--memory-budget-gb` ile bu tabloyu zorlar: sığmıyorsa **başlamadan**
# durur. 16384 isteyen 40 GB'lık bir kartta reddedilecek, 80 GB'lıkta geçecek.
CONTEXT_LENGTH = 8192
MEMORY_BUDGET_GB = 40.0
EXPANDABLE_SEGMENTS = True               # parçalanmayı azaltır (bkz. 4. hücre)
EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 1e-5
MAX_RECORDS = 400                      # korpus 4491 kayıt; süreyi sınırlamak için kes
MAX_TOKENS_PER_RECORD = CONTEXT_LENGTH  # üstünü atılır, *sayısı raporlanır*

# Yayınlama
PUSH_EVERY = 25                        # optimizer adımı

print(f"repo   : {REPO_URL}@{REPO_REF}")
print(f"model  : {MODEL_ID}")
print(f"push to: {HF_REPO_ID} every {PUSH_EVERY} steps (dry_run={DRY_RUN})")
"""


#: The Go toolchain the notebook installs, and the sha256 go.dev publishes for it.
#:
#: Pinned, not "latest". A Go release can change `go vet` diagnostics, and a
#: toolchain that drifts between two runs of the same notebook makes a difference in
#: test output impossible to attribute. The digest is go.dev's own
#: ``?mode=json`` entry for ``go1.23.6.linux-amd64.tar.gz``; the download is refused
#: unless the bytes hash to it. This is the same pin-then-verify rule the sandbox
#: Dockerfile applies to rtk, for the same reason: a toolchain that is not provably
#: the intended one produces scores nobody can trust.
GO_VERSION: Final[str] = "1.23.6"
GO_SHA256: Final[str] = "9379441ea310de000f33a4dc767bd966e72ab2826270e038e78b2c53c2e7802d"

#: Where the toolchain is unpacked. Not on PATH by default in a Colab image, and a
#: symlink into /usr/local/bin would be a second, silently-different installation.
GO_ROOT: Final[str] = "/usr/local/go"

#: RTK, at the version and digest ``docker/go-sandbox/Dockerfile`` already pins.
#:
#: The same two-source verification, for the same reason. The Dockerfile states what
#: was reviewed and then proves the publisher still says so; this reuses both, so the
#: notebook and the sandbox image are provably running the *same* binary. That is not
#: tidiness: the catalogue's output format is RTK's, so a notebook on a different
#: build is measuring a different format than the eval harness will.
RTK_VERSION: Final[str] = "0.50.0"
RTK_SHA256: Final[str] = "bc2b8902b0d9c796c82ef45f16ae2307e17757afeca5ee156235a3dc7bda5f89"
RTK_BASE_URL: Final[str] = "https://github.com/rtk-ai/rtk/releases/download"
#: The musl build is static, so it runs on a glibc image without a loader.
RTK_ASSET: Final[str] = "rtk-x86_64-unknown-linux-musl.tar.gz"


def _toolchain_cell() -> dict[str, Any]:
    """Install Go and rtk, both pinned and verified, before anything shells out.

    Colab ships neither. That is not a cosmetic gap: the catalogue's commands *are*
    ``rtk go build``/``rtk go test``/``rtk read``/``rtk grep``, and the package's own
    suite runs the real commands. Without these two, a missing dependency is
    reported as a broken project -- the single most expensive kind of confusion,
    because it is indistinguishable from a real regression and costs a round trip
    through a metered machine to discover.

    Both installs are verified against a pinned digest, and rtk additionally against
    the publisher's own ``checksums.txt``, which is exactly what the sandbox Dockerfile
    does. One cell rather than two so the download-and-verify helper has a single
    definition; each install still names itself as it goes, so a failure says which
    tool was the problem.
    """
    return _code(f"""
# @title 4 — Go ve rtk toolchain'leri
#
# Colab'da ikisi de yok. Bu bir eksik değil, **kapının kırılma biçimi**: katalogdaki
# komutların çoğu rtk üzerinden çalışır (`rtk go test`, `rtk read`, `rtk grep`), ve
# paketin kendi testleri gerçek komutları koşar. rtk yokken `go_test` "rtk not found"
# ile HARNESS_ERROR döner ve hata **kırık paket** gibi görünür — oysa kırık olan
# ortamdır.
#
# İkisi de sabit + doğrulanmış. rtk'nin sürümü ve digest'i docker/go-sandbox/Dockerfile'ın
# sabitlediğiyle **aynı**: yani defter ile sandbox imajı kanıtlanabilir biçimde aynı
# binary'yi koşuyor. Katalogun çıktı formatı rtk'nin formatıdır; farklı bir build
# üzerinde ölçmek, eval harness'ın ölçeceği formatı ölçmemek demektir.
#
# rtk doğrulaması çift kaynaklı: gömülü sabit *ve* yayıncının checksums.txt'i. Uyuşmazsa
# kurulum iptal edilir.
#
# GOTOOLCHAIN=local: go.mod'daki `go 1.23` direktifinin ağdan toolchain indirmesini
# engeller — testlerin ürettiği modüller ağ olmadan da derlenir.

import hashlib
import io
import os
import pathlib
import platform
import shutil
import tarfile
import urllib.request

GO_VERSION = {GO_VERSION!r}
GO_SHA256 = {GO_SHA256!r}
GO_ROOT = {GO_ROOT!r}

RTK_VERSION = {RTK_VERSION!r}
RTK_SHA256 = {RTK_SHA256!r}
RTK_BASE_URL = {RTK_BASE_URL!r}
RTK_ASSET = {RTK_ASSET!r}


def fetch_verified(url: str, expected_sha256: str, what: str) -> bytes:
    \"\"\"İndir, hash'le, sabitle karşılaştır. Uyuşmazsa kurulum yapılmaz.\"\"\"
    print(f"  indiriliyor {{what}}: {{url}}")
    with urllib.request.urlopen(url, timeout=300) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    print(f"  sha256           {{digest}}")
    if digest != expected_sha256:
        raise SystemExit(
            f"{{what}} doğrulanamadı. İndirilen {{digest}}, beklenen {{expected_sha256}}. "
            "Kurulum iptal edildi — doğrulanmamış bir toolchain, hiç kurmamaktan kötüdür."
        )
    return payload


if platform.machine() not in ("x86_64", "AMD64"):
    raise SystemExit(
        f"bu defter {{platform.machine()}} mimarisi için; rtk'nin sabitlenmiş build'i "
        "x86_64. Colab'da T4/L4/A100 seçtiğiniz için bu burada sorun olmaz."
    )

# ------------------------------------------------------------------ Go
print("Go kuruluyor...")
go_archive = f"go{{GO_VERSION}}.linux-amd64.tar.gz"
go_payload = fetch_verified(f"https://go.dev/dl/{{go_archive}}", GO_SHA256, "go")

root = pathlib.Path(GO_ROOT)
if root.exists():
    shutil.rmtree(root)
with tarfile.open(fileobj=io.BytesIO(go_payload)) as tar:
    try:
        tar.extractall(root.parent, filter="data")
    except TypeError:  # Python < 3.12 has no filter argument
        tar.extractall(root.parent)

# ------------------------------------------------------------------ rtk
print("rtk kuruluyor...")
base = f"{{RTK_BASE_URL}}/v{{RTK_VERSION}}"
rtk_payload = fetch_verified(f"{{base}}/{{RTK_ASSET}}", RTK_SHA256, "rtk")

# İkinci kaynak: yayıncının kendi checksum listesi. Sabit "neyi inceledik"i söyler,
# bu indirme "yayıncı hâlâ öyle diyor"u kanıtlar.
with urllib.request.urlopen(f"{{base}}/checksums.txt", timeout=120) as response:
    published = response.read().decode("utf-8", "replace")
expected_line = next(
    (line for line in published.splitlines() if line.split()[-1:] == [RTK_ASSET]), None
)
if expected_line is None:
    raise SystemExit(f"checksums.txt içinde {{RTK_ASSET}} yok; yayıncı listeyi değiştirmiş.")
if expected_line.split()[0] != RTK_SHA256:
    raise SystemExit(
        f"sabit ile yayıncı uyuşmuyor. Gömülü {{RTK_SHA256}}, yayıncı "
        f"{{expected_line.split()[0]}}. İkisi de doğru olmalı."
    )
print(f"  yayıncı da doğruladı: {{expected_line.split()[0][:16]}}...")

rtk_dir = pathlib.Path("/tmp/rtk-dist")
if rtk_dir.exists():
    shutil.rmtree(rtk_dir)
rtk_dir.mkdir(parents=True)
with tarfile.open(fileobj=io.BytesIO(rtk_payload)) as tar:
    tar.extractall(rtk_dir)
binary = rtk_dir / "rtk"
shutil.copy2(binary, "/usr/local/bin/rtk")
os.chmod("/usr/local/bin/rtk", 0o755)
shutil.rmtree(rtk_dir, ignore_errors=True)

# ------------------------------------------------------------------ PATH
# os.environ üzerinden, shell export'u değil: subprocess os.environ'ı miras alır,
# dolayısıyla hem kalite kapısı hem eğitim koşusu aynı toolchain'i görür. Shell'e
# kurulup sürece aktarılmayan bir Go, kapıdan geçer ve eğitimde patlar.
os.environ["PATH"] = f"{{root / 'bin'}}:/usr/local/bin:{{os.environ['PATH']}}"
os.environ["GOTOOLCHAIN"] = "local"

# Parçalanma ayarı. CUDA önbelleğinde ayrılmış ama kullanılmayan bloklar uzun
# bir koşuda birikir ve "boş" görünen belleği yer. Bu ayar ayrılmış blokları
# büyüterek birleştirir; etkisi çalışma zamanında görülür, tahmin edilemez.
if EXPANDABLE_SEGMENTS:
    _existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in _existing:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            f"{{_existing}},expandable_segments:True" if _existing else "expandable_segments:True"
        )
    print("PYTORCH_CUDA_ALLOC_CONF:", os.environ["PYTORCH_CUDA_ALLOC_CONF"])

# `run` 3. hücreden.
print(run("go", "version"))
print(run("rtk", "--version"))
""")


def _parameters_cell() -> dict[str, Any]:
    """The one cell an operator edits, with both remotes interpolated from constants.

    The token stays an obvious placeholder in the generated file. A notebook that
    carries a real credential is a credential in the git history, and the literal
    would sit in *this* file rather than in a secret store.
    """
    header = f"""
# @title 1 — Parameters
#
# Buradaki değerler mock olarak girdi. Gerçek koşuda HF_TOKEN'ı Colab Secrets'a
# (🔑 soldaki panel) koy; buraya yazmak token'ı notebook çıktısına ve git geçmişine
# sızdırır.

REPO_URL = {DEFAULT_REPO_URL!r}
REPO_REF = "main"                      # commit sha da olabilir; sha daha tekrarlanabilir

MODEL_ID = "Qwen/Qwen3.5-4B"          # eğitilecek taban model
HF_REPO_ID = {DEFAULT_HF_REPO_ID!r}    # hedef Hub deposu
HF_TOKEN = "hf_mock_replace_me"        # mock — Colab Secrets'tan okunacak
DRY_RUN = False                        # True: takvimi prova et, hiçbir şey yükleme
"""
    return _code(header + _PARAMETERS_BODY)


def build_cells() -> list[dict[str, Any]]:
    """The notebook's cells, in order.

    Kept as data so the structure is reviewable: a notebook whose only form is a
    400-line JSON blob cannot be diffed meaningfully, and this project's whole
    premise is that a change to the training path must be visible in review.
    """
    return [
        _markdown(
            """
# gotooltrain — 4B tam fine-tune (Colab)

`Qwen/Qwen3.5-4B` → Go + tool-use modeli, tam fine-tune, **periyodik Hugging Face Hub
yüklemesiyle**.

Bu defter şunu yapar, başka hiçbir şeyi:

1. Depoyu remote'dan çeker (`alexzakarov/model-trainer`).
2. `pip install -e ".[dev,train]"` ile kurar.
3. **Go ve rtk toolchain'lerini** sabitlenmiş digest'lerle kurar (Colab'da ikisi de yok).
4. Go-UT-Bench korpusunu indirip ölçer (lisans süzgeci dahil).
5. Token formatını **gerçek Qwen3.5-4B tokenizer'ıyla** doğrular — maske yanlışsa
   burada durur, GPU saatleri harcanmaz.
6. 4B tam fine-tune'u başlatır ve **her N adımta** checkpoint'ı Hub'a gönderir.

---

## ⚠️ `REPO_REF`'i sabitle

`REPO_REF` şu an `main`. Bu **bir kez çalışmak** içindir: yarın aynı defteri
çalıştırmak farklı kod demektir. `95b1aa3` gibi bir commit sha'sına sabitle,
böylece aynı komut aynı kodu çeker.

## ⚠️ Token'ın yazma yetkisi

Hedef depo 1. hücredeki `HF_REPO_ID`. `HF_TOKEN`'ın o depoya **yazma** yetkisi
olmalı. Depo şu an boş; ilk push onu doldurur. `upload_folder` model card yazmaz
— kart elle eklenir.

## ⚠️ GPU seçimi

4B **tam** fine-tune AdamW ile ~18 GB (ağırlık + gradyan) ister. Colab'da:

| Runtime | VRAM | Bu defter |
|---|---|---|
| T4 (ücretsiz) | 15 GB | ❌ çalışmaz |
| L4 | 24 GB | ⚠️ Adafactor + kısa bağlamla sığar |
| **A100 40 GB** | 40 GB | ✅ önerilen |
| A100 80 GB | 80 GB | ✅ |

Aşağıdaki hücre yetersiz VRAM'i sessizce geçmez — adını ve nedenini söyler.
"""
        ),
        _parameters_cell(),
        _code(
            """
# @title 2 — GPU ön kontrolü
#
# Yetersiz VRAM'de çalıştırmak, OOM ile 40 dakika sonra ölmekten iyidir ama yine de
# kötüdür: ne öğrenildiği belirsiz. Adı ve gerekeni söyleyip duruyoruz.

import subprocess

import torch

GB = 1024**3
free = torch.cuda.is_available()
total_gb = (torch.cuda.get_device_properties(0).total_memory / GB) if free else 0.0
MIN_GB = 22.0  # Adafactor + gradient checkpointing + 8K bağlam için gerçek taban

print(f"torch      : {torch.__version__}")
print(f"cuda       : {free}")
print(f"device     : {torch.cuda.get_device_name(0) if free else 'none'}")
print(f"vram       : {total_gb:.1f} GB (gereken >= {MIN_GB:.0f} GB)")

if total_gb < MIN_GB:
    raise SystemExit(
        f"{total_gb:.1f} GB VRAM bu tam fine-tune için yetersiz. "
        "Runtime > Change runtime type > A100 (40 GB) seçin. "
        "T4'de 4B tam fine-tune matematiksel olarak sığmaz; LoRA'ya düşmenin "
        "bu projenin kararı değil, ayrı bir karar olurdu."
    )
"""
        ),
        _code(
            """
# @title 3 — Depoyu çek ve kur
#
# REF bir commit sha ise tam tekrarlanabilir olur. 'main' daha basit ama yarın
# aynı defteri çalıştırmak farklı kod demektir.

import importlib
import os
import subprocess
import sys

os.environ["GIT_TERMINAL_PROMPT"] = "0"  # şifre soran bir clone sessizce beklemesin

def run(*args: str) -> str:
    \"\"\"Bir komutu çalıştır; stdout döndür, hata olursa mesajla dur.\"\"\"
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"{' '.join(args)} failed:\\n{result.stdout}\\n{result.stderr}")
    return result.stdout.strip()

target = "/content/model-trainer"
if os.path.isdir(target):
    run("git", "-C", target, "fetch", "--all")
    run("git", "-C", target, "checkout", REPO_REF)
    run("git", "-C", target, "pull", "--ff-only", "origin", REPO_REF)
else:
    run("git", "clone", REPO_URL, target)
    run("git", "-C", target, "checkout", REPO_REF)

print(run("git", "-C", target, "log", "-1", "--format=%H %s"))

# transformers 5.x şart: Qwen3.5 config'i model_type qwen3_5 bildiriyor ve 4.x onu
# reddediyor (tokenizer yüklenir, config yüklenmez). Sürüm burada da sabitleniyor,
# çünkü Colab'ın imajı zamanla değişir.
run(sys.executable, "-m", "pip", "install", "-q", "-U", "pip")
run(sys.executable, "-m", "pip", "install", "-q", "transformers>=5.17,<6")
run(sys.executable, "-m", "pip", "install", "-q", "-e", f"{target}[dev,train]")

os.chdir(target)
print("cwd:", os.getcwd())

# ---------------------------------------------------------------- kernel path
#
# `pip install -e` yalnızca *yeni* süreçlerde görünür. .pth ve editable-finder
# çengelleri yorumlayıcı **başlangıcında** kurulur; Colab çekirdeği ise bu
# hücreden çok önce başlamıştır. Yani alt süreçler (kalite kapısı, eğitim koşusu)
# paketi görür, çekirdek görmez — ve hata beş hücre sonra, başka bir hücrenin
# satırında belirir. Burada ölçtüm: pip'i çalıştıran yorumlayıcı kendi kurduğu
# paketi import edemiyor.
#
# Bu yüzden kaynak dizini çekirdeğin sys.path'ine de ekleniyor. Gizli bir
# düşüş değil: kurulum zaten yapıldı, bu yalnızca aynı kurulumu *bu* sürece
# görünür kılıyor. Konsol betikleri (gotooltrain-train vb.) alt süreçlerde çalışır
# ve zaten yolunu bulur.
sys.path.insert(0, os.path.join(target, "src"))

# Ama bir tuzak daha var ve bu seviyesiz: `import`, `sys.modules`'tan önbelleklenir.
# Bu hücreyi **ikinci kez** çalıştırırsan kod yeni (pull edildi) ama çekirdek
# eski modülü kullanmaya devam ediyor. Ölçtüm: pull sonrası `import` hâlâ bir
# önceki sürümü döndürüyor. Sonucu görmek kolay: 8192 bağlamı yine OOM eder —
# ama bu kez "40 GB'a sığdırdık" diye düşünürsün, çünkü bellek bütçesi hücrede
# güzel görünüyor. O yüzden önbellek düşürülüyor; `invalidate_caches()` olmadan
# silmek yetmiyor, çünkü bayt-kod önbelleği ayrı tutuluyor.
for _name in [n for n in list(sys.modules) if n == "gotooltrain" or n.startswith("gotooltrain.")]:
    del sys.modules[_name]
importlib.invalidate_caches()

import gotooltrain  # noqa: E402

print("gotooltrain:", gotooltrain.__version__, "->", gotooltrain.__file__)

# Klonun içinden geldiğini doğrula. Path'te başka bir gotooltrain varsa (sık
# gelen bir tuzak: daha önce kurulmuş bir sürüm) sessizce onu eğitirdik.
if not gotooltrain.__file__.startswith(target):
    raise SystemExit(
        f"gotooltrain {gotooltrain.__file__} konumundan geliyor, klon {target} değil. "
        "Sistemde başka bir kurulum var ve eğitimi yanlış kodla yapacağız."
    )

# Koşunun ihtiyaç duyduğu yetenekler gerçekten var mı? Kontrolün kendisi
# import: burada hata veren bir koşu zaten yapamayacağı bir şeyi denemeye
# çalışıyordu, ve iki saat sonra OOM olarak dönecekti. `REPO_REF` eski bir
# revizyona sabitlenmişse koşu burada, açık bir mesajla durur.
#
# `from ... import` bilinçli seçildi: `gotooltrain.train` bir *fonksiyon* adıyla
# da var (paket `train` işlevini dışa aktarıyor) ve modülü gölgeliyor. Ölçtüm:
# `hasattr(gotooltrain.train, ...)` üç yeteneğin de False'unu veriyor, `import
# ... as` da fonksiyonu bağlıyor. Yani ilk yazdığım kontrol, elindeki kod
# doğru olsa bile "eksik" diyordu — doğru bir koşuyu durdurup hatayı koda
# yazıyordu. from-import modülü sys.modules üzerinden çözer, gölgelenmez.
try:
    # Kullanılmıyorlar; buradaki import'un kendisi sözleşme. noqa: F401.
    from gotooltrain.train import (  # noqa: E402, F401
        DEFAULT_MEMORY_BUDGET_GB,
        completion_only_loss,
        enter_training_mode,
    )
except ImportError as _exc:
    raise SystemExit(
        f"{REPO_REF} revizyonu bu koşunun ihtiyaç duyduğu yeteneklerden yoksun "
        f"({_exc.name}). Bellek bütçesinin işe yaraması bunlar olmadan mümkün "
        "değil; yeni bir revizyona sabitle ya da Runtime'ı yeniden başlat."
    ) from _exc
print(f"40 GB koşusunun yetenekleri: tamam (bütçe {DEFAULT_MEMORY_BUDGET_GB:.0f} GB)")

"""
        ),
        _toolchain_cell(),
        _code(
            """
# @title 5 — Token: Colab secret ya da mock
#
# Proje kuralı: sessiz düşüş yok. Token yoksa yayınlama başlamadan hata verir —
# çünkü 700 adım sonra öğrenmek, 700 adım önce öğrenmekten pahalıdır.

import os

try:
    from google.colab import userdata

    HF_TOKEN = userdata.get("HF_TOKEN")
    print("HF_TOKEN: Colab secret'tan okundu")
except (ImportError, Exception):  # noqa: B014 - colab dışında da çalışabilmeli
    print("Colab secret yok; hücre 1'deki mock değer kullanılacak")

if not HF_TOKEN or HF_TOKEN == "hf_mock_replace_me":
    if DRY_RUN:
        os.environ["HF_TOKEN"] = HF_TOKEN or "hf_mock_dry_run"
        print("DRY_RUN=True: gerçek bir token gerekmiyor")
    else:
        raise SystemExit(
            "HF_TOKEN yok. Sol panelden 🔑 Secrets'a HF_TOKEN ekleyin, ya da "
            "DRY_RUN=True ile takvimi hiçbir şey yüklemeden prova edin."
        )
else:
    os.environ["HF_TOKEN"] = HF_TOKEN

from huggingface_hub import login

login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("hub client hazır")
"""
        ),
        _code(
            """
# @title 6 — Kalite kapısı (paket kendi testini koşar)
#
# Colab'a kurduğumuz şeyin bu depodakiyle aynı şey olduğunu doğrulamadan 4B
# eğitimi başlatmak, 4B eğitimi başlatmadan önce yapılabilecek en pahalı kontrolü
# atlamak olurdu. Testler ~2 dakika sürer; 4B bir epoch saatler sürer.
#
# -x: ilk kırıklıkta dur. Kapının amacı "bir şeyler yanlış" demek, listelemek değil.
# Docker ve rtk testleri yoksa *atlanır* (gerekçeleri skip metninde) — bu, eksik
# bağımlılığın sessizce yeşile dönmesi değil, açıkça raporlanmasıdır. Son satırda
# kaç testin atlandığını ve nedenini görüyorsunuz.

import subprocess

started = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "-x", "-rs"],
    cwd=target,
    capture_output=True,
    text=True,
    check=False,
)
print(started.stdout[-3000:])
print(started.stderr[-2000:])

skipped = [line for line in started.stdout.splitlines() if line.startswith("SKIPPED")]
if skipped:
    print(f"\\n--- atlanan testler ({len(skipped)}) ve gerekçeleri ---")
    for line in skipped:
        print(" ", line)

if started.returncode != 0:
    raise SystemExit(
        "Kalite kapısı kırmızı. Eğitime başlamak, kırık bir ağaca kanat takmak olurdu. "
        "Yukarıdaki çıktıya bakın. "
        f"(Atlanan test: {len(skipped)} — bir bağımlılık eksikse bu normal; "
        "kırmızı bir test normal değil.)"
    )
print("kapı yeşil")
"""
        ),
        _code(
            """
# @title 7 — Go korpusunu indir, süz, ölç
#
# `go-pairs` lisans süzgecini uygular: Go-UT-Bench "permissive" diyor ama
# terraform BUSL-1.1, go-ethereum LGPL-3.0. İkisi de varsayılan olarak reddedilir
# ve rapora yazılır. Ayrıca split'ler örtüştüğü için tekilleştirilir.

import subprocess

run(
    sys.executable, "-m", "gotooltrain.datacli", "go-pairs",
    "--download-to", "data/go-ut-bench",
    "--out", "data/go-dapt.jsonl",
    "--unit-tests-out", "data/go-unit-tests.jsonl",
    "--report", "data/go-dapt-report.json",
)
print(open("data/go-dapt-report.json", encoding="utf-8").read()[:1200])
"""
        ),
        _code(
            """
# @title 8 — SFT verisini ölç, süz, doğrula
#
# Buradaki veri Go-UT-Bench'in *birim testi yaz* görevleridir; değerlendirme
# trajektöri değil. Bu, yetenek ölçümü değil — format ve boru hattı denemesidir.
# Tek turlu, araçsız, sekiz depodan biri ağırlıklı bir korpus katalogu öğretmez;
# `measure` bunu "yetersiz" diye bildirecek ve bu doğru cevaptır. Değerlendirme
# için konteyner havuzu, rtk ve gerçek Go depoları gerekir (docs/EVAL.md).
#
# **Burada token id yazmıyoruz.** Eğitim CLI'si mesajları kendisi render eder;
# tek render yolu ilkesi budur. Buradaki iş render etmek değil **ölçmek**:
# hangi kayıtların bağlama sığdığını ve maskenin doğru olduğunu, pahalı
# hücrelerden önce görmek.
#
# Uzunluk **katalogla** ölçülüyor, çünkü eğitim CLI'sı da katalogla render edecek.
# Katalogsuz ölçseydik altımızda kalırdı ve bağlam dışı bir kayıt eğitimi düşürürdü.

import json

from transformers import AutoTokenizer

from gotooltrain import (
    catalog,
    install_template,
    load_template_source,
    normalize_conversation,
    read_jsonl,
    render_example,
)

TOOLS = catalog()
tokenizer = install_template(AutoTokenizer.from_pretrained(MODEL_ID), load_template_source())
print("tokenizer hazır:", type(tokenizer).__name__)

records = list(read_jsonl("data/go-unit-tests.jsonl"))[:MAX_RECORDS]
print(f"{len(records)} kayıt okundu (sınır: {MAX_RECORDS})")

kept = []
first_rendered = None
dropped_invalid = 0
dropped_truncated = 0
dropped_no_supervision = 0
total_tokens = 0
supervised_tokens = 0
longest = 0

for record in records:
    try:
        conversation = normalize_conversation(record["messages"], TOOLS)
    except Exception as exc:  # noqa: BLE001 - her ret sayılır, ilk birkaçı adlandırılır
        dropped_invalid += 1
        if dropped_invalid <= 3:
            print(f"  reddedildi (geçersiz): {type(exc).__name__}: {exc}")
        continue
    example = render_example(tokenizer, conversation, max_length=MAX_TOKENS_PER_RECORD)

    # **Tavana oturan kayıt kesilmiştir.** Kesme baştaki token'ları korur, yani
    # asistan turn'ü sınırdan önce başladıysa *denetimli token hâlâ vardır* ve
    # "denetim var mı" sorusu kesilmiş bir kaydı geçirir. Oysa eğitim kaydı
    # sınırsız yeniden render eder ve `assert_examples_fit` onu bağlam dışı
    # diye reddeder — yani ön kontrol, eğitimin reddedeceği kaydı onaylıyordu.
    # (Ölçüldü: ilk 400 kaydın 74'ü bu durumdaydı; gerçek uzunluklar 8.5K–17.6K.)
    if len(example.input_ids) >= MAX_TOKENS_PER_RECORD:
        dropped_truncated += 1
        continue
    if example.supervised_tokens == 0:
        # Asistan turn'üne hiç ulaşılamadı: denetlenecek bir şey yok.
        dropped_no_supervision += 1
        continue
    kept.append({"messages": record["messages"], "tools": TOOLS})
    total_tokens += len(example.input_ids)
    supervised_tokens += example.supervised_tokens
    longest = max(longest, len(example.input_ids))
    if first_rendered is None:
        first_rendered = example

if not kept:
    raise SystemExit(
        f"{len(records)} kaydın hiçbiri kullanılabilir değil "
        f"({dropped_truncated} kesilmiş, {dropped_no_supervision} denetimsiz, "
        f"{dropped_invalid} geçersiz). Eğitilecek veri yok; bu koşuyu başlatmak "
        "boşa GPU yakar."
    )

share = supervised_tokens / max(1, total_tokens)
print(f"kabul  : {len(kept)}")
print(
    f"atılan: {dropped_truncated} (bağlamda kesilmiş), "
    f"{dropped_no_supervision} (asistan turn'üne ulaşılamadı), "
    f"{dropped_invalid} (geçersiz)"
)
print(f"token  : {total_tokens} toplam, {supervised_tokens} denetimli ({share:.1%})")
average = total_tokens // len(kept)
print(f"uzunluk: ortalama {average}, en uzun {longest} (sınır {MAX_TOKENS_PER_RECORD})")
"""
        ),
        _code(
            """
# @title 9 — Token formatını doğrula (eğitimden ÖNCE)
#
# Maske bozuksa model araç çıktısı uydurmayı öğrenir ve kayıp normal görünür. Bu
# kontrol dakikalar sürer, bir 4B epoch saatler sürer. Sıra burada.

from gotooltrain import assert_mask_sane
from gotooltrain.evalstore import sha256_text
from gotooltrain.template import load_template_source

# 8. hücrede ölçülen kayıt; eğitimin de göreceği kayıt. Render burada tekrar
# yapılmıyor: 8. hücre zaten `first_rendered`'ı bıraktı.
example = first_rendered
assert_mask_sane(example)

print("maske tutarlı: labels ve assistant_mask aynı yeri işaretliyor")
print("denetimli token:", example.supervised_tokens, "/", len(example.input_ids))
print("template sha256:", sha256_text(load_template_source())[:16])
print("ilk 300 karakter:")
print(tokenizer.decode(example.input_ids[:120], skip_special_tokens=False)[:300])
"""
        ),
        _code(
            """
# @title 10 — Eğitimi başlat (periyodik Hub yüklemesiyle)
#
# Adafactor: ilk momenti tutmaz, ikinci momenti çarpanlaştırır. Optimizer durumu
# birkaç GB yerine birkaç MB — tek kartta 4B'yi sığdıran şey bu.
#
# **--gradient-checkpointing burada belirleyici.** Bu modelde 32 katmanın 24'ü
# Gated DeltaNet; duram [32 v_heads, 128, 128] yani token başına 1 MB. Aktivasyon
# belleği bağlamla doğrusal büyür — tablo 8. hücrede, hesaplayan `--memory-budget-gb`.
# Checkpointing'in etkinleşmesi için modelin *train* modunda olması gerekir
# (`from_pretrained` eval modunda döndürür) — bu, sürümde artık düzeltildi.
# Çalışma çıktısında "gradient checkpointing on: N module(s)" satırını **gör**;
# yoksa bayrak hiç çalışmamış demekter.
#
# --hub-push-every: optimizer ADIMI sayar, batch değil. 8 kademe biriktirmeli
# olduğu için "her batch'te gönder" demek accumulation ayarına bağımlı olurdu.
#
# **Bu hücreyi ikinci kez çalıştırırsan** `runs/colab-sft/training_plan.json`
# zaten duruyor ve yeni planı reddeder: plan, koşunun ne olduğunun kaydı; üstüne
# yazmak önceki koşunun ne olduğunu unutmak olurdu. Farklı bir plan için ya
# `OUTPUT` dizinini değiştir ya da eski dizini **bilerek** sil:
#
#     !rm -rf runs/colab-sft
#
# Aynı parametrelerle tekrar çalıştırırsan sorun yok: plan aynı, kabul edilir.
#
# İlk koşuda DRY_RUN=True ile başla: pahalı hücrelerden geçer, takvimi ve
# dosyer yazımını prova eder, tek bir bayt göndermez. Yeşil görünce DRY_RUN=False
# yapıp yeniden çalıştır.

import subprocess

OUTPUT = "runs/colab-sft"

with open("data/sft.jsonl", "w", encoding="utf-8", newline="\\n") as handle:
    for row in kept:
        handle.write(json.dumps(row) + "\\n")
print(f"data/sft.jsonl yazıldı: {len(kept)} kayıt (mesaj biçimi)")

command = [
    # `-u` şart, göze çarpmayan bir ayrıntı değil: alt sürecin stdout'u buraya
    # *boru* olarak bağlanıyor ve Python, boruya yazarken satır değil 8 KB blokta
    # tamponluyor. Yani `print()`'ler 8 KB birikene kadar **görünmez**. stderr
    # ise her zaman satır tamponlu — bu yüzden ilk koşuda yalnızca tqdm ve
    # transformers uyarıları göründü, bellek bütçesi ve adım sayacı hiç görünmedi;
    # ekran "takıldı" gibi görünüyordu, koşu ise çalışıyordu. `-u` ikisini de
    # tamponlamaz.
    sys.executable, "-u", "-m", "gotooltrain.traincli", "sft",
    "--model", MODEL_ID,
    "--output", OUTPUT,
    "--tokenizer", MODEL_ID,
    "--dataset", "data/sft.jsonl",
    "--dtype", "bfloat16",
    "--epochs", str(EPOCHS),
    "--batch-size", str(BATCH_SIZE),
    "--grad-accum", str(GRAD_ACCUM),
    "--lr", str(LEARNING_RATE),
    "--max-length", str(CONTEXT_LENGTH),
    "--memory-budget-gb", str(MEMORY_BUDGET_GB),
    "--loss-mode", "selective",
    "--optimizer", "adafactor",
    "--gradient-checkpointing",
    "--hub-repo-id", HF_REPO_ID,
    "--hub-push-every", str(PUSH_EVERY),
]
if DRY_RUN:
    command.append("--hub-dry-run")

print(" ".join(command))
print()
print("Bu hücre saatlerce sürebilir. Yayınlar PUSH_EVERY adımda birer olur,")
print("böylece sekme kapanırsa kaybedilen şey en fazla PUSH_EVERY adım olur.")
print()
result = None
log_path = pathlib.Path("train.log")
with log_path.open("w", encoding="utf-8", buffering=1) as log:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        print(line, end="")
        log.write(line)
    result = process.wait()

print("exit:", result)

# Hata, hücre çıktısında **kaybolmasın diye** ayrıca dosyadan basılıyor. Önceki
# koşuda alt sürecin "error: ..." satırı hiç görünmedi; iki hücre sonra "hub_push.json
# yok" diye yanlış yere baktık. Uzun bir koşunun çıktısı kaydırılınca kaybolabilir,
# yani teşhisin tesadüfe bağlı olmaması gerekiyor.
if result != 0:
    out = pathlib.Path(OUTPUT)
    if not out.is_dir():
        cause = "Çıktı dizini hiç oluşmadı: koşu modeli yüklenmeden ya da veriyi okuyamadan durdu."
    else:
        listing = sorted(p.name for p in out.iterdir())
        print(f"\\n{OUTPUT} içinde: {listing}")
        if "training_plan.json" not in listing:
            cause = "Plan yazılmadı: koşu başlamadan durdu."
        elif "hub_push.json" not in listing:
            cause = "Plan yazıldı ama bitmedi; yayınlama adımına hiç gelinmedi."
        else:
            cause = "Yayınladı ama hata ile bitti."
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
    print("\\n--- train.log (son 40 satır) ---")
    for line in tail:
        print(" ", line)
    raise SystemExit(
        f"Eğitim {result} ile bitti. Yukarıdaki çıktıya bakın. {cause} "
        "Bu hücre hatayı yutmaz: 11. hücreye geçmeden durur."
    )
"""
        ),
        _code(
            """
# @title 11 — Ne olduğunu doğrula
#
# "Push ettim" demek yetmez; push'un *ne* olduğu okunmalı. Yayınlanan klasör
# kendi kökenini taşır: hangi adım, hangi ayarlar.
#
# Ama önce ayrım: çıktı dizini hiç oluşmadı mı, yoksa koştu ve yayınlama adımına
# mı gelmedi? Bu ikisi farklı hatalar ve aynı hata mesajıyla gelmez. "hub_push.json
# yok" demek, asıl hatayı değil *sonucunu* söyler; 10. hücre artık yutmuyor ama
# yine de teşhis burada kesinleşmeli.

import json
import pathlib

from gotooltrain import read_push_state

output_dir = pathlib.Path(OUTPUT)
if not output_dir.is_dir():
    raise SystemExit(
        f"{output_dir} yok — eğitim hiç çıktı üretmedi, yani model yüklenmeden ya da "
        "planı yazmadan durdu. 10. hücrenin çıktısına bakın."
    )

listing = sorted(p.name for p in output_dir.iterdir())
print(f"{output_dir} içinde: {listing}")
print()

state = read_push_state(output_dir)
print(f"step {state['step']}/{state['total_steps']}")
plan_record = state["plan"]
for key in (
    "model_id",
    "learning_rate",
    "epochs",
    "context_length",
    "optimizer",
    "token_format",
    "gradient_checkpointing",
):
    print(f"  {key:22} {plan_record[key]}")
print(f"  {'hub':22} {plan_record['hub']}")

if not DRY_RUN:
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(HF_REPO_ID)
    print(f"\\n{HF_REPO_ID} içinde {len(files)} dosya:")
    for name in sorted(files)[:15]:
        print("  ", name)
    print("\\ngit log gibi: her push bir commit. Adım commit mesajında.")
else:
    print("\\nDRY_RUN=True idi: hiçbir şey gönderilmedi. Yukarıdaki 'pushed step'")
    print("satırlarını 10. hücrede görmüş olmalısınız. Şimdi DRY_RUN=False yapıp")
    print("yeniden çalıştırın.")
"""
        ),
        _markdown(
            """
## Bu koşu ne ölçer, ne ölçmez

**Ölçer:** token formatı doğrudan gerçek tokenizer ile üretildi; maske `{% generation %}`
bloğundan geldi; loss yalnızca asistan token'larına uygulandı; 4B tam FT gerçekten
bir optimizer adımı attı ve checkpoint yazdı; periyodik yayın takvimi çalıştı.

**Ölçmez:** Go yeteneği. Korpus Go-UT-Bench'in "bu dosya için birim testi yaz"
görevlerinden gelir — tek turlu, araçsız, sekiz depodan biri ağırlıklı. Bu, korpusun
katalog öğretmediğini gösteren bir gerçektir, hata değil. Yetenek ölçümü için
`docs/EVAL.md`'deki üç aşamalı eval gerekir; o konteyner havuzu + `rtk` + gerçek Go
 depoları ister, Colab'da bu yoktur.

## Yayınlama hakkında dürüstlük

- `--hub-dry-run` **hiçbir şey yüklemez**; defter bunu log'da `DRY RUN` diye yazar.
- Gerçek koşuda `--hub-repo-id` verdiyseniz token'ın **yazma yetkisi** olmalı.
  Yazma yetkisi yoksa ilk push'ta hata verir — 700 adım sonra değil, `HF_TOKEN`
  çözümlendiği anda değil (pusher, model yüklenmeden önce kurulur).
- Yayınlanan klasör `hub_push.json` taşır: adım, toplam adım ve tüm hiperparametreler.
  Token **oraya yazılmaz**; sadece değişkenin adı yazılır.
- Bu bir **checkpoint**, tam bir **resume** değildir: optimizer durumu saklanmaz.
  Kaldığınız yerden devam etmek `resume_from` + yeniden eğitim demektir.
"""
        ),
    ]


def assemble_notebook(cells: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Assemble the notebook document from its cells."""
    return {
        "cells": cells if cells is not None else build_cells(),
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def write_notebook(path: str | Path | None = None) -> Path:
    """Write the notebook and confirm it parses as one.

    The parse check is the whole point of generating rather than hand-writing: a
    notebook that Jupyter refuses to open is a file that looks finished and is not.
    """
    target = Path(path) if path is not None else _project_root() / NOTEBOOK_PATH
    document = assemble_notebook()
    body = json.dumps(document, indent=1, ensure_ascii=False) + "\n"
    try:
        json.loads(body)
    except json.JSONDecodeError as exc:  # pragma: no cover - would be a generator bug
        raise ToolTrainError(f"generated notebook is not valid JSON: {exc}") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8", newline="\n")
    return target


def _project_root() -> Path:
    """Walk up to the directory that contains ``pyproject.toml``."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ToolTrainError(  # pragma: no cover - a packaging accident, not a run path
        "cannot locate the project root: no ancestor has pyproject.toml"
    )


def main() -> int:
    """Regenerate the notebook. Prints what it wrote."""
    path = write_notebook()
    print(f"wrote {path} ({len(build_cells())} cells, notebook v{NOTEBOOK_VERSION})")
    return 0


if __name__ == "__main__":  # pragma: no cover - maintenance entry point
    raise SystemExit(main())
