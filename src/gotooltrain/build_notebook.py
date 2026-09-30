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

# Kaldığın yerden devam?
#
# Boş bırakırsan **taban modelden** başlarsın: Qwen/Qwen3.5-4B, sıfırdan.
# Bu, ilk koşu için doğru olan.
#
# Hugging Face'te bu projeden çıkmış bir checkpoint varsa buraya o repo id'sini yaz
# (yukarıdaki HF_REPO_ID ile aynısı), ağırlıklar oradan yüklenir. Ama neyi geri
# getirdiğini bil: yayınlanan klasör yalnızca `save_pretrained` çıktısıdır —
# ağırlık ve tokenizer. **Optimizer durumu, scheduler, adım sayacı ve push
# geçmişi saklanmaz.** Yani:
#
#   * Adafactor'ın momentleri sıfırdan başlar → ilk birkaç adımın gradyanı
#     daha büyük, bir an için geri gidebilirsin.
#   * Adım sayacı 1'den başlar → daha önce gördüğü örnekleri **ikinci kez** görürsün.
#   * Yayın geçmişi boş sayılır → ilk push yeni sayılır.
#
# Bu bir "kaldığı yerden devam" değil, "o ağırlıklarla yeniden eğitim".
# Gerçek devam için optimizer durumu da saklanmalı; şu an saklanmıyor.
#
# DİKKAT: boş bırakırsan ve repoda zaten bir checkpoint varsa, ilk push'ta (25.
# adım) o checkpoint'in üzerine **yazılır** — 25 adım ilerlemiş taban model
# göndereceksin. Geri dönüşü olmayan bir hata. Varsayılan "auto" bunu kendiliğinden
# çözer: 5. hücre repoyu sorar, ağırlık varsa devam eder, yoksa sıfırdan başlar.
# Yine de yazılıyor, çünkü bu hücrenin ne yapacağını okumadan çalıştırmak da
# bir tercih, ve 12. hücre ne gönderileceğini yine de söyler.

RESUME_FROM = "auto"   # "auto" = repoda checkpoint varsa devam et | "" = bilerek
                       # sıfırdan | bir repo id ya da revizyon = elle seç

# Tercih optimizasyonu (12-13. hücreler) için her görevin kaç kez deneneceği.
# 1 OLABİLMEZ: tercih, birinin geçip birinin kaldığı iki denemeden doğar. Tek
# denemede "başarılı/başarısız" çifti yoktur, `preferences` boş döner.
N_SAMPLES = 4

# --- boru hattı ayarları ---------------------------------------------------
#
# Bunlar 2. hücredeki tek komuta gider; sıra ve koşullar `pipeline.py` içinde.

OPTIMIZER = "adafactor"
SFT_OUTPUT = "runs/colab-sft"

# Tercih aşaması, görev dosyası (`data/eval/tasks.jsonl`) ve ornekleme yapacak
# çalışan bir model gerektirir. O katman kurulana kadar False: atlanan aşama
# *atlandığını* söyler, sahte bir başarı üretmez.
RUN_PREFERENCE_STAGE = False

# Kalite kapısı her koşuda yeniden kurulmaz; dakikalar verir ve sonucu
# değiştirmez. Kapatmak bir tercih olduğu için burada adı var.
RUN_QUALITY_GATE = True

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
print(f"resume : {RESUME_FROM or 'yok - taban modelden sıfırdan'}")
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
# @title 3 — Go ve rtk toolchain'leri
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

# ---------------------------------------------------------------- rtk kurulumu
#
# rtk kurulu ama başlatılmamışsa her komuta "[rtk] /!\\ No hook installed" uyarısı
# basar. Bu, **paketin kendi testlerinden birini kırar**: başarılı bir `go build`
# sessiz olmalı ve uyarı yazıyor. Yani kurulum eksik değil eksiksiz sayılıyordu ve
# kalite kapısı Colab'da da düşüyordu — hata modelden değil, iki satır eksik
# kurulumdan geliyordu. Sandbox imajı bunu yapıyor, o yüzden orada görünmüyor.
_initialised = subprocess.run(
    ["rtk", "init", "-g"], capture_output=True, text=True, check=False
)
print("rtk init -g:", _initialised.returncode)
if _initialised.stdout.strip():
    print(" ", _initialised.stdout.strip()[:400])

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
# @title 2 — Depoyu çek ve kur
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
# @title 4 — Tüm boru hattı, tek hucrede
#
# Buradaki tek hucre butun boru hattini calistiriyor: kalite kapisi, korpus, veri
# olcumu, SFT, degerlendirme, tercih ciftleri, DPO. Sira, kosullar ve hata
# durumlari `src/gotooltrain/pipeline.py` icinde; burada yalnizca ayarlar var.
#
# Neden tek hucre: bir defteri on bir hucreye bolmek, sirayi **okuyan kisinin
# hafizasina** yaziyor. Bir hucre dusunce sonrakiler yine calisiyor ve hangi
# asamanin gercekten gerceklestigi belirsizlesiyor. Burada her asamanin adi bastan
# yazilir, ciktisi canli akar ve **bir asama basarisiz olursa boru hatti durur** --
# yarisi calismis bir kosu, tamamlanmis gibi gorunmez.
#
# Her asamanin logu `runs/pipeline/<asama>.log` altinda. Cikti kaydirilinca teshis
# kaybolmaz; hata olursa o asamanin son satirlari hucrede basilir.
#
# Once `--dry-run` ile bir dene: komutlar yazilir, hicbir sey calismaz.
#
# Calisma suresi: kalite kapisi + korpus + SFT saatler; degerlendirme ve DPO bunlarin
# ustune eklenir. Yayinlar PUSH_EVERY adimda birer olur.

import json
import pathlib
import subprocess
import sys

command = [
    sys.executable, "-u", "-m", "gotooltrain.pipeline",
    "--model", MODEL_ID,
    "--context-length", str(CONTEXT_LENGTH),
    "--memory-budget-gb", str(MEMORY_BUDGET_GB),
    "--optimizer", OPTIMIZER,
    "--epochs", str(EPOCHS),
    "--batch-size", str(BATCH_SIZE),
    "--grad-accum", str(GRAD_ACCUM),
    "--learning-rate", str(LEARNING_RATE),
    "--max-records", str(MAX_RECORDS),
    "--sft-output", SFT_OUTPUT,
]

# Tercih zinciri, degerlendirme icin bir gorev dosyasi ve ornekleme yapacak calisan
# bir model ister. Yoksa bu asamalar **acikca** atlanir; sahte bir basari uretilmez.
if not RUN_PREFERENCE_STAGE:
    command += ["--no-eval", "--no-dpo"]

# Kalite kapisi her kosuda yeniden kurulmaz; tekrar etmek dakikalar verir ve
# sonucu degistirmez. Kapatmak bir tercih, sessizce yapilmaz.
if not RUN_QUALITY_GATE:
    command.append("--no-gate")

# Tek bir `--extra-sft`. argparse REMAINDER, ilk göründüğü yerden sonrasını alır:
# iki kez geçince ikincisi ilkincinin içine düşer ve eğitim komutu `--extra-sft`
# diye bilinmeyen bir bayrak görür. Buradaki bütün sıfırda biten bayraklar tek listeye
# toplanıyor.
passthrough = []
if RESUME_FROM:
    passthrough += ["--resume-from", RESUME_FROM]

if DRY_RUN:
    command.append("--dry-run")
else:
    # Kuru koşuda hiçbir şey yüklenmez, bu yüzden yayın bayraklari da gonderilmez.
    passthrough += ["--hub-repo-id", HF_REPO_ID, "--hub-push-every", str(PUSH_EVERY)]
if passthrough:
    command += ["--extra-sft", *passthrough]

print(" ".join(command))
print()

result = subprocess.run(command, check=False)
print()
print("Boru hatti bitti. Ozet ve loglar: runs/pipeline/")
if result.returncode:
    # Boru hatti hangi asamada dustugunu `runs/pipeline/last_failure.json`'a yazar.
    # Burada tekrar okunuyor: cikti binlerce satir olabilir ve okuyan kisi o asamayi
    # bulmak icin geri kaydirmak zorunda kalmamali. Kayit yoksa dosya sistemi
    # sorunu demektir; bu yuzden ayri bir mesajla soyleniyor.
    failure = pathlib.Path("runs/pipeline/last_failure.json")
    if failure.is_file():
        info = json.loads(failure.read_text(encoding="utf-8"))
        stage = info["stage"]
        log = pathlib.Path("runs/pipeline") / f"{stage}.log"
        print(f"\\nDURAN ASAMA: {stage} (cikis kodu {info['exit_code']})")
        print(f"log: {log}")
        print("\\n--- son 25 satir ---")
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]:
            print("  ", line)
    else:
        print(f"\\nDURAN ASAMA: bilinmiyor; {failure} yazilmadi.")
    raise SystemExit(
        f"Boru hatti {result.returncode} ile durdu. Bu hucre hatayi yutmaz; "
        "asama kendi loguna yaziliyor ve hicbir sey gonderilmedi."
    )
"""
        ),
        _markdown(
            """
## Bu koşu ne ölçer, ne ölçmez

**Ölçer:** token formatı doğrudan gerçek tokenizer ile üretildi; maske `{% generation %}`
bloğundan geldi; loss yalnızca asistan token'larına uygulandı; 4B tam FT gerçekten
bir optimizer adımı attı ve checkpoint yazdı; periyodik yayın takvimi çalıştı.

**Ölçmez:** Go yeteneği. Korpus Go-UT-Bench'in "bu dosya için birim testi yaz"
görevlerinden gelir - tek turlu, araçsız, sekiz depodan biri ağırlıklı. Bu, korpusun
katalog öretmediğini gösteren bir gerçektir, hata değil.

12-13. hücreler bir tercih aşaması sunar, ama **görev dosyası olmadan durur** ve bunu
söyler. Bunun sebebi yetenek eksikliği değil, girdi eksikliği: tercih çifti ancak
bir görevin `verification` komutu çalıştırılıp bir deneme geçerken bir başkası
kalırken doğar. Go-UT-Bench `(depo, dosya, kod)` verir; o katman
`src/gotooltrain/tasks.py`'nin işidir; mimari karar `docs/EVAL.md`'de. Sıfır çift
üretip "DPO çalıştı" demek, hiç karşılaştırma yapılmamışken başarı göstermek
olurdu — o yüzden hücre adıyla durur.

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
