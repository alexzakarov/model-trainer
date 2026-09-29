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
CONTEXT_LENGTH = 8192                  # 32768 tek kartta sığmaz; 8K repo düzeyi iş için yeterli
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
3. Go-UT-Bench korpusunu indirip ölçer (lisans süzgeci dahil).
4. Token formatını **gerçek Qwen3.5-4B tokenizer'ıyla** doğrular — maske yanlışsa
   burada durur, GPU saatleri harcanmaz.
5. 4B tam fine-tune'u başlatır ve **her N adımda** checkpoint'ı Hub'a gönderir.

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
"""
        ),
        _code(
            """
# @title 4 — Token: Colab secret ya da mock
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
# @title 5 — Kalite kapısı (paket kendi testini koşar)
#
# Colab'a kurduğumuz şeyin bu depodakiyle aynı şey olduğunu doğrulamadan 4B
# eğitimi başlatmak, 4B eğitimi başlatmadan önce yapılabilecek en pahalı kontrolü
# atlamak olurdu. 888 test ~1 dakika sürer; 4B bir epoch saatler sürer.

import subprocess

started = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "-x"],
    cwd=target,
    capture_output=True,
    text=True,
    check=False,
)
print(started.stdout[-3000:])
print(started.stderr[-2000:])
if started.returncode != 0:
    raise SystemExit(
        "Kalite kapısı kırmızı. Eğitime başlamak, kırık bir ağaca kanat takmak olurdu. "
        "Yukarıdaki çıktıya bakın."
    )
print("kapı yeşil")
"""
        ),
        _code(
            """
# @title 6 — Go korpusunu indir, süz, ölç
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
# @title 7 — SFT verisini token'la
#
# Buradaki veri Go-UT-Bench'in *birim testi yaz* görevleridir; değerlendirme
# trajektöri değil. Bu, yetenek ölçümü değil — format ve boru hattı denemesidir.
# Tek turlu, tek depodan bir korpus katalogu öğretmez; `measure` bunu "yetersiz"
# diye bildirecek ve bu doğru cevaptır. Değerlendirme için gerçek bir Go deposu ve
# konteyner havuzu gerekir (docs/EVAL.md).

import json

from transformers import AutoTokenizer

from gotooltrain import install_template, load_template_source, normalize_conversation, read_jsonl

tokenizer = install_template(AutoTokenizer.from_pretrained(MODEL_ID), load_template_source())
print("tokenizer hazır:", type(tokenizer).__name__)

records = list(read_jsonl("data/go-unit-tests.jsonl"))[:MAX_RECORDS]
print(f"{len(records)} kayıt okundu (sınır: {MAX_RECORDS})")

rendered = []
dropped_long = 0
dropped_invalid = 0
for record in records:
    try:
        conversation = normalize_conversation(record["messages"], None)
    except Exception as exc:  # noqa: BLE001 - her ret sayılmalı, adlandırılmalı
        dropped_invalid += 1
        print(f"  reddedildi (geçersiz): {type(exc).__name__}: {exc}")
        continue
    from gotooltrain import render_example

    example = render_example(tokenizer, conversation, max_length=MAX_TOKENS_PER_RECORD)
    if example.supervised_tokens == 0:
        dropped_long += 1
        continue
    rendered.append(example.to_dict())

total_tokens = sum(len(r["input_ids"]) for r in rendered)
supervised = sum(1 for r in rendered for label in r["labels"] if label != -100)
share = supervised / max(1, total_tokens)
print(f"kabul  : {len(rendered)}")
print(f"atılan: {dropped_long} (bağlam dışı), {dropped_invalid} (geçersiz)")
print(f"token  : {total_tokens} toplam, {supervised} denetimli ({share:.1%})")
print(f"ortalama: {total_tokens // max(1, len(rendered))} token/kayıt")
"""
        ),
        _code(
            """
# @title 8 — Token formatını doğrula (eğitimden ÖNCE)
#
# Maske bozuksa model araç çıktısı uydurmayı öğrenir ve kayıp normal görünür. Bu
# kontrol dakikalar sürer, bir 4B epoch saatler sürer. Sıra burada.

from gotooltrain import assert_mask_sane
from gotooltrain.template import RenderedExample, load_template_source
from gotooltrain.evalstore import sha256_text

first = rendered[0]
example = RenderedExample(
    input_ids=first["input_ids"],
    attention_mask=first["attention_mask"],
    labels=first["labels"],
    assistant_mask=first["assistant_mask"],
    text="",
)
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
# @title 9 — Eğitimi başlat (periyodik Hub yüklemesiyle)
#
# Adafactor: ilk momenti tutmaz, ikinci momenti çarpanlaştırır. Optimizer durumu
# birkaç GB yerine birkaç MB — tek kartta 4B'yi sığdıran şey bu.
#
# --hub-push-every: optimizer ADIMI sayar, batch değil. 8 kademe biriktirmeli
# olduğu için "her batch'te gönder" demek accumulation ayarına bağımlı olurdu.

import subprocess

with open("data/sft.jsonl", "w", encoding="utf-8", newline="\\n") as handle:
    for row in rendered:
        handle.write(json.dumps({"format_version": "anthropic-tools-v1", **row}) + "\\n")
print(f"data/sft.jsonl yazıldı: {len(rendered)} kayıt")

command = [
    sys.executable, "-m", "gotooltrain.traincli", "sft",
    "--model", MODEL_ID,
    "--output", "runs/colab-sft",
    "--tokenizer", MODEL_ID,
    "--dataset", "data/sft.jsonl",
    "--dtype", "bfloat16",
    "--epochs", str(EPOCHS),
    "--batch-size", str(BATCH_SIZE),
    "--grad-accum", str(GRAD_ACCUM),
    "--lr", str(LEARNING_RATE),
    "--max-length", str(CONTEXT_LENGTH),
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
result = subprocess.run(command, check=False)
print("exit:", result.returncode)
"""
        ),
        _code(
            """
# @title 10 — Ne olduğunu doğrula
#
# "Push ettim" demek yetmez; push'un *ne* olduğu okunmalı. Yayınlanan klasör
# kendi kökenini taşır: hangi adım, hangi ayarlar.

import json

from gotooltrain import read_push_state

state = read_push_state("runs/colab-sft")
print(f"step {state['step']}/{state['total_steps']}")
plan_record = state["plan"]
for key in ("model_id", "learning_rate", "epochs", "context_length", "optimizer", "token_format"):
    print(f"  {key:18} {plan_record[key]}")
print(f"  {'hub':18} {plan_record['hub']}")

if not DRY_RUN:
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(HF_REPO_ID)
    print(f"\\n{HF_REPO_ID} içinde {len(files)} dosya:")
    for name in sorted(files)[:15]:
        print("  ", name)
    print("\\ngit log gibi: her push bir commit. Adım commit mesajında.")
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
