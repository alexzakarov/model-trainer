# 4 GB VRAM'a Sığan Açık Kaynak Kod (Coder) Modelleri

Araştırma tarihi: **29 Eylül 2026**. Kapsam: açık ağırlıklı, **toplam ≤ 4 GB VRAM** içinde
çalışabilen kod/yazılım-geliştirme modelleri, başarı sırasıyla.

## Yöntem ve veri kaynakları

Boyutlar **GGUF quant dosya boyutları** (llama.cpp) ve bunların üstüne KV cache + çalışma
zamanı eklenerek hesaplanan gerçek VRAM tüketimidir. Birincil kaynaklar:

- Hugging Face model kartları (Qwen, OpenBMB, unsloth, bartowski, HuggingFace/SmolLM)
- Ollama kütüphane sayfaları (resmî quant boyutları)
- Qwen blog/Hugging Face kartı (Qwen3.5 serisi, Mart 2026)
- Qwen2.5-Coder teknik raporu (arXiv 2409.12186), SmolLM3 model kartı
- LM-Kit model kataloğu, CodeSOTA (LiveCodeBench), Yerel model VRAM rehberleri

> **Not:** İlk arama turunda `benchlm.ai`, `lmmarketcap.com`, `ertas.ai`, `techsy.io`,
> `canirun.ai` gibi karşılaştırma siteleri model adları ve skorları arasında ciddi tutarsızlık
> gösterdi (ör. "Qwen3.6-27B", "Kimi K2.6", "Gemma 4 26B A4B", "LFM2.5"). Bu kaynaklar
> **kullanılmadı**; sıralama yalnızca doğrulanabilir kaynaklara dayanıyor.

## Kritik uyarı: 4 GB bütçe aslında bağlam sınırıdır

Ağırlıklar 4 GB'a sığsa bile **KV cache** belleği de VRAM'dan yer yer. Gerçek bütçe:

| Bağlam | ~4B model toplam VRAM |
|---|---|
| 4K | ~3.5–4.0 GB ✅ |
| 8K | ~4.5–5.0 GB ⚠️ |
| 32K | ~7.1 GB ❌ |
| 128K | ~15–19 GB ❌ |

Yani 4 GB kartta **pratikte 4K–8K bağlam** ile çalışacak. Uzun dosya bağlamı isteyen kod
görevleri bu modellerde sınırlı kalır.

## Sıralama

| # | Model | Param | GGUF (Q4_K_M) | ~Toplam VRAM (4K) | Lisans | Kod notu |
|---|---|---|---|---|---|---|
| 1 | **Qwen3.5-4B** | 4.33B | 3.01 GB (IQ4_XS 2.67 / Q4_0 2.78) | ~3.9–4.3 GB | Apache-2.0 | Sınıfının en güçlüsü. Hibrit Gated DeltaNet + attention, 262K bağlam, native multimodal, tool use. 4 GB'a sığması için **IQ4_XS** önerilir. |
| 2 | **Qwen3-4B-Instruct-2507** | 4B | 2.50 GB | ~3.8 GB | Apache-2.0 | **Doğrulanmış en yüksek kod skoru:** LiveCodeBench v4 = 52.9 (aynı sınıftaki SmolLM3-3B 30.0, Qwen2.5-3B 10.5). 32K bağlam. |
| 3 | **SmolLM3-3B** | 3B | 1.78 GB | ~3.0 GB | Apache-2.0 | HuggingFace, 128K bağlam, tam açık eğitim reçetesi. Kod: LCB v4 30.0. Çok daha küçük olduğu için 8K+ bağlam veya agentic döngü için en rahat seçenek. |
| 4 | **MiniCPM5-2B** | 2.52B | 1.6 GB | ~2.8 GB | Apache-2.0 | OpenBMB, 7 Eylül 2026. 128K bağlam, standart `LlamaForCausalLM` (özel kernel yok), XML tool-call, kod + agentic tool use odaklı. Üretici "2B sınıfı açık kaynak SOTA" diyor (kendi karşılaştırma seti). |
| 5 | **Qwen2.5-Coder-3B-Instruct** | 3B | ~1.9 GB | ~3.1 GB | Apache-2.0 | Bu boyutta **kod-özel** tek güçlü aday. FIM (fill-in-middle) desteği. Kasım 2024 modeli; LCB v4 10.5 ile yeni nesil genel modellerin gerisinde. |
| 6 | **Phi-4-mini-reasoning** | 3.8B | 2.4 GB | ~3.7 GB | MIT | İnsanEval 74.4, GSM8K 88.6 — 3.8B'de param başına çok güçlü. Düşünme (reasoning) modu kodda yavaşlatır. |
| 7 | **Ministral 3 3B** | 3.85B | 2.4 GB | ~3.8 GB | Apache-2.0 | Mistral, 256K bağlam, tool calling, vision. |
| 8 | **Gemma 3 4B** | 4B | 2.5 GB | ~3.8 GB | Gemma | 128K bağlam, güçlü genel muhakeme; kod tarafında 4B sınıfının alt ucunda. |
| 9 | **Qwen3.5-2B** | 2B | 1.5 GB | ~2.7 GB | Apache-2.0 | Aynı mimari, çok daha küçük; sadece çok kısa kod işleri veya cihaz-kenarı için. |
| 10 | **MiniCPM5-1B** | 1.08B | 688 MB | ~1.9 GB | Apache-2.0 | 1B sınıfı SOTA iddiası, 131K bağlam. Kod için ancak FIM/şablon işleri. |

## 4 GB'a **sığmayan**lar (bilinçli olarak dışlandı)

| Model | Param | Q4_K_M | Neden |
|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct | 7B | 4.1 GB | Ağırlıklar tek başına bütçeyi aşıyor; KV cache sığmıyor. IQ3/IQ2 ile ancak ciddi kalite kaybı. |
| Qwen3.5-9B | 9B | 5.1–6.2 GB | Bütçe dışı. |
| Qwen3.5-27B / 35B-A3B | 27B / 35B | 17–22 GB | Bütçe dışı. |
| Qwen3-Coder-Next | 80B MoE (3B aktif) | ~45 GB | MoE = tüm uzmanlar bellekte; 3B aktif yanıltıcı. |
| Devstral Small 2 / Small 3 | 24B | ~14 GB | Bütçe dışı. |
| North Mini Code (Cohere) | 30B MoE (3B aktif) | ~17 GB | MoE; kod odaklı ama 30B. |
| GLM-Edge 4B | — | — | Doğrulanabilir resmî kart/quant bulunamadı; listeye almadım. |

## Önemli bulgu: 4B altında kod-özel model kalmadı

2026'da bu bütçedeki en iyi modeller **genel amaçlı** modeller; Qwen2.5-Coder soyunun 3B
varyantı bile yeni nesil genel modellerin gerisinde kaldı. Kod-özel ailelerin hepsi
(Devstral, Qwen3-Coder-Next, North Mini Code, GLM-4.7-Flash) en küçük varyantlarında bile
24B+ ve MoE. Yani **"4 GB'da en iyi coder" sorusunun cevabı, en iyi genel model"**.

## Öneri

- **Kod kalitesi önceliği, 4 GB sabit:** `Qwen3.5-4B` (IQ4_XS) — mimari olarak en güncel,
  tool calling ve uzun bağlam desteği var; 8K üzeri bağlamda `Qwen3-4B-Instruct-2507`
  (2.5 GB) daha fazla yer bırakır.
- **VRAM payı ~3 GB'a düşse:** `SmolLM3-3B` veya yeni `MiniCPM5-2B` — ikisi de 128K
  bağlamda çalışabilir, 4 GB kartta agentic kod döngüsüne daha çok alan kalır.
- **Sadece FIM / satır-içi tamamlama (editör eklentisi):** `Qwen2.5-Coder-3B-Instruct` ya da
  `Qwen2.5-Coder-1.5B` (Q5_1 1.2 GB).

## Doğrulama notu

Buradaki skorlar model kartları ve derleme kaynaklarından; bağımsız üçüncü taraf
leaderboard'lar (AA Intelligence Index vb.) 2B-4B aralığını yayınlamıyor. Kesin sıralama
için hedef iş yüklerini (repo düzeltme, birim test yazma, FIM) bu modellerle kıyaslamak
gerekir.
