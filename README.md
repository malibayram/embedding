# Embedding Research

Dil işleme çalışmaları için optimize edilmiş embedding modelleri geliştirme ve mevcut modellerin embedding katmanlarını verimli kullanma projesi.

---

## Projenin Amacı ve Kapsamı

Bu proje, doğal dil işleme çalışmalarında kullanılmak üzere iki temel hedefe odaklanmaktadır:

1. Yüksek kaliteli, dil özelliklerine duyarlı embedding modelleri geliştirmek
2. Mevcut büyük dil modellerinin embedding katmanlarını pretrain gerektirmeden etkili bir şekilde kullanmak

## Araştırma Alanları

### 1. Transfer Learning Yaklaşımları

- Mevcut modellerin embedding katmanlarının analizi
- Ağırlık transferi metodolojileri
- Layer adaptation teknikleri
- Fine-tuning stratejileri

### 2. Embedding Optimizasyonu

- Dil spesifik özellik çıkarımı
- Boyut optimizasyonu
- Hafıza verimli temsiller
- Hız optimizasyonu

### 3. Yeni Metodolojiler

- Hibrit embedding yaklaşımları
- Dinamik embedding oluşturma
- Bağlam-duyarlı adaptasyon
- Zero-shot transfer teknikleri

## Deneysel Çalışmalar

### Kullanılan Modeller

- BERT türevleri
- GPT modelleri
- XLM-R
- Custom architectures

### Değerlendirme Metrikleri

- Anlamsal benzerlik
- Task performansı
- Hesaplama verimliliği
- Bellek kullanımı

```python
# Örnek Kullanım - Gemma Model with Device Support
from gemma_model import GemmaForCausalLM, get_config_for_270m_tr_tokenizer
from turkish_tokenizer import TurkishTokenizer
import torch

# Determine best available device (CUDA, MPS, or CPU)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Initialize model with device specification
config = get_config_for_270m_tr_tokenizer(dtype="float32")
tokenizer = TurkishTokenizer()
model = GemmaForCausalLM(
    config=config,
    tokenizer=tokenizer,
    device=device  # Model will be automatically moved to this device
)

# Generate text
response = model.generate(
    prompts="Merhaba, nasılsın?",
    output_len=50,
    temperature=0.7,
    top_p=0.9,
    top_k=40
)
print(f"Response: {response}")

# Move model to different device if needed
if device == "cpu" and torch.cuda.is_available():
    model.to_device("cuda")
    print(f"Model moved to: {model.get_device()}")
```

## Teknik Altyapı

### Temel Bileşenler

- Embedding ekstraktörleri
- Model adaptörleri
- Optimizasyon araçları
- Değerlendirme suite'i

### Desteklenen Özellikler

- Çoklu model desteği
- Batch processing
- GPU optimizasyonu
- Memory-mapping

### Device Support

The Gemma model implementation supports multiple devices:

- **CUDA**: For NVIDIA GPUs with CUDA support
- **MPS**: For Apple Silicon (M1/M2) GPUs
- **CPU**: Fallback for systems without GPU support

The model automatically handles device placement during initialization and all operations are performed on the specified device.

## Deneysel Sonuçlar

### Performans Metrikleri

- Embedding kalitesi
- İşlem süresi
- Bellek kullanımı
- Task başarımı

### Karşılaştırmalı Analizler

- Model bazlı karşılaştırmalar
- Yaklaşım değerlendirmeleri
- Ablasyon çalışmaları
- Use-case analizleri

## Geliştirme ve Katkıda Bulunma

### Araştırma Alanları

- Yeni embedding teknikleri
- Optimizasyon yaklaşımları
- Değerlendirme metodolojileri
- Use-case geliştirmeleri

### Katkı Süreci

1. Repository'yi fork edin
2. Yeni bir research branch'i oluşturun
3. Deneysel çalışmanızı yapın
4. Sonuçları raporlayın
5. Pull request açın

## Yol Haritası

### Kısa Vadeli Hedefler

- Temel model analizleri
- İlk adaptasyon deneyleri
- Baseline oluşturma
- Metrik belirleme

### Uzun Vadeli Hedefler

- Yeni metodolojiler
- Performans optimizasyonu
- Geniş ölçekli testler
- Pratik uygulamalar

## Lisans

MIT

---

**Not:** Bu proje, aktif araştırma ve geliştirme aşamasındadır. Detaylı teknik dokümantasyon, deney sonuçları ve metodoloji açıklamaları için [Wiki](wiki) sayfasını ziyaret edebilirsiniz.
