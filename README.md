# Ses Tanıma ile Ortamdaki Kişi Sayısını Tespit Eden Gömülü Sistem
*(Not: Bu yazı yapay zeka ile oluşturulmuştur.)*

Bu proje, bir kapalı ortamdaki ses sinyallerini analiz ederek, konuşan kişi sayısını gerçek zamanlı olarak tespit eden bir gömülü sistem prototipidir. Proje, kamera kullanımının mahremiyet veya yetersiz ışık nedeniyle uygun olmadığı durumlar için enerji verimliliği ve otomasyon çözümleri sunmayı amaçlamaktadır.

Bu çalışma, **TÜBİTAK 2209-A - Üniversite Öğrencileri Araştırma Projeleri Desteği Programı** kapsamında desteklenmiştir.

---

[English Version Below](#english-version)

---

## Türkçe Açıklama

### 🎯 Projenin Amacı

Projenin temel amacı, bir mekandaki konuşmacı sayısını ses analizi yoluyla tespit etmektir. Tespit edilen bu bilgi, akıllı binalardaki aydınlatma, ısıtma ve iklimlendirme sistemlerini otomatik olarak kontrol ederek enerji tasarrufu sağlamak gibi çeşitli otomasyon senaryolarında kullanılabilir. Sistem, mahremiyeti ön planda tutan, kamera tabanlı sistemlere etkili bir alternatif olarak tasarlanmıştır.

### 🖼️ Görseller

**Devre Şeması:**
[![Image](https://i.hizliresim.com/dciq0h9.png)](https://hizliresim.com/dciq0h9)

### 🛠️ Teknoloji Mimarisi

#### Donanım
* **İşlemci:** Raspberry Pi 5 (8GB)
* **Mikrofon:** MAX9814 Mikrofon Amplifikatörü (Otomatik Kazanç Kontrollü)
* **ADC:** MCP3204 12-bit Analog-Dijital Çevirici
* **Ekran:** Raspberry Pi 5 inç Dokunmatik Ekran (Geliştirme ve çıktı görselleştirme için)
* **Diğer:** Resmi Raspberry Pi 27W Güç Adaptörü, Breadboard, Jumper Kablolar

#### Yazılım
* **Dil:** Python 3
* **Ana Kütüphaneler:**
  * `pyannote.audio`: Konuşmacı tespiti (diarization) için. `pyannote/speaker-diarization-3.1` modeli kullanılmıştır.
  * `spidev`: Raspberry Pi ve MCP3204 ADC arasında SPI haberleşmesi için.
  * `librosa` & `soundfile`: Ses dosyalarını okuma ve işleme için.
  * `noisereduce`: Gürültü azaltma için.
  * `webrtcvad`: Ses aktivite tespiti (VAD) için.
  * `numpy`: Sayısal işlemler için.
  * `matplotlib`: Çıktı grafiklerini oluşturmak için.

### ⚙️ Sistem Mimarisi

Sistem, 4 ana modülden oluşan bir iş akışına sahiptir:
1.  **Ses Yakalama (`capture.py`):** MAX9814 ve MCP3204 aracılığıyla analog ses sinyalini yakalar ve dijitalleştirir.
2.  **Ses Önişleme (`onisleme.py`):** Yakalanan sesten gürültüyü azaltır, sinyali normalleştirir ve sessiz kısımları atar.
3.  **Analiz (`analiz.py`):** `pyannote.audio` modelini kullanarak temizlenmiş ses segmentindeki konuşmacıları tespit eder ve etiketler.
4.  **Çıktı (`output.py`):** Analiz sonuçlarını terminalde gösterir ve konuşmacı zaman çizelgesi grafiği oluşturur.

### 🚀 Kurulum

1.  **Donanım Kurulumu:** Yukarıdaki devre şemasını referans alarak bileşenleri bir breadboard üzerinde birleştirin.
2.  **Raspberry Pi Yapılandırması:**
    * Raspberry Pi OS (64-bit) işletim sistemini kurun.
    * Terminali açın ve `sudo raspi-config` komutunu çalıştırın.
    * `Interface Options` -> `SPI` menüsüne gidin ve SPI arayüzünü etkinleştirin.
3.  **Yazılım Bağımlılıkları:**
    * Proje dosyalarını klonlayın: `git clone [Projenizin GitHub linkini buraya ekleyin]`
    * Proje dizinine gidin: `cd [proje-klasor-adi]`
    * Gerekli Python kütüphanelerini kurun:
      ```bash
      pip install -r requirements.txt
      ```

### 🔧 Yapılandırma

Bu projenin çalışması için bir Hugging Face kullanıcı belirtecine (token) ihtiyacınız vardır.

1.  [Hugging Face](https://huggingface.co/settings/tokens) sitesinden bir `READ` yetkisine sahip token oluşturun.
2.  Oluşturduğunuz token'ı `config.py` dosyasındaki `HUGGINGFACE_TOKEN` değişkenine atayın:
    ```python
    # config.py
    HUGGINGFACE_TOKEN = "hf_SizinHuggingFaceTokeninizBurayaGelecek"
    ```

### ⚡ Kullanım

Proje `main.py` betiği üzerinden çalıştırılır. İki ana modu vardır:

1.  **Canlı Ses İşleme:** Mikrofondan anlık olarak ses kaydı alır ve işler.
    ```bash
    python3 main.py --live
    ```
    * Belirli bir süre kayıt yapmak için:
    ```bash
    python3 main.py --live --duration 60 # 60 saniye kayıt yapar
    ```

2.  **Dosyadan Ses İşleme:** Mevcut bir `.wav` dosyasını işler.
    ```bash
    python3 main.py --input-file /path/to/your/audio.wav
    ```

### 📂 Proje Yapısı
```
.
├── capture.py        # Ses yakalama ve ADC kontrol modülü
├── onisleme.py       # Ses önişleme (gürültü azaltma, VAD) modülü
├── analiz.py         # Konuşmacı tespiti (diarization) modülü
├── output.py         # Sonuçları görselleştirme ve raporlama modülü
├── main.py           # Ana betik, tüm modülleri yönetir
├── config.py         # Yapılandırma ayarları (API token, ses parametreleri)
├── requirements.txt  # Gerekli Python kütüphaneleri
└── README.md         # Bu dosya
```

### 🙏 Teşekkür

Bu proje, **TÜBİTAK Bilim İnsanı Destek Programları Başkanlığı (BİDEB)** tarafından yürütülen **2209-A Üniversite Öğrencileri Araştırma Projeleri Desteği Programı** kapsamında desteklenmiştir.

### 📄 Lisans

Bu proje MIT Lisansı ile lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.

---
---

## <a name="english-version"></a>English Version
*(Note: This text was created with artificial intelligence.)*

### 🎯 Project Goal

The primary goal of this project is to detect the number of speakers in an environment through audio analysis. This information can be used in various automation scenarios, such as controlling lighting, heating, and air conditioning systems in smart buildings to save energy. The system is designed as an effective, privacy-first alternative to camera-based systems.

### 🖼️ Visuals

**Circuit Diagram:**
[![Image](https://i.hizliresim.com/dciq0h9.png)](https://hizliresim.com/dciq0h9)

### 🛠️ Technology Stack

#### Hardware
* **Processor:** Raspberry Pi 5 (8GB)
* **Microphone:** MAX9814 Microphone Amplifier (with Automatic Gain Control)
* **ADC:** MCP3204 12-bit Analog-to-Digital Converter
* **Display:** Raspberry Pi 5" Touchscreen Display (for development and output visualization)
* **Other:** Official Raspberry Pi 27W Power Supply, Breadboard, Jumper Wires

#### Software
* **Language:** Python 3
* **Core Libraries:**
  * `pyannote.audio`: For speaker diarization. The `pyannote/speaker-diarization-3.1` model was used.
  * `spidev`: For SPI communication between Raspberry Pi and MCP3204 ADC.
  * `librosa` & `soundfile`: For reading and processing audio files.
  * `noisereduce`: For noise reduction.
  * `webrtcvad`: For Voice Activity Detection (VAD).
  * `numpy`: For numerical operations.
  * `matplotlib`: For generating output graphs.

### ⚙️ System Architecture

The system has a workflow consisting of 4 main modules:
1.  **Audio Capture (`capture.py`):** Captures and digitizes the analog audio signal via MAX9814 and MCP3204.
2.  **Audio Preprocessing (`onisleme.py`):** Reduces noise, normalizes the signal, and removes silent parts from the captured audio.
3.  **Analysis (`analiz.py`):** Detects and labels speakers in the cleaned audio segment using the `pyannote.audio` model.
4.  **Output (`output.py`):** Displays the analysis results in the terminal and generates a speaker timeline graph.

### 🚀 Installation

1.  **Hardware Setup:** Assemble the components on a breadboard according to the circuit diagram above.
2.  **Raspberry Pi Configuration:**
    * Install Raspberry Pi OS (64-bit).
    * Open the terminal and run `sudo raspi-config`.
    * Navigate to `Interface Options` -> `SPI` and enable the SPI interface.
3.  **Software Dependencies:**
    * Clone the project files: `git clone [Add your project's GitHub link here]`
    * Navigate to the project directory: `cd [project-folder-name]`
    * Install the required Python libraries:
      ```bash
      pip install -r requirements.txt
      ```

### 🔧 Configuration

This project requires a Hugging Face user token to work.

1.  Create a token with `READ` permissions from the [Hugging Face](https://huggingface.co/settings/tokens) website.
2.  Assign the created token to the `HUGGINGFACE_TOKEN` variable in the `config.py` file:
    ```python
    # config.py
    HUGGINGFACE_TOKEN = "hf_YourHuggingFaceTokenGoesHere"
    ```

### ⚡ Usage

The project is run via the `main.py` script. It has two main modes:

1.  **Live Audio Processing:** Records and processes audio from the microphone in real-time.
    ```bash
    python3 main.py --live
    ```
    * To record for a specific duration:
    ```bash
    python3 main.py --live --duration 60 # Records for 60 seconds
    ```

2.  **File-based Audio Processing:** Processes an existing `.wav` file.
    ```bash
    python3 main.py --input-file /path/to/your/audio.wav
    ```

### 📂 Project Structure
```
.
├── capture.py        # Audio capture and ADC control module
├── onisleme.py       # Audio preprocessing (noise reduction, VAD) module
├── analiz.py         # Speaker diarization module
├── output.py         # Results visualization and reporting module
├── main.py           # Main script, manages all modules
├── config.py         # Configuration settings (API token, audio parameters)
├── requirements.txt  # Required Python libraries
└── README.md         # This file
```

### 🙏 Acknowledgments

This project has been supported by the **TÜBİTAK Scientist Support Programs Directorate (BİDEB)** under the **2209-A University Students Research Projects Support Program**.

### 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
