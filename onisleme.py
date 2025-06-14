import os
import wave
import numpy as np
import librosa
import noisereduce as nr
import soundfile as sf
from scipy import signal
import webrtcvad
import io
import struct
import collections

class AudioPreprocessingModule:
    def __init__(self, 
                 sample_rate=16000,
                 channels=1,
                 vad_mode=1,
                 vad_frame_duration_ms=30,
                 noise_reduction_strength=0.3,
                 target_db=-15,
                 output_dir="preprocessed_audio"):
        """
        Ses önişleme modülü için başlangıç yapılandırması
        
        Parametreler:
            sample_rate (int): Örnekleme hızı (Hz)
            channels (int): Kanal sayısı (mono=1, stereo=2)
            vad_mode (int): WebRTC VAD saldırganlık modu (0-3, 3 en saldırgan)
            vad_frame_duration_ms (int): VAD çerçeve süresi (ms)
            noise_reduction_strength (float): Gürültü azaltma gücü (0-1)
            target_db (float): Normalleştirme hedef dB seviyesi
            output_dir (str): İşlenmiş ses dosyalarının dizini
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.vad_mode = vad_mode
        self.vad_frame_duration_ms = vad_frame_duration_ms
        self.noise_reduction_strength = noise_reduction_strength
        self.target_db = target_db
        
        # VAD ayarları
        self.vad = webrtcvad.Vad(self.vad_mode)
        
        # Çıkış klasörü
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_file(self, input_file, output_file=None):
        """
        Ses dosyasını işler ve önişleme yapar
        
        Parametreler:
            input_file (str): İşlenecek ses dosyasının yolu
            output_file (str, optional): Çıktı dosyasının yolu. None ise otomatik oluşturulur.
            
        Döndürür:
            str: İşlenmiş ses dosyasının yolu
        """
        print(f"Ses dosyası işleniyor: {input_file}")
        
        # Çıktı dosya adını oluştur (belirtilmemişse)
        if output_file is None:
            base_name = os.path.basename(input_file)
            output_file = os.path.join(self.output_dir, f"processed_{base_name}")
        
        # Ses dosyasını yükle
        audio_data, sr = librosa.load(input_file, sr=self.sample_rate, mono=True)
        
        # Önişleme adımlarını uygula
        processed_audio = self._apply_preprocessing(audio_data)
        
        # İşlenmiş sesi kaydet
        sf.write(output_file, processed_audio, self.sample_rate)
        print(f"İşlenmiş ses kaydedildi: {output_file}")
        
        return output_file, processed_audio
    
    def process_audio_data(self, audio_data, sr=None):
        """
        Ham ses verisini işler ve önişleme yapar
        
        Parametreler:
            audio_data (numpy.ndarray): İşlenecek ses verisi
            sr (int, optional): Ses verisinin örnekleme hızı. None ise default kullanılır.
            
        Döndürür:
            numpy.ndarray: İşlenmiş ses verisi
        """
        if sr is None:
            sr = self.sample_rate
        
        # Örnekleme hızını kontrol et ve gerekirse yeniden örnekle
        if sr != self.sample_rate:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
            
        # Önişleme adımlarını uygula
        processed_audio = self._apply_preprocessing(audio_data)
        
        return processed_audio
    
    def _apply_preprocessing(self, audio_data):
        """
        Ses verisine tüm önişleme adımlarını uygular
        
        Parametreler:
            audio_data (numpy.ndarray): İşlenecek ses verisi
            
        Döndürür:
            numpy.ndarray: İşlenmiş ses verisi
        """
        # 1. Gürültü temizleme
        denoised_audio = self._apply_noise_reduction(audio_data)
        
        # 2. Ses seviyesi normalleştirme
        normalized_audio = self._apply_normalization(denoised_audio)
        
        # 3. Ses Aktivite Tespiti (VAD) ve sessiz bölümleri atla
        voice_only_audio = self._apply_vad(normalized_audio)
        
        return voice_only_audio
    
    def _apply_noise_reduction(self, audio_data):
        """
        Ses verisindeki gürültüyü temizler
        
        Parametreler:
            audio_data (numpy.ndarray): İşlenecek ses verisi
            
        Döndürür:
            numpy.ndarray: Gürültüsü azaltılmış ses verisi
        """
        try:
            # Stationary noise reduction (sabit gürültü azaltma)
            denoised_audio = nr.reduce_noise(
                y=audio_data,
                sr=self.sample_rate,
                prop_decrease=self.noise_reduction_strength,
                stationary=True,
                n_std_thresh_stationary=1.5
            )
            
            # Belirli frekans bandlarında ek filtreleme (bandpass filter)
            # İnsan sesi için daha geniş bir aralık kullanıyoruz
            sos = signal.butter(6, [60, 7800], 'bandpass', fs=self.sample_rate, output='sos')
            filtered_audio = signal.sosfilt(sos, denoised_audio)
            
            return filtered_audio
            
        except Exception as e:
            print(f"Gürültü temizleme sırasında hata: {str(e)}")
            return audio_data  # Hata durumunda orijinal ses verisini döndür
    
    def _apply_normalization(self, audio_data):
        """
        Ses seviyesini normalleştirir
        
        Parametreler:
            audio_data (numpy.ndarray): İşlenecek ses verisi
            
        Döndürür:
            numpy.ndarray: Normalleştirilmiş ses verisi
        """
        try:
            # Mevcut dB seviyesini hesapla
            if np.max(np.abs(audio_data)) > 0:
                current_db = 20 * np.log10(np.max(np.abs(audio_data)))
            else:
                current_db = -80  # Çok düşük ses seviyesi varsayımı
                
            # Gerekli dB değişimini hesapla
            db_change = self.target_db - current_db
            
            # Ses seviyesini ayarla
            normalized_audio = audio_data * (10 ** (db_change / 20))
            
            # Clipping'i önle (-1 ile 1 arasında kalsın)
            normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
            
            # Ek güçlendirme (eğer ses hala çok düşükse)
            if np.max(np.abs(normalized_audio)) < 0.5:
                normalized_audio = normalized_audio * 1.5
                normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
            
            return normalized_audio
            
        except Exception as e:
            print(f"Ses seviyesi normalleştirme sırasında hata: {str(e)}")
            return audio_data  # Hata durumunda orijinal ses verisini döndür
    
    def _apply_vad(self, audio_data):
        """
        WebRTC VAD kullanarak ses aktivitesi tespiti yapar ve sessiz bölümleri atar
        
        Parametreler:
            audio_data (numpy.ndarray): İşlenecek ses verisi
            
        Döndürür:
            numpy.ndarray: Yalnızca ses aktivitesi olan bölümleri içeren ses verisi
        """
        try:
            # VAD çerçeve boyutunu hesapla
            frame_size = int(self.sample_rate * self.vad_frame_duration_ms / 1000)
            
            # Ses verisini kısa çerçevelere böl
            audio_frames = []
            for i in range(0, len(audio_data) - frame_size, frame_size):
                frame = audio_data[i:i + frame_size]
                audio_frames.append(frame)
            
            # Her çerçevenin ses aktivitesi içerip içermediğini kontrol et
            is_speech = []
            for frame in audio_frames:
                # VAD, int16 formatında PCM verisi bekler
                frame_pcm = (frame * 32768).astype(np.int16).tobytes()
                
                try:
                    # VAD, bazı çerçeve boyutlarını kabul eder (10, 20, 30 ms)
                    has_speech = self.vad.is_speech(frame_pcm, self.sample_rate)
                    is_speech.append(has_speech)
                except Exception:
                    # Geçersiz çerçeve boyutu durumunda, aktivite var kabul et
                    is_speech.append(True)
            
            # Ses aktivitesi içeren çerçeveleri birleştir
            voice_frames = [audio_frames[i] for i in range(len(audio_frames)) if is_speech[i]]
            
            if not voice_frames:
                print("Ses aktivitesi tespit edilemedi, orijinal veri korunuyor.")
                return audio_data
            
            voice_only_audio = np.concatenate(voice_frames)
            
            return voice_only_audio
            
        except Exception as e:
            print(f"VAD işlemi sırasında hata: {str(e)}")
            return audio_data  # Hata durumunda orijinal ses verisini döndür
    
    def batch_process_files(self, input_dir, pattern="*.wav"):
        """
        Belirtilen dizindeki tüm ses dosyalarını toplu olarak işler
        
        Parametreler:
            input_dir (str): İşlenecek ses dosyalarının bulunduğu dizin
            pattern (str): Dosya arama deseni
            
        Döndürür:
            list: İşlenmiş dosya yollarının listesi
        """
        import glob
        
        input_files = glob.glob(os.path.join(input_dir, pattern))
        processed_files = []
        
        for input_file in input_files:
            try:
                processed_file, _ = self.process_file(input_file)
                processed_files.append(processed_file)
            except Exception as e:
                print(f"Dosya işleme hatası ({input_file}): {str(e)}")
        
        return processed_files


if __name__ == "__main__":
    # Modülü test etmek için örnek kullanım
    import sys
    
    if len(sys.argv) < 2:
        print("Kullanım: python3 preprocessing_module.py <input_audio_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    preprocess_module = AudioPreprocessingModule(
        sample_rate=16000,
        vad_mode=1,
        noise_reduction_strength=0.3,
        target_db=-15
    )
    
    processed_file, processed_audio = preprocess_module.process_file(input_file)
    print(f"İşleme tamamlandı. Çıktı dosyası: {processed_file}")
    
    # Sonuçları analiz et
    print(f"Orijinal dosya uzunluğu: {librosa.get_duration(filename=input_file)} saniye")
    print(f"İşlenmiş dosya uzunluğu: {len(processed_audio) / preprocess_module.sample_rate} saniye")