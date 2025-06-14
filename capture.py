import os
import wave
import time
import queue
import threading
import numpy as np
import spidev
import datetime
from config import AUDIO_CONFIG, SPI_CONFIG
import soundfile as sf

class AudioCaptureModule:
    def __init__(self, 
                 sample_rate=AUDIO_CONFIG["sample_rate"], 
                 channels=AUDIO_CONFIG["channels"], 
                 chunk_size=AUDIO_CONFIG["chunk_size"], 
                 segment_duration=AUDIO_CONFIG["segment_duration"],
                 overlap_duration=AUDIO_CONFIG["overlap_duration"],
                 output_dir=AUDIO_CONFIG["output_dir"],
                 spi_bus=SPI_CONFIG["bus"],
                 spi_device=SPI_CONFIG["device"]):
        """
        Ses yakalama modülü için başlangıç yapılandırması
        
        Parametreler:
            sample_rate (int): Örnekleme hızı (Hz)
            channels (int): Kanal sayısı (mono=1, stereo=2)
            chunk_size (int): Her okumada alınacak ses verisi boyutu
            segment_duration (int): Her segmentin süresi (saniye)
            overlap_duration (int): Ardışık segmentler arasındaki örtüşme süresi (saniye)
            output_dir (str): Kaydedilen ses dosyalarının dizini
            spi_bus (int): SPI bus numarası (Raspberry Pi'da genellikle 0)
            spi_device (int): SPI cihaz numarası (CE0 için 0, CE1 için 1)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.segment_duration = segment_duration
        self.overlap_duration = overlap_duration
        self.bit_depth = 12  # MCP3204 12-bit ADC
        
        # Segment bilgileri
        self.frames_per_segment = int(self.sample_rate * self.segment_duration)
        self.frames_per_overlap = int(self.sample_rate * self.overlap_duration)
        
        # SPI yapılandırması
        self.spi = spidev.SpiDev()
        self.spi.open(spi_bus, spi_device)
        self.spi.max_speed_hz = 1000000  # 1 MHz
        
        # Ses kayıt ve segment yönetimi için gerekli değişkenler
        self.is_recording = False
        self.stop_requested = False
        
        # Kaydedilen ses verileri için kuyruk
        self.audio_queue = queue.Queue()
        
        # Kaydedilen ses dosyalarının kaydedileceği dizin
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def adc_oku(self, ch):
        """MCP3204 ADC'den veri okur"""
        if ch > 3: return -1
        komut = [0b00000110 | (ch >> 2), (ch & 0b11) << 6, 0x00]
        cevap = self.spi.xfer2(komut)
        sonuc = ((cevap[1] & 0x0F) << 8) | cevap[2]
        return sonuc
    
    def normalize(self, samples):
        """Ses örneklerini normalize eder"""
        samples = np.array(samples)
        samples = samples - np.mean(samples)
        samples = samples / np.max(np.abs(samples))
        return samples
    
    def start_recording(self):
        """Ses yakalamayı başlatır ve bir iş parçacığında (thread) çalıştırır"""
        if self.is_recording:
            print("Kayıt zaten devam ediyor.")
            return
        
        self.stop_requested = False
        self.is_recording = True
        
        # İş parçacıklarını başlat
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.processing_thread = threading.Thread(target=self._process_segments)
        
        self.recording_thread.daemon = True
        self.processing_thread.daemon = True
        
        self.recording_thread.start()
        self.processing_thread.start()
        
        print("Ses kaydı başlatıldı.")
    
    def stop_recording(self):
        """Ses kayıt işlemini durdurur"""
        if not self.is_recording:
            print("Kayıt zaten durdurulmuş.")
            return
        
        self.stop_requested = True
        self.recording_thread.join()
        self.processing_thread.join()
        
        self.is_recording = False
        print("Ses kaydı durduruldu.")
    
    def _record_audio(self):
        """Ses verisini sürekli olarak kuyrukta saklar"""
        samples_buffer = []
        samples_count = 0
        start_time = time.perf_counter()
        
        try:
            while not self.stop_requested:
                # ADC'den örnek al
                sample = self.adc_oku(2)  # CH2'den oku
                samples_buffer.append(sample)
                samples_count += 1
                
                # Belirli bir örnek sayısına ulaşınca kuyruğa ekle
                if len(samples_buffer) >= self.chunk_size:
                    self.audio_queue.put(samples_buffer.copy())
                    samples_buffer = []
                
                # Örnekleme hızını korumak için bekle
                # Her örnek için beklemek yerine, toplam örnek sayısına göre hedef zamanı hesaplayıp bekleyelim
                t_hedef = start_time + (samples_count / self.sample_rate)
                while time.perf_counter() < t_hedef:
                    pass  # zaman dolana kadar bekle
                    
        except Exception as e:
            print(f"Ses kaydı sırasında hata oluştu: {str(e)}")
        
        # Kalan verileri kuyruğa ekle
        if samples_buffer:
            self.audio_queue.put(samples_buffer)
    
    def _process_segments(self):
        """Ses segmentlerini işler ve dosyalara kaydeder"""
        all_samples = []
        last_segment_end = 0
        segment_count = 0
        
        try:
            while not self.stop_requested or not self.audio_queue.empty():
                try:
                    # 0.5 saniye bekleyerek ses verisini al
                    samples_chunk = self.audio_queue.get(timeout=0.5)
                    all_samples.extend(samples_chunk)
                    
                    # Segment oluşturma koşulunu kontrol et
                    total_samples = len(all_samples)
                    
                    # Segment süresine ulaşıldıysa veya kayıt durdurulduysa segment oluştur
                    if total_samples >= last_segment_end + self.frames_per_segment or (self.stop_requested and total_samples > 0):
                        # Segment başlangıç pozisyonu (örtüşme için)
                        if last_segment_end > 0:
                            start_pos = last_segment_end - self.frames_per_overlap
                        else:
                            start_pos = 0
                            
                        # Segment bitiş pozisyonu
                        if self.stop_requested:
                            # Kayıt durdurulduysa kalan tüm veriyi kullan
                            end_pos = total_samples
                        else:
                            end_pos = start_pos + self.frames_per_segment
                        
                        # Segment verisini ayıkla
                        segment_data = all_samples[start_pos:end_pos]
                        
                        # Minimum segment süresi kontrolü (0.5 saniye)
                        min_samples = int(0.5 * self.sample_rate)
                        if len(segment_data) >= min_samples:
                            # Segmenti kaydet
                            self._save_segment(segment_data, segment_count)
                            segment_count += 1
                        
                        # Son segment bitiş konumunu güncelle
                        last_segment_end = end_pos
                        
                        # Belleği temizle (artık gerekmeyen eski verileri kaldır)
                        if len(all_samples) > end_pos + self.frames_per_overlap:
                            # Bir miktar örtüşme bırak
                            keep_from = end_pos - self.frames_per_overlap
                            all_samples = all_samples[keep_from:]
                            last_segment_end -= keep_from
                
                except queue.Empty:
                    # Kuyruk boşsa kısa bir süre bekle
                    time.sleep(0.1)
                    continue
        
        except Exception as e:
            print(f"Segment işleme sırasında hata oluştu: {str(e)}")
    
    def _save_segment(self, segment_data, segment_number):
        """Ses segmentini WAV dosyası olarak kaydeder"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"segment_{timestamp}_{segment_number}.wav")
        
        # Verileri normalize et (float array)
        normalized_data = self.normalize(segment_data)
        
        # Soundfile kullanarak WAV dosyası olarak kaydet
        try:
            sf.write(filename, normalized_data, self.sample_rate)
            print(f"Segment {segment_number} kaydedildi (soundfile): {filename}")
            return filename
        except Exception as e:
            print(f"Soundfile ile kayıt sırasında hata: {str(e)}")
            # Hata durumunda wave kütüphanesini kullanarak kayıt etme (yedek)
            try:
                # 16-bit PCM formatına dönüştür (wave kütüphanesi için)
                pcm_data = (normalized_data * 32767).astype(np.int16)
                
                with wave.open(filename, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(2)  # 16-bit = 2 bytes
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(pcm_data.tobytes())
                print(f"Segment {segment_number} kaydedildi (wave - yedek): {filename}")
                return filename
            except Exception as wave_e:
                print(f"Wave ile kayıt sırasında hata: {str(wave_e)}")
                return None
    
    def __del__(self):
        """Nesne yok edilirken kaynakları temizle"""
        if self.is_recording:
            self.stop_recording()
        
        if self.spi:
            self.spi.close()


if __name__ == "__main__":
    # Modülü test etmek için örnek kullanım
    capture_module = AudioCaptureModule(
        sample_rate=16000,
        channels=1,
        segment_duration=60,
        overlap_duration=3
    )
    
    try:
        print("Ses kayıt modülü test ediliyor. Kaydı durdurmak için Ctrl+C tuşuna basın.")
        capture_module.start_recording()
        
        # 60 saniye boyunca kayıt yap (test amaçlı)
        time.sleep(60)
        
    except KeyboardInterrupt:
        print("Kayıt kullanıcı tarafından durduruldu.")
    finally:
        capture_module.stop_recording()
        print("Kayıt modülü testi tamamlandı.")
