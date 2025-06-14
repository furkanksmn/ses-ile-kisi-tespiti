import os
import time
import argparse
import shutil
from datetime import datetime
from capture import AudioCaptureModule
from onisleme import AudioPreprocessingModule
from analiz import AudioAnalyzer
# from speakeridentity import SpeakerIdentityDB, SpeakerMatchingModule # KALDIRILDI
from output import OutputManager

class SpeakerDiarizationSystem:
    def __init__(self,
                 auth_token: str = None,
                 output_dir: str = "results",
                 sample_rate: int = 16000,
                 segment_duration: int = 60,
                 overlap_duration: int = 3):
        """
        Konuşmacı diarization sistemi ana sınıfı
        
        Parametreler:
            auth_token (str): Hugging Face API token'ı
            output_dir (str): Çıktı dosyalarının kaydedileceği dizin
            sample_rate (int): Örnekleme hızı (Hz)
            segment_duration (int): Segment süresi (saniye)
            overlap_duration (int): Örtüşme süresi (saniye)
        """
        # Çıktı dizinlerini oluştur
        self.output_dir = output_dir
        self.captured_dir = os.path.join(output_dir, "captured")
        self.preprocessed_dir = os.path.join(output_dir, "preprocessed")
        self.diarization_dir = os.path.join(output_dir, "diarization")
        
        for directory in [self.captured_dir, self.preprocessed_dir,
                         self.diarization_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Modülleri başlat
        print("Sistem modülleri başlatılıyor...")
        
        # Ses yakalama modülü
        self.capture_module = AudioCaptureModule(
            sample_rate=sample_rate,
            segment_duration=segment_duration,
            overlap_duration=overlap_duration,
            output_dir=self.captured_dir
        )
        
        # Önişleme modülü
        self.preprocess_module = AudioPreprocessingModule(
            sample_rate=sample_rate,
            output_dir=self.preprocessed_dir
        )
        
        # Diarization modülü
        self.diarization_module = AudioAnalyzer(
            output_dir=self.diarization_dir
        )
        
        # # Konuşmacı kimlik veritabanı # KALDIRILDI
        # self.speaker_db = SpeakerIdentityDB(
        #     db_file=os.path.join(self.output_dir, "speaker_db.json"),
        #     output_dir=os.path.join(self.output_dir, "speaker_profiles")
        # )
        
        # # Konuşmacı eşleştirme modülü # KALDIRILDI
        # self.speaker_matcher = SpeakerMatchingModule(
        #     speaker_db=self.speaker_db
        # )
        
        # Sonuç yönetimi modülü (Sadece zaman çizelgesi için)
        self.output_manager = OutputManager(
            output_dir=self.output_dir # Zaman çizelgesi için ana çıktı dizinini kullan
        )

        # Diarizasyon sonuçlarını geçici olarak saklamak için
        self.current_diarization_results = []
        
        print("Tüm modüller başarıyla başlatıldı.")
    
    def _archive_captured_segments(self):
        if not os.path.exists(self.captured_dir):
            return

        captured_files = [f for f in os.listdir(self.captured_dir) if f.endswith('.wav')]
        if not captured_files:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = os.path.join(self.output_dir, "captured_history", timestamp)
        os.makedirs(archive_dir, exist_ok=True)

        for f in captured_files:
            src = os.path.join(self.captured_dir, f)
            dst = os.path.join(archive_dir, f)
            shutil.move(src, dst)
            print(f"{len(captured_files)} eski dosya 'captured_history/{timestamp}' klasörüne taşındı.")
    
    def process_live_audio(self, duration: int = None):
        """
        Canlı ses kaydını işler
        
        Parametreler:
            duration (int, optional): Kayıt süresi (saniye)
        """
        self._archive_captured_segments() # Önceki ses kayıtlarını taşı

        try:
            print("Canlı ses kaydı başlatılıyor...")
            self.capture_module.start_recording()
            
            if duration:
                print(f"{duration} saniye boyunca kayıt yapılacak...")
                time.sleep(duration)
                self.capture_module.stop_recording()
            else:
                print("Kaydı durdurmak için Ctrl+C tuşuna basın...")
                while True:
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            print("\nKayıt kullanıcı tarafından durduruldu.")
            self.capture_module.stop_recording()
        
        # Kaydedilen segmentleri işle
        self._process_captured_segments()
    
    def process_audio_file(self, audio_file: str):
        """
        Kayıtlı ses dosyasını işler
        
        Parametreler:
            audio_file (str): Ses dosyasının yolu
        """
        print(f"Ses dosyası işleniyor: {audio_file}")
        
        # Dosyayı önişle
        processed_file, _ = self.preprocess_module.process_file(audio_file)
        
        # Diarization işlemi
        diarization, _ = self.diarization_module.process_audio(processed_file) # embeddings artık kullanılmıyor
        
        # Diarizasyon sonuçlarını output_manager'a ekle
        self.current_diarization_results = diarization
        for segment, track, speaker in diarization.itertracks(yield_label=True):
            self.output_manager.add_segment_result(
                segment_id=f"{os.path.basename(audio_file)}_{track}",
                speaker_id=speaker, # Doğrudan pyannote'un speaker ID'si
                start_time=segment.start,
                end_time=segment.end,
                confidence=1.0, # Pyannote doğrudan confidence vermeyebilir, varsayılan 1.0
                embedding=None # Artık embedding kullanmıyoruz
            )
        
        # Raporları oluştur (sadece terminal çıktısı ve grafik)
        self._generate_output()
    
    def _process_captured_segments(self):
        """Kaydedilen ses segmentlerini işler"""
        print("Kaydedilen segmentler işleniyor...")
        
        captured_files = [f for f in os.listdir(self.captured_dir) if f.endswith('.wav')]
        
        all_segments_for_timeline = [] # Tüm segmentleri tek bir yerden toplamak için
        
        for segment_file in captured_files:
            segment_path = os.path.join(self.captured_dir, segment_file)
            
            # Segmenti önişle
            processed_file, _ = self.preprocess_module.process_file(segment_path)
            
            # Diarization işlemi
            diarization, _ = self.diarization_module.process_audio(processed_file) # embeddings artık kullanılmıyor
            
            # Diarizasyon sonuçlarını output_manager'a ekle
            for segment, track, speaker in diarization.itertracks(yield_label=True):
                 # Segment ID'sini benzersiz yapmak için dosya adını dahil et
                segment_full_id = f"{os.path.basename(segment_file).replace('.wav', '')}_{track}"
                
                self.output_manager.add_segment_result(
                    segment_id=segment_full_id,
                    speaker_id=speaker, # Doğrudan pyannote'un speaker ID'si
                    start_time=segment.start,
                    end_time=segment.end,
                    confidence=1.0, # Pyannote doğrudan confidence vermeyebilir, varsayılan 1.0
                    embedding=None # Artık embedding kullanmıyoruz
                )
                
                # Tüm segmentleri zaman çizelgesi için birleştir
                all_segments_for_timeline.append({
                    "speaker_id": speaker,
                    "start_time": segment.start,
                    "end_time": segment.end
                })

        # Tüm segmentleri topladıktan sonra, bunları kullanarak toplam konuşmacı sayısını ve zaman çizelgesini oluştur
        self.current_diarization_results_combined = all_segments_for_timeline
        self._generate_output()
    
    def _generate_output(self):
        """Terminal çıktısını ve görselleştirmeyi oluşturur"""
        print("Sonuçlar değerlendiriliyor...")
        
        # Toplam benzersiz konuşmacı sayısını belirle
        unique_speakers = set()
        if hasattr(self, 'current_diarization_results'): # Tek dosya modu
            for segment, track, speaker in self.current_diarization_results.itertracks(yield_label=True):
                unique_speakers.add(speaker)
        elif hasattr(self, 'current_diarization_results_combined'): # Toplu segment modu
            for segment_info in self.current_diarization_results_combined:
                unique_speakers.add(segment_info["speaker_id"])

        total_speakers = len(unique_speakers)
        print(f"\nTespit Edilen Toplam Konuşmacı Sayısı: {total_speakers}")
        
        # Zaman çizelgesini oluştur
        timeline_file = self.output_manager.generate_timeline()
        if timeline_file:
            print(f"Konuşmacı Zaman Çizelgesi kaydedildi: {timeline_file}")
        else:
            print("Konuşmacı Zaman Çizelgesi oluşturulamadı.")


def main():
    parser = argparse.ArgumentParser(description="Konuşmacı Diarization Sistemi")
    parser.add_argument("--auth-token", help="Hugging Face API token'ı")
    parser.add_argument("--output-dir", default="results", help="Çıktı dizini")
    parser.add_argument("--live", action="store_true", help="Canlı ses kaydı modu")
    parser.add_argument("--duration", type=int, help="Kayıt süresi (saniye)")
    parser.add_argument("--input-file", help="İşlenecek ses dosyası")
    
    args = parser.parse_args()
    
    system = SpeakerDiarizationSystem(
        auth_token=args.auth_token,
        output_dir=args.output_dir
    )
    
    try:
        if args.live:
            system.process_live_audio(args.duration)
        elif args.input_file:
            system.process_audio_file(args.input_file)
        else:
            print("Hata: --live veya --input-file parametrelerinden biri gerekli.")
            parser.print_help()
            
    except Exception as e:
        print(f"Hata: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())