import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import librosa
import soundfile as sf
from pyannote.audio import Pipeline
from pyannote.core import Segment, Annotation
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import io
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional, Union
from config import HUGGINGFACE_TOKEN

class AudioAnalyzer:
    def __init__(self,
                 model_name: str = "pyannote/speaker-diarization-3.1",
                 # embedding_model: str = "pyannote/embedding-3.1", # KALDIRILDI
                 min_speakers: int = 1,
                 max_speakers: int = 4,
                 min_speech_duration: float = 0.5,
                 min_silence_duration: float = 0.5,
                 overlap_threshold: float = 0.5,
                 output_dir: str = "results"):
        """
        Konuşmacı diarization modülü
        
        Parametreler:
            model_name (str): Kullanılacak diarization modeli
            # embedding_model (str): Konuşmacı gömülüm vektörleri için model # KALDIRILDI
            min_speakers (int): Minimum konuşmacı sayısı tahmini
            max_speakers (int): Maksimum konuşmacı sayısı tahmini
            min_speech_duration (float): Minimum konuşma süresi
            min_silence_duration (float): Minimum sessizlik süresi
            overlap_threshold (float): Kesişim eşiği
            output_dir (str): Diarization sonuçlarının kaydedileceği dizin
        """
        self.model_name = model_name
        # self.embedding_model = embedding_model # KALDIRILDI
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.overlap_threshold = overlap_threshold
        
        # Çıkış dizinini oluştur
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("Diarization modelleri yükleniyor...")
        try:
            # PyAnnote diarization pipeline
            self.diarization_pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=HUGGINGFACE_TOKEN
            )
            
            # Konuşmacı sayısı ve segmentasyon parametrelerini ayarla
            self.diarization_pipeline.instantiate({
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": 10
                }
            })
            
            print("Diarization modelleri başarıyla yüklendi.")
        except Exception as e:
            print(f"Model yükleme hatası: {str(e)}")
            print("PyAnnote modelleri yüklenirken bir sorun oluştu.")
            print("Lütfen Hugging Face API token'ınızı kontrol edin veya internetinizi kontrol edin.")
            raise e
    
    def process_audio(self, audio_file: str, num_speakers: Optional[int] = None) -> Tuple[Annotation, Dict]:
        """
        Ses dosyasında konuşmacı diarization işlemi yapar
        
        Parametreler:
            audio_file (str): İşlenecek ses dosyasının yolu
            num_speakers (int, optional): Belirtilirse, konuşmacı sayısı sabitleniyor
            
        Döndürür:
            tuple: (diarization, embeddings_dict)
              - diarization: PyAnnote Annotation nesnesi
              - embeddings_dict: Konuşmacı kimliklerine göre gömülüm vektörleri (şimdi boş bir dict olacak)
        """
        print(f"Diarization işlemi başlatılıyor: {audio_file}")
        
        try:
            # Diarization işlemini çalıştır
            if num_speakers is not None:
                diarization = self.diarization_pipeline(
                    audio_file,
                    num_speakers=num_speakers
                )
            else:
                diarization = self.diarization_pipeline(
                    audio_file,
                    min_speakers=self.min_speakers,
                    max_speakers=self.max_speakers
                )
            
            # # Konuşmacı gömülümleri için özellik çıkarımı # KALDIRILDI
            # embeddings_dict = self._extract_speaker_embeddings(audio_file, diarization)
            
            # Sonuçları kaydet
            output_file = os.path.join(
                self.output_dir, 
                f"diarization_{os.path.basename(audio_file).replace('.wav', '.json')}"
            )
            with open(output_file, "w") as f:
                f.write(str(diarization)) # Diarization Annotation objesini string olarak kaydeder

            # Terminale PyAnnote'un raw çıktısını yazdır
            print("\nPyAnnote Diarizasyon Çıktısı:")
            for segment, track, speaker in diarization.itertracks(yield_label=True):
                print(f"[ {segment.start:.3f} --> {segment.end:.3f}] {track} {speaker}")
            print("-" * 40)
            
            print(f"Diarization tamamlandı. {len(diarization.labels())} konuşmacı tespit edildi.")
            return diarization, {} # Boş bir embedding sözlüğü döndür
            
        except Exception as e:
            print(f"Diarization işlemi sırasında hata: {str(e)}")
            raise e
    
    # # _extract_speaker_embeddings metodu tamamen KALDIRILDI
    # def _extract_speaker_embeddings(self, audio_file: str, diarization: Annotation) -> Dict[str, np.ndarray]:
    #     """
    #     Her konuşmacı için gömülüm vektörlerini çıkarır
    #     """
    #     return {} # Boş bir sözlük döndür
    
    def process_batch(self, audio_files: List[str], num_speakers: Optional[int] = None) -> Dict[str, Tuple[Annotation, Dict]]:
        """
        Birden fazla ses dosyasını toplu olarak işler
        
        Parametreler:
            audio_files (list): İşlenecek ses dosyalarının yolları
            num_speakers (int, optional): Sabit konuşmacı sayısı
            
        Döndürür:
            dict: Dosya adlarına göre diarization sonuçları
        """
        results = {}
        
        for audio_file in audio_files:
            try:
                diarization, embeddings = self.process_audio(audio_file, num_speakers)
                results[audio_file] = (diarization, embeddings)
            except Exception as e:
                print(f"Dosya işleme hatası ({audio_file}): {str(e)}")
        
        return results
    
    def visualize_diarization(self, audio_file: str, diarization: Annotation, output_file: Optional[str] = None) -> str:
        """
        Diarization sonuçlarını görselleştirir
        
        Parametreler:
            audio_file (str): Ses dosyasının yolu
            diarization (Annotation): PyAnnote diarization sonucu
            output_file (str, optional): Çıktı dosyasının yolu
            
        Döndürür:
            str: Kaydedilen görselleştirme dosyasının yolu
        """
        # Ses dosyasının süresini al
        audio_duration = librosa.get_duration(filename=audio_file)
        
        # Görselleştirme için matplotlib figürü oluştur
        plt.figure(figsize=(12, 4))
        ax = plt.gca()
        
        # Her konuşmacı için farklı renk
        colors = plt.cm.tab10(np.linspace(0, 1, len(diarization.labels())))
        
        # Konuşmacı segmentlerini çiz
        for i, (segment, track, speaker) in enumerate(diarization.itertracks(yield_label=True)):
            color_idx = list(diarization.labels()).index(speaker)
            rect = Rectangle(
                (segment.start, 0), 
                segment.end - segment.start, 
                1, 
                color=colors[color_idx],
                alpha=0.6,
                label=speaker if speaker not in plt.gca().get_legend_handles_labels()[1] else None
            )
            ax.add_patch(rect)
            
            # Konuşmacı etiketini segment üzerine yaz
            ax.text(
                segment.start + (segment.end - segment.start) / 2, 
                0.5, 
                speaker,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=8,
                color='black'
            )
        
        # Eksen ayarları
        ax.set_xlim(0, audio_duration)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel('Zaman (saniye)')
        ax.set_title(f'Konuşmacı Diarization: {os.path.basename(audio_file)}')
        ax.legend(loc='upper right')
        
        # Görselleştirmeyi kaydet
        if output_file is None:
            output_file = os.path.join(
                self.output_dir, 
                f"diarization_viz_{os.path.basename(audio_file).replace('.wav', '.png')}"
            )
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        print(f"Diarization görselleştirmesi kaydedildi: {output_file}")
        return output_file
    
    def get_speaker_timeline(self, diarization: Annotation) -> Dict[str, List[Tuple[float, float]]]:
        """
        Her konuşmacı için konuşma zaman aralıklarını çıkarır
        
        Parametreler:
            diarization (Annotation): PyAnnote diarization sonucu
            
        Döndürür:
            dict: Konuşmacı bazlı zaman aralıkları
        """
        timeline = {}
        
        for segment, track, speaker in diarization.itertracks(yield_label=True):
            if speaker not in timeline:
                timeline[speaker] = []
            
            timeline[speaker].append((segment.start, segment.end))
        
        # Zaman aralıklarını sırala
        for speaker in timeline:
            timeline[speaker].sort(key=lambda x: x[0])
        
        return timeline
    
    def get_speaker_stats(self, diarization: Annotation) -> Dict[str, Dict]:
        """
        Her konuşmacı için konuşma istatistiklerini hesaplar
        
        Parametreler:
            diarization (Annotation): PyAnnote diarization sonucu
            
        Döndürür:
            dict: Konuşmacı bazlı istatistikler
        """
        stats = {}
        
        # Konuşmacı zaman aralıklarını al
        timeline = self.get_speaker_timeline(diarization)
        
        # Her konuşmacı için istatistikleri hesapla
        for speaker, intervals in timeline.items():
            total_duration = sum(end - start for start, end in intervals)
            avg_segment_duration = total_duration / len(intervals) if intervals else 0
            
            stats[speaker] = {
                'toplam_konusma_suresi': total_duration,
                'segment_sayisi': len(intervals),
                'ortalama_segment_suresi': avg_segment_duration,
                'ilk_konusma': intervals[0][0] if intervals else None,
                'son_konusma': intervals[-1][1] if intervals else None
            }
        
        return stats


if __name__ == "__main__":
    # Modülü test etmek için örnek kullanım
    import sys
    
    if len(sys.argv) < 2:
        print("Kullanım: python3 diarization_module.py <input_audio_file> [auth_token]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    auth_token = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        diarization_module = AudioAnalyzer(
            # auth_token=auth_token, # Sadece config.py'den alınmalı
            min_speakers=1,
            max_speakers=5
        )
        
        diarization, embeddings = diarization_module.process_audio(input_file)
        
        # Sonuçları görselleştir
        diarization_module.visualize_diarization(input_file, diarization)
        
        # Konuşmacı istatistiklerini göster
        stats = diarization_module.get_speaker_stats(diarization)
        for speaker, speaker_stats in stats.items():
            print(f"\nKonuşmacı: {speaker}")
            for stat_name, stat_value in speaker_stats.items():
                print(f"  {stat_name}: {stat_value}")
        
        print(f"\nToplam {len(stats)} konuşmacı tespit edildi.")
        
    except Exception as e:
        print(f"Hata: {str(e)}")
        sys.exit(1)