import os
import json
import datetime
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

class NumpyEncoder(json.JSONEncoder):
    """NumPy veri tiplerini JSON'a dönüştürmek için özel encoder"""
    def default(self, obj):
        # Artık np.ndarray işlemeye gerek yok, sadece int/float kontrolü yeterli.
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        # elif isinstance(obj, np.ndarray): # KALDIRILDI
        #     return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)

class OutputManager:
    def __init__(self, 
                 output_dir: str = "output_results",
                 report_format: str = "json",
                 timeline_format: str = "png"):
        """
        Sonuç yönetimi ve çıktı modülü
        
        Parametreler:
            output_dir (str): Çıktı dosyalarının kaydedileceği dizin
            report_format (str): Rapor formatı (json, txt, html) (Artık kullanılmıyor ama uyumluluk için tutulabilir)
            timeline_format (str): Zaman çizelgesi formatı (png, pdf)
        """
        self.output_dir = output_dir
        self.report_format = report_format
        self.timeline_format = timeline_format
        
        # Çıkış dizinini oluştur
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Rapor verilerini saklamak için yapılar
        self.speaker_segments = defaultdict(list)  # Konuşmacı bazlı segmentler
        self.global_timeline = []  # Tüm konuşma olayları
        self.speaker_stats = {}  # Konuşmacı istatistikleri
    
    def add_segment_result(self, 
                          segment_id: str,
                          speaker_id: str,
                          start_time: float,
                          end_time: float,
                          confidence: float,
                          embedding: Optional[np.ndarray] = None): # embedding artık kullanılmıyor
        """
        Bir segment sonucunu ekler
        
        Parametreler:
            segment_id (str): Segment ID'si
            speaker_id (str): Konuşmacı ID'si
            start_time (float): Başlangıç zamanı (saniye)
            end_time (float): Bitiş zamanı (saniye)
            confidence (float): Güven skoru
            embedding (np.ndarray, optional): Konuşmacı gömülüm vektörü (Şimdi None olarak geçirilecek)
        """
        # Segment bilgisini ekle
        segment_info = {
            "segment_id": segment_id,
            "start_time": float(start_time),
            "end_time": float(end_time),
            "duration": float(end_time - start_time),
            "confidence": float(confidence)
        }
        
        # Konuşmacı bazlı segmentleri güncelle
        self.speaker_segments[speaker_id].append(segment_info)
        
        # Global zaman çizelgesini güncelle
        self.global_timeline.append({
            "speaker_id": speaker_id,
            **segment_info
        })
        
        # Konuşmacı istatistiklerini güncelle (bu bölüm timeline için hala gerekli olabilir)
        if speaker_id not in self.speaker_stats:
            self.speaker_stats[speaker_id] = {
                "total_duration": 0.0,
                "segment_count": 0,
                "avg_confidence": 0.0,
                "first_seen": float(start_time),
                "last_seen": float(end_time)
            }
        
        stats = self.speaker_stats[speaker_id]
        stats["total_duration"] += segment_info["duration"]
        stats["segment_count"] += 1
        stats["avg_confidence"] = (
            (stats["avg_confidence"] * (stats["segment_count"] - 1) + confidence) / 
            stats["segment_count"]
        )
        stats["first_seen"] = min(stats["first_seen"], float(start_time))
        stats["last_seen"] = max(stats["last_seen"], float(end_time))
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Detaylı bir rapor oluşturur (Artık ana akışta çağrılmıyor)
        """
        # Bu metot artık main.py'den çağrılmayacak. İsterseniz içeriğini boşaltabilir veya silebilirsiniz.
        print("generate_report çağrıldı ama main akışta kullanılmıyor.")
        return ""
    
    def _save_txt_report(self, report_data: Dict, output_file: str):
        """TXT formatında rapor kaydeder (Artık ana akışta çağrılmıyor)"""
        pass
    
    def _save_html_report(self, report_data: Dict, output_file: str):
        """HTML formatında rapor kaydeder (Artık ana akışta çağrılmıyor)"""
        pass
    
    def generate_timeline(self, output_file: Optional[str] = None) -> str:
        """
        Konuşmacı zaman çizelgesini görselleştirir
        
        Parametreler:
            output_file (str, optional): Çıktı dosyasının yolu
            
        Döndürür:
            str: Görselleştirme dosyasının yolu
        """
        if output_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                self.output_dir, 
                f"speaker_timeline_{timestamp}.{self.timeline_format}"
            )
        
        # Zaman çizelgesi verilerini sırala
        timeline_data = sorted(self.global_timeline, key=lambda x: x["start_time"])
        
        if not timeline_data:
            print("Görselleştirilecek veri yok.")
            return None
        
        # Görselleştirme için matplotlib figürü oluştur
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        
        # Konuşmacı sayısını ve renklerini belirle
        speakers = sorted(list(self.speaker_stats.keys())) # Konuşmacıları sırala
        colors = plt.cm.tab10(np.linspace(0, 1, len(speakers)))
        
        # Zaman aralığını belirle
        min_time = min(event["start_time"] for event in timeline_data)
        max_time = max(event["end_time"] for event in timeline_data)
        
        # Her konuşmacı için zaman çizelgesini çiz
        for i, speaker_id in enumerate(speakers):
            y_pos = i + 0.5
            
            # Bu konuşmacıya ait segmentleri bul
            speaker_segments = [event for event in timeline_data if event["speaker_id"] == speaker_id]
            
            for segment in speaker_segments:
                # Segment çubuğunu çiz
                rect = Rectangle(
                    (segment["start_time"], y_pos - 0.4),
                    segment["duration"],
                    0.8,
                    color=colors[i],
                    alpha=0.7,
                    label=speaker_id if i == 0 else None # Sadece ilk çubukta label koy
                )
                ax.add_patch(rect)
                
                # Segment üzerine süre bilgisini yaz
                if segment["duration"] > 0.5: # Daha kısa segmentler için metin yazma
                    ax.text(
                        segment["start_time"] + segment["duration"] / 2,
                        y_pos,
                        f"{segment['duration']:.1f}s",
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=8,
                        color='black'
                    )
        
        # Eksen ayarları
        ax.set_yticks(range(1, len(speakers) + 1))
        ax.set_yticklabels(speakers)
        ax.set_xlabel('Zaman (saniye)')
        ax.set_title('Konuşmacı Zaman Çizelgesi')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Zaman etiketlerini ekle (min_time ve max_time'ı kullanarak)
        # Sadece min_time ve max_time arasındaki gerçek saniye değerlerini kullanın.
        # datetime.datetime.fromtimestamp kullanımı hatalı olabilir, zaman çizelgesi 0'dan başlamalı.
        # Basitçe saniye cinsinden etiketler gösterelim.
        
        num_ticks = 10 # Gösterilecek etiket sayısı
        tick_interval = (max_time - min_time) / (num_ticks - 1) if num_ticks > 1 else 1
        time_ticks = [min_time + i * tick_interval for i in range(num_ticks)]
        time_labels = [f"{t:.1f}" for t in time_ticks] # Sadece saniye değerlerini göster
        plt.xticks(time_ticks, time_labels, rotation=45)

        # Legend'ı ekle
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles)) # Sadece benzersiz label'ları al
        ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
        
        # Görselleştirmeyi kaydet
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Zaman çizelgesi kaydedildi: {output_file}")
        return output_file
    
    def export_to_excel(self, output_file: Optional[str] = None) -> str:
        """
        Sonuçları Excel dosyasına aktarır (Artık ana akışta çağrılmıyor)
        """
        # Bu metot artık main.py'den çağrılmayacak. İsterseniz içeriğini boşaltabilir veya silebilirsiniz.
        print("export_to_excel çağrıldı ama main akışta kullanılmıyor.")
        return ""