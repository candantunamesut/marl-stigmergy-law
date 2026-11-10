# Multi-Agent Collective Survival & Resource Sharing Simulation
**Kollektif Hayatta Kalma, Kaynak Paylaşımı ve Çok Ajanlı Pekiştime/Evrimsel Simülasyon Sistemi**

## Proje Özeti
Bu proje, kaynak kıtlığı bulunan bir ızgara dünyasında (20x20 Grid) bağımsız ajanların hayatta kalma, kaynak toplama ve takım içi işbirliği dinamiklerini inceleyen bir Çok Ajanlı Pekiştirmeli Öğrenme (MARL) ve Nöroevrim (Neuroevolution / GA) simülasyonudur.

---

## Yapılan Güncellemeler ve Yeni Kurallar

1. **Başlangıç ve Maksimum Can (60 HP):**
   - Ajanlar oyuna `60.0 HP` sağlıkla başlar (`MAX_HEALTH = 60.0`).
   - Sağlık oranı gözlem vektörüne `[0, 1]` aralığında normalleştirilerek verilir.

2. **Yemek Bulma ve Ödül Sistemi (+60 HP & +60 Ödül, %10 Takım Payı):**
   - Bir ajan yiyecek hücresine ulaştığında **+60.0 HP** can kazanır (maksimum 60 ile sınırlandırılır).
   - Yemeği bulan ajan **+60.0 puan** ödül kazanır (`FOOD_FOUND = 60.0`).
   - Takım paylaşımı kuralı uyarınca, **hayatta olan diğer tüm takım arkadaşları yemeğin %10'u kadar (+6.0 puan) ödül kazanır**.

3. **Harita Sisteminin Kaldırılması (Sadeleştirilmiş Hareket Uzayı):**
   - Karışıklığa ve can kaybına yol açan paylaşımlı harita (shared map), haritayı açma (`READ_MAP`) ve haritaya işaretleme (`WRITE_TO_MAP`) eylemleri tamamen kaldırıldı.
   - Ajanların aksiyon uzayı 5 temel harekete indirgendi:
     - `0: MOVE_UP` (Yukarı)
     - `1: MOVE_DOWN` (Aşağı)
     - `2: MOVE_LEFT` (Sola)
     - `3: MOVE_RIGHT` (Sağa)
     - `4: STAY` (Olduğu yerde bekle)
   - Gözlem uzayı harita skaları çıkarılarak **21 boyuta** güncellendi (`[0:2]` koordinatlar, `[2]` can oranı, `[3:12]` çevredeki ajanlar 3x3, `[12:21]` çevredeki yiyecekler 3x3).

4. **Metabolik Can Kaybı (0.2 HP / Adım):**
   - Adım başına sağlık kaybı `0.2 HP` olarak ayarlandı. Ajanlar aç kalmadan ~300 adım boyunca çevreyi keşfedebilir.

5. **Takım Arkadaşı Ölüm Cezası (-10 HP):**
   - Bir ajan canı bittiğinde öldüğünde (`DEATH = -100.0` ceza puanı), **hayatta olan diğer tüm takım arkadaşları -10.0 HP can kaybeder**.
   - Bu kural ajanların takım arkadaşlarının hayatta kalmasına dolaylı olarak bağlı olmasını sağlar.

---

## Dosya Yapısı

- `config.py`: Ortam hiperparametreleri (can limitleri, ödül katsayıları, ceza değerleri).
- `env.py`: PettingZoo `ParallelEnv` standardına uygun 20x20 çok ajanlı çevre motoru.
- `model.py`: 21-boyutlu gözlem ve 5-aksiyonlu PyTorch Actor-Critic sinir ağı (Nöroevrim için düz ağırlık arayüzü içerir).
- `train.py`: Çok Ajanlı PPO (Proximal Policy Optimization) eğitim boru hattı.
- `train_evolution.py`: Genetik Algoritma (GA / Neuroevolution) ile popülasyon bazlı politika optimizasyonu.
- `compare_algorithms.py`: Random Baseline, PPO ve Nöroevrim politikalarını özdeş test seed'leri üzerinde karşılaştıran benchmark aracı.
- `visualize.py`: Simülasyonu terminalde canlı ızgara olarak adım adım gösterir ve tarayıcıda izlenebilir zengin bir etkileşimli HTML oynatıcısı (`simulation_playback.html`) üretir.
- `simulation_playback.html`: Adım adım kaydırıcı, oynat/durdur butonları, hız seçenekleri, renkli can barları ve olay kayıtları içeren HTML arayüzü.
- `test_env.py`: Tüm yeni kuralların doğruluğunu test eden birim testi.
- `record_detailed_episodes.py`: Ajanların episode bazında tüm hareket ve olaylarını metin ve JSON olarak kaydeder.

---

## Çalıştırma Komutları

### 1. Ortam Kurallarını Test Etme
```bash
py test_env.py
```

### 2. PPO Modeli Eğitimi
```bash
py train.py
```

### 3. Nöroevrim (Genetik Algoritma) Eğitimi
```bash
py train_evolution.py
```

### 4. Algoritma Karşılaştırma Benchmark'ı
```bash
py compare_algorithms.py
```

### 5. Görsel Simülasyon ve HTML Oynatıcı Oluşturma
```bash
py visualize.py --steps 80
```
Bu komut terminalde simülasyonu çalıştırır ve doğrudan tarayıcınızda açabileceğiniz `simulation_playback.html` dosyasını oluşturur.
