from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import random
import fuzzywuzzy.fuzz
import fuzzywuzzy.process
from data import egitim_verisi
from knowledge_base import HAYVAN_BILGILERI 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import sqlite3
import json
import datetime


# VERİTABANI YÖNETİMİ (SQLite)

DATABASE = 'chatbot_gecmis.db'

def get_db_connection():
    """Veritabanı bağlantısını kurar ve tabloyu oluşturur."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row 
    return conn

def init_db():
    """Uygulama ilk kez çalıştığında veritabanı tablosunu oluşturur."""
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS sohbetler (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            baslik TEXT NOT NULL,
            gecmis TEXT NOT NULL,
            tarih TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# 1. KRİTİK FONKSİYON TANIMI (pickle için zorunlu)

def tokenize_ve_kok_bul(metin):
    """Bu fonksiyon, model yüklenirken pickle tarafından aranır."""
    return metin.lower().split()

# 2. MODELLERİ YÜKLEME
try:
    with open('yeni_model.pkl', 'rb') as f:
        pipeline = pickle.load(f) 
    with open('yeni_le.pkl', 'rb') as f:
        le = pickle.load(f)
except FileNotFoundError:
    print("Hata: Model dosyaları bulunamadı. Lütfen önce model.py dosyasını yeniden çalıştırın.")


# 3. YARDIMCI FONKSİYONLAR (Varlık ve Metrik Tanıma)

def varlik_tanima(soru):
    """Soru içindeki tüm geçerli hayvan adlarını liste veya tek bir string olarak bulur."""
    hayvan_isimleri = list(HAYVAN_BILGILERI.keys())
    bulunan_hayvanlar = []
    
    for hayvan in hayvan_isimleri:
        if fuzzywuzzy.fuzz.partial_ratio(hayvan, soru.lower()) >= 80:
            if hayvan not in bulunan_hayvanlar:
                bulunan_hayvanlar.append(hayvan)
    
    if len(bulunan_hayvanlar) > 1:
        return bulunan_hayvanlar
    elif len(bulunan_hayvanlar) == 1:
        return bulunan_hayvanlar[0]
    
    return None

def extract_metric(text):
    """Kullanıcı girdisinden metrik adını (hız, ağırlık, ömür, güç, gebelik) çıkarır. Çekimli kelimeler için kural tabanlıdır."""
    text = text.lower()
    
    # 1. Ağırlık Kontrolü (Ağırlık, Ağırlığını, Kilo, Kilosu, Ağır)
    if "ağırlı" in text or "kilo" in text or "ağır" in text:
        return "ağırlık_kg", "ağırlık"
        
    # 2. Boyut Kontrolü (Boyut, Uzunluk, Büyük, Boyu)
    if "boyut" in text or "uzun" in text or "büyük" in text or "boyu" in text:
        return "boyut_m", "boyut"
        
    # 3. Ömür Kontrolü (Ömür, Yaşam, Yaşar)
    if "ömür" in text or "yaşam" in text or "yaşar" in text:
        return "omur", "ömür"
        
    # 4. Güç Kontrolü (Güçlü, Gücü)
    if "güç" in text or "güçlü" in text:
        return "guc_puani", "güç puanı"
        
    # 5. Gebelik Kontrolü (Gebelik, Hamilelik)
    if "gebelik" in text or "hamilelik" in text:
        return "gebelik_gun", "gebelik süresi"
        
    # 6. Hız Kontrolü (Hız, Hızlı, Koşar) (Varsayılan olarak son)
    if "hız" in text or "hızlı" in text or "koşar" in text:
        return "hiz", "hız"
            
    # Eğer spesifik metrik bulunamazsa varsayılan olarak "hız" döner
    return "hiz", "hız" 

def perform_comparison(hayvan_isimleri, metrik_key, metrik_tr):
    """Çoklu hayvanlar arasında belirlenen metriğe göre karşılaştırma yapar."""
    if len(hayvan_isimleri) < 2:
        return "Karşılaştırma yapmak için lütfen en az iki hayvan adı belirtin."

    karsilastirma_listesi = []
    
    for hayvan in hayvan_isimleri:
        bilgi = HAYVAN_BILGILERI.get(hayvan)
        deger = bilgi.get(metrik_key) if bilgi else None
        
        if deger is not None:
            try:
                # Değeri sayısal hale getir
                sayisal_deger = float(str(deger).replace(',', '.'))
                karsilastirma_listesi.append({'isim': hayvan.capitalize(), metrik_key: sayisal_deger})
            except ValueError:
                continue

    if not karsilastirma_listesi:
        return f"Üzgünüm, sorunuzdaki hayvanların '{metrik_tr}' bilgisini bulamadım veya değerler karşılaştırılamıyor."
        
    karsilastirma_listesi.sort(key=lambda x: x[metrik_key], reverse=True)
    
    # Birim tespiti
    birim = {'hiz': ' km/s', 'ağırlık_kg': ' kg', 'boyut_m': ' m', 'omur': ' yıl', 'gebelik_gun': ' gün', 'guc_puani': ' puan'}.get(metrik_key, '')

    yanit = f"İstediğiniz hayvanların {metrik_tr.upper()} sıralaması: \n\n"
    sira = 1
    for item in karsilastirma_listesi:
        yanit += f"{sira}. {item['isim']}: {item[metrik_key]}{birim}\n"
        sira += 1
        
    en_iyi = karsilastirma_listesi[0]['isim']
    yanit += f"\nSonuç: Bu gruptaki en yüksek {metrik_tr} değerine sahip hayvan **{en_iyi}**'dır."
    
    return yanit


def find_superlative(metrik_key, metrik_tr):
    """Veri tabanındaki en yüksek değere sahip hayvanı bulur (En Hızlı, En Güçlü vb.)."""
    en_iyi_hayvan = None
    en_yuksek_deger = -1
    
    birim = {'hiz': ' km/s', 'ağırlık_kg': ' kg', 'boyut_m': ' m', 'omur': ' yıl', 'gebelik_gun': ' gün', 'guc_puani': ' puan'}.get(metrik_key, '')

    for hayvan_adi, bilgiler in HAYVAN_BILGILERI.items():
        deger = bilgiler.get(metrik_key)
        
        if deger is not None:
            try:
                sayisal_deger = float(str(deger).replace(',', '.'))
                
                if sayisal_deger > en_yuksek_deger:
                    en_yuksek_deger = sayisal_deger
                    en_iyi_hayvan = hayvan_adi.capitalize()
            except ValueError:
                continue
    
    if en_iyi_hayvan:
        yanit = (
            f"Tüm veri tabanında, **{metrik_tr.capitalize()}** özelliğinde en yüksek değere sahip hayvan:\n\n"
            f"**{en_iyi_hayvan}**\n"
            f"Değer: **{en_yuksek_deger}{birim}**"
        )
        return yanit
    else:
        return f"Üzgünüm, {metrik_tr} özelliğine sahip herhangi bir hayvan bulamadım."


def get_random_answer(intent_tag):
    """Eğitim verisinden rastgele bir cevap çeker."""
    for intent in egitim_verisi["intentler"]:
        if intent["tag"] == intent_tag:
            return random.choice(intent["cevaplar"])
    return "Anlayamadım."


# 4. YANIT ÜRETME FONKSİYONU (YZ Mantığı)
def yz_botu_yanitla(kullanici_sorusu):
    
    # 1. Niyet Tahmini
    tahmin_sonucu = pipeline.predict([kullanici_sorusu])
    tahmin_niyeti = le.inverse_transform(tahmin_sonucu)[0]
    
    tahmin_olasiliklari = pipeline.predict_proba([kullanici_sorusu])[0]
    en_yuksek_olasilik = np.max(tahmin_olasiliklari)

    # --- ÖNCELİKLİ ANAHTAR KELİME KONTROLÜ (Misclassification Önleyici)
    kullanici_sorusu_alt = kullanici_sorusu.lower()
    
    # Sayı Sorgulama (Kullanıcı girdisine birebir odaklanır)
    if "kaç hayvan" in kullanici_sorusu_alt or "hayvan sayısı" in kullanici_sorusu_alt or "kaç hyavan" in kullanici_sorusu_alt:
        tahmin_niyeti = "hayvan_sayisi_sorgulama"
    # Kapsam Sorgulama
    elif "kategori" in kullanici_sorusu_alt or "kapsam" in kullanici_sorusu_alt or "türleri" in kullanici_sorusu_alt:
        tahmin_niyeti = "kapsam_sorgulama"
    # --- KONTROL SONU ---

    # 2. Varlık Tanıma
    hayvan_varliklari = varlik_tanima(kullanici_sorusu)
    hayvan_adi = hayvan_varliklari[0] if isinstance(hayvan_varliklari, list) else hayvan_varliklari
    
    # 3. Güven Kontrolü
    if en_yuksek_olasilik < 0.10 and tahmin_niyeti not in ["merhaba", "nasılsın", "yardım", "tesekkur", "hos_cakal", "niyet_takim", "niyet_sadik", "niyet_kim", "hayvan_sayisi_sorgulama", "kapsam_sorgulama"]:
        return f"Sorunuzu tam olarak anlayamadım. (Tahmin: {tahmin_niyeti}, Eminiyet: {en_yuksek_olasilik:.2f}). Lütfen daha net ifade edin."

    # A. TEMEL NİYETLER
    if tahmin_niyeti in ["merhaba", "nasılsın", "yardım", "tesekkur", "hos_cakal", "niyet_takim", "niyet_sadik", "niyet_kim"]:
        return get_random_answer(tahmin_niyeti)
                
    # B. KARMAŞIK NİYETLER
    
    # 1. Karşılaştırma Niyeti Kontrolü
    elif tahmin_niyeti == "karsilastirma" and isinstance(hayvan_varliklari, list) and len(hayvan_varliklari) >= 2:
        metrik_key, metrik_tr = extract_metric(kullanici_sorusu)
        return perform_comparison(hayvan_varliklari, metrik_key, metrik_tr)
        
    # 2. EN İYİ (Superlative) Sorgulama Niyeti Kontrolü 
    elif tahmin_niyeti == "en_iyi_sorgulama":
        metrik_key, metrik_tr = extract_metric(kullanici_sorusu)
        return find_superlative(metrik_key, metrik_tr)

    # 3. Hayvan Sayısı Sorgulama Niyeti Kontrolü (Keyword Override buraya yönlendirdi)
    elif tahmin_niyeti == "hayvan_sayisi_sorgulama":
        toplam_sayi = len(HAYVAN_BILGILERI)
        return f"Şu anda bilgi tabanımda **{toplam_sayi}** farklı hayvan hakkında detaylı bilgi bulunmaktadır. Bu, oldukça geniş bir kapsamdır!"

    # 4. Kapsam ve Kategori Sayısı Sorgulama Niyeti Kontrolü (Keyword Override buraya yönlendirdi)
    elif tahmin_niyeti == "kapsam_sorgulama":
        kategori_sayilari = {}
        toplam_sayi = 0
        
        for bilgiler in HAYVAN_BILGILERI.values():
            ana_sinif = bilgiler.get('ana_sinif', 'Bilinmeyen')
            kategori_sayilari[ana_sinif] = kategori_sayilari.get(ana_sinif, 0) + 1
            toplam_sayi += 1
            
        yanit = f"Bilgi tabanım, toplam **{toplam_sayi}** hayvanı kapsamakta ve bu hayvanlar başlıca şu ana sınıflara ayrılmaktadır:\n\n"
        
        for kategori, sayi in sorted(kategori_sayilari.items(), key=lambda item: item[1], reverse=True):
             yanit += f"**{kategori}**: {sayi} tür\n"
             
        yanit += "\nBu listelenen tüm hayvanlar hakkında karşılaştırmalı bilgiye sahibim."
        
        return yanit

    # 5. Ağırlık Sorgulama Niyeti Kontrolü
    elif hayvan_adi and tahmin_niyeti == "ağırlık_sorgulama":
        hayvan_adi = hayvan_adi.lower()
        bilgiler = HAYVAN_BILGILERI.get(hayvan_adi)
        
        if bilgiler and 'ağırlık_kg' in bilgiler:
            agirlik = bilgiler['ağırlık_kg']
            return f"**{hayvan_adi.capitalize()}** hayvanının ağırlığı: **{agirlik} kg**."
        elif bilgiler:
            return f"'{hayvan_adi}' için ağırlık bilgisi bulunmuyor."
        else:
            return f"Üzgünüm, '{hayvan_adi}' hakkında detaylı bilgi bulamadım."
            
    # 6. Yetenek Sorgulama Niyeti Kontrolü
    elif hayvan_adi and tahmin_niyeti == "yetenek_sorgulama":
        hayvan_adi = hayvan_adi.lower()
        bilgiler = HAYVAN_BILGILERI.get(hayvan_adi)
        
        if bilgiler and 'yetenekler' in bilgiler:
            yetenek = bilgiler['yetenekler']
            return f"{hayvan_adi.capitalize()}'ın yetenekleri: {yetenek}"
        elif bilgiler:
            return f"'{hayvan_adi}' için yetenek bilgisi bulunmuyor."
        else:
            return f"Üzgünüm, '{hayvan_adi}' için yetenek bilgisi bulamadım."
            
    # 7. Kategori Sorgulama Niyeti Kontrolü (Listeleyen kısım)
    elif tahmin_niyeti == "kategori_sorgulama":
        kategoriler = ["memeli", "kuş", "sürüngen", "balık", "amfibi", "omurgasız"]
        
        istenen_kategori = None
        for kategori in kategoriler:
            if kategori in kullanici_sorusu_alt: 
                istenen_kategori = kategori.capitalize()
                break
        
        if istenen_kategori:
            filtrelenmis_hayvanlar = [
                adi.capitalize() for adi, bilgiler in HAYVAN_BILGILERI.items()
                if bilgiler.get('ana_sinif') == istenen_kategori
            ]

            if filtrelenmis_hayvanlar:
                sayi = len(filtrelenmis_hayvanlar)
                liste_str = ", ".join(filtrelenmis_hayvanlar)
                
                yanit_metni = (
                    f"Bilgi tabanında **{istenen_kategori}** sınıfına ait toplam **{sayi}** hayvan bulundu.\n\n"
                    f"**Liste:** {liste_str}"
                )
                return yanit_metni
            else:
                return f"Üzgünüm, bilgi tabanında **{istenen_kategori}** sınıfına ait hayvan bulamadım."
        else:
            return "Hangi hayvan kategorisini (Memeli, Kuş, Sürüngen vb.) listelememi istersiniz?"


    # C. BİLGİ ÇEKME NİYETLERİ
    
    # 1. TÜM BİLGİLERİ ÇEKME NİYETİ
    elif hayvan_adi and tahmin_niyeti == "tum_bilgiler":
        try:
            hayvan_adi_lower = hayvan_adi.lower()
            bilgi_seti = HAYVAN_BILGILERI[hayvan_adi_lower]
            
            # Yeni metrikleri de dahil eden detaylı format
            yanit = f"**{hayvan_adi.capitalize()}** hakkında genel bilgiler:\n\n"
            yanit += f"**Sınıflandırma:** {bilgi_seti.get('siniflandirma', 'Bilinmiyor')}\n"
            yanit += f"**Ana Sınıf:** {bilgi_seti.get('ana_sinif', 'Bilinmiyor')}\n"
            yanit += f"**Beslenme:** {bilgi_seti.get('beslenme', 'Bilinmiyor')}\n"
            yanit += f"**Yaşam Alanı:** {bilgi_seti.get('yasama_alani', 'Bilinmiyor')}\n"
            yanit += f"**Ömür:** {bilgi_seti.get('omur', 'Bilinmiyor')} yıl\n"
            yanit += f"**Hız (Max):** {bilgi_seti.get('hiz', 'Bilinmiyor')} km/s\n"
            yanit += f"**Ağırlık:** {bilgi_seti.get('ağırlık_kg', 'Bilinmiyor')} kg\n"
            yanit += f"**Boyut:** {bilgi_seti.get('boyut_m', 'Bilinmiyor')} m\n"
            yanit += f"**Koruma Durumu:** {bilgi_seti.get('koruma_durumu', 'Bilinmiyor')}\n"
            yanit += f"**Güç Puanı:** {bilgi_seti.get('guc_puani', 'Bilinmiyor')}\n"
            yanit += f"**Gebelik Süresi:** {bilgi_seti.get('gebelik_gun', 'Bilinmiyor')} gün\n" 
            yanit += f"**Sosyal Yaşam:** {bilgi_seti.get('sosyal_yasam', 'Bilinmiyor')}\n" 
            yanit += f"**Yetenekler:** {bilgi_seti.get('yetenekler', 'Bilinmiyor')}"
            
            return yanit
            
        except KeyError:
            return f"Üzgünüm, '{hayvan_adi.capitalize()}' hakkında detaylı bilgiyi bulamadım."
            
    # 2. BASİT BİLGİ ÇEKME NİYETLERİ (Beslenme, Yaşam Alanı vb.)
    elif hayvan_adi and tahmin_niyeti in ["beslenme", "yasama_alani", "omur", "siniflandirma", "boyut"]:
        try:
            bilgi = HAYVAN_BILGILERI[hayvan_adi][tahmin_niyeti]
            return bilgi 
        except KeyError:
            return f"Üzgünüm, {hayvan_adi.capitalize()} hayvanı hakkında {tahmin_niyeti} bilgisini bulamadım."
            
    # Eğer niyet hayvan sorusu ama hayvan adı yoksa (FALLBACK)
    elif tahmin_niyeti in ["beslenme", "yasama_alani", "omur", "siniflandirma", "karsilastirma", "yetenek_sorgulama", "tum_bilgiler", "ağırlık_sorgulama", "en_iyi_sorgulama"] and not hayvan_adi:
        return "Hangi hayvan hakkında bilgi istediğinizi belirtir misiniz?"
        
    return "Üzgünüm, beklenmedik bir hata oluştu veya bu konuya henüz eğitim almadım."


# 5. FLASK UYGULAMASI VE API ROTLARI

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Ana sohbet API'si. Kullanıcı mesajını alır ve bot yanıtını döndürür."""
    data = request.get_json()
    kullanici_mesaji = data.get('message', '')
    bot_yaniti = yz_botu_yanitla(kullanici_mesaji)
    return jsonify({'response': bot_yaniti})

@app.route('/api/chats', methods=['GET'])
def get_chats():
    """Tüm kaydedilmiş sohbetlerin listesini (başlıkları) döndürür."""
    conn = get_db_connection()
    chats = conn.execute("SELECT id, baslik, strftime('%d.%m.%Y %H:%M', tarih) as tarih_okunur FROM sohbetler ORDER BY id DESC").fetchall()
    conn.close()
    
    chats_list = [dict(chat) for chat in chats]
    return jsonify(chats_list)

@app.route('/api/chat/<int:chat_id>', methods=['GET'])
def load_chat(chat_id):
    """Belirli bir sohbet ID'sine ait geçmişi yükler."""
    conn = get_db_connection()
    chat = conn.execute('SELECT gecmis FROM sohbetler WHERE id = ?', (chat_id,)).fetchone()
    conn.close()
    
    if chat is None:
        return jsonify({'error': 'Sohbet bulunamadı'}), 404
        
    return jsonify(json.loads(chat['gecmis']))

@app.route('/api/save', methods=['POST'])
def save_chat():
    """Mevcut sohbet geçmişini kaydeder."""
    data = request.get_json()
    baslik = data.get('baslik', 'İsimsiz Sohbet')
    gecmis = json.dumps(data['gecmis'])
    
    conn = get_db_connection()
    conn.execute(
        'INSERT INTO sohbetler (baslik, gecmis, tarih) VALUES (?, ?, datetime("now", "localtime"))',
        (baslik, gecmis)
    )
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Sohbet başarıyla kaydedildi!'}), 201

@app.route('/api/chat/<int:chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    """Belirli bir sohbet ID'sine ait kaydı siler."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT id FROM sohbetler WHERE id = ?', (chat_id,))
    if cursor.fetchone() is None:
        conn.close()
        return jsonify({'error': 'Sohbet bulunamadı'}), 404
    
    cursor.execute('DELETE FROM sohbetler WHERE id = ?', (chat_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Sohbet başarıyla silindi!'}), 200


if __name__ == '__main__':
    app.run(debug=True, port=5002, use_reloader=False)