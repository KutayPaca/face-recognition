import cv2 
import os
import time
from deepface import DeepFace

# Çalışma dizinini yazdır
print("Çalışma dizini:", os.getcwd())

# Dosya yolu
img_path = r"C:\PythonCodes\PythonProjects\face_recognition_deepface\Kutay.jpg"

# Dosyanın gerçekten var olup olmadığını kontrol et
if not os.path.exists(img_path):
    print(f"HATA: '{img_path}' dosyası bulunamadı!")
    exit()
else:
    print(f"'{img_path}' dosyası bulundu, okumaya çalışılıyor...")
    img = cv2.imread(img_path)
    
    if img is None:
        print("HATA: OpenCV resmi okuyamıyor! Dosya bozuk olabilir.")
        exit()
    else:
        print("Başarıyla okundu, ekrana gösteriliyor...")
        cv2.imshow("Test Görseli", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Bilinen yüzlerin bulunduğu dizin
db_path = "known_faces"
os.makedirs(db_path, exist_ok=True)  # Dizin yoksa oluştur

# Bilinen yüzleri işleme
img_save_path = os.path.join(db_path, "Kutay.jpg")
if not os.path.exists(img_save_path):
    cv2.imwrite(img_save_path, img)
    print(f"Resim {img_save_path} olarak kaydedildi.")
else:
    print("Resim zaten veritabanında mevcut.")

# Kamera akışını başlat
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Kamera açılamadı.")
    exit()

frame_count = 0
last_identity = None  # Son tanınan kimlik
last_update_time = time.time()  # Son güncelleme zamanı

while True:
    ret, frame = video_capture.read()
    
    if not ret:
        print("Kameradan görüntü alınamadı.")
        break

    if frame_count % 24 == 0:
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = DeepFace.find(frame_rgb, db_path=db_path, model_name="Facenet", enforce_detection=True)
            print(results)
        except Exception as e:
            print(f"Yüz tanıma hatası: {e}")
            results = []
            
        if isinstance(results, list) and len(results) > 0:
            for result in results:
                if "identity" in result and not result["identity"].empty:
                    identity_value = result["identity"].iloc[0]
                    name = os.path.basename(identity_value)
                    
                    if name == "Kutay.jpg":
                        last_identity = "Kutay"
                        last_update_time = time.time()
                    break
        else:
            last_identity = "Bilinmiyor"
            last_update_time = time.time()

    if last_identity is not None and time.time() - last_update_time < 1:
        color = (0, 255, 0) if last_identity != "Bilinmiyor" else (0, 0, 255)
        cv2.putText(frame, last_identity, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
