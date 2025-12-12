import numpy as np
import cv2
import requests
import matplotlib.pyplot as plt

#Dwa zdjęcia do wyboru
#IMG_URL = "https://upload.wikimedia.org/wikipedia/commons/3/3f/Fronalpstock_big.jpg"
IMG_URL = "https://upload.wikimedia.org/wikipedia/commons/1/1d/Penguin_in_Antarctica_jumping_out_of_the_water.jpg"


def download_image(url):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "image/*"
    }
    r = requests.get(url, headers=headers, timeout=20, allow_redirects=True)

    data = np.frombuffer(r.content, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    
    return img

def show_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,6))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title("Wczytane zdjęcie")
    plt.show()

def show_histograms(img):
    #Wyświetla histogram całego zdjęcia i kanałów R/G/B.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r = img[:,:,2]
    g = img[:,:,1]
    b = img[:,:,0]

    plt.figure(figsize=(10,8))

    # histogram luminancji
    plt.subplot(2,2,1)
    plt.title("Histogram luminancji")
    plt.hist(gray.ravel(), bins=256, range=(0,255))
    plt.xlim([0,255])

    # R
    plt.subplot(2,2,2)
    plt.title("Histogram R")
    plt.hist(r.ravel(), bins=256, range=(0,255))
    plt.xlim([0,255])

    # G
    plt.subplot(2,2,3)
    plt.title("Histogram G")
    plt.hist(g.ravel(), bins=256, range=(0,255))
    plt.xlim([0,255])

    # B
    plt.subplot(2,2,4)
    plt.title("Histogram B")
    plt.hist(b.ravel(), bins=256, range=(0,255))
    plt.xlim([0,255])

    plt.tight_layout()
    plt.show()

def assess_quality(img):
    
    #Prosty algorytm oceny jakości na podstawie histogramu:
    # clipping w cieniach (0)
    # clipping w światłach (255)
    # zakres tonalny (p1–p99)
    # odchylenie standardowe (kontrast)
    # średnia jasność (ekspozycja)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flat = gray.ravel()
    total = len(flat)

    # Histogram globalny
    pct_black = np.sum(flat < 10) / total
    pct_white = np.sum(flat > 245) / total
    mean = np.mean(flat)
    median = np.median(flat)
    std = np.std(flat)

    # zakres centralny
    mid_tones = np.sum((flat > 85) & (flat < 170)) / total

    # Histogram RGB
    r_mean = np.mean(img[:,:,2])
    g_mean = np.mean(img[:,:,1])
    b_mean = np.mean(img[:,:,0])
    color_range = max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean)

    issues = []

    # Reguły
    if pct_black > 0.02:
        issues.append("Zgniecione cienie.")

    if pct_white > 0.02:
        issues.append("Przepalenia.")

    if std < 50:
        issues.append(f"Niski kontrast (std = {std:.1f}).")

    if mid_tones > 0.85:
        issues.append(f"Histogram skupiony w środku ({mid_tones*100:.1f}%).")

    # asymetria histogramu
    if mean < median - 10:
        issues.append("Histogram przesunięty w lewo (za ciemno).")
    elif mean > median + 10:
        issues.append("Histogram przesunięty w prawo (za jasno).")

    # zafarb kolorystyczny
    if color_range > 15:
        issues.append("Nierównowaga kanałów RGB (możliwy zafarb).")

    # --- Ocena punktowa ---
    score = 100
    score -= pct_black * 150
    score -= pct_white * 150
    score -= max(0, 50 - std) * 1.2
    score -= max(0, mid_tones - 0.85) * 120
    score -= max(0, color_range - 15) * 0.8

    if mean < median - 10:
        score -= 5
    elif mean > median + 10:
        score -= 5

    score = max(0, min(100, int(score)))
    return score, issues

def main():
    print("Pobieranie zdjęcia z internetu...")
    img = download_image(IMG_URL)
    print("Zdjęcie wczytane. Rozmiar:", img.shape)

    show_image(img)
    show_histograms(img)

    score, issues = assess_quality(img)

    print("\nOcena jakości zdjęcia:", score, "/ 100")
    print("Wykryte problemy:")
    if issues:
        for i in issues:
            print(" -", i)
    else:
        print(" - Brak problemów wykrytych na podstawie histogramu.")

if __name__ == "__main__":
    main()