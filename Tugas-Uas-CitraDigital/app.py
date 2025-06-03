import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image, ImageEnhance

app = Flask(__name__)

# ================================
#  Konfigurasi folder upload
# ================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'upload')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Buat folder upload jika belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# ================================
#  Fungsi Sharpening (Mempertajam Citra, pada warna)
# ================================
def apply_sharpening(pil_img):
    """
    Menerima PIL Image (mode RGB), 
    mengkonversi ke array BGR, menerapkan kernel sharpening pada tiap channel, 
    lalu mengembalikan PIL Image (RGB).
    """
    # Konversi PIL RGB → NumPy array BGR
    arr_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Define kernel sharpening sederhana
    kernel_sharp = np.array([[0, -1,  0],
                             [-1, 5, -1],
                             [0, -1,  0]])
    # Terapkan filter2D pada BGR (automatis men‐filter di tiap channel)
    sharpened_bgr = cv2.filter2D(src=arr_bgr, ddepth=-1, kernel=kernel_sharp)

    # Kembalikan ke PIL RGB
    sharpened_rgb = cv2.cvtColor(sharpened_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(sharpened_rgb)


# ================================
#  Fungsi Denoising (Mereduksi Noise, tetap di warna asli)
# ================================
def apply_denoising(pil_img):
    """
    Menerima PIL Image (mode RGB), terapkan dua metode denoising:
      1) fastNlMeansDenoisingColored (mengurangi noise pada gambar full‐color)
      2) medianBlur pada tiap channel (grayscale hanya untuk demonstrasi)
    Mengembalikan dict dua varian hasil denoising.
    """
    # Konversi PIL RGB → NumPy array BGR
    arr_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 1) Denoising Warna (NL Means) → hasil berwarna
    denoised_color = cv2.fastNlMeansDenoisingColored(arr_bgr, None,
                                                      h=10, hColor=10,
                                                      templateWindowSize=7,
                                                      searchWindowSize=21)
    denoise_color_pil = Image.fromarray(cv2.cvtColor(denoised_color, cv2.COLOR_BGR2RGB))

    # 2) Median Blur pada grayscale (opsional—hanya contoh jika ingin output BW)
    gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY)
    denoise_gray_arr = cv2.medianBlur(gray, ksize=5)
    denoise_gray_pil = Image.fromarray(denoise_gray_arr)

    return {
        'denoise_color': denoise_color_pil,
        'denoise_gray':  denoise_gray_pil
    }


# ================================
#  Fungsi Peningkatan Kualitas Gambar
#  (Grayscale, Kontras, Gaussian Blur, Crop, Portrait, Sharpen, Denoise)
# ================================
def improve_image_quality(original_img, base_filename):
    """
    Menghasilkan dan menyimpan beberapa varian:
    - grayscale
    - contrast (lebih ekstrim → diwarna)
    - gaussian blur (diwarna)
    - crop (200×300 di tengah, tetap diwarna)
    - portrait mode (lebar 150px di tengah, diwarna)
    - sharpening (diwarna)
    - denoising color (fastNlMeans) → diwarna
    - denoising grayscale (medianBlur) → BW
    Mengembalikan dict:
      {
        'grayscale':       'upload/grayscale_<base_filename>',
        'contrast':        'upload/contrast_<base_filename>',
        'blur':            'upload/blur_<base_filename>',
        'crop':            'upload/crop_<base_filename>',
        'portrait':        'upload/portrait_<base_filename>',
        'sharpen':         'upload/sharpen_<base_filename>',
        'denoise_color':   'upload/denoise_color_<base_filename>',
        'denoise_gray':    'upload/denoise_gray_<base_filename>'
      }
    """
    steps = {}

    # 1) Grayscale (menghasilkan BW)
    img_gray = original_img.convert('L')
    steps['grayscale'] = img_gray.copy()

    # 2) Kontras (lebih ekstrim) → pada gambar warna (RGB)
    img_contrast = original_img.convert('RGB')  # pastikan mode RGB
    enhancer = ImageEnhance.Contrast(img_contrast)
    img_contrast = enhancer.enhance(3.0)
    steps['contrast'] = img_contrast.copy()

    # 3) Gaussian Blur (lebih kuat) → pada gambar warna
    arr_color = cv2.cvtColor(np.array(original_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    blurred_bgr = cv2.GaussianBlur(arr_color, (25, 25), 10)
    blurred_rgb = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2RGB)
    img_blur = Image.fromarray(blurred_rgb)
    steps['blur'] = img_blur.copy()

    # 4) Crop (200×300 di tengah) → pada gambar warna
    img_crop_color = original_img.convert('RGB')
    w, h = img_crop_color.size
    crop_box = (
        (w - 200) // 2,
        (h - 300) // 2,
        (w + 200) // 2,
        (h + 300) // 2
    )
    img_crop = img_crop_color.crop(crop_box)
    steps['crop'] = img_crop.copy()

    # 5) Portrait Mode (lebar 150px di tengah) → pada gambar warna
    img_portrait_color = original_img.convert('RGB')
    pw = 150
    cx = img_portrait_color.size[0] // 2
    portrait_box = (
        cx - pw // 2,
        0,
        cx + pw // 2,
        img_portrait_color.size[1]
    )
    img_portrait = img_portrait_color.crop(portrait_box)
    steps['portrait'] = img_portrait.copy()

    # 6) Sharpening → pada gambar warna
    steps['sharpen'] = apply_sharpening(original_img).copy()

    # 7) Denoising (2 varian)
    denoise_dict = apply_denoising(original_img)
    steps['denoise_color'] = denoise_dict['denoise_color'].copy()
    steps['denoise_gray']  = denoise_dict['denoise_gray'].copy()

    # Simpan semua hasil ke folder static/upload/
    saved = {}
    for step_name, pil_img in steps.items():
        out_fname = f"{step_name}_{base_filename}"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_fname)
        pil_img.save(out_path)
        # Path relatif yang nantinya dipakai di url_for('static', filename=...)
        saved[step_name] = f"upload/{out_fname}"

    return saved


# ================================
#  Fungsi Edge Detection (Canny)
# ================================
def apply_edge_detection(image_path):
    """
    – Membaca image_path (contoh: '/.../static/upload/foo.jpg')
    – Mengkonversi ke grayscale dan menyimpannya sebagai '<basename>_ed_gray.jpg'
    – Menerapkan Canny → menyimpannya sebagai '<basename>_edge.jpg'
    – Mengembalikan dict:
        {
          'ed_gray': 'upload/<basename>_ed_gray.jpg',
          'edge':    'upload/<basename>_edge.jpg'
        }
    """
    # Baca dengan OpenCV (BGR)
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Simpan grayscale hasil edge detection
    base, ext = os.path.splitext(os.path.basename(image_path))
    gray_fname = f"{base}_ed_gray{ext}"
    gray_path = os.path.join(app.config['UPLOAD_FOLDER'], gray_fname)
    cv2.imwrite(gray_path, gray)

    # Edge Detection (Canny)
    edges = cv2.Canny(gray, 100, 200)
    edge_fname = f"{base}_edge{ext}"
    edge_path = os.path.join(app.config['UPLOAD_FOLDER'], edge_fname)
    cv2.imwrite(edge_path, edges)

    return {
        'ed_gray': f"upload/{gray_fname}",
        'edge':    f"upload/{edge_fname}"
    }


# ================================
#  Route Utama: '/', GET dan POST
# ================================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 1) Pastikan ada file di form
        if 'file' not in request.files:
            return redirect(url_for('index'))

        file = request.files['file']
        if file.filename == '':
            return redirect(url_for('index'))

        # 2) Simpan file asli ke static/upload/
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 3) Baca dengan OpenCV untuk segmentasi warna
        original_bgr = cv2.imread(filepath)
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

        # 4) Simpan versi RGB (untuk ditampilkan di browser)
        rgb_fname = f"rgb_{filename}"
        rgb_path = os.path.join(app.config['UPLOAD_FOLDER'], rgb_fname)
        Image.fromarray(original_rgb).save(rgb_path)

        # ================================
        # 5) Segmentasi WARNA KUNING (HSV + inRange)
        # ================================
        hsv = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([10, 50, 50])
        upper_yellow = np.array([45, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # ================================
        # 6) Opening (erosi + dilasi)
        # ================================
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # ================================
        # 7) Visualisasi Masking: 
        #    - piksel opening > 0 → kuning
        #    - else → latar ungu (pink)
        # ================================
        purple_bg   = np.full_like(original_rgb, (255, 0, 255))
        yellow_full = np.full_like(original_rgb, (255, 255, 0))
        masked_vis = np.where(
            opening[:, :, np.newaxis] > 0,
            yellow_full,
            purple_bg
        )
        mask_vis_fname = f"mask_vis_{filename}"
        mask_vis_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_vis_fname)
        Image.fromarray(masked_vis).save(mask_vis_path)

        # ================================
        # 8) Simpan hasil Opening (grayscale → RGB)
        # ================================
        opening_colored = cv2.cvtColor(opening, cv2.COLOR_GRAY2RGB)
        opening_fname = f"opening_{filename}"
        opening_path = os.path.join(app.config['UPLOAD_FOLDER'], opening_fname)
        Image.fromarray(opening_colored).save(opening_path)

        # ================================
        # 9) Improve Image Quality (grayscale, contrast, blur, crop, portrait, sharpen, denoise)
        # ================================
        pil_orig = Image.open(filepath).convert('RGB')
        improve_fnames = improve_image_quality(pil_orig, filename)
        #   improve_fnames = {
        #       'grayscale':      'upload/grayscale_<filename>',
        #       'contrast':       'upload/contrast_<filename>',
        #       'blur':           'upload/blur_<filename>',
        #       'crop':           'upload/crop_<filename>',
        #       'portrait':       'upload/portrait_<filename>',
        #       'sharpen':        'upload/sharpen_<filename>',
        #       'denoise_color':  'upload/denoise_color_<filename>',
        #       'denoise_gray':   'upload/denoise_gray_<filename>'
        #   }

        # ================================
        # 10) Edge Detection (Canny)
        # ================================
        edge_fnames = apply_edge_detection(filepath)
        #   edge_fnames = {
        #       'ed_gray': 'upload/<basename>_ed_gray.ext',
        #       'edge':    'upload/<basename>_edge.ext'
        #   }

        # ================================
        # 11) Kumpulkan semua paths ke template
        # ================================
        all_steps = {
            # 1) Segmentasi:
            'mask':     f"upload/{mask_vis_fname}",
            'opening':  f"upload/{opening_fname}",

            # 2) Improve Image Quality (warna):
            'grayscale_iq':     improve_fnames['grayscale'],       # BW
            'contrast_iq':      improve_fnames['contrast'],        # Warna
            'blur_iq':          improve_fnames['blur'],            # Warna
            'crop_iq':          improve_fnames['crop'],            # Warna
            'portrait_iq':      improve_fnames['portrait'],        # Warna
            'sharpen_iq':       improve_fnames['sharpen'],         # Warna
            'denoise_color_iq': improve_fnames['denoise_color'],   # Warna
            'denoise_gray_iq':  improve_fnames['denoise_gray'],    # BW (hasil medianBlur)

            # 3) Edge Detection:
            'ed_gray': edge_fnames['ed_gray'],    # BW
            'edge':    edge_fnames['edge']        # BW
        }

        return render_template(
            'index.html',
            original_filename = f"upload/{rgb_fname}",
            step_images       = all_steps
        )

    # Jika GET, tampilkan halaman upload saja
    return render_template('index.html')


if __name__ == '__main__':
    print("Menjalankan Flask pada port 8000...")
    app.run(debug=True, port=8000)