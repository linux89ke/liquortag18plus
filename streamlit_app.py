import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import os
import re
import zipfile
import concurrent.futures
import pandas as pd
from bs4 import BeautifulSoup

# ─── Page Config & Styling ────────────────────────────────────────────────────────
st.set_page_config(page_title="18+ Tag Generator", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    * { font-family: 'DM Sans', sans-serif; }
    [data-testid="stAppViewContainer"] { background: #FFF8F2; }
    [data-testid="stSidebar"] { background: #1A1A1A !important; }
    [data-testid="stSidebar"] * { color: #fff !important; }
    .stButton > button[kind="primary"], .stDownloadButton > button { background: #F68B1E !important; color: #fff !important; border: none !important; border-radius: 6px; font-weight: 700; width: 100%; }
    .orange-bar { height: 3px; background: linear-gradient(90deg, #F68B1E, #ffb347); border-radius: 2px; margin-bottom: 1.2rem; }
    div[data-testid="stImage"] img { border: 1.5px solid #F0D5B8; border-radius: 6px; }
    .preview-empty { height: 240px; display: flex; align-items: center; justify-content: center; border: 2px dashed #F0D5B8; border-radius: 8px; color: #bbb; background: #FFFAF5; font-size: 0.875rem; }
</style>
""", unsafe_allow_html=True)

# ─── Constants & Setup ────────────────────────────────────────────────────────────
TAG_FILE = "NSFW-18++-Tag.png"
TARGET_CANVAS_SIZE = (800, 800)
VERTICAL_PADDING = 50

TAG_PATH = TAG_FILE if os.path.exists(TAG_FILE) else os.path.join(os.path.dirname(__file__), TAG_FILE)
TAG_MISSING = not os.path.exists(TAG_PATH)

st.sidebar.markdown("## Settings")
remove_old_tags = st.sidebar.checkbox("Remove existing 18+ tags", value=True)
marketplace = st.sidebar.radio("Region", ["Kenya", "Uganda"])
MARKET_BASE = "https://www.jumia.co.ke" if marketplace == "Kenya" else "https://www.jumia.ug"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

# ─── Session State Initialization ─────────────────────────────────────────────────
# Prevents the UI from resetting when a download button is clicked
tabs_states = ["files", "excel", "urls", "skus", "category", "single"]
for tab in tabs_states:
    if f"results_{tab}" not in st.session_state:
        st.session_state[f"results_{tab}"] = None
        st.session_state[f"orig_{tab}"] = None
if "single_name" not in st.session_state:
    st.session_state["single_name"] = "tagged_image_1.jpg"


# ─── High-Performance Image Processing ────────────────────────────────────────────
def remove_existing_tag(img):
    """Fast numpy array masking to detect and overwrite existing red tags."""
    img_rgb = img.convert('RGB')
    data = np.array(img_rgb)
    h, w, _ = data.shape
    sh, sw = int(h * 0.35), int(w * 0.35)
    tr = data[0:sh, w-sw:w]
    
    red_mask = (tr[:, :, 0] > 160) & (tr[:, :, 1] < 80) & (tr[:, :, 2] < 80) & (tr[:, :, 0].astype(int) - tr[:, :, 1].astype(int) > 100)
    
    if np.sum(red_mask) > 30:
        coords = np.argwhere(red_mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        pad = 25
        y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
        y1, x1 = min(sh, y1 + pad), min(sw, x1 + pad)
        data[y0:y1, (w - sw) + x0:(w - sw) + x1] = [255, 255, 255]
        
        if img.mode == 'RGBA':
            orig = np.array(img)
            orig[:, :, 0:3] = data
            return Image.fromarray(orig, 'RGBA')
        return Image.fromarray(data, 'RGB')
    return img

def crop_white_space(img):
    """Calculates boundaries of non-white pixels to crop dead space."""
    arr = np.array(img.convert('RGB'))
    mask = arr < 245
    if not mask.any(): return img
    coords = np.argwhere(mask.any(axis=-1))
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img.crop((x0, y0, x1, y1))

@st.cache_resource
def load_tag_image():
    if TAG_MISSING: return None
    return Image.open(TAG_PATH).convert("RGBA").resize(TARGET_CANVAS_SIZE, Image.Resampling.LANCZOS)

def compose_image(product_img, tag_img, apply_remove=True):
    """Composes the final 800x800 image."""
    img = remove_existing_tag(product_img) if apply_remove else product_img.copy()
    img = crop_white_space(img)
    
    cw, ch = TARGET_CANVAS_SIZE
    avail_h, avail_w = ch - 2 * VERTICAL_PADDING, cw - 2 * VERTICAL_PADDING
    sw, sh = img.size
    
    scale = min(avail_w / sw, avail_h / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    result = Image.new("RGB", TARGET_CANVAS_SIZE, (255, 255, 255))
    
    px, py = (cw - nw) // 2, (ch - nh) // 2
    if img.mode == 'RGBA':
        result.paste(img, (px, py), img)
    else:
        result.paste(img, (px, py))
        
    result.paste(tag_img, (0, 0), tag_img)
    return result

def process_bulk(products, tag):
    """Threaded image processing to utilize multiple CPU cores."""
    def _process(item):
        img, name = item
        res = compose_image(img, tag, apply_remove=remove_old_tags)
        return res, name
    
    with concurrent.futures.ThreadPoolExecutor() as ex:
        results = list(ex.map(_process, products))
    return results

def img_to_bytes(img, fmt="JPEG"):
    buf = BytesIO()
    img.save(buf, format=fmt, quality=95)
    return buf.getvalue()

def build_zip(pairs, fmt="JPEG"):
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for img, name in pairs:
            zf.writestr(f"{name}.jpg", img_to_bytes(img, fmt))
    return buf.getvalue()

# ─── Fast HTTP Scraping (No Selenium required) ────────────────────────────────────
def fetch_image_from_url(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGBA")
    except:
        return None

def search_by_sku(sku):
    """Scrapes directly via HTTP. 10x faster than Selenium."""
    try:
        url = f"{MARKET_BASE}/catalog/?q={sku}"
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        
        link = soup.select_one("article.prd a.core")
        if not link: return None
        
        target_url = link.get('href')
        if target_url.startswith('/'): target_url = MARKET_BASE + target_url
            
        r_prod = requests.get(target_url, headers=HEADERS, timeout=10)
        p_soup = BeautifulSoup(r_prod.text, 'html.parser')
        
        og_img = p_soup.find('meta', property='og:image')
        if og_img and og_img.get('content'):
            return fetch_image_from_url(og_img['content'])
            
    except Exception:
        pass
    return None

def scrape_category(category_url, max_items=30):
    """Scrapes category page with robust SKU extraction from URL slugs."""
    try:
        r = requests.get(category_url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        results = []
        
        articles = soup.find_all('article', class_='prd')
        
        for i, art in enumerate(articles):
            if len(results) >= max_items:
                break
                
            # 1. Try to get SKU from article tag
            sku = art.get('data-sku') or art.get('data-id')
            
            # 2. Try to get SKU from the anchor tag
            core_link = art.find('a', class_='core')
            if not sku and core_link:
                sku = core_link.get('data-id') or core_link.get('data-sku')
                
                # 3. POWERFUL FALLBACK: Extract SKU directly from the URL string
                if not sku:
                    href = core_link.get('href', '')
                    match = re.search(r'-([a-zA-Z0-9]+)\.html', href, re.IGNORECASE)
                    if match:
                        sku = match.group(1)
            
            # Clean up the SKU
            if sku:
                sku = str(sku).strip().split()[0]
                sku = re.sub(r'[^\w-]', '', sku) # Remove special characters
            
            img_tag = art.find('img', class_='img')
            if img_tag:
                src = img_tag.get('data-src') or img_tag.get('src', '')
                if src and 'data:image' not in src:
                    name = sku if sku else f"product_{i+1}"
                    results.append((name, src))
                    
        return results
    except Exception as e:
        st.error(f"Scraping error: {e}")
        return []

# ─── UI & Interface ───────────────────────────────────────────────────────────────
def show_grid_and_download(results, originals=None, zip_name="tagged_images.zip", tab_id=""):
    # Limit preview to 8 to prevent Streamlit MediaFileStorageError
    preview_limit = min(8, len(results))
    st.markdown(f"**Previewing {preview_limit} of {len(results)} generated images:**")
    
    cols = st.columns(4)
    for i in range(preview_limit):
        img, name = results[i]
        cols[i % 4].image(img, caption=name, width="stretch")

    st.markdown("---")
    dl_col, orig_col = st.columns(2)
    
    with dl_col:
        st.download_button(
            label=f"Download {len(results)} Tagged Image(s)",
            data=build_zip(results),
            file_name=zip_name,
            mime="application/zip",
            type="primary",
            icon=":material/download:",
            key=f"dl_tag_{tab_id}"
        )
    if originals:
        with orig_col:
            st.download_button(
                label=f"Download Originals",
                data=build_zip(originals),
                file_name=zip_name.replace(".zip", "_originals.zip"),
                mime="application/zip",
                icon=":material/folder_zip:",
                key=f"dl_orig_{tab_id}"
            )

st.title("18+ Tag Generator")
st.markdown('<div class="orange-bar"></div>', unsafe_allow_html=True)

if TAG_MISSING:
    st.error(f"Overlay file not found: {TAG_FILE} — place it in the same directory as this script.")
    st.stop()

tag_img_cached = load_tag_image()

tab_single, tab_files, tab_excel, tab_urls, tab_skus, tab_category = st.tabs([
    "Single Image", "Multiple Images", "Excel", "URLs", "SKUs", "Category Scrape"
])

# ─── 1. Single Image ───
with tab_single:
    col_in, col_out = st.columns(2, gap="large")
    with col_in:
        uploaded = st.file_uploader("Select image", type=["png", "jpg", "jpeg", "webp"])
        url_input = st.text_input("Or paste an image URL", placeholder="https://...")
        
        if st.button("Process Single Image", type="primary"):
            if uploaded:
                img = Image.open(uploaded).convert("RGBA")
                base_name = uploaded.name.rsplit('.', 1)[0]
                if not base_name.endswith("_1"): base_name += "_1"
                st.session_state.single_name = f"{base_name}.jpg"
                st.session_state.results_single = compose_image(img, tag_img_cached, apply_remove=remove_old_tags)
            elif url_input.strip():
                img = fetch_image_from_url(url_input.strip())
                if img:
                    st.session_state.single_name = "fetched_image_1.jpg"
                    st.session_state.results_single = compose_image(img, tag_img_cached, apply_remove=remove_old_tags)

    with col_out:
        if st.session_state.results_single is not None:
            st.image(st.session_state.results_single, width="stretch")
            st.download_button(
                label="Download Image", 
                data=img_to_bytes(st.session_state.results_single), 
                file_name=st.session_state.single_name, 
                mime="image/jpeg", 
                type="primary",
                icon=":material/download:",
                key="dl_single"
            )
        else:
            st.markdown('<div class="preview-empty">Preview will appear here</div>', unsafe_allow_html=True)

# ─── 2. Multiple Images (Device) ───
with tab_files:
    files = st.file_uploader("Select images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    if st.button("Process Uploaded Files", type="primary") and files:
        products = []
        originals = []
        for f in files:
            base_name = f.name.rsplit('.', 1)[0]
            if not base_name.endswith("_1"): base_name = f"{base_name}_1"
            products.append((Image.open(f).convert("RGBA"), base_name))
            originals.append((Image.open(f).convert("RGB"), base_name))
            
        with st.spinner(f"Processing {len(products)} image(s)..."):
            st.session_state.results_files = process_bulk(products, tag_img_cached)
            st.session_state.orig_files = originals
            
    if st.session_state.results_files:
        st.success(f"{len(st.session_state.results_files)} image(s) ready.")
        show_grid_and_download(st.session_state.results_files, originals=st.session_state.orig_files, zip_name="device_uploads.zip", tab_id="files")

# ─── 3. Excel ───
with tab_excel:
    xl = st.file_uploader("Select Excel file", type=["xlsx", "xls"])
    if xl:
        df = pd.read_excel(xl)
        url_col = next((c for c in df.columns if any(k in str(c).lower() for k in ['url', 'link', 'image'])), None)
        sku_col = next((c for c in df.columns if 'sku' in str(c).lower()), None)
        name_col = next((c for c in df.columns if any(k in str(c).lower() for k in ['name', 'title'])), df.columns[1] if len(df.columns) > 1 else None)
        url_col = url_col or df.columns[0]
        
        if st.button("Process Excel Data", type="primary"):
            prog = st.progress(0, text="Fetching images...")
            fetched = []
            
            def fetch_row(args):
                idx, row = args
                name = None
                if sku_col and pd.notna(row.get(sku_col)):
                    name = str(row[sku_col]).strip()
                if not name and name_col and pd.notna(row.get(name_col)):
                    name = re.sub(r'[^\w\s-]', '', str(row[name_col])).strip().replace(' ', '_')
                if not name:
                    name = f"product_{idx}"
                
                if not name.endswith("_1"): name = f"{name}_1"
                    
                img = None
                if url_col and pd.notna(row.get(url_col)):
                    img = fetch_image_from_url(str(row[url_col]))
                if not img and sku_col and pd.notna(row.get(sku_col)):
                    img = search_by_sku(str(row[sku_col]).strip())
                return img, name
                
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                futs = [ex.submit(fetch_row, r) for r in df.iterrows()]
                for i, f in enumerate(concurrent.futures.as_completed(futs)):
                    img, name = f.result()
                    if img: fetched.append((img, name))
                    prog.progress((i + 1) / len(df), text=f"Fetched {i + 1} of {len(df)}...")
                    
            if fetched:
                with st.spinner("Applying tags..."):
                    st.session_state.results_excel = process_bulk([(img, name) for img, name in fetched], tag_img_cached)
                    st.session_state.orig_excel = [(img.convert("RGB"), name) for img, name in fetched]
            else:
                st.error("No images found.")

    if st.session_state.results_excel:
        show_grid_and_download(st.session_state.results_excel, originals=st.session_state.orig_excel, zip_name="excel_tagged.zip", tab_id="excel")

# ─── 4. URLs ───
with tab_urls:
    urls_text = st.text_area("Paste image URLs, one per line", height=150)
    if urls_text.strip() and st.button("Fetch URLs", type="primary"):
        urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
        prog = st.progress(0, text="Fetching images...")
        
        fetched = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(fetch_image_from_url, u) for u in urls]
            for i, f in enumerate(concurrent.futures.as_completed(futs)):
                img = f.result()
                if img: fetched.append((img, f"url_image_{i+1}_1"))
                prog.progress((i + 1) / len(urls))
                
        if fetched:
            st.session_state.results_urls = process_bulk([(img, name) for img, name in fetched], tag_img_cached)
            st.session_state.orig_urls = [(img.convert("RGB"), name) for img, name in fetched]

    if st.session_state.results_urls:
        show_grid_and_download(st.session_state.results_urls, originals=st.session_state.orig_urls, zip_name="url_tagged.zip", tab_id="urls")

# ─── 5. SKUs ───
with tab_skus:
    skus_text = st.text_area("Enter product SKUs, one per line", height=150)
    if skus_text.strip() and st.button("Fetch SKUs", type="primary"):
        skus = [s.strip() for s in skus_text.splitlines() if s.strip()]
        prog = st.progress(0, text="Fetching SKUs...")
        
        fetched = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(search_by_sku, s): s for s in skus}
            for i, f in enumerate(concurrent.futures.as_completed(futs)):
                img = f.result()
                if img: 
                    sku_name = futs[f]
                    name = f"{sku_name}_1" if not sku_name.endswith("_1") else sku_name
                    fetched.append((img, name))
                prog.progress((i + 1) / len(skus))
                
        if fetched:
            st.session_state.results_skus = process_bulk([(img, name) for img, name in fetched], tag_img_cached)
            st.session_state.orig_skus = [(img.convert("RGB"), name) for img, name in fetched]
        else:
            st.error("No verified images found for the provided SKUs.")

    if st.session_state.results_skus:
        show_grid_and_download(st.session_state.results_skus, originals=st.session_state.orig_skus, zip_name="sku_tagged.zip", tab_id="skus")

# ─── 6. Category Scrape ───
with tab_category:
    cat_url = st.text_input("Category URL")
    max_items = st.slider("Maximum items to scrape", 10, 100, 30, step=10)
    
    if cat_url.strip() and st.button("Scrape Category", type="primary"):
        with st.spinner("Scraping catalog data..."):
            scraped = scrape_category(cat_url.strip(), max_items)
            
        if scraped:
            prog = st.progress(0, text="Downloading images...")
            fetched = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                futs = [ex.submit(lambda x: (x[0], fetch_image_from_url(x[1])), item) for item in scraped]
                for i, f in enumerate(concurrent.futures.as_completed(futs)):
                    name, img = f.result()
                    if img: 
                        final_name = f"{name}_1" if not name.endswith("_1") else name
                        fetched.append((img, final_name))
                    prog.progress((i + 1) / len(scraped))
                    
            if fetched:
                with st.spinner("Applying tags..."):
                    st.session_state.results_category = process_bulk([(img, name) for img, name in fetched], tag_img_cached)
                    st.session_state.orig_category = [(img.convert("RGB"), name) for img, name in fetched]
            else:
                st.error("No images could be downloaded.")
        else:
            st.error("No products found on that page.")

    if st.session_state.results_category:
        show_grid_and_download(
            st.session_state.results_category, 
            originals=st.session_state.orig_category, 
            zip_name="category_tagged.zip", 
            tab_id="category"
        )
