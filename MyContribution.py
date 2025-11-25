# -------------------- VISUALIZATION --------------------

def show_fashion_samples(base_folder, categories, num_samples=5, thumb_size=(320,320), cols=5, caption=None):
    """
    Improved visualizer: show colorful images from local dataset with a polished grid UI.
    - base_folder: root dataset folder containing category subfolders (e.g., dataset/T-shirt_top/)
    - categories: list of category names (must match folder names or mapping)
    - num_samples: how many images PER category to display (total per category)
    - thumb_size: (w,h) size to resize thumbnail for display
    - cols: grid columns (images per row). If num_samples > cols, multiple rows produced.
    - caption: optional short text to show under the grid (e.g., the LLM recommendation line)
    """
    # mapping from FASHION_LABELS to local folder names
    label_to_folder = {
        "T-shirt/top": "T-shirt_top",
        "Trouser": "Trouser",
        "Pullover": "Pullover",
        "Dress": "Dress",
        "Coat": "Coat",
        "Sandal": "Sandal",
        "Shirt": "Shirt",
        "Sneaker": "Sneaker",
        "Bag": "Bag",
        "Ankle boot": "Ankle_boot"
    }

    # Collect images to display: (img_array, category_label, filename)
    grid_items = []
    for category in categories:
        folder_name = label_to_folder.get(category, category)
        folder_path = os.path.join(base_folder, folder_name)
        if not os.path.isdir(folder_path):
            # skip missing categories silently
            continue

        image_files = [f for f in os.listdir(folder_path)if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.avif'))]
        if not image_files:
            continue

        # choose up to num_samples images (randomize for variety)
        chosen = random.sample(image_files, min(num_samples, len(image_files)))
        for fn in chosen:
            p = os.path.join(folder_path, fn)
            img_bgr = cv2.imread(p)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            # resize while keeping aspect ratio, pad to thumb_size
            h, w = img_rgb.shape[:2]
            target_w, target_h = thumb_size
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w*scale), int(h*scale)
            img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # create white background and center the resized image
            canvas = 255 * np.ones((target_h, target_w, 3), dtype=np.uint8)
            x_off = (target_w - new_w)//2
            y_off = (target_h - new_h)//2
            canvas[y_off:y_off+new_h, x_off:x_off+new_w] = img_resized
            grid_items.append((canvas, category, fn))

    if not grid_items:
        print("[Info] No images available to display.")
        return

    # compute grid layout
    total = len(grid_items)
    rows = ceil(total / cols)
    fig_w = cols * (thumb_size[0] / 80)   # scale factor for figure size
    fig_h = rows * (thumb_size[1] / 120)
    plt.figure(figsize=(fig_w, fig_h + (0.4 if caption else 0.0)))
    plt.suptitle("Suggested Outfit Visuals", fontsize=18, fontweight='bold', y=0.98)

    for idx, (img, category, fn) in enumerate(grid_items):
        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(img)
        ax.axis('off')
        # category label below image (centered)
        ax.set_title(category, fontsize=10, fontweight='semibold')
        # add subtle border rectangle
        rect = Rectangle((0,0), 1, 1, transform=ax.transAxes, fill=False, edgecolor="#2b7a78", linewidth=2, clip_on=False)
        ax.add_patch(rect)

    # optional caption (original LLM text or explanation)
    if caption:
        plt.figtext(0.5, 0.02, caption, wrap=True, horizontalalignment='center', fontsize=10, color="#333333")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.4, top=0.92)
    plt.show(block=True)

# -------------------- HELPER: PARSE OUTFIT TEXT --------------------

def extract_categories_from_text(outfit_text):
    """
    Return categories in LLM order, but resolve collisions:
      - If both T-shirt/top and Shirt detected, keep only T-shirt/top.
    """
    mapping_keywords = {
        "t-shirt":"T-shirt/top","tshirt":"T-shirt/top","tee":"T-shirt/top",
        "shirt":"Shirt",
        "trouser":"Trouser","trousers":"Trouser","pants":"Trouser","jeans":"Trouser",
        "pullover":"Pullover","sweater":"Pullover","jumper":"Pullover","hoodie":"Pullover",
        "dress":"Dress",
        "coat":"Coat","jacket":"Coat","blazer":"Coat",
        "sandal":"Sandal","flipflop":"Sandal",
        "sneaker":"Sneaker","trainers":"Sneaker",
        "boot":"Ankle boot","boots":"Ankle boot",
        "bag":"Bag","backpack":"Bag","purse":"Bag","tote":"Bag"
    }

    # 1) check explicit VisualCategories line first (preserve order)
    m = re.search(r'VisualCategories\s*:\s*(.+)', outfit_text, flags=re.IGNORECASE)
    if m:
        listed = [c.strip() for c in m.group(1).split(',') if c.strip()]
        out = []
        for c in listed:
            if c in FASHION_LABELS and c not in out:
                out.append(c)
        if out:
            # resolve collision: prefer T-shirt/top over Shirt
            if "T-shirt/top" in out and "Shirt" in out:
                out = [x for x in out if x != "Shirt"]
            return out

    # 2) otherwise scan text left-to-right for keywords and labels
    s = outfit_text.lower()
    # normalize common variants to avoid double-matching later
    s = s.replace("t-shirt", "tshirt").replace("t-shirt", "tshirt")  # normalize hyphen variants

    found = []
    # detect explicit full-label occurrences (e.g., 't-shirt/top' may not appear verbatim; check label tokens)
    for label in FASHION_LABELS:
        lowlabel = label.lower()
        if lowlabel.replace("-", " ").replace("/", " ") in s and label not in found:
            found.append(label)

    # find keyword occurrences and record position
    positions = []
    for kw, lab in mapping_keywords.items():
        pos = s.find(kw)
        if pos != -1:
            positions.append((pos, lab))

    # combine and sort by first occurrence position
    # also include those already detected from full labels with pos=inf if no position found
    combined = []
    for i, lab in enumerate(found):
        combined.append((s.find(lab.lower().replace("-", " ").replace("/", " ")), lab))
    for pos, lab in positions:
        combined.append((pos, lab))

    # sort by position and collect unique labels preserving first-seen order
    combined_sorted = sorted(combined, key=lambda x: x[0] if x[0] >= 0 else float("inf"))
    ordered = []
    for pos, lab in combined_sorted:
        if lab not in ordered:
            ordered.append(lab)

    # Resolve collision: prefer T-shirt/top over Shirt
    if "T-shirt/top" in ordered and "Shirt" in ordered:
        ordered = [x for x in ordered if x != "Shirt"]

    # Last-resort fuzzy token scanning (keeps order)
    if not ordered:
        words = re.findall(r"[a-zA-Z\-]+", s)
        for w in words:
            match = difflib.get_close_matches(w, list(mapping_keywords.keys()), n=1, cutoff=0.85)
            if match:
                lab = mapping_keywords[match[0]]
                if lab not in ordered:
                    ordered.append(lab)

    return ordered
