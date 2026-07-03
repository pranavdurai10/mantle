# =============================================================================
# Author        : Pranav Durai
# Role          : Research Fellow
# Affiliation   : Stanford Center for Innovation in In Vivo Imaging
#                 Stanford University School of Medicine, Stanford, CA 94305
# Collaboration : Dr. Gary B. Doran
#                 Jet Propulsion Laboratory, California Institute of Technology,
#                 Pasadena, CA 91109
#
# Description : Interactive SAM2 ground-truth annotation tool (Qt5 GUI)
#               – Stage 1: parameter tuning on a random image sample
#               – Stage 2: streaming batch SAM2 inference → RLE JSON export
#               – Stage 3: review, edit, and binary mask export
# =============================================================================

"""
Mantle: SOTA Ground Truth GUI Tool

Architecture (3-stage pipeline)
--------------------------------
Stage 1  |  Page 1  |  Parameter Tuning
         |          |  Tune SAM2 params on a random sample before committing.
         |          |  Includes JSON dir selector for annotation output.

Stage 2  |  Page 1  |  Batch Inference + JSON Export (streaming, QThread)
         |          |  Streams image → SAM2 → RLE JSON → disk. Zero RAM
         |          |  accumulation. One JSON per image, same stem as source.
         |          |  SAM2AutomaticMaskGenerator instantiated ONCE per run.

Stage 3  |  Page 2  |  Review + Edit + Binary Export
         |          |  Loads one image + JSON at a time. RLE decoded on the
         |          |  fly. Click-to-toggle visibility. Save writes back to
         |          |  JSON. Final export composites visible masks → binary PNG.

Dependencies
------------
    pip install torch torchvision opencv-python pycocotools \
                PyQt5 tqdm huggingface_hub
    pip install git+https://github.com/facebookresearch/sam2.git
"""

import sys, os, json, random
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QFileDialog, QFrame, QProgressBar,
    QCheckBox, QStackedWidget, QLineEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from huggingface_hub import hf_hub_download
from pycocotools import mask as mask_utils


# Helpers

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def encode_mask(binary_mask: np.ndarray) -> dict:
    """Encode a boolean/uint8 HxW mask to COCO RLE."""
    m = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(m)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def decode_mask(rle: dict) -> np.ndarray:
    """Decode COCO RLE back to boolean HxW mask."""
    rle_copy = {'counts': rle['counts'].encode('utf-8'), 'size': rle['size']}
    return mask_utils.decode(rle_copy).astype(bool)

def apply_clahe(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def json_path_for(image_fname: str, json_dir: str) -> str:
    stem = Path(image_fname).stem
    return os.path.join(json_dir, stem + '.json')


# QThread worker for Stage 2 batch inference 

class BatchInferenceWorker(QThread):
    progress   = pyqtSignal(int, str)   # (index, filename)
    finished   = pyqtSignal(int, int)   # (processed, skipped)
    error      = pyqtSignal(str)

    def __init__(self, model, image_list, input_dir, json_dir,
                 params, use_clahe, parent=None):
        super().__init__(parent)
        self.model      = model
        self.image_list = image_list
        self.input_dir  = input_dir
        self.json_dir   = json_dir
        self.params     = params          # dict of param_name → int value
        self.use_clahe  = use_clahe
        self._abort     = False

    def abort(self):
        self._abort = True

    def run(self):
        set_seed(42)
        os.makedirs(self.json_dir, exist_ok=True)

        # Instantiate generator ONCE for the whole run
        gen = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=self.params['points_per_side'],
            pred_iou_thresh=self.params['pred_iou_thresh'] / 100.0,
            stability_score_thresh=self.params['stability_score_thresh'] / 100.0,
            min_mask_region_area=self.params['min_mask_region_area'],
        )

        processed = skipped = 0
        max_area_pct = self.params['max_mask_area_pct'] / 100.0

        for i, fname in enumerate(self.image_list):
            if self._abort:
                break

            img_path  = os.path.join(self.input_dir, fname)
            out_path  = json_path_for(fname, self.json_dir)

            bgr = cv2.imread(img_path)
            if bgr is None:
                skipped += 1
                self.progress.emit(i + 1, f"[SKIP] {fname}")
                continue

            rgb = apply_clahe(bgr) if self.use_clahe else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = bgr.shape[:2]
            max_area = max_area_pct * h * w

            try:
                with torch.inference_mode():
                    raw_masks = gen.generate(rgb)
            except Exception as e:
                skipped += 1
                self.progress.emit(i + 1, f"[ERROR] {fname}: {e}")
                continue

            # Filter by max_area, encode to RLE, build annotation
            annotations = []
            for m in raw_masks:
                if m['area'] <= max_area:
                    annotations.append({
                        'id':               len(annotations),
                        'area':             int(m['area']),
                        'predicted_iou':    float(m['predicted_iou']),
                        'stability_score':  float(m['stability_score']),
                        'segmentation':     encode_mask(m['segmentation']),
                        'visible':          True,
                    })

            record = {
                'source':       fname,
                'image_shape':  [h, w],
                'clahe_used':   self.use_clahe,
                'params':       self.params,
                'masks':        annotations,
            }

            with open(out_path, 'w') as f:
                json.dump(record, f, separators=(',', ':'))

            processed += 1
            self.progress.emit(i + 1, fname)

        self.finished.emit(processed, skipped)


# Main GUI

DARK_BG    = "#0d0d0d"
PANEL_BG   = "#141414"
SIDEBAR_BG = "#1a1a1a"
BORDER     = "#2a2a2a"
ACCENT     = "#e86c2f"          # Mars-orange
ACCENT2    = "#5b9bd5"          # Steel blue
TEXT       = "#e0e0e0"
TEXT_DIM   = "#777777"
SUCCESS    = "#2e7d32"
DANGER     = "#c62828"

MONO = "'Menlo', 'Courier New', monospace"

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #rgb or #rrggbb to rgba(r,g,b,alpha) for Qt5-safe transparency."""
    h = hex_color.lstrip('#')
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

BTN_BASE = f"""
    QPushButton {{
        background: #222; color: {TEXT}; border: 1px solid {BORDER};
        border-radius: 4px; padding: 6px 12px; font-size: 12px;
        font-family: 'Menlo', 'Courier New', monospace;
    }}
    QPushButton:hover  {{ background: #2e2e2e; border-color: {ACCENT}; }}
    QPushButton:pressed {{ background: #111; }}
    QPushButton:disabled {{ color: #444; border-color: #222; }}
"""

SLIDER_STYLE = f"""
    QSlider::groove:horizontal {{
        height: 4px; background: #2a2a2a; border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {ACCENT}; width: 14px; height: 14px;
        margin: -5px 0; border-radius: 7px;
    }}
    QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
"""


class MantleGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        set_seed(42)
        self.setWindowTitle("Mantle - Ground Truth Tool v4")
        self.setMinimumSize(1920, 980)
        self._apply_palette()

        self.device = (
            "mps"  if torch.backends.mps.is_available()  else
            "cuda" if torch.cuda.is_available()           else
            "cpu"
        )
        print(f"[Mantle] Device: {self.device}")

        # State
        self.input_dir   = ""
        self.json_dir    = ""
        self.output_dir  = ""
        self.image_list  = []
        self.current_idx = 0
        self.current_img_bgr   = None
        self.current_json_data = None   # Stage 3: loaded JSON for current image
        self.decoded_masks = []          # cache: list of bool HxW arrays, decoded once on load
        self.mask_selection_active = False
        self.batch_worker = None

        self.VIEW_SIZE = 480
        self.sam2_model = None           # lazy-loaded on first use

        self._init_ui()

    # Palette

    def _apply_palette(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background: {DARK_BG}; color: {TEXT}; }}
            QLabel {{ color: {TEXT}; }}
            QCheckBox {{ color: {TEXT}; }}
            QProgressBar {{
                border: 1px solid {BORDER}; border-radius: 3px;
                background: #1a1a1a; text-align: center; color: {TEXT};
                height: 18px;
            }}
            QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
            QScrollArea {{ border: none; }}
        """)

    # SAM2 

    def _ensure_sam2(self):
        if self.sam2_model is not None:
            return
        print("[Mantle] Loading SAM2 checkpoint …")
        try:
            ckpt = hf_hub_download(repo_id="facebook/sam2-hiera-large",
                                   filename="sam2_hiera_large.pt")
        except Exception:
            ckpt = "sam2_hiera_large.pt"
        self.sam2_model = build_sam2("sam2_hiera_l.yaml", ckpt, device=self.device)
        print("[Mantle] SAM2 ready.")

    # UI scaffold

    def _init_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        vbox = QVBoxLayout(root); vbox.setSpacing(0); vbox.setContentsMargins(0,0,0,0)

        # Navbar
        nav = QFrame()
        nav.setFixedHeight(52)
        nav.setStyleSheet(f"background:{PANEL_BG}; border-bottom:1px solid {BORDER};")
        nav_h = QHBoxLayout(nav); nav_h.setContentsMargins(16,0,16,0)

        title = QLabel("MANTLE")
        title.setStyleSheet(f"color:{ACCENT}; font-size:15px; font-weight:700; "
                            f"font-family:'Menlo','Courier New',monospace; letter-spacing:3px;")
        nav_h.addWidget(title)
        nav_h.addStretch()

        self.btn_p1 = self._nav_btn("01  PARAMETERS + BATCH")
        self.btn_p2 = self._nav_btn("02  REVIEW + EXPORT")
        self.btn_p1.clicked.connect(lambda: self._switch(0))
        self.btn_p2.clicked.connect(lambda: self._switch(1))
        nav_h.addWidget(self.btn_p1); nav_h.addWidget(self.btn_p2)
        vbox.addWidget(nav)

        self.pages = QStackedWidget()
        self.page1 = QWidget(); self.page2 = QWidget()
        self._init_page1(); self._init_page2()
        self.pages.addWidget(self.page1); self.pages.addWidget(self.page2)
        vbox.addWidget(self.pages)
        self._switch(0)

    def _nav_btn(self, text):
        b = QPushButton(text)
        b.setFixedHeight(36)
        b.setStyleSheet(f"""
            QPushButton {{
                background:transparent; color:{TEXT_DIM};
                border:none; font-size:11px; font-family:'Menlo','Courier New',monospace;
                letter-spacing:1px; padding: 0 18px;
            }}
            QPushButton:hover {{ color:{TEXT}; }}
        """)
        return b

    def _switch(self, idx):
        self.pages.setCurrentIndex(idx)
        self.btn_p1.setStyleSheet(self.btn_p1.styleSheet().split("color:")[0])
        self.btn_p2.setStyleSheet(self.btn_p2.styleSheet().split("color:")[0])
        for i, btn in enumerate([self.btn_p1, self.btn_p2]):
            c = ACCENT if i == idx else TEXT_DIM
            border = ACCENT if i == idx else "transparent"
            btn.setStyleSheet(f"""
                QPushButton {{
                    background:transparent; color:{c};
                    border:none; border-bottom:2px solid {border};
                    font-size:11px; font-family:'Menlo','Courier New',monospace;
                    letter-spacing:1px; padding:0 18px;
                }}
            """)

    # Page 1: Parameter Tuning + Batch Inference

    def _init_page1(self):
        h = QHBoxLayout(self.page1); h.setSpacing(0); h.setContentsMargins(0,0,0,0)

        # sidebar
        sidebar = QFrame()
        sidebar.setFixedWidth(300)
        sidebar.setStyleSheet(f"background:{SIDEBAR_BG}; border-right:1px solid {BORDER};")
        s = QVBoxLayout(sidebar); s.setContentsMargins(16,16,16,16); s.setSpacing(10)

        s.addWidget(self._section_label("DIRECTORIES"))
        self.lbl_input  = self._dir_label("No input dir selected")
        self.lbl_json   = self._dir_label("No JSON dir selected")
        s.addWidget(self._dir_btn("Select Input Dir",  self._sel_input))
        s.addWidget(self.lbl_input)
        s.addWidget(self._dir_btn("Select JSON Dir",   self._sel_json))
        s.addWidget(self.lbl_json)

        s.addWidget(self._sep())
        s.addWidget(self._section_label("SAM2 PARAMETERS"))

        self.params = {}
        self._add_slider(s, "points_per_side",        8,   128,  32)
        self._add_slider(s, "pred_iou_thresh",         0,   100,  70,  pct=True)
        self._add_slider(s, "stability_score_thresh",  0,   100,  85,  pct=True)
        self._add_slider(s, "min_mask_region_area",    0,  2000, 100)
        self._add_slider(s, "max_mask_area_pct",       1,   100,  80,  pct=True)

        self.clahe_chk = QCheckBox("Enable CLAHE Enhancement")
        self.clahe_chk.setStyleSheet(f"color:{TEXT}; font-size:11px;")
        self.clahe_chk.stateChanged.connect(self._p1_update_clahe)
        s.addWidget(self.clahe_chk)

        s.addWidget(self._sep())
        s.addWidget(self._section_label("ACTIONS"))

        self.btn_test = self._action_btn("Test on Current Image", ACCENT)
        self.btn_test.clicked.connect(self._p1_run_inference)
        s.addWidget(self.btn_test)

        self.btn_batch_start = self._action_btn("Start Batch Inference", SUCCESS)
        self.btn_batch_start.clicked.connect(self._batch_start)
        s.addWidget(self.btn_batch_start)

        self.btn_batch_abort = self._action_btn("Abort Batch", DANGER)
        self.btn_batch_abort.clicked.connect(self._batch_abort)
        self.btn_batch_abort.setEnabled(False)
        s.addWidget(self.btn_batch_abort)

        s.addWidget(self._sep())
        self.lbl_batch_status = QLabel("Idle")
        self.lbl_batch_status.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px; "
                                            f"font-family:'Menlo','Courier New',monospace;")
        self.lbl_batch_status.setWordWrap(True)
        s.addWidget(self.lbl_batch_status)

        self.p1_progress = QProgressBar()
        self.p1_progress.setValue(0)
        s.addWidget(self.p1_progress)

        s.addStretch()
        h.addWidget(sidebar)

        # viewer area
        viewer = QWidget()
        vl = QVBoxLayout(viewer); vl.setContentsMargins(20,16,20,12); vl.setSpacing(12)

        row = QHBoxLayout(); row.setSpacing(16)
        self.p1_src     = self._pane(row, "SOURCE")
        self.p1_clahe   = self._pane(row, "CLAHE")
        self.p1_mask    = self._pane(row, "MASK")
        self.p1_overlay = self._pane(row, "OVERLAY")
        vl.addLayout(row)

        nav_row = QHBoxLayout()
        b_prev = self._nav_arrow("Prev"); b_next = self._nav_arrow("Next")
        self.p1_counter = QLabel("—")
        self.p1_counter.setAlignment(Qt.AlignCenter)
        self.p1_counter.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px; "
                                      f"font-family:'Menlo','Courier New',monospace;")
        b_prev.clicked.connect(self._prev); b_next.clicked.connect(self._next)
        nav_row.addStretch(); nav_row.addWidget(b_prev)
        nav_row.addWidget(self.p1_counter); nav_row.addWidget(b_next); nav_row.addStretch()
        vl.addLayout(nav_row)

        h.addWidget(viewer)

    # Page 2: Review + Edit + Binary Export

    def _init_page2(self):
        h = QHBoxLayout(self.page2); h.setSpacing(0); h.setContentsMargins(0,0,0,0)

        # Sidebar
        sidebar = QFrame()
        sidebar.setFixedWidth(300)
        sidebar.setStyleSheet(f"background:{SIDEBAR_BG}; border-right:1px solid {BORDER};")
        s = QVBoxLayout(sidebar); s.setContentsMargins(16,16,16,16); s.setSpacing(10)

        s.addWidget(self._section_label("DIRECTORIES"))
        self.lbl_p2_input  = self._dir_label("No input dir selected")
        self.lbl_p2_json   = self._dir_label("No JSON dir selected")
        self.lbl_p2_output = self._dir_label("No output dir selected")
        s.addWidget(self._dir_btn("Select Input Dir",  self._sel_p2_input))
        s.addWidget(self.lbl_p2_input)
        s.addWidget(self._dir_btn("Select JSON Dir",   self._sel_p2_json))
        s.addWidget(self.lbl_p2_json)
        s.addWidget(self._dir_btn("Select Output Dir", self._sel_p2_output))
        s.addWidget(self.lbl_p2_output)

        s.addWidget(self._sep())
        s.addWidget(self._section_label("MASK INFO"))
        self.lbl_mask_info = QLabel("Load an image to begin")
        self.lbl_mask_info.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px; "
                                         f"font-family:'Menlo','Courier New',monospace;")
        self.lbl_mask_info.setWordWrap(True)
        s.addWidget(self.lbl_mask_info)

        s.addWidget(self._sep())
        s.addWidget(self._section_label("ACTIONS"))

        self.btn_sel = self._action_btn("Mask Selection [OFF]", ACCENT2)
        self.btn_sel.clicked.connect(self._toggle_selection)
        s.addWidget(self.btn_sel)

        self.btn_save_json = self._action_btn("Save Edits to JSON", ACCENT2)
        self.btn_save_json.clicked.connect(self._save_json)
        s.addWidget(self.btn_save_json)

        self.btn_export = self._action_btn("Export All Binary Masks", SUCCESS)
        self.btn_export.clicked.connect(self._export_binary)
        s.addWidget(self.btn_export)

        self.btn_delete = self._action_btn("Delete Pair [W]", DANGER)
        self.btn_delete.clicked.connect(self._delete_pair)
        s.addWidget(self.btn_delete)

        s.addWidget(self._sep())
        s.addWidget(self._section_label("IMAGE-JSON PAIR"))

        jump_row = QHBoxLayout()
        self.jump_input = QLineEdit()
        self.jump_input.setPlaceholderText("Pair number…")
        self.jump_input.setFocusPolicy(Qt.ClickFocus)
        self.jump_input.setStyleSheet(f"""
            QLineEdit {{
                background: #1e1e1e; color: {TEXT}; border: 1px solid {BORDER};
                border-radius: 4px; padding: 5px 8px; font-size: 11px;
                font-family: {MONO};
            }}
            QLineEdit:focus {{ border-color: {ACCENT}; }}
        """)
        self.jump_input.returnPressed.connect(self._jump_to_pair)
        btn_jump = self._action_btn("Jump", ACCENT)
        btn_jump.setFixedWidth(60)
        btn_jump.clicked.connect(self._jump_to_pair)
        jump_row.addWidget(self.jump_input)
        jump_row.addWidget(btn_jump)
        s.addLayout(jump_row)

        s.addWidget(self._sep())
        self.lbl_p2_status = QLabel("Idle")
        self.lbl_p2_status.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px; "
                                          f"font-family:'Menlo','Courier New',monospace;")
        self.lbl_p2_status.setWordWrap(True)
        s.addWidget(self.lbl_p2_status)

        self.p2_progress = QProgressBar(); self.p2_progress.setValue(0)
        s.addWidget(self.p2_progress)
        s.addStretch()
        h.addWidget(sidebar)

        # Viewer
        viewer = QWidget()
        vl = QVBoxLayout(viewer); vl.setContentsMargins(20,16,20,12); vl.setSpacing(12)

        row = QHBoxLayout(); row.setSpacing(16)
        self.p2_src     = self._pane(row, "SOURCE IMAGE")
        self.p2_mask    = self._pane(row, "BINARY MASK")
        self.p2_overlay = self._pane(row, "OVERLAY - click to toggle mask")
        self.p2_overlay.mousePressEvent = self._on_click
        self.p2_overlay.setCursor(Qt.ArrowCursor)
        vl.addLayout(row)

        nav_row = QHBoxLayout()
        b_prev = self._nav_arrow("Prev"); b_next = self._nav_arrow("Next")
        self.p2_counter = QLabel("—")
        self.p2_counter.setAlignment(Qt.AlignCenter)
        self.p2_counter.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px; "
                                      f"font-family:'Menlo','Courier New',monospace;")
        b_prev.clicked.connect(self._prev); b_next.clicked.connect(self._next)
        nav_row.addStretch(); nav_row.addWidget(b_prev)
        nav_row.addWidget(self.p2_counter); nav_row.addWidget(b_next); nav_row.addStretch()
        vl.addLayout(nav_row)
        h.addWidget(viewer)

    # Widget factories

    def _section_label(self, text):
        l = QLabel(text)
        l.setStyleSheet(f"color:{TEXT_DIM}; font-size:9px; letter-spacing:2px; "
                        f"font-family:'Menlo','Courier New',monospace; margin-top:4px;")
        return l

    def _sep(self):
        f = QFrame(); f.setFrameShape(QFrame.HLine)
        f.setStyleSheet(f"color:{BORDER};"); return f

    def _dir_label(self, text):
        l = QLabel(text)
        l.setStyleSheet(f"color:{TEXT_DIM}; font-size:9px; "
                        f"font-family:'Menlo','Courier New',monospace;")
        l.setWordWrap(True); return l

    def _dir_btn(self, text, slot):
        b = QPushButton(text); b.setStyleSheet(BTN_BASE)
        b.clicked.connect(slot); return b

    def _action_btn(self, text, color):
        b = QPushButton(text)
        bg      = hex_to_rgba(color, 0.10)
        bg_hov  = hex_to_rgba(color, 0.25)
        border  = hex_to_rgba(color, 0.35)
        b.setStyleSheet(f"""
            QPushButton {{
                background:{bg}; color:{color}; border:1px solid {border};
                border-radius:4px; padding:8px; font-size:12px;
                font-family:{MONO};
            }}
            QPushButton:hover {{ background:{bg_hov}; }}
            QPushButton:disabled {{ color:#444444; border-color:#333333; background:#111111; }}
        """)
        return b

    def _nav_arrow(self, text):
        b = QPushButton(text); b.setFixedSize(110, 34); b.setStyleSheet(BTN_BASE)
        return b

    def _pane(self, layout, title):
        v = QVBoxLayout()
        lbl_title = QLabel(title)
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet(f"color:{TEXT_DIM}; font-size:9px; letter-spacing:2px; "
                                f"font-family:'Menlo','Courier New',monospace; margin-bottom:4px;")
        v.addWidget(lbl_title)
        pane = QLabel()
        pane.setFixedSize(self.VIEW_SIZE, self.VIEW_SIZE)
        pane.setStyleSheet(f"border:1px solid {BORDER}; background:#0a0a0a;")
        pane.setAlignment(Qt.AlignCenter)
        v.addWidget(pane); layout.addLayout(v)
        return pane

    def _add_slider(self, layout, name, lo, hi, default, pct=False):
        row = QHBoxLayout()
        lbl = QLabel(name.replace('_', ' '))
        lbl.setStyleSheet(f"color:{TEXT}; font-size:10px; "
                          f"font-family:'Menlo','Courier New',monospace;")
        val_lbl = QLabel()
        val_lbl.setFixedWidth(45)
        val_lbl.setAlignment(Qt.AlignRight)
        val_lbl.setStyleSheet(f"color:{ACCENT}; font-size:10px; "
                              f"font-family:'Menlo','Courier New',monospace;")
        sl = QSlider(Qt.Horizontal)
        sl.setRange(lo, hi); sl.setValue(default)
        sl.setStyleSheet(SLIDER_STYLE)

        def _upd(v):
            val_lbl.setText(f"{v/100:.2f}" if pct else str(v))
        _upd(default)
        sl.valueChanged.connect(_upd)

        layout.addWidget(lbl); row.addWidget(sl); row.addWidget(val_lbl)
        layout.addLayout(row)
        self.params[name] = sl

    # Directory selectors

    def _pick_dir(self, title):
        res = QFileDialog.getExistingDirectory(self, title)
        if isinstance(res, tuple): res = res[0]
        return res or ""

    def _sel_input(self):
        d = self._pick_dir("Select Input Directory")
        if d:
            self.input_dir = d
            self.lbl_input.setText(d)
            self.image_list = sorted([
                f for f in os.listdir(d)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            self.current_idx = 0
            self._load_image()
            self.p1_progress.setMaximum(len(self.image_list))

    def _sel_json(self):
        d = self._pick_dir("Select JSON Annotation Directory")
        if d: self.json_dir = d; self.lbl_json.setText(d)

    def _sel_p2_input(self):
        d = self._pick_dir("Select Input Directory (Images)")
        if d:
            self.input_dir = d
            self.lbl_p2_input.setText(d)
            self.image_list = sorted([
                f for f in os.listdir(d)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            self.current_idx = 0
            self._load_image()

    def _sel_p2_json(self):
        d = self._pick_dir("Select JSON Annotation Directory")
        if d: self.json_dir = d; self.lbl_p2_json.setText(d)

    def _sel_p2_output(self):
        d = self._pick_dir("Select Binary Mask Output Directory")
        if d: self.output_dir = d; self.lbl_p2_output.setText(d)

    # Image loaders

    def _load_image(self):
        if not self.image_list: return
        fname    = self.image_list[self.current_idx]
        path     = os.path.join(self.input_dir, fname)
        self.current_img_bgr = cv2.imread(path)
        if self.current_img_bgr is None: return

        pix = QPixmap(path).scaled(self.VIEW_SIZE, self.VIEW_SIZE,
                                   Qt.KeepAspectRatio, Qt.SmoothTransformation)
        page = self.pages.currentIndex()
        counter_text = f"{self.current_idx+1} / {len(self.image_list)} - {fname}"

        if page == 0:
            self.p1_src.setPixmap(pix)
            self.p1_counter.setText(counter_text)
            self._p1_update_clahe()
        else:
            self.p2_src.setPixmap(pix)
            self.p2_counter.setText(counter_text)
            self._p2_load_json(fname)

    def _prev(self):
        if self.current_idx > 0:
            self.current_idx -= 1; self._load_image()

    def _next(self):
        if self.current_idx < len(self.image_list) - 1:
            self.current_idx += 1; self._load_image()

    # Page 1 - Logic

    def _p1_update_clahe(self):
        if self.current_img_bgr is None: return
        if self.clahe_chk.isChecked():
            rgb = apply_clahe(self.current_img_bgr)
            self.p1_clahe.setPixmap(self._arr_to_pix(rgb, rgb=True))
        else:
            self.p1_clahe.clear()

    def _get_params_dict(self):
        return {k: v.value() for k, v in self.params.items()}

    def _p1_run_inference(self):
        if self.current_img_bgr is None: return
        self._ensure_sam2()
        set_seed(42)
        p = self._get_params_dict()
        rgb = apply_clahe(self.current_img_bgr) if self.clahe_chk.isChecked() \
              else cv2.cvtColor(self.current_img_bgr, cv2.COLOR_BGR2RGB)

        gen = SAM2AutomaticMaskGenerator(
            model=self.sam2_model,
            points_per_side=p['points_per_side'],
            pred_iou_thresh=p['pred_iou_thresh'] / 100.0,
            stability_score_thresh=p['stability_score_thresh'] / 100.0,
            min_mask_region_area=p['min_mask_region_area'],
        )
        with torch.inference_mode():
            raw = gen.generate(rgb)

        h, w = self.current_img_bgr.shape[:2]
        max_a = (p['max_mask_area_pct'] / 100.0) * h * w
        masks = [m for m in raw if m['area'] <= max_a]
        self._render_masks(masks, self.p1_mask, self.p1_overlay)

    # Batch inference

    def _batch_start(self):
        if not self.input_dir:
            self.lbl_batch_status.setText("Warning: No input dir selected."); return
        if not self.json_dir:
            self.lbl_batch_status.setText("Warning: No JSON dir selected."); return
        if not self.image_list:
            self.lbl_batch_status.setText("Warning: Image list is empty."); return

        self._ensure_sam2()
        self.p1_progress.setMaximum(len(self.image_list))
        self.p1_progress.setValue(0)
        self.btn_batch_start.setEnabled(False)
        self.btn_batch_abort.setEnabled(True)
        self.btn_test.setEnabled(False)

        self.batch_worker = BatchInferenceWorker(
            model=self.sam2_model,
            image_list=self.image_list,
            input_dir=self.input_dir,
            json_dir=self.json_dir,
            params=self._get_params_dict(),
            use_clahe=self.clahe_chk.isChecked(),
        )
        self.batch_worker.progress.connect(self._on_batch_progress)
        self.batch_worker.finished.connect(self._on_batch_done)
        self.batch_worker.start()

    def _on_batch_progress(self, idx, fname):
        self.p1_progress.setValue(idx)
        self.lbl_batch_status.setText(
            f"{idx}/{len(self.image_list)}\n{Path(fname).name}"
        )
        QApplication.processEvents()

    def _on_batch_done(self, processed, skipped):
        self.btn_batch_start.setEnabled(True)
        self.btn_batch_abort.setEnabled(False)
        self.btn_test.setEnabled(True)
        self.lbl_batch_status.setText(
            f"Done\n{processed} saved - {skipped} skipped\n→ {self.json_dir}"
        )

    def _batch_abort(self):
        if self.batch_worker: self.batch_worker.abort()
        self.lbl_batch_status.setText("Aborted.")
        self.btn_batch_start.setEnabled(True)
        self.btn_batch_abort.setEnabled(False)
        self.btn_test.setEnabled(True)

    # Page 2 - Logic

    def _p2_load_json(self, fname):
        self.current_json_data = None
        if not self.json_dir:
            self.lbl_mask_info.setText("No JSON dir set."); return

        jpath = json_path_for(fname, self.json_dir)
        if not os.path.exists(jpath):
            self.decoded_masks = []
            self.p2_mask.clear(); self.p2_overlay.clear()
            self.lbl_mask_info.setText(f"No JSON found:\n{Path(jpath).name}")
            return

        with open(jpath) as f:
            self.current_json_data = json.load(f)

        # Decode all RLE masks once into cache
        self.decoded_masks = [
            decode_mask(m['segmentation'])
            for m in self.current_json_data['masks']
        ]

        n = len(self.current_json_data['masks'])
        vis = sum(1 for m in self.current_json_data['masks'] if m.get('visible', True))
        self.lbl_mask_info.setText(
            f"Masks: {n} - Visible: {vis}\n"
            f"Shape: {self.current_json_data['image_shape']}\n"
            f"CLAHE: {self.current_json_data.get('clahe_used', '?')}"
        )
        self._p2_render()

    def _p2_render(self):
        if self.current_img_bgr is None or self.current_json_data is None: return
        h, w = self.current_img_bgr.shape[:2]
        overlay = cv2.cvtColor(self.current_img_bgr, cv2.COLOR_BGR2RGB).copy()

        # Vectorized composite — stack all visible masks into one bool array in a
        # single np.any() call instead of 500+ individual indexed writes
        visible_indices = [
            i for i, m in enumerate(self.current_json_data['masks'])
            if m.get('visible', True)
        ]

        if visible_indices:
            # Shape: (N_visible, H, W) — build once, collapse once
            stack   = np.stack([self.decoded_masks[i] for i in visible_indices], axis=0)
            combined = np.any(stack, axis=0).astype(np.uint8) * 255

            # Overlay: broadcast Mars-orange onto all visible mask pixels at once
            mask_union = combined.astype(bool)
            overlay[mask_union] = (overlay[mask_union] * 0.55 + 
                                   np.array([50, 205, 50]) * 0.45).astype(np.uint8)
        else:
            combined = np.zeros((h, w), dtype=np.uint8)

        q_mask = QImage(combined.tobytes(), w, h, w, QImage.Format_Grayscale8)
        q_ov   = QImage(overlay.tobytes(), w, h, 3*w, QImage.Format_RGB888)
        self.p2_mask.setPixmap(
            QPixmap.fromImage(q_mask).scaled(self.VIEW_SIZE, self.VIEW_SIZE,
                                              Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.p2_overlay.setPixmap(
            QPixmap.fromImage(q_ov).scaled(self.VIEW_SIZE, self.VIEW_SIZE,
                                           Qt.KeepAspectRatio, Qt.SmoothTransformation))

        vis = sum(1 for m in self.current_json_data['masks'] if m.get('visible', True))
        self.lbl_mask_info.setText(
            f"Masks: {len(self.current_json_data['masks'])} - Visible: {vis}\n"
            f"Shape: {self.current_json_data['image_shape']}\n"
            f"CLAHE: {self.current_json_data.get('clahe_used', '?')}"
        )

    def _toggle_selection(self):
        self.mask_selection_active = not self.mask_selection_active
        color = DANGER if self.mask_selection_active else ACCENT2
        label = "Mask Selection [ON]" if self.mask_selection_active else "Mask Selection [OFF]"
        self.btn_sel.setText(label)
        self.p2_overlay.setCursor(Qt.CrossCursor if self.mask_selection_active else Qt.ArrowCursor)
        bg     = hex_to_rgba(color, 0.10)
        bg_hov = hex_to_rgba(color, 0.25)
        border = hex_to_rgba(color, 0.35)
        self.btn_sel.setStyleSheet(f"""
            QPushButton {{
                background:{bg}; color:{color}; border:1px solid {border};
                border-radius:4px; padding:8px; font-size:12px;
                font-family:{MONO};
            }}
            QPushButton:hover {{ background:{bg_hov}; }}
        """)

    def _on_click(self, event):
        if not self.mask_selection_active or self.current_json_data is None: return
        if self.current_img_bgr is None: return

        lsz = self.p2_overlay.size()
        h, w = self.current_img_bgr.shape[:2]
        scale   = min(lsz.width() / w, lsz.height() / h)
        new_w   = int(w * scale); new_h = int(h * scale)
        off_x   = (lsz.width()  - new_w) // 2
        off_y   = (lsz.height() - new_h) // 2
        rx = int((event.x() - off_x) / scale)
        ry = int((event.y() - off_y) / scale)

        if not (0 <= rx < w and 0 <= ry < h): return

        # Find all masks covering this pixel — use pre-decoded cache
        hits = []
        for i, m in enumerate(self.current_json_data['masks']):
            if self.decoded_masks[i][ry, rx]:
                hits.append(m)

        if not hits: return

        # Sort by area ascending; toggle smallest visible first
        hits.sort(key=lambda x: x['area'])
        toggled = False
        for m in hits:
            if m.get('visible', True):
                m['visible'] = False
                toggled = True
                break
        if not toggled:
            hits[-1]['visible'] = True   # restore largest if all hidden

        self._p2_render()

    def _save_json(self):
        if not self.current_json_data or not self.json_dir: return
        fname = self.image_list[self.current_idx]
        jpath = json_path_for(fname, self.json_dir)
        with open(jpath, 'w') as f:
            json.dump(self.current_json_data, f, separators=(',', ':'))
        self.lbl_p2_status.setText(f"Saved\n{Path(jpath).name}")

    def _delete_pair(self):
        if not self.image_list: return
        fname = self.image_list[self.current_idx]
        img_path  = os.path.join(self.input_dir, fname)
        json_path = json_path_for(fname, self.json_dir)

        if os.path.exists(img_path):
            os.remove(img_path)
        if os.path.exists(json_path):
            os.remove(json_path)

        self.image_list.pop(self.current_idx)
        if not self.image_list:
            self.p2_src.clear(); self.p2_mask.clear(); self.p2_overlay.clear()
            self.lbl_p2_status.setText("No images remaining.")
            return

        # Stay at same index (now points to next image) or clamp to last
        self.current_idx = min(self.current_idx, len(self.image_list) - 1)
        self.lbl_p2_status.setText(f"Deleted {fname}")
        self._load_image()

    def _jump_to_pair(self):
        text = self.jump_input.text().strip()
        if not text.isdigit():
            self.lbl_p2_status.setText("Warning: Enter a valid number.")
            self.jump_input.clearFocus()
            return
        idx = int(text) - 1   # 1-indexed input → 0-indexed
        if not (0 <= idx < len(self.image_list)):
            self.lbl_p2_status.setText(
                f"Warning: Out of range. Valid: 1 - {len(self.image_list)}.")
            self.jump_input.clearFocus()
            return
        self.current_idx = idx
        self.jump_input.clear()
        self.jump_input.clearFocus()
        self._load_image()

    def _export_binary(self):
        if not self.json_dir or not self.output_dir:
            self.lbl_p2_status.setText("Warning: Set JSON dir and output dir first."); return
        if not self.image_list:
            self.lbl_p2_status.setText("Warning: No images loaded."); return

        os.makedirs(self.output_dir, exist_ok=True)
        self.p2_progress.setMaximum(len(self.image_list))
        self.p2_progress.setValue(0)
        exported = skipped = 0

        for i, fname in enumerate(tqdm(self.image_list, desc="Exporting")):
            jpath = json_path_for(fname, self.json_dir)
            if not os.path.exists(jpath):
                skipped += 1; self.p2_progress.setValue(i+1); continue

            with open(jpath) as f:
                data = json.load(f)

            sh = data['image_shape']; h, w = sh[0], sh[1]
            binary = np.zeros((h, w), dtype=np.uint8)
            for m in data['masks']:
                if m.get('visible', True):
                    binary[decode_mask(m['segmentation'])] = 255

            out_path = os.path.join(self.output_dir, fname)
            cv2.imwrite(out_path, binary)
            exported += 1
            self.p2_progress.setValue(i+1)
            QApplication.processEvents()

        self.lbl_p2_status.setText(
            f"Export done\n{exported} masks - {skipped} skipped\n→ {self.output_dir}"
        )

    # Keyboard shortcuts

    def keyPressEvent(self, event):
        if self.pages.currentIndex() != 1:
            super().keyPressEvent(event); return
        key = event.key()
        if key == Qt.Key_A:
            self._prev()
        elif key == Qt.Key_D:
            self._next()
        elif key == Qt.Key_S:
            self._save_json()
        elif key == Qt.Key_W:
            self._delete_pair()
        else:
            super().keyPressEvent(event)

    # Shared render helpers

    def _render_masks(self, masks, mask_pane, overlay_pane):
        """Render raw SAM2 mask dicts (Page 1 preview only)."""
        if self.current_img_bgr is None: return
        h, w = self.current_img_bgr.shape[:2]
        combined = np.zeros((h, w), dtype=np.uint8)
        overlay  = cv2.cvtColor(self.current_img_bgr, cv2.COLOR_BGR2RGB).copy()
        red      = np.zeros_like(overlay)

        for m in masks:
            combined[m['segmentation']] = 255
            red[m['segmentation']] = [220, 80, 30]

        cv2.addWeighted(red, 0.45, overlay, 0.55, 0, overlay)
        q_mask = QImage(combined.tobytes(), w, h, w, QImage.Format_Grayscale8)
        q_ov   = QImage(overlay.tobytes(), w, h, 3*w, QImage.Format_RGB888)
        mask_pane.setPixmap(QPixmap.fromImage(q_mask).scaled(
            self.VIEW_SIZE, self.VIEW_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        overlay_pane.setPixmap(QPixmap.fromImage(q_ov).scaled(
            self.VIEW_SIZE, self.VIEW_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _arr_to_pix(self, arr, rgb=False):
        h, w = arr.shape[:2]
        fmt  = QImage.Format_RGB888 if rgb else QImage.Format_Grayscale8
        bpl  = 3*w if rgb else w
        q    = QImage(arr.tobytes(), w, h, bpl, fmt)
        return QPixmap.fromImage(q).scaled(self.VIEW_SIZE, self.VIEW_SIZE,
                                           Qt.KeepAspectRatio, Qt.SmoothTransformation)


# Entry point

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Menlo", 10))
    win = MantleGUI()
    win.show()
    sys.exit(app.exec_())