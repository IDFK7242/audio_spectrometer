import json
import queue
import sys
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pyqtgraph as pg
import sounddevice as sd
from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import median_filter
from scipy.signal import firwin, get_window, lfilter

# ------------------------ Utilities & Shared Config ------------------------

ROOT_DIR = Path(__file__).resolve().parent
PROFILES_PATH = ROOT_DIR / "profiles.json"


def db(x, eps=1e-12):
    return 20.0 * np.log10(np.maximum(x, eps))


def load_shared_config(path: Path = PROFILES_PATH) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def get_active_freqs(profile: dict, mod_mode: str) -> list[float]:
    freqs4 = profile["freqs4"]
    if mod_mode == "2fsk":
        return [float(freqs4[0]), float(freqs4[3])]
    if mod_mode == "4fsk":
        return [float(f) for f in freqs4]
    f0, f3 = float(freqs4[0]), float(freqs4[3])
    step = (f3 - f0) / 15.0
    return [float(round(f0 + i * step)) for i in range(16)]


def decode_hex_to_text(hex_str: str) -> str:
    decoded = []
    for i in range(0, len(hex_str) - 1, 2):
        try:
            val = int(hex_str[i : i + 2], 16)
            if val == 0x5F:
                decoded.append(" ")
            elif 32 <= val <= 126:
                decoded.append(chr(val))
            else:
                decoded.append("▯")
        except ValueError:
            pass
    return "".join(decoded)


# ------------------------ Headless DSP / Demodulator Engine ------------------------


class DemodulatorEngine:
    """
    Pure NumPy/SciPy DSP & M-FSK demodulator engine decoupled from Qt so it can be
    tested headlessly without a microphone or GUI.
    """

    def __init__(self, config: dict, sr: int = 44100, hop: int = 512, nfft: int = 2048, seconds: float = 8.0):
        self.config = config
        self.profiles = config["profiles"]
        self.vis_defaults = config["visualizerDefaults"]

        self.sr = int(sr)
        self.hop = int(hop)
        self.nfft = int(nfft)
        self.seconds = float(seconds)

        self.profile_key = "canary" if "canary" in self.profiles else next(iter(self.profiles))
        self.mod_mode = self.profiles[self.profile_key].get("defaultMode", "4fsk")
        self.rx_enabled = True

        # Per-profile & VIS RAM autosave
        self.saved_settings = {
            k: dict(v["rx"]) for k, v in self.profiles.items()
        }
        self.saved_settings["VIS"] = dict(self.vis_defaults)

        init_rx = self.saved_settings[self.profile_key]
        self.hp_fc = float(init_rx["hp"])
        self.lp_fc = float(init_rx["lp"])
        self.current_fmax = float(init_rx["fmax"])
        self.gate_margin = float(init_rx["gate"])

        # FIR Filter config (odd taps for highpass)
        self.hp_taps = 257
        self.lp_taps = 257
        self.fir_window = ("kaiser", 8.0)

        # Demodulator & framing state
        self.rx_state = "IDLE"
        self.current_rx_hex = ""
        self.message_history: list[tuple[str, str]] = []  # list of (hex_payload, decoded_txt)
        self.last_char_time = 0.0
        self.message_timeout = 1.5  # seconds of silence to finalize a message
        self.startup_guard_sec = 1.0

        self.rx_sym_idx = 1
        self.rx_current_val = 0
        self.rx_next_sample_time = 0.0
        self.prev_fsk_state = -1
        self.prev_valid_signal = False

        # Smoothed spectrogram levels
        self.smooth_cmax = -40.0
        self.last_noise_floor_db = -90.0
        self.last_loud_db = -40.0

        self._init_dsp_buffers()
        self.update_filters()

    def _init_dsp_buffers(self):
        self.freqs_all = np.fft.rfftfreq(self.nfft, 1.0 / self.sr)
        self.audio_time = 0.0
        self.audio_samples_processed = 0
        self.overlap_buf = np.zeros(self.nfft, dtype=np.float32)
        self.hanning_win = get_window("hann", self.nfft).astype(np.float32)

        self.time_bins = max(16, int((self.seconds * self.sr) / self.hop))
        self.freq_bins = len(self.freqs_all)
        self.spec_data = np.full((self.time_bins, self.freq_bins), -120.0, dtype=np.float32)
        self.noise_spec = None

    def set_sample_rate(self, sr: int):
        self.sr = int(sr)
        self._init_dsp_buffers()
        self.clamp_filters(self.hp_fc, self.lp_fc)
        self.update_filters()

    def get_active_profile(self) -> dict:
        return self.profiles[self.profile_key]

    def get_active_freqs(self) -> list[float]:
        return get_active_freqs(self.get_active_profile(), self.mod_mode)

    def clamp_filters(self, hp: float, lp: float) -> tuple[float, float]:
        max_lp = 0.49 * self.sr
        lp = max(300.0, min(float(lp), max_lp))
        hp = max(10.0, min(float(hp), lp - 200.0))
        self.hp_fc = hp
        self.lp_fc = lp
        return hp, lp

    def update_filters(self):
        hp_cut, lp_cut = self.clamp_filters(self.hp_fc, self.lp_fc)
        self.hc = firwin(
            numtaps=self.hp_taps,
            cutoff=hp_cut,
            fs=self.sr,
            pass_zero="highpass",
            window=self.fir_window,
        ).astype(np.float32)
        self.hzi = np.zeros(len(self.hc) - 1, dtype=np.float32)

        self.lc = firwin(
            numtaps=self.lp_taps,
            cutoff=lp_cut,
            fs=self.sr,
            pass_zero="lowpass",
            window=self.fir_window,
        ).astype(np.float32)
        self.lzi = np.zeros(len(self.lc) - 1, dtype=np.float32)

    def reset_demod_state(self, clear_current_buffer: bool = False):
        self.rx_state = "IDLE"
        self.prev_valid_signal = False
        self.prev_fsk_state = -1
        if clear_current_buffer:
            self.current_rx_hex = ""

    def finalize_current_message(self) -> bool:
        """Archive the active message (stripping its leading 'F' preamble) and reset buffer."""
        if not self.current_rx_hex:
            return False
        payload_hex = self.current_rx_hex[1:] if len(self.current_rx_hex) > 1 else ""
        self.current_rx_hex = ""
        if payload_hex:
            decoded = decode_hex_to_text(payload_hex)
            self.message_history.append((payload_hex, decoded))
            if len(self.message_history) > 20:
                self.message_history = self.message_history[-20:]
            return True
        return False

    def clear_all_messages(self):
        self.current_rx_hex = ""
        self.message_history.clear()
        self.reset_demod_state()

    def process_block(self, block: np.ndarray) -> bool:
        """
        Process one audio block. Returns True if demodulator state or decoded text changed.
        """
        ui_dirty = False
        prev_state = self.rx_state

        block = np.asarray(block, dtype=np.float32)
        self.audio_samples_processed += len(block)
        self.audio_time = self.audio_samples_processed / self.sr

        # FIR Bandpass
        block, self.hzi = lfilter(self.hc, [1.0], block, zi=self.hzi)
        block, self.lzi = lfilter(self.lc, [1.0], block, zi=self.lzi)

        self.overlap_buf = np.roll(self.overlap_buf, -len(block))
        self.overlap_buf[-len(block) :] = block

        # RFFT
        windowed = self.overlap_buf * self.hanning_win
        S_mag_raw = np.abs(np.fft.rfft(windowed))
        S_db_raw = db(S_mag_raw)

        loud_db_raw = float(np.max(S_db_raw))
        current_noise_floor_db = (
            float(db(self.noise_spec.mean())) if self.noise_spec is not None else -90.0
        )

        # Smart noise floor tracking
        if self.noise_spec is None:
            self.noise_spec = S_mag_raw.copy()
        else:
            if loud_db_raw < current_noise_floor_db + 15.0:
                self.noise_spec = 0.95 * self.noise_spec + 0.05 * S_mag_raw
            self.noise_spec = np.maximum(self.noise_spec, 1e-12)

        S_mag = np.maximum(S_mag_raw - 0.7 * self.noise_spec, 0.05 * self.noise_spec)
        S_db = db(S_mag)

        self.spec_data = np.roll(self.spec_data, -1, axis=0)
        self.spec_data[-1, :] = S_db

        # Smooth spectrogram ceiling (fast attack, slow decay)
        target_cmax = max(-40.0, loud_db_raw + 20.0)
        alpha = 0.35 if target_cmax > self.smooth_cmax else 0.04
        self.smooth_cmax = (1.0 - alpha) * self.smooth_cmax + alpha * target_cmax
        self.last_noise_floor_db = current_noise_floor_db
        self.last_loud_db = loud_db_raw

        if not self.rx_enabled:
            return False

        sp = self.get_active_profile()
        active_freqs = self.get_active_freqs()

        if self.mod_mode == "16fsk":
            bits_per_sym = 4
            syms_per_hex = 1
            peak_radius = 1
            guard_radius = 0
            guard_offset = (sp["freqs4"][3] - sp["freqs4"][0]) / 30.0
            scoop_dur = 0.015
            wait_pad = 0.05
        elif self.mod_mode == "4fsk":
            bits_per_sym = 2
            syms_per_hex = 2
            peak_radius = 2
            guard_radius = 2
            guard_offset = float(sp.get("guardOffset", (sp["freqs4"][1] - sp["freqs4"][0]) / 2.0))
            scoop_dur = float(sp["scoopDur"])
            wait_pad = 0.10
        else:
            bits_per_sym = 1
            syms_per_hex = 4
            peak_radius = 2
            guard_radius = 2
            guard_offset = float(sp.get("guardOffset", (sp["freqs4"][1] - sp["freqs4"][0]) / 2.0))
            scoop_dur = float(sp["scoopDur"])
            wait_pad = 0.10

        sym_interval = float(sp["toneDur"]) + float(sp["gapDur"])
        sample_delay = max(0.05, (float(sp["toneDur"]) - scoop_dur) * 0.45)

        valid_signal = False
        current_fsk_state = -1
        best_all_sym = 0

        if self.audio_time > self.startup_guard_sec:
            def get_local_max(target_f, radius):
                idx = int(np.abs(self.freqs_all - target_f).argmin())
                lo = max(0, idx - radius)
                hi = min(len(S_db_raw), idx + radius + 1)
                return float(np.max(S_db_raw[lo:hi]))

            band_mask = (self.freqs_all >= self.hp_fc) & (self.freqs_all <= self.lp_fc)
            current_median_db = (
                float(np.median(S_db_raw[band_mask]))
                if np.any(band_mask)
                else float(np.median(S_db_raw))
            )
            absolute_thresh = current_median_db + (14.0 if self.mod_mode == "16fsk" else 10.0)

            vals_db = []
            best_valid_db = -1e9

            for sym_i, target_f in enumerate(active_freqs):
                val_db = get_local_max(target_f, peak_radius)
                vals_db.append(val_db)

                g_low = max(get_local_max(target_f - guard_offset, guard_radius), current_median_db)
                g_high = max(get_local_max(target_f + guard_offset, guard_radius), current_median_db)

                is_valid = (
                    val_db > absolute_thresh
                    and val_db > g_low + self.gate_margin
                    and val_db > g_high + self.gate_margin
                )
                if is_valid and val_db > best_valid_db:
                    best_valid_db = val_db
                    valid_signal = True
                    current_fsk_state = sym_i

            best_all_sym = int(np.argmax(vals_db))

        # Edge detector (phase correction)
        edge_detected = False
        if valid_signal and self.prev_valid_signal and current_fsk_state != self.prev_fsk_state:
            edge_detected = True
        elif valid_signal and not self.prev_valid_signal:
            edge_detected = True

        if edge_detected and self.rx_state == "LOCKED":
            expected_edge = self.rx_next_sample_time - sample_delay
            error = self.audio_time - expected_edge
            if abs(error) < 0.06:
                self.rx_next_sample_time += error * 0.4

        self.prev_fsk_state = current_fsk_state
        self.prev_valid_signal = valid_signal

        # PLL state machine
        if self.rx_state == "IDLE":
            if self.current_rx_hex and (self.audio_time - self.last_char_time) >= self.message_timeout:
                if self.finalize_current_message():
                    ui_dirty = True
                else:
                    # Discarded single stray nibble
                    ui_dirty = True

            if valid_signal:
                self.rx_state = "LOCKED"
                self.rx_sym_idx = syms_per_hex - 1
                self.rx_current_val = 0
                self.rx_next_sample_time = self.audio_time + sample_delay

        elif self.rx_state == "LOCKED":
            if self.audio_time >= self.rx_next_sample_time:
                if not valid_signal and self.rx_sym_idx == (syms_per_hex - 1):
                    self.rx_state = "IDLE"
                else:
                    sym = current_fsk_state if valid_signal else best_all_sym
                    self.rx_current_val |= sym << (self.rx_sym_idx * bits_per_sym)
                    self.rx_sym_idx -= 1

                    if self.rx_sym_idx < 0:
                        char_hex = hex(self.rx_current_val)[2:].upper()
                        self.current_rx_hex += char_hex
                        if len(self.current_rx_hex) > 121:
                            # Keep preamble + last 120 hex chars (even length)
                            self.current_rx_hex = self.current_rx_hex[0] + self.current_rx_hex[-120:]
                        self.last_char_time = self.audio_time
                        ui_dirty = True

                        self.rx_sym_idx = syms_per_hex - 1
                        self.rx_current_val = 0

                        self.rx_state = "WAIT_NEXT_CHAR"
                        self.rx_next_sample_time = (
                            self.audio_time + (float(sp["toneDur"]) - sample_delay) + wait_pad
                        )
                    else:
                        self.rx_next_sample_time += sym_interval

            if self.rx_state == "LOCKED" and self.audio_time > self.rx_next_sample_time + 0.5:
                self.rx_state = "IDLE"

        elif self.rx_state == "WAIT_NEXT_CHAR":
            if self.audio_time >= self.rx_next_sample_time:
                if not valid_signal or self.audio_time >= (self.rx_next_sample_time + 0.06):
                    self.rx_state = "IDLE"

        if self.rx_state != prev_state:
            ui_dirty = True

        return ui_dirty


# ------------------------ Student Link Sync Dialog ------------------------


class StudentLinkDialog(QDialog):
    def __init__(self, base_url: str, profile_key: str, mod_mode: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Classroom Student Sync Link")
        self.setMinimumWidth(480)
        self.base_url = base_url.rstrip("/") + "/"
        self.profile_key = profile_key
        self.mod_mode = mod_mode

        layout = QVBoxLayout(self)

        info = QLabel(
            "Share this link with students so their devices open directly on the "
            "exact same profile and modulation mode:"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("Student UI Language:"))
        self.combo_lang = QComboBox()
        self.combo_lang.addItem("Română (RO)", userData="ro")
        self.combo_lang.addItem("English (EN)", userData="en")
        self.combo_lang.currentIndexChanged.connect(self._refresh_url)
        lang_row.addWidget(self.combo_lang)
        lang_row.addStretch()
        layout.addLayout(lang_row)

        self.url_edit = QLineEdit()
        self.url_edit.setReadOnly(True)
        self.url_edit.setFont(QFont("Monospace", 11))
        layout.addWidget(self.url_edit)

        btn_row = QHBoxLayout()
        btn_copy = QPushButton("Copy Link to Clipboard")
        btn_copy.clicked.connect(self._copy_url)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_copy)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        self.lbl_copied = QLabel("")
        self.lbl_copied.setStyleSheet("color: #00e5ff;")
        layout.addWidget(self.lbl_copied)

        self._refresh_url()

    def _refresh_url(self):
        lang = self.combo_lang.currentData()
        qs = urlencode({"profile": self.profile_key, "mode": self.mod_mode, "lang": lang})
        self.url_edit.setText(f"{self.base_url}?{qs}")
        self.lbl_copied.setText("")

    def _copy_url(self):
        QApplication.clipboard().setText(self.url_edit.text())
        self.lbl_copied.setText("Copied to clipboard!")


# ------------------------ Main Professor PyQt6 App ------------------------


class AcousticSpectrometerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = load_shared_config()
        self.engine = DemodulatorEngine(self.config, sr=44100, hop=512, nfft=2048, seconds=8.0)

        self.setWindowTitle("Acoustic Spectrometer & Bird-Language Receiver (Professor)")
        self.resize(1300, 780)

        self.current_mode = "RX"
        self.projector_mode = False
        self.freq_lines = []
        self.q_in: queue.Queue = queue.Queue()
        self.stream = None
        self.mic_error_msg = ""

        self.setup_ui()
        self.populate_microphones()
        self.update_mod_controls_for_profile()
        self.update_freq_guides()
        self.plot_widget.setYRange(0, self.engine.current_fmax, padding=0)

        # Open default microphone safely
        self.start_audio_stream(self.combo_mic.currentData())

        # GUI Update Timer (~60 Hz)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(16)

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Left Panel (Controls)
        left_panel = QWidget()
        left_panel.setFixedWidth(360)
        left_layout = QVBoxLayout(left_panel)
        layout.addWidget(left_panel)

        # Status Pill / Signal Lock LED
        self.lbl_status_pill = QLabel("● LISTENING")
        self.lbl_status_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status_pill.setFont(QFont("Monospace", 11, QFont.Weight.Bold))
        self.lbl_status_pill.setStyleSheet(
            "background: #1d2b2d; color: #00e5ff; padding: 6px; border-radius: 4px; border: 1px solid #00adb5;"
        )
        left_layout.addWidget(self.lbl_status_pill)

        # Microphone Selector
        group_mic = QGroupBox("Microphone Input")
        mic_layout = QHBoxLayout(group_mic)
        self.combo_mic = QComboBox()
        self.combo_mic.currentIndexChanged.connect(self.on_mic_change)
        btn_refresh_mic = QPushButton("↻")
        btn_refresh_mic.setFixedWidth(32)
        btn_refresh_mic.setToolTip("Refresh audio devices")
        btn_refresh_mic.clicked.connect(self.populate_microphones)
        mic_layout.addWidget(self.combo_mic)
        mic_layout.addWidget(btn_refresh_mic)
        left_layout.addWidget(group_mic)

        # Operation Mode Toggle
        group_mode = QGroupBox("Operation Mode")
        mode_layout = QVBoxLayout(group_mode)
        self.radio_rx = QRadioButton("Data RX Mode (Classic & Bird FSK)")
        self.radio_vis = QRadioButton("Full Visualizer Mode (Images / Free Audio)")
        self.radio_rx.setChecked(True)
        self.radio_rx.toggled.connect(self.on_mode_change)
        mode_layout.addWidget(self.radio_rx)
        mode_layout.addWidget(self.radio_vis)
        left_layout.addWidget(group_mode)

        # Profile & Modulation Selector
        group_profile = QGroupBox("Acoustic Profile & Modulation")
        prof_layout = QVBoxLayout(group_profile)

        self.combo_profile = QComboBox()
        for key, prof in self.engine.profiles.items():
            self.combo_profile.addItem(prof["labelPy"], userData=key)
        self.combo_profile.setCurrentIndex(list(self.engine.profiles.keys()).index(self.engine.profile_key))
        self.combo_profile.currentIndexChanged.connect(self.on_profile_change)
        prof_layout.addWidget(self.combo_profile)

        self.radio_16fsk = QRadioButton("16-FSK (16 Notes - Quiet Room)")
        self.radio_4fsk = QRadioButton("4-FSK (4 Notes - Balanced)")
        self.radio_2fsk = QRadioButton("2-FSK (2 Notes - Noise-Safe / Classic)")
        self.radio_4fsk.setChecked(True)

        self.radio_16fsk.toggled.connect(self.on_mod_change)
        self.radio_4fsk.toggled.connect(self.on_mod_change)
        self.radio_2fsk.toggled.connect(self.on_mod_change)

        prof_layout.addWidget(self.radio_16fsk)
        prof_layout.addWidget(self.radio_4fsk)
        prof_layout.addWidget(self.radio_2fsk)

        btn_sync_link = QPushButton("Student Sync Link (Render URL)")
        btn_sync_link.clicked.connect(self.show_student_link_dialog)
        prof_layout.addWidget(btn_sync_link)

        left_layout.addWidget(group_profile)

        # Decoded Messages Output (Scrollable, Projector-ready)
        self.txt_output = QTextEdit()
        self.txt_output.setReadOnly(True)
        self.txt_output.setFont(QFont("Monospace", 12, QFont.Weight.Bold))
        self.txt_output.setStyleSheet(
            "color: #00e5ff; background: #121212; padding: 8px; border: 1px solid #333; border-radius: 5px;"
        )
        self.txt_output.setMinimumHeight(150)
        left_layout.addWidget(self.txt_output)

        # Output controls row
        out_btn_row = QHBoxLayout()
        btn_clear = QPushButton("Clear Messages")
        btn_clear.clicked.connect(self.clear_text)
        self.chk_projector = QCheckBox("Projector Font")
        self.chk_projector.toggled.connect(self.on_projector_toggle)
        out_btn_row.addWidget(btn_clear)
        out_btn_row.addWidget(self.chk_projector)
        left_layout.addLayout(out_btn_row)

        # DSP Sliders
        self.group_sliders = QGroupBox(f"DSP Settings ({self.engine.sr} Hz)")
        form = QFormLayout(self.group_sliders)

        self.sl_fmax = self.create_slider(1000, int(self.engine.sr / 2), int(self.engine.current_fmax), self.on_fmax_change)
        self.sl_hp = self.create_slider(10, 10000, int(self.engine.hp_fc), self.on_filter_change)
        self.sl_lp = self.create_slider(500, int(self.engine.sr / 2 - 500), int(self.engine.lp_fc), self.on_filter_change)
        self.sl_gate = self.create_slider(2, 60, int(self.engine.gate_margin), self.on_gate_change)

        self.lbl_fmax_val = QLabel(f"{int(self.engine.current_fmax)} Hz")
        self.lbl_hp_val = QLabel(f"{int(self.engine.hp_fc)} Hz")
        self.lbl_lp_val = QLabel(f"{int(self.engine.lp_fc)} Hz")
        self.lbl_gate_val = QLabel(f"{int(self.engine.gate_margin)} dB")

        form.addRow("Max Freq:", self._slider_row(self.sl_fmax, self.lbl_fmax_val))
        form.addRow("HP Cutoff:", self._slider_row(self.sl_hp, self.lbl_hp_val))
        form.addRow("LP Cutoff:", self._slider_row(self.sl_lp, self.lbl_lp_val))
        form.addRow("Gate Margin:", self._slider_row(self.sl_gate, self.lbl_gate_val))

        btn_reset_dsp = QPushButton("Reset DSP Defaults")
        btn_reset_dsp.clicked.connect(self.reset_dsp_defaults)
        form.addRow(btn_reset_dsp)
        left_layout.addWidget(self.group_sliders)

        # Right Panel (Spectrogram)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Frequency", units="Hz")
        layout.addWidget(self.plot_widget)

        self.img = pg.ImageItem()
        self.plot_widget.addItem(self.img)

        colormap = pg.colormap.get("magma")
        self.img.setColorMap(colormap)

        self.refresh_display()

    def _slider_row(self, slider, label):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        label.setFixedWidth(62)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        h.addWidget(slider)
        h.addWidget(label)
        return w

    def create_slider(self, min_val, max_val, init_val, callback):
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setMinimum(min_val)
        sl.setMaximum(max_val)
        sl.setValue(init_val)
        sl.valueChanged.connect(callback)
        return sl

    def populate_microphones(self):
        self.combo_mic.blockSignals(True)
        self.combo_mic.clear()
        self.combo_mic.addItem("System Default Microphone", userData=None)
        try:
            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) > 0:
                    name = dev.get("name", f"Device {idx}")
                    sr_hint = int(dev.get("default_samplerate", 44100))
                    self.combo_mic.addItem(f"[{idx}] {name} ({sr_hint} Hz)", userData=idx)
        except Exception as exc:
            self.mic_error_msg = f"Audio device query failed: {exc}"
        self.combo_mic.blockSignals(False)

    def start_audio_stream(self, device_idx):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        self.mic_error_msg = ""
        candidate_rates = [44100]
        try:
            dev_info = sd.query_devices(device_idx, "input")
            default_sr = int(dev_info.get("default_samplerate", 44100))
            if default_sr not in candidate_rates:
                candidate_rates.append(default_sr)
        except Exception:
            candidate_rates.append(48000)

        last_err = None
        for sr in candidate_rates:
            try:
                if sr != self.engine.sr:
                    self.engine.set_sample_rate(sr)
                    self._update_slider_bounds_for_sr()
                self.stream = sd.InputStream(
                    device=device_idx,
                    samplerate=self.engine.sr,
                    channels=1,
                    blocksize=self.engine.hop,
                    callback=self.audio_callback,
                )
                self.stream.start()
                self.group_sliders.setTitle(f"DSP Settings ({self.engine.sr} Hz)")
                self.refresh_display()
                return
            except Exception as exc:
                last_err = exc

        self.mic_error_msg = f"Could not open microphone: {last_err}"
        self.refresh_display()

    def _update_slider_bounds_for_sr(self):
        nyq = int(self.engine.sr / 2)
        self.sl_fmax.setMaximum(nyq)
        self.sl_lp.setMaximum(max(500, nyq - 500))

    def on_mic_change(self):
        self.start_audio_stream(self.combo_mic.currentData())

    def audio_callback(self, indata, frames, time_info, status):
        block = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        self.q_in.put(block)

    def update_mod_controls_for_profile(self):
        prof = self.engine.get_active_profile()
        supported = prof.get("supportedModes", ["2fsk"])

        self.radio_16fsk.setEnabled("16fsk" in supported)
        self.radio_4fsk.setEnabled("4fsk" in supported)
        self.radio_2fsk.setEnabled("2fsk" in supported)

        if self.engine.mod_mode not in supported:
            self.engine.mod_mode = prof.get("defaultMode", supported[0])

        self.radio_16fsk.blockSignals(True)
        self.radio_4fsk.blockSignals(True)
        self.radio_2fsk.blockSignals(True)
        self.radio_16fsk.setChecked(self.engine.mod_mode == "16fsk")
        self.radio_4fsk.setChecked(self.engine.mod_mode == "4fsk")
        self.radio_2fsk.setChecked(self.engine.mod_mode == "2fsk")
        self.radio_16fsk.blockSignals(False)
        self.radio_4fsk.blockSignals(False)
        self.radio_2fsk.blockSignals(False)

    def update_freq_guides(self):
        for line in self.freq_lines:
            self.plot_widget.removeItem(line)
        self.freq_lines.clear()

        if not self.engine.rx_enabled:
            return

        mod = self.engine.mod_mode
        alpha = 95 if mod == "16fsk" else 145
        color = (255, 167, 38, alpha) if self.engine.profile_key == "classic" else (0, 173, 181, alpha)
        label_hex = "#ffb74d" if self.engine.profile_key == "classic" else "#00adb5"
        pen = pg.mkPen(color=color, width=1, style=Qt.PenStyle.DashLine)

        for idx, f in enumerate(self.engine.get_active_freqs()):
            if mod == "16fsk":
                label_txt = f" {idx:X} ({int(f)})"
                pos = 0.05 if idx % 2 == 0 else 0.16
            elif mod == "4fsk":
                label_txt = f" {idx:02b} ({int(f)}Hz)"
                pos = 0.08
            else:
                label_txt = f" BIT {idx} ({int(f)}Hz)"
                pos = 0.08

            line = pg.InfiniteLine(
                pos=f,
                angle=0,
                pen=pen,
                label=label_txt,
                labelOpts={"position": pos, "color": label_hex, "fill": (0, 0, 0, 150)},
            )
            self.plot_widget.addItem(line)
            self.freq_lines.append(line)

    def apply_settings_dict(self, settings: dict):
        hp, lp = self.engine.clamp_filters(settings["hp"], settings["lp"])
        fmax = min(float(settings["fmax"]), self.engine.sr / 2.0)
        gate = float(settings["gate"])

        self.sl_hp.blockSignals(True)
        self.sl_lp.blockSignals(True)
        self.sl_fmax.blockSignals(True)
        self.sl_gate.blockSignals(True)

        self.sl_hp.setValue(int(hp))
        self.sl_lp.setValue(int(lp))
        self.sl_fmax.setValue(int(fmax))
        self.sl_gate.setValue(int(gate))

        self.sl_hp.blockSignals(False)
        self.sl_lp.blockSignals(False)
        self.sl_fmax.blockSignals(False)
        self.sl_gate.blockSignals(False)

        self.engine.current_fmax = fmax
        self.engine.gate_margin = gate

        self.lbl_hp_val.setText(f"{int(hp)} Hz")
        self.lbl_lp_val.setText(f"{int(lp)} Hz")
        self.lbl_fmax_val.setText(f"{int(fmax)} Hz")
        self.lbl_gate_val.setText(f"{int(gate)} dB")

        self.engine.noise_spec = None
        self.engine.update_filters()
        self.plot_widget.setYRange(0, self.engine.current_fmax, padding=0)

    def _active_save_key(self) -> str:
        return self.engine.profile_key if self.current_mode == "RX" else "VIS"

    def on_mode_change(self):
        if self.radio_rx.isChecked():
            self.current_mode = "RX"
            self.engine.rx_enabled = True
            self.apply_settings_dict(self.engine.saved_settings[self.engine.profile_key])
        else:
            self.current_mode = "VIS"
            self.engine.rx_enabled = False
            self.engine.reset_demod_state()
            self.apply_settings_dict(self.engine.saved_settings["VIS"])

        self.update_freq_guides()
        self.refresh_display()

    def on_profile_change(self):
        self.engine.profile_key = self.combo_profile.currentData()
        self.engine.finalize_current_message()
        self.engine.reset_demod_state(clear_current_buffer=True)
        self.update_mod_controls_for_profile()
        if self.engine.rx_enabled:
            self.apply_settings_dict(self.engine.saved_settings[self.engine.profile_key])
            self.update_freq_guides()
        self.refresh_display()

    def on_mod_change(self):
        if self.radio_16fsk.isChecked():
            self.engine.mod_mode = "16fsk"
        elif self.radio_4fsk.isChecked():
            self.engine.mod_mode = "4fsk"
        else:
            self.engine.mod_mode = "2fsk"

        self.engine.finalize_current_message()
        self.engine.reset_demod_state(clear_current_buffer=True)
        self.update_freq_guides()
        self.refresh_display()

    def on_fmax_change(self, val):
        self.engine.current_fmax = float(val)
        self.lbl_fmax_val.setText(f"{int(val)} Hz")
        self.plot_widget.setYRange(0, self.engine.current_fmax, padding=0)
        self.engine.saved_settings[self._active_save_key()]["fmax"] = int(val)

    def on_filter_change(self):
        hp, lp = self.engine.clamp_filters(self.sl_hp.value(), self.sl_lp.value())
        if int(hp) != self.sl_hp.value():
            self.sl_hp.blockSignals(True)
            self.sl_hp.setValue(int(hp))
            self.sl_hp.blockSignals(False)
        if int(lp) != self.sl_lp.value():
            self.sl_lp.blockSignals(True)
            self.sl_lp.setValue(int(lp))
            self.sl_lp.blockSignals(False)

        self.lbl_hp_val.setText(f"{int(hp)} Hz")
        self.lbl_lp_val.setText(f"{int(lp)} Hz")
        self.engine.update_filters()
        key = self._active_save_key()
        self.engine.saved_settings[key]["hp"] = int(hp)
        self.engine.saved_settings[key]["lp"] = int(lp)

    def on_gate_change(self, val):
        self.engine.gate_margin = float(val)
        self.lbl_gate_val.setText(f"{int(val)} dB")
        self.engine.saved_settings[self._active_save_key()]["gate"] = int(val)

    def reset_dsp_defaults(self):
        key = self._active_save_key()
        if key == "VIS":
            defaults = dict(self.engine.vis_defaults)
        else:
            defaults = dict(self.engine.profiles[key]["rx"])
        self.engine.saved_settings[key] = defaults
        self.apply_settings_dict(defaults)

    def on_projector_toggle(self, checked: bool):
        self.projector_mode = checked
        size = 20 if checked else 12
        self.txt_output.setFont(QFont("Monospace", size, QFont.Weight.Bold))
        self.refresh_display()

    def clear_text(self):
        self.engine.clear_all_messages()
        self.refresh_display()

    def show_student_link_dialog(self):
        base_url = self.config.get("renderUrl", "https://audio-spectrometer.onrender.com/")
        dlg = StudentLinkDialog(base_url, self.engine.profile_key, self.engine.mod_mode, self)
        dlg.exec()

    def update_loop(self):
        blocks_processed = 0
        ui_dirty = False

        while not self.q_in.empty() and blocks_processed < 20:
            try:
                block = self.q_in.get_nowait()
            except queue.Empty:
                break
            if self.engine.process_block(block):
                ui_dirty = True
            blocks_processed += 1

        if ui_dirty:
            self.refresh_display()

        if blocks_processed > 0:
            f_idx = int(np.abs(self.engine.freqs_all - self.engine.current_fmax).argmin())
            f_idx = max(1, f_idx)
            display_data = self.engine.spec_data[:, :f_idx]

            if self.engine.rx_enabled:
                display_data = median_filter(display_data, size=3)

            actual_fmax = float(self.engine.freqs_all[f_idx - 1])
            self.img.setRect(QRectF(-self.engine.seconds, 0, self.engine.seconds, actual_fmax))

            c_min = max(-130.0, self.engine.last_noise_floor_db - 15.0)
            c_max = max(c_min + 20.0, self.engine.smooth_cmax)
            self.img.setImage(display_data, autoLevels=False, levels=(c_min, c_max))

    def refresh_display(self):
        if self.mic_error_msg:
            self.lbl_status_pill.setText("● MIC ERROR")
            self.lbl_status_pill.setStyleSheet(
                "background: #3b1717; color: #ff6b6b; padding: 6px; border-radius: 4px; border: 1px solid #d9534f;"
            )
            self.txt_output.setPlainText(self.mic_error_msg)
            return

        if not self.engine.rx_enabled:
            self.lbl_status_pill.setText("● VISUALIZER MODE")
            self.lbl_status_pill.setStyleSheet(
                "background: #2b2214; color: #ffb74d; padding: 6px; border-radius: 4px; border: 1px solid #ff9800;"
            )
            self.txt_output.setPlainText(
                "MODE: FULL VISUALIZER\n\nFull spectrum monitoring active.\nPlay spectrogram images or live audio!"
            )
            return

        mode_tag = f"{self.engine.profile_key.upper()} · {self.engine.mod_mode.upper()}"
        if self.engine.rx_state != "IDLE":
            self.lbl_status_pill.setText(f"● RECEIVING ({mode_tag})")
            self.lbl_status_pill.setStyleSheet(
                "background: #14331d; color: #69f0ae; padding: 6px; border-radius: 4px; border: 1px solid #00c853;"
            )
        else:
            self.lbl_status_pill.setText(f"● LISTENING ({mode_tag})")
            self.lbl_status_pill.setStyleSheet(
                "background: #1d2b2d; color: #00e5ff; padding: 6px; border-radius: 4px; border: 1px solid #00adb5;"
            )

        lines = []
        active_hex = self.engine.current_rx_hex[1:] if len(self.engine.current_rx_hex) > 1 else ""
        if active_hex:
            active_txt = decode_hex_to_text(active_hex)
            if self.projector_mode:
                lines.append(f"LIVE: {active_txt}")
            else:
                lines.append(f"[LIVE] TXT: {active_txt}\n       HEX: {active_hex}")

        if self.engine.message_history:
            if lines:
                lines.append("-" * 24)
            for idx, (h, t) in enumerate(reversed(self.engine.message_history), start=1):
                msg_num = len(self.engine.message_history) - idx + 1
                if self.projector_mode:
                    lines.append(f"#{msg_num}: {t}")
                else:
                    lines.append(f"#{msg_num} TXT: {t}\n   HEX: {h}")

        if not lines:
            lines.append(f"[{mode_tag}]\nWaiting for student signal...")

        self.txt_output.setPlainText("\n".join(lines))

    def closeEvent(self, event):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, Qt.GlobalColor.black)
    palette.setColor(palette.ColorRole.WindowText, Qt.GlobalColor.white)
    app.setPalette(palette)

    window = AcousticSpectrometerApp()
    window.show()
    sys.exit(app.exec())
