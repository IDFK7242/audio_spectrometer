import re
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ui import DemodulatorEngine, decode_hex_to_text, get_active_freqs, load_shared_config


def text_to_hex(text: str) -> str:
    text = text.replace(" ", "_")
    return "".join(f"{ord(ch):02X}" for ch in text)


def synthesize_transmission(
    hex_payload: str,
    profile: dict,
    mod_mode: str,
    sr: int = 44100,
    lead_silence: float = 1.2,
    tail_silence: float = 1.8,
) -> np.ndarray:
    """
    Synthesizes an acoustic transmission matching index.html (wakeup tone + 'F' preamble + payload)
    for deterministic headless loopback testing of DemodulatorEngine.
    """
    full_hex = "F" + hex_payload.upper()
    active_freqs = get_active_freqs(profile, mod_mode)
    tone_dur = float(profile["toneDur"])
    gap_dur = float(profile["gapDur"])
    char_gap = float(profile["charGap"])
    gap16 = float(profile["gap16"])
    sym_interval = tone_dur + gap_dur
    is_square = profile.get("waveform") == "square"
    is_16 = mod_mode == "16fsk"

    # Estimate total duration
    sync_dur = 0.50 if is_square else 0.45
    sync_stride = 0.60 if is_square else 0.58
    est_dur = lead_silence + sync_stride + len(full_hex) * (4 * sym_interval + char_gap + 0.2) + tail_silence
    buf = np.zeros(int(est_dur * sr), dtype=np.float32)

    def add_tone(start_t: float, dur: float, freq: float, amp: float = 0.25, square: bool = False, scoop_ratio: float = 1.0, scoop_dur: float = 0.0):
        i0 = int(round(start_t * sr))
        n = int(round(dur * sr))
        if n <= 0 or i0 + n > len(buf):
            return
        t = np.arange(n, dtype=np.float64) / sr
        if not square and scoop_dur > 0 and scoop_ratio != 1.0:
            n_scoop = min(n, int(round(scoop_dur * sr)))
            f_inst = np.full(n, freq, dtype=np.float64)
            if n_scoop > 0:
                f_start = freq * scoop_ratio
                f_inst[:n_scoop] = np.geomspace(f_start, freq, n_scoop)
            phase = 2.0 * np.pi * np.cumsum(f_inst) / sr
            sig = np.sin(phase)
        else:
            phase = 2.0 * np.pi * freq * t
            sig = np.sign(np.sin(phase)) if square else np.sin(phase)

        # Smooth 15ms attack/release envelope
        env = np.ones(n, dtype=np.float64)
        ramp = min(n // 4, int(0.015 * sr))
        if ramp > 0:
            env[:ramp] = np.linspace(0.0, 1.0, ramp)
            env[-ramp:] = np.linspace(1.0, 0.0, ramp)
        buf[i0 : i0 + n] += (amp * sig * env).astype(np.float32)

    t_cur = lead_silence
    # Wakeup tone
    add_tone(t_cur, sync_dur, float(profile["syncFreq"]), amp=0.22, square=False)
    t_cur += sync_stride

    for idx, ch in enumerate(full_hex):
        val = int(ch, 16)
        is_end_of_byte = idx > 0 and idx % 2 == 0
        scoop_dur = 0.015 if is_16 else float(profile["scoopDur"])
        scoop_ratio = 1.0 if (is_square or is_16) else float(profile["scoopRatio"])

        if mod_mode == "16fsk":
            add_tone(t_cur, tone_dur, active_freqs[val], square=False, scoop_ratio=scoop_ratio, scoop_dur=scoop_dur)
            t_cur += tone_dur + gap16 + (0.08 if is_end_of_byte else 0.0)
        elif mod_mode == "4fsk":
            syms = [(val >> 2) & 0x3, val & 0x3]
            for k, s in enumerate(syms):
                add_tone(
                    t_cur + k * sym_interval,
                    tone_dur,
                    active_freqs[s],
                    square=is_square,
                    scoop_ratio=scoop_ratio,
                    scoop_dur=scoop_dur,
                )
            t_cur += sym_interval + tone_dur + char_gap
        else:
            for bit_pos in range(3, -1, -1):
                bit = (val >> bit_pos) & 1
                k = 3 - bit_pos
                add_tone(
                    t_cur + k * sym_interval,
                    tone_dur,
                    active_freqs[bit],
                    square=is_square,
                    scoop_ratio=scoop_ratio,
                    scoop_dur=scoop_dur,
                )
            t_cur += 3 * sym_interval + tone_dur + char_gap

    end_sample = min(len(buf), int((t_cur + tail_silence) * sr))
    # Add realistic low-level room noise floor
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 1e-4, size=end_sample).astype(np.float32)
    return buf[:end_sample] + noise


def run_engine_on_signal(engine: DemodulatorEngine, signal: np.ndarray):
    hop = engine.hop
    for i in range(0, len(signal) - hop + 1, hop):
        engine.process_block(signal[i : i + hop])


class TestSyncAndFrontend(unittest.TestCase):
    def test_profiles_json_synced_with_html(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "sync_profiles.py"), "--check"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)

    def test_html_translations_complete_and_js_valid(self):
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        # Extract inline script
        m = re.search(r"<script>(.*?)</script>", html, re.DOTALL)
        self.assertIsNotNone(m, "Inline <script> not found in index.html")
        js_code = m.group(1)

        # Check all data-translate and data-placeholder keys exist in both en and ro
        keys = set(re.findall(r'data-(?:translate|placeholder)="([^"]+)"', html))
        self.assertGreater(len(keys), 10)
        for key in keys:
            occurrences = len(re.findall(rf"\b{re.escape(key)}\s*:", js_code))
            self.assertGreaterEqual(
                occurrences,
                2,
                msg=f"Translation key '{key}' should exist in both EN and RO dictionaries",
            )

        # Validate JS syntax with node if installed
        node_check = subprocess.run(
            ["node", "--check", "-"],
            input=js_code,
            capture_output=True,
            text=True,
        )
        self.assertEqual(node_check.returncode, 0, msg=f"JS syntax error:\n{node_check.stderr}")


class TestDSPLoopback(unittest.TestCase):
    def setUp(self):
        self.config = load_shared_config()

    def _configure_engine(self, profile_key: str, mod_mode: str) -> DemodulatorEngine:
        eng = DemodulatorEngine(self.config, sr=44100, hop=512, nfft=2048, seconds=8.0)
        eng.profile_key = profile_key
        eng.mod_mode = mod_mode
        rx = eng.profiles[profile_key]["rx"]
        eng.hp_fc = float(rx["hp"])
        eng.lp_fc = float(rx["lp"])
        eng.gate_margin = float(rx["gate"])
        eng.update_filters()
        return eng

    def test_classic_2fsk_square_loopback(self):
        eng = self._configure_engine("classic", "2fsk")
        payload = text_to_hex("HI")
        sig = synthesize_transmission(payload, eng.get_active_profile(), "2fsk", sr=eng.sr)
        run_engine_on_signal(eng, sig)
        self.assertEqual(len(eng.message_history), 1)
        self.assertEqual(eng.message_history[0][1], "HI")

    def test_canary_4fsk_bird_loopback(self):
        eng = self._configure_engine("canary", "4fsk")
        payload = text_to_hex("SALUT")
        sig = synthesize_transmission(payload, eng.get_active_profile(), "4fsk", sr=eng.sr)
        run_engine_on_signal(eng, sig)
        self.assertEqual(len(eng.message_history), 1)
        self.assertEqual(eng.message_history[0][1], "SALUT")

    def test_blackbird_16fsk_bird_loopback(self):
        eng = self._configure_engine("blackbird", "16fsk")
        payload = text_to_hex("OK")
        sig = synthesize_transmission(payload, eng.get_active_profile(), "16fsk", sr=eng.sr)
        run_engine_on_signal(eng, sig)
        self.assertEqual(len(eng.message_history), 1)
        self.assertEqual(eng.message_history[0][1], "OK")

    def test_goldcrest_2fsk_bird_loopback(self):
        eng = self._configure_engine("goldcrest", "2fsk")
        payload = text_to_hex("A1")
        sig = synthesize_transmission(payload, eng.get_active_profile(), "2fsk", sr=eng.sr)
        run_engine_on_signal(eng, sig)
        self.assertEqual(len(eng.message_history), 1)
        self.assertEqual(eng.message_history[0][1], "A1")

    def test_multi_message_framing_no_preamble_shift(self):
        """Regression test for Bug P1: back-to-back messages must both strip their own 'F' preamble."""
        eng = self._configure_engine("canary", "4fsk")
        prof = eng.get_active_profile()
        sig1 = synthesize_transmission(text_to_hex("HI"), prof, "4fsk", sr=eng.sr, lead_silence=1.2, tail_silence=2.0)
        sig2 = synthesize_transmission(text_to_hex("OK"), prof, "4fsk", sr=eng.sr, lead_silence=0.2, tail_silence=2.0)
        combined = np.concatenate([sig1, sig2])
        run_engine_on_signal(eng, combined)
        self.assertEqual(len(eng.message_history), 2)
        self.assertEqual(eng.message_history[0][1], "HI")
        self.assertEqual(eng.message_history[1][1], "OK")

    def test_filter_inversion_guard(self):
        """Regression test for Bug P3: HP cutoff can never exceed LP - 200 Hz."""
        eng = self._configure_engine("canary", "4fsk")
        hp, lp = eng.clamp_filters(6000, 4000)
        self.assertLessEqual(hp, lp - 200.0)


if __name__ == "__main__":
    unittest.main()
