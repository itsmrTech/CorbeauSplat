import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from .base_engine import BaseEngine
from .i18n import tr
from app.scripts.setup_dependencies import install_extractor_360, get_venv_360_python, uninstall_extractor_360, resolve_project_root

_MASK_SUFFIX = ".mask.png"
_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def find_source_image(mask_path: Path) -> Optional[Path]:
    """Return the extracted image a ``<name>.mask.png`` mask belongs to, or None.

    Handles both naming conventions: an appended suffix
    (``frame_001.jpg.mask.png`` -> ``frame_001.jpg``) and a replaced extension
    (``frame_001.mask.png`` -> ``frame_001.jpg`` / ``.png`` / ``.jpeg``).
    """
    name = mask_path.name
    if not name.lower().endswith(_MASK_SUFFIX):
        return None
    parent = mask_path.parent
    base = name[: -len(_MASK_SUFFIX)]  # "frame_001.jpg" or "frame_001"

    # Appended-suffix convention: base already carries the image extension.
    appended = parent / base
    if appended.suffix.lower() in _IMAGE_EXTS and appended.exists():
        return appended

    # Replaced-extension convention: try each known image extension.
    stem = appended.stem if appended.suffix else base
    for ext in _IMAGE_EXTS:
        candidate = parent / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def operator_mask_fraction(mask_path: Path) -> float:
    """Fraction of a mask covered by the (minority) masked-out region.

    The 360 extractor marks the detected operator/object as one extreme of the
    mask and the background as the other. This returns ~0 for a uniform "keep
    everything" mask and a positive fraction when an operator region is present
    — independent of whether the tool marks the operator black or white (it
    supports mask inversion) and of whether the mask is stored as luminance or
    in an alpha channel.
    """
    from PIL import Image
    import numpy as np

    with Image.open(mask_path) as im:
        bands = im.getbands()
        signal = np.asarray(im.convert("L"), dtype=np.uint8)
        if signal.size and "A" in bands:
            alpha = np.asarray(im.getchannel("A"), dtype=np.uint8)
            # Use whichever channel actually carries the mask (more contrast).
            if int(alpha.max()) - int(alpha.min()) > int(signal.max()) - int(signal.min()):
                signal = alpha

    if signal.size == 0:
        return 0.0
    high = float(np.count_nonzero(signal >= 128)) / signal.size
    return min(high, 1.0 - high)


class Extractor360Engine(BaseEngine):
    def __init__(self, logger_callback=None):
        super().__init__("360Extractor", logger_callback)
        self.root_dir = Path(resolve_project_root())
        self.engines_dir = self.root_dir / "engines"
        self.extractor_dir = self.engines_dir / "extractor_360"
        self.venv_python = Path(get_venv_360_python())
        self.script_path = self.extractor_dir / "src" / "main.py"

    def is_installed(self):
        """Checks if venv and script exist"""
        return self.venv_python.exists() and self.script_path.exists()

    def install(self):
        """Installs via setup_dependencies"""
        install_extractor_360()

    def uninstall(self):
        """Uninstalls"""
        uninstall_extractor_360()

    def run_extraction(self, input_path, output_dir, params, progress_callback=None, log_callback=None, status_callback=None, check_cancel_callback=None):
        """
        Runs the extraction CLI.
        params: dict of arguments mirroring CLI args
        """
        if status_callback: status_callback(tr("status_extracting_360", "Extraction vidéo 360°..."))
        if not self.is_installed():
            if log_callback: log_callback("Error: 360Extractor not installed.")
            return False

        cmd = [
            self.venv_python,
            self.script_path,
            "--input", input_path,
            "--output", output_dir
        ]

        # Map params to CLI args
        # interval
        if "interval" in params:
            cmd.extend(["--interval", str(params["interval"])])
        
        # format
        if "format" in params:
            cmd.extend(["--format", params["format"]])
            
        # resolution
        if "resolution" in params:
            cmd.extend(["--resolution", str(params["resolution"])])
            
        # camera-count
        if "camera_count" in params:
            cmd.extend(["--camera-count", str(params["camera_count"])])
            
        # quality
        if "quality" in params:
            cmd.extend(["--quality", str(params["quality"])])
            
        # layout
        if "layout" in params:
            cmd.extend(["--layout", params["layout"]])
            
        # AI options. The operator-drop mode needs the YOLO masks, so it
        # implies --ai-mask even if the user did not tick "Mask Operator".
        drop_operator = params.get("drop_operator", False)
        if params.get("ai_mask", False) or drop_operator:
            cmd.append("--ai-mask")
            if drop_operator and not params.get("ai_mask", False) and log_callback:
                log_callback("Operator drop enabled - turning on AI masking to detect the operator.")

        if params.get("ai_skip", False):
            cmd.append("--ai-skip")
            
        if params.get("adaptive", False):
            cmd.append("--adaptive")
            if "motion_threshold" in params:
                cmd.extend(["--motion-threshold", str(params["motion_threshold"])])

        if log_callback:
            # Use map(str, ...) to handle Path objects in the list
            log_callback(f"Command: {' '.join(map(str, cmd))}")

        # Run process
        # We use Popen to capture stdout/stderr for progress
        env = os.environ.copy()
        # Isolate from the main app's PYTHONPATH to avoid package conflicts
        env.pop("PYTHONPATH", None)
        
        # Ensure all arguments are strings for subprocess
        cmd_str = [str(arg) for arg in cmd]

        # Use BaseEngine's Template Method for process execution
        def line_handler(line: str):
            if log_callback:
                log_callback(line)
            if "%" in line and progress_callback:
                try:
                    if "[" in line and "%]" in line:
                        part = line.split("[")[1].split("%]")[0]
                        progress_callback(int(part.strip()))
                except (ValueError, IndexError):
                    pass

        if status_callback:
            status_callback(tr("status_extracting_360", "Extraction vidéo 360°..."))

        returncode = self._execute_command(cmd_str, env=env, cwd=str(self.extractor_dir), line_callback=line_handler)

        if check_cancel_callback and check_cancel_callback():
            if log_callback:
                log_callback("Process cancelled by user.")
            return False

        if returncode == 0 and drop_operator:
            if status_callback:
                status_callback(tr("status_360_drop_operator", "Removing operator faces..."))
            threshold = float(params.get("operator_drop_threshold", 0.005))
            try:
                self._drop_operator_frames(output_dir, threshold, log_callback)
            except Exception as e:
                if log_callback:
                    log_callback(f"Operator drop: post-processing failed: {e}")

        if status_callback:
            status_callback(tr("status_ready", "Traitement terminé !"))
        return returncode == 0

    def _drop_operator_frames(self, output_dir, threshold, log_callback=None) -> Tuple[int, int]:
        """Delete extracted faces whose operator mask covers at least ``threshold``.

        Relies on the masks produced by the AI masking step. For every
        ``*.mask.png`` whose masked (operator) region covers at least
        ``threshold`` of the image, the image and its mask are removed; faces
        with no operator are left untouched. Returns (dropped, kept).
        """
        out = Path(output_dir)
        masks = sorted(out.rglob("*" + _MASK_SUFFIX))
        if not masks:
            if log_callback:
                log_callback("Operator drop: no masks found - was AI masking enabled? Nothing removed.")
            return 0, 0

        dropped = kept = 0
        for mask in masks:
            try:
                fraction = operator_mask_fraction(mask)
            except Exception as e:
                if log_callback:
                    log_callback(f"Operator drop: could not read mask {mask.name}: {e}")
                continue

            image = find_source_image(mask)
            if fraction >= threshold:
                if image is not None:
                    image.unlink(missing_ok=True)
                mask.unlink(missing_ok=True)
                dropped += 1
                if log_callback:
                    target = image.name if image is not None else mask.name
                    log_callback(f"Operator drop: removed {target} (operator {fraction * 100:.1f}%)")
            else:
                kept += 1

        if log_callback:
            log_callback(f"Operator drop: removed {dropped} face(s) containing the operator, kept {kept}.")
        return dropped, kept
