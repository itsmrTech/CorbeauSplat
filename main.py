#!/usr/bin/env python3
import sys
import argparse
import time
import shutil
import subprocess
import signal
from pathlib import Path as _Path

# ─────────────────────────────────────────────────────────────────────────────
# GUI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _set_macos_dock_icon(icon_path: _Path):
    try:
        from AppKit import NSApplication, NSImage
        ns_image = NSImage.alloc().initWithContentsOfFile_(str(icon_path))
        if ns_image:
            NSApplication.sharedApplication().setApplicationIconImage_(ns_image)
    except Exception:
        pass


def _launch_gui():
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QIcon
    from PyQt6.QtCore import QTimer

    app = QApplication(sys.argv)

    assets = _Path(__file__).parent / "assets"
    png_path  = assets / "icon.png"
    icns_path = assets / "icon.icns"
    icon_file = png_path if png_path.exists() else icns_path
    if icon_file.exists():
        app.setWindowIcon(QIcon(str(icon_file)))

    dock_src = icns_path if icns_path.exists() else png_path
    if dock_src.exists():
        QTimer.singleShot(0, lambda: _set_macos_dock_icon(dock_src))

    window = ColmapGUI()
    window.show()
    sys.exit(app.exec())


# ─────────────────────────────────────────────────────────────────────────────
# Imports (after GUI guard so headless runs don't pull Qt)
# ─────────────────────────────────────────────────────────────────────────────

from app.core.i18n import tr
from app.core.params import ColmapParams
from app.core.engine import ColmapEngine
from app.core.brush_engine import BrushEngine
from app.core.sharp_engine import SharpEngine
from app.core.superplat_engine import SuperSplatEngine
from app.core.system import check_dependencies
from app.gui.main_window import ColmapGUI

# ─────────────────────────────────────────────────────────────────────────────
# Brush defaults and presets
# ─────────────────────────────────────────────────────────────────────────────

BRUSH_DEFAULTS = {
    "total_steps": 30000,
    "sh_degree": 3,
    "start_iter": 0,
    "refine_every": 200,
    "growth_grad_threshold": 0.003,
    "growth_select_fraction": 0.2,
    "growth_stop_iter": 15000,
    "max_splats": 10_000_000,
    "checkpoint_interval": 7000,
    "max_resolution": 0,
    "with_viewer": False,
    "refine_mode": False,
}

BRUSH_PRESETS = {
    "fast": {
        "total_steps": 7000, "refine_every": 100,
        "growth_grad_threshold": 0.01, "growth_select_fraction": 0.2,
        "growth_stop_iter": 6000,
    },
    "std": {
        "total_steps": 30000, "refine_every": 200,
        "growth_grad_threshold": 0.003, "growth_select_fraction": 0.2,
        "growth_stop_iter": 15000,
    },
    "dense": {
        "total_steps": 50000, "refine_every": 100,
        "growth_grad_threshold": 0.0005, "growth_select_fraction": 0.6,
        "growth_stop_iter": 40000,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def get_parser():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="CorbeauSplat — Pipeline Gaussian Splatting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Without arguments, the graphical interface is launched.\n"
            "Each subcommand has its own help: main.py <command> --help\n\n"
            "Examples:\n"
            "  python3 main.py pipeline -i video.mp4 -o ~/projets --type video --preset dense\n"
            "  python3 main.py colmap   -i video.mp4 -o ~/projets\n"
            "  python3 main.py brush    -i ~/projets/scene -o ~/projets/scene --preset dense\n"
            "  python3 main.py sharp    -i photo.jpg -o ~/out\n"
            "  python3 main.py view     -i splat.ply\n"
            "  python3 main.py upscale  -i image.png -o ~/out --scale 4\n"
            "  python3 main.py 4dgs     -i ~/videos -o ~/out\n"
            "  python3 main.py extract360 -i 360.mp4 -o ~/out\n"
        ),
    )
    parser.add_argument("--gui", action="store_true", help="Force the graphical interface to launch")

    subs = parser.add_subparsers(dest="command", metavar="COMMANDE")

    # ── pipeline ──────────────────────────────────────────────────────────────
    p = subs.add_parser(
        "pipeline",
        help="Full pipeline: COLMAP → Brush in a single command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # From a video\n"
            "  python3 main.py pipeline -i video.mp4 -o ~/projets --type video\n\n"
            "  # From photos, high-quality preset\n"
            "  python3 main.py pipeline -i ~/photos -o ~/projets --preset dense\n\n"
            "  # With Glomap and a project name\n"
            "  python3 main.py pipeline -i ~/photos -o ~/projets --project_name scene --use_glomap\n"
        ),
    )
    p.add_argument("--input",  "-i", required=True, help="Source video or image folder")
    p.add_argument("--output", "-o", required=True, help="Parent output folder")
    p.add_argument("--project_name", default="Untitled", help="Project subfolder name (default: Untitled)")
    # COLMAP
    p.add_argument("--type", choices=["images", "video"], default="images",
                   help="Input type (default: images)")
    p.add_argument("--fps",  type=int, default=5,   help="Video extraction FPS (default: 5)")
    p.add_argument("--camera_model", default="SIMPLE_RADIAL",
                   choices=["SIMPLE_PINHOLE","PINHOLE","SIMPLE_RADIAL","RADIAL","OPENCV","OPENCV_FISHEYE"],
                   help="COLMAP camera model (default: SIMPLE_RADIAL)")
    p.add_argument("--undistort",  action="store_true", help="Undistort images after reconstruction")
    p.add_argument("--use_glomap", action="store_true", help="Use Glomap instead of the COLMAP mapper")
    p.add_argument("--matcher_type", choices=["exhaustive","sequential","vocab_tree"], default="exhaustive",
                   help="Matching strategy (default: exhaustive)")
    p.add_argument("--match_gpu_streams", type=int, default=2,
                   help="Parallel GPU matching streams; raises GPU use, same quality (default: 2)")
    p.add_argument("--max_image_size", type=int, default=3200,
                   help="Max image resolution for COLMAP (default: 3200)")
    # Brush
    p.add_argument("--preset", choices=["default","fast","std","dense"], default="default",
                   help="Brush training preset (default: default)")
    p.add_argument("--iterations", type=int,   default=None, metavar="N",
                   help="Total Brush iterations (overrides preset)")
    p.add_argument("--sh_degree",  type=int,   default=None, choices=range(1,5),
                   help="Spherical Harmonics degree 1-4 (default: 3)")
    p.add_argument("--device", default="auto",
                   choices=["auto","mps","cuda","cpu"], help="Brush device (default: auto)")
    p.add_argument("--with_viewer", action="store_true", help="Open the interactive viewer after training")
    p.add_argument("--ply_name",    default=None,        help="Output PLY filename")

    # ── colmap ────────────────────────────────────────────────────────────────
    p = subs.add_parser("colmap", help="COLMAP pipeline (video/images → dataset)")
    p.add_argument("--input",  "-i", required=True, help="Source video or image folder")
    p.add_argument("--output", "-o", required=True, help="Output folder")
    p.add_argument("--type", choices=["images", "video"], default="images", help="Input type (default: images)")
    p.add_argument("--fps",  type=int, default=5,         help="Video extraction FPS (default: 5)")
    p.add_argument("--project_name", default="Untitled",  help="Project subfolder name")
    # Options de base
    p.add_argument("--camera_model", default="SIMPLE_RADIAL",
                   choices=["SIMPLE_PINHOLE","PINHOLE","SIMPLE_RADIAL","RADIAL","OPENCV","OPENCV_FISHEYE"],
                   help="COLMAP camera model (default: SIMPLE_RADIAL)")
    p.add_argument("--undistort",  action="store_true", help="Undistort images after reconstruction")
    p.add_argument("--use_glomap", action="store_true", help="Use Glomap instead of the COLMAP mapper")
    # Feature extraction
    p.add_argument("--no_single_camera",  action="store_true", help="Disable single camera mode")
    p.add_argument("--max_image_size",    type=int,   default=3200, help="Max image resolution (default: 3200)")
    p.add_argument("--max_num_features",  type=int,   default=8192, help="Max features per image (default: 8192)")
    p.add_argument("--estimate_affine_shape", action="store_true", help="Estimate affine shape of features")
    p.add_argument("--no_domain_size_pooling", action="store_true", help="Disable domain size pooling")
    # Feature matching
    p.add_argument("--matcher_type", choices=["exhaustive","sequential","vocab_tree"], default="exhaustive",
                   help="Matching strategy (default: exhaustive)")
    p.add_argument("--match_gpu_streams", type=int, default=2,
                   help="Parallel GPU matching streams; raises GPU use, same quality (default: 2)")
    p.add_argument("--max_ratio",    type=float, default=0.8,  help="Max Lowe ratio (default: 0.8)")
    p.add_argument("--max_distance", type=float, default=0.7,  help="Max distance (default: 0.7)")
    p.add_argument("--no_cross_check", action="store_true", help="Disable cross-check")
    # Mapper
    p.add_argument("--min_model_size",    type=int, default=10, help="Min model size (default: 10)")
    p.add_argument("--min_num_matches",   type=int, default=15, help="Min number of matches (default: 15)")
    p.add_argument("--multiple_models",   action="store_true",  help="Allow multiple models")
    p.add_argument("--no_refine_focal",   action="store_true",  help="Do not refine focal length")
    p.add_argument("--refine_principal",  action="store_true",  help="Refine principal point")
    p.add_argument("--no_refine_extra",   action="store_true",  help="Do not refine extra params")

    # ── brush ─────────────────────────────────────────────────────────────────
    p = subs.add_parser("brush", help="Gaussian Splat training (Brush)")
    p.add_argument("--input",  "-i", required=True, help="COLMAP dataset folder")
    p.add_argument("--output", "-o", required=True, help="Output folder")
    p.add_argument("--preset", choices=["default","fast","std","dense"], default="default",
                   help="Parameter preset (default: default)")
    p.add_argument("--iterations", type=int,   default=None, metavar="N",
                   help="Total iterations (preset default: 30000)")
    p.add_argument("--sh_degree",  type=int,   default=None, choices=range(1,5),
                   help="Spherical Harmonics degree 1-4 (default: 3)")
    p.add_argument("--device",     default="auto",
                   choices=["auto","mps","cuda","cpu"], help="Device (default: auto)")
    p.add_argument("--refine_mode", action="store_true", help="Refine mode (resumes from last checkpoint)")
    p.add_argument("--with_viewer", action="store_true", help="Open the interactive viewer")
    p.add_argument("--ply_name",   default=None,      help="Output PLY filename")
    p.add_argument("--custom_args", default=None,     help="Additional arguments passed to brush")
    # Paramètres avancés (None = utilise la valeur du preset ou du défaut)
    p.add_argument("--start_iter",              type=int,   default=None, help="Starting iteration (default: 0)")
    p.add_argument("--refine_every",            type=int,   default=None, help="Densification every N iters (default: 200)")
    p.add_argument("--growth_grad_threshold",   type=float, default=None, help="Densification gradient threshold (default: 0.003)")
    p.add_argument("--growth_select_fraction",  type=float, default=None, help="Densification selection fraction (default: 0.2)")
    p.add_argument("--growth_stop_iter",        type=int,   default=None, help="Stop densification at iteration (default: 15000)")
    p.add_argument("--max_splats",              type=int,   default=None, help="Max number of Gaussians (default: 10 000 000)")
    p.add_argument("--checkpoint_interval",     type=int,   default=None, help="Save every N iters (default: 7000)")
    p.add_argument("--max_resolution",          type=int,   default=None, help="Max training resolution, 0=auto (default: 0)")

    # ── sharp ─────────────────────────────────────────────────────────────────
    p = subs.add_parser("sharp", help="Single Image/Video → 3D Splat (ML-Sharp)")
    p.add_argument("--input",  "-i", required=True, help="Image, image folder, or video")
    p.add_argument("--output", "-o", required=True, help="Output folder")
    p.add_argument("--mode",   choices=["image","video"], default="image",
                   help="Mode: single image or video (default: image)")
    p.add_argument("--checkpoint", "-c", default=None, help="Path to a .pt checkpoint")
    p.add_argument("--device", default="default",
                   choices=["default","mps","cpu","cuda"], help="Device (default: default)")
    p.add_argument("--skip_frames", type=int, default=1,
                   help="[video mode] Process 1 frame out of N (default: 1)")
    p.add_argument("--upscale", action="store_true",
                   help="Upscale images before prediction (requires upscayl-bin)")
    p.add_argument("--verbose", action="store_true", help="Show detailed Sharp output")

    # ── view ──────────────────────────────────────────────────────────────────
    p = subs.add_parser("view", help="Visualize a .ply in SuperSplat")
    p.add_argument("--input",     "-i", required=True, help=".ply file or folder")
    p.add_argument("--port",      type=int, default=3000, help="SuperSplat port (default: 3000)")
    p.add_argument("--data_port", type=int, default=8000, help="Data server port (default: 8000)")
    p.add_argument("--no_ui",     action="store_true", help="Hide the SuperSplat interface")
    p.add_argument("--cam_pos",   default=None, metavar="X,Y,Z", help="Initial camera position")
    p.add_argument("--cam_rot",   default=None, metavar="X,Y,Z", help="Initial camera rotation (degrees)")

    # ── upscale ───────────────────────────────────────────────────────────────
    p = subs.add_parser("upscale", help="Image upscaling via upscayl-bin (NCNN)")
    p.add_argument("--input",  "-i", required=True, help="Image or image folder")
    p.add_argument("--output", "-o", required=True, help="Output folder")
    p.add_argument("--model",  default="realesrgan-x4plus",
                   help="upscayl model ID (default: realesrgan-x4plus)")
    p.add_argument("--scale",  type=int, choices=[2, 3, 4], default=4,
                   help="Upscale factor (default: 4)")
    p.add_argument("--format", choices=["png","jpg","webp"], default="png",
                   help="Output format (default: png)")
    p.add_argument("--tile",        type=int, default=0,
                   help="VRAM tile size in px, 0=auto (default: 0)")
    p.add_argument("--tta",         action="store_true", help="Enable Test-Time Augmentation")
    p.add_argument("--compression", type=int, default=0,
                   help="Output compression level 0-9 (default: 0)")

    # ── 4dgs ──────────────────────────────────────────────────────────────────
    p = subs.add_parser("4dgs", help="4D Gaussian Splatting dataset preparation (Nerfstudio)")
    p.add_argument("--input",  "-i", required=True,
                   help="Folder containing multi-camera videos")
    p.add_argument("--output", "-o", required=True, help="Output folder")
    p.add_argument("--fps",    type=int, default=5,  help="Video extraction FPS (default: 5)")
    p.add_argument("--colmap_only", action="store_true",
                   help="Run COLMAP only on an already-extracted dataset")

    # ── extract360 ────────────────────────────────────────────────────────────
    p = subs.add_parser("extract360", help="Extract 360° video into COLMAP-ready multi-camera images")
    p.add_argument("--input",  "-i", required=True, help="360° video file")
    p.add_argument("--output", "-o", required=True, help="Output folder")
    p.add_argument("--interval",        type=float, default=1.0,
                   help="Interval between frames in seconds (default: 1.0)")
    p.add_argument("--format",          default="jpg",
                   help="Output image format (default: jpg)")
    p.add_argument("--resolution",      type=int,   default=2048,
                   help="Resolution of extracted images (default: 2048)")
    p.add_argument("--camera_count",    type=int,   default=6,
                   help="Number of virtual cameras (default: 6)")
    p.add_argument("--quality",         type=int,   default=95,
                   help="JPEG quality 0-100 (default: 95)")
    p.add_argument("--layout",          default="equirectangular",
                   help="Projection layout (default: equirectangular)")
    p.add_argument("--ai_mask",         action="store_true", help="Enable AI masking")
    p.add_argument("--drop_operator",   action="store_true",
                   help="Drop each extracted face where the operator is detected (uses AI masking) instead of masking it")
    p.add_argument("--operator_drop_threshold", type=float, default=0.005,
                   help="Min fraction of a face the operator must cover to drop it (default: 0.005)")
    p.add_argument("--ai_skip",         action="store_true", help="Enable AI frame skipping")
    p.add_argument("--adaptive",        action="store_true", help="Motion-adaptive extraction")
    p.add_argument("--motion_threshold", type=float, default=0.3,
                   help="Motion threshold for adaptive extraction (default: 0.3)")

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Run functions
# ─────────────────────────────────────────────────────────────────────────────

def run_colmap(args):
    params = ColmapParams(
        camera_model=args.camera_model,
        single_camera=not args.no_single_camera,
        max_image_size=args.max_image_size,
        max_num_features=args.max_num_features,
        estimate_affine_shape=args.estimate_affine_shape,
        domain_size_pooling=not args.no_domain_size_pooling,
        max_ratio=args.max_ratio,
        max_distance=args.max_distance,
        cross_check=not args.no_cross_check,
        min_model_size=args.min_model_size,
        multiple_models=args.multiple_models,
        ba_refine_focal_length=not args.no_refine_focal,
        ba_refine_principal_point=args.refine_principal,
        ba_refine_extra_params=not args.no_refine_extra,
        min_num_matches=args.min_num_matches,
        matcher_type=args.matcher_type,
        match_gpu_streams=args.match_gpu_streams,
        undistort_images=args.undistort,
        use_glomap=args.use_glomap,
    )

    print(tr("cli_start_colmap"))
    print(tr("cli_input", args.input))
    print(tr("cli_output", args.output))

    engine = ColmapEngine(
        params, args.input, args.output, args.type, args.fps,
        project_name=args.project_name,
        logger_callback=print,
        progress_callback=lambda x: print(tr("cli_progression", x)),
    )

    success, msg = engine.run()
    if success:
        print(tr("cli_success", msg))
    else:
        print(tr("cli_error", msg))
        sys.exit(1)


def run_brush(args):
    params = dict(BRUSH_DEFAULTS)

    if args.preset != "default":
        params.update(BRUSH_PRESETS[args.preset])

    # Explicit args override preset (only when provided by user)
    if args.iterations is not None:           params["total_steps"] = args.iterations
    if args.sh_degree is not None:            params["sh_degree"] = args.sh_degree
    if args.start_iter is not None:           params["start_iter"] = args.start_iter
    if args.refine_every is not None:         params["refine_every"] = args.refine_every
    if args.growth_grad_threshold is not None: params["growth_grad_threshold"] = args.growth_grad_threshold
    if args.growth_select_fraction is not None: params["growth_select_fraction"] = args.growth_select_fraction
    if args.growth_stop_iter is not None:     params["growth_stop_iter"] = args.growth_stop_iter
    if args.max_splats is not None:           params["max_splats"] = args.max_splats
    if args.checkpoint_interval is not None:  params["checkpoint_interval"] = args.checkpoint_interval
    if args.max_resolution is not None:       params["max_resolution"] = args.max_resolution

    params["device"] = args.device
    params["refine_mode"] = args.refine_mode
    params["with_viewer"] = args.with_viewer
    if args.custom_args: params["custom_args"] = args.custom_args
    if args.ply_name:    params["ply_name"] = args.ply_name

    print(tr("cli_start_brush"))
    print(tr("cli_input", args.input))
    print(tr("cli_output", args.output))
    print(f"  Preset     : {args.preset}")
    print(f"  Steps      : {params['total_steps']}")
    print(f"  SH degree  : {params['sh_degree']}")
    print(f"  Device     : {params['device']}")

    engine = BrushEngine(logger_callback=print)

    try:
        returncode = engine.train(args.input, args.output, params=params)
        if returncode == 0:
            print(tr("msg_success"))
        else:
            print(tr("msg_error"))
            sys.exit(1)
    except KeyboardInterrupt:
        print(tr("cli_stopping"))
        engine.stop()


def run_sharp(args):
    engine = SharpEngine(logger_callback=print)

    params = {
        "checkpoint": args.checkpoint,
        "device": args.device,
        "verbose": args.verbose,
    }

    if args.mode == "image":
        print(tr("cli_start_sharp"))
        print(tr("cli_input", args.input))
        print(tr("cli_output", args.output))

        try:
            returncode = engine.predict(args.input, args.output, params=params)
            if returncode == 0:
                print(tr("msg_success"))
            else:
                print(tr("msg_error"))
                sys.exit(1)
        except KeyboardInterrupt:
            print(tr("cli_stopping"))
            engine.stop()

    else:  # video mode
        _run_sharp_video(args, engine, params)


def _run_sharp_video(args, engine, params):
    import tempfile

    video_path = _Path(args.input)
    output_dir = _Path(args.output)
    skip = max(1, args.skip_frames)

    print(f"Sharp video: {video_path.name} (1 frame / {skip})")
    print(tr("cli_output", args.output))

    frames_dir = output_dir / "temp_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Extraire les frames
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg_bin, "-y", "-i", str(video_path),
        "-vf", f"select=not(mod(n\\,{skip}))",
        "-vsync", "vfr", "-q:v", "1",
        str(frames_dir / "frame_%04d.png"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}")
        sys.exit(1)

    frames = sorted(frames_dir.glob("*.png"))
    total = len(frames)
    if total == 0:
        print("No frames extracted.")
        sys.exit(1)

    print(f"{total} frames extracted.")
    success_count = 0

    try:
        for idx, frame_path in enumerate(frames, 1):
            print(f"  Frame {idx}/{total}: {frame_path.name}")
            frame_out = output_dir / frame_path.stem
            returncode = engine.predict(str(frame_path), str(frame_out), params)
            if returncode == 0:
                ply_files = list(frame_out.rglob("*.ply"))
                if ply_files:
                    shutil.copy2(ply_files[0], output_dir / f"{frame_path.stem}.ply")
                    success_count += 1
            if frame_out.exists():
                shutil.rmtree(frame_out)
    except KeyboardInterrupt:
        print(tr("cli_stopping"))
        engine.stop()
    finally:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)

    print(f"Complete: {success_count}/{total} frames converted.")
    if success_count == 0:
        sys.exit(1)


def run_supersplat(args):
    engine = SuperSplatEngine()

    import os
    if os.path.isfile(args.input):
        data_dir = os.path.dirname(args.input)
        filename = os.path.basename(args.input)
    else:
        data_dir = args.input
        filename = ""

    ok, msg = engine.start_data_server(data_dir, port=args.data_port)
    if not ok:
        print(f"{tr('msg_error')}: {msg}")
        sys.exit(1)
    print(msg)

    ok, msg = engine.start_supersplat(port=args.port)
    if not ok:
        print(f"{tr('msg_error')}: {msg}")
        engine.stop_all()
        sys.exit(1)
    print(msg)

    # Build URL with optional params
    url = f"http://localhost:{args.port}"
    url_params = []
    if filename:
        data_url = f"http://localhost:{args.data_port}/{filename}"
        url_params.append(f"load={data_url}")
    if args.no_ui:
        url_params.append("noui")
    if args.cam_pos:
        url_params.append(f"cameraPosition={args.cam_pos.strip()}")
    if args.cam_rot:
        url_params.append(f"cameraRotation={args.cam_rot.strip()}")
    if url_params:
        url += "?" + "&".join(url_params)

    print(f"\nOpen: {url}\n")
    print("Press Ctrl+C to stop the servers.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(tr("cli_server_stop"))
        engine.stop_all()


def run_upscale(args):
    from app.core.upscale_engine import UpscaleEngine

    engine = UpscaleEngine(logger_callback=print)

    if not engine.is_installed():
        print("Error: upscayl-bin not found. Install it from the Upscale tab in the graphical interface.")
        sys.exit(1)

    upsampler = engine.load_model(
        model_id=args.model,
        scale=args.scale,
        output_format=args.format,
        tile=args.tile,
        tta=args.tta,
        compression=args.compression,
    )
    if not upsampler:
        print("Error: could not load model.")
        sys.exit(1)

    import os
    input_path = _Path(args.input)
    output_path = _Path(args.output)

    print(f"Upscale x{args.scale} — model: {args.model}")
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")

    try:
        if input_path.is_dir():
            success, msg = engine.upscale_folder(
                str(input_path), str(output_path),
                cancel_check=None, **upsampler,
            )
        else:
            success = engine.upscale_image(str(input_path), str(output_path / input_path.name), upsampler)
            msg = "Upscale complete." if success else "Upscale failed."
    except KeyboardInterrupt:
        print(tr("cli_stopping"))
        sys.exit(0)

    print(f"{'Success' if success else 'Error'}: {msg}")
    if not success:
        sys.exit(1)


def run_4dgs(args):
    from app.core.four_dgs_engine import FourDGSEngine

    engine = FourDGSEngine(logger_callback=print)

    if not args.colmap_only and not _Path(args.input).exists():
        print(f"Error: source folder not found: {args.input}")
        sys.exit(1)

    print("Preparing 4DGS dataset")
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")

    try:
        if args.colmap_only:
            print("COLMAP-only mode.")
            success = engine.run_colmap(args.output)
        else:
            print(f"  FPS    : {args.fps}")
            success = engine.process_dataset(args.input, args.output, fps=args.fps)
    except KeyboardInterrupt:
        print(tr("cli_stopping"))
        engine.stop()
        sys.exit(0)

    print("Complete." if success else "Error during processing.")
    if not success:
        sys.exit(1)


def run_extract360(args):
    from app.core.extractor_360_engine import Extractor360Engine

    engine = Extractor360Engine(logger_callback=print)

    if not engine.is_installed():
        print("Error: 360° extractor not installed. Enable it from the 360° tab in the graphical interface.")
        sys.exit(1)

    params = {
        "interval":         args.interval,
        "format":           args.format,
        "resolution":       args.resolution,
        "camera_count":     args.camera_count,
        "quality":          args.quality,
        "layout":           args.layout,
        "ai_mask":          args.ai_mask,
        "drop_operator":    args.drop_operator,
        "operator_drop_threshold": args.operator_drop_threshold,
        "ai_skip":          args.ai_skip,
        "adaptive":         args.adaptive,
        "motion_threshold": args.motion_threshold,
    }

    print("360° video extraction")
    print(f"  Input       : {args.input}")
    print(f"  Output      : {args.output}")
    print(f"  Interval    : {args.interval}s")
    print(f"  Resolution  : {args.resolution}px")
    print(f"  Cameras     : {args.camera_count}")

    try:
        success = engine.run_extraction(
            args.input, args.output, params,
            log_callback=print,
            progress_callback=lambda x: print(f"  Progress: {x}%"),
        )
    except KeyboardInterrupt:
        print(tr("cli_stopping"))
        engine.stop()
        sys.exit(0)

    print("Complete." if success else "Error during extraction.")
    if not success:
        sys.exit(1)


def run_pipeline(args):
    """Pipeline complet COLMAP → Brush."""

    _sep = lambda title: print(f"\n{'─' * 50}\n  {title}\n{'─' * 50}")

    # ── Étape 1 : COLMAP ──────────────────────────────────────────────────────
    _sep("Step 1/2 — COLMAP reconstruction")
    print(f"  Input       : {args.input}")
    print(f"  Output      : {args.output}")
    print(f"  Project     : {args.project_name}")
    print(f"  Type        : {args.type}")
    if args.type == "video":
        print(f"  FPS         : {args.fps}")

    colmap_params = ColmapParams(
        camera_model=args.camera_model,
        matcher_type=args.matcher_type,
        match_gpu_streams=args.match_gpu_streams,
        max_image_size=args.max_image_size,
        undistort_images=args.undistort,
        use_glomap=args.use_glomap,
    )

    colmap_engine = ColmapEngine(
        colmap_params, args.input, args.output, args.type, args.fps,
        project_name=args.project_name,
        logger_callback=print,
        progress_callback=lambda x: print(f"  Progress: {x}%"),
    )

    try:
        success, msg = colmap_engine.run()
    except KeyboardInterrupt:
        print(tr("cli_stopping"))
        colmap_engine.stop()
        sys.exit(0)

    if not success:
        print(f"\nCOLMAP error: {msg}")
        sys.exit(1)

    dataset_path = _Path(args.output) / args.project_name
    print(f"\nDataset ready: {dataset_path}")

    # ── Étape 2 : Brush ───────────────────────────────────────────────────────
    _sep("Step 2/2 — Brush training")

    brush_params = dict(BRUSH_DEFAULTS)

    if args.preset != "default":
        brush_params.update(BRUSH_PRESETS[args.preset])

    if args.iterations is not None: brush_params["total_steps"] = args.iterations
    if args.sh_degree is not None:  brush_params["sh_degree"] = args.sh_degree
    brush_params["device"] = args.device
    brush_params["with_viewer"] = args.with_viewer
    if args.ply_name: brush_params["ply_name"] = args.ply_name

    print(f"  Dataset     : {dataset_path}")
    print(f"  Preset      : {args.preset}")
    print(f"  Steps       : {brush_params['total_steps']}")
    print(f"  SH degree   : {brush_params['sh_degree']}")
    print(f"  Device      : {brush_params['device']}")

    brush_engine = BrushEngine(logger_callback=print)

    try:
        returncode = brush_engine.train(str(dataset_path), str(dataset_path), params=brush_params)
    except KeyboardInterrupt:
        print(tr("cli_stopping"))
        brush_engine.stop()
        sys.exit(0)

    if returncode == 0:
        print(f"\nPipeline complete. Splat available in: {dataset_path}")
    else:
        print(f"\nBrush returned an error (code {returncode}).")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

DISPATCH = {
    "pipeline":    run_pipeline,
    "colmap":      run_colmap,
    "brush":       run_brush,
    "sharp":       run_sharp,
    "view":        run_supersplat,
    "upscale":     run_upscale,
    "4dgs":        run_4dgs,
    "extract360":  run_extract360,
}


def main():
    parser = get_parser()
    args = parser.parse_args()

    # No subcommand + no --gui → GUI par défaut
    if not args.command and not args.gui:
        _launch_gui()
        return

    if args.gui:
        _launch_gui()
        return

    missing_deps = check_dependencies()
    if missing_deps:
        print(f"Warning: missing dependencies: {', '.join(missing_deps)}")

    handler = DISPATCH.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
