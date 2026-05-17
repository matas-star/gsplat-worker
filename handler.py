#!/usr/bin/env python3
"""
RunPod Serverless Handler — Video → 3D Gaussian Splatting
Pipeline: ffmpeg frames → COLMAP SfM → GSplat training → .ply
Self-bootstrapping: installs dependencies on first run.
"""

import os
import sys
import uuid
import shutil
import subprocess
import tempfile
from pathlib import Path

import requests


def bootstrap():
    """Install all dependencies. Called once at worker startup."""
    deps_ok = True
    
    # Check and install system deps
    for pkg, check_cmd in [("colmap", ["which", "colmap"]), ("ffmpeg", ["which", "ffmpeg"])]:
        try:
            subprocess.run(check_cmd, check=True, capture_output=True)
        except Exception:
            print(f"[bootstrap] Installing {pkg}...")
            try:
                subprocess.run(["apt-get", "update", "-qq"], check=True, timeout=300)
                subprocess.run(["apt-get", "install", "-y", "-qq", pkg, "wget", "curl"], check=True, timeout=300)
            except Exception as e:
                print(f"[bootstrap] WARNING: {pkg} install failed: {e}")
                deps_ok = False

    # Install Python deps
    py_deps = ["gsplat", "nerfview", "viser", "opencv-python-headless", "plyfile"]
    for dep in py_deps:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", dep], check=True, timeout=600)
        except Exception as e:
            print(f"[bootstrap] WARNING: {dep} install failed: {e}")
            deps_ok = False

    # Clone gsplat repo for SimpleTrainer
    if not Path("/app/gsplat").exists():
        try:
            subprocess.run(["git", "clone", "--depth", "1", 
                          "https://github.com/nerfstudio-project/gsplat.git",
                          "/app/gsplat"], check=True, timeout=600)
        except Exception as e:
            print(f"[bootstrap] WARNING: gsplat clone failed: {e}")

    # Download train script
    for fname in ["train_gsplat.py"]:
        if not Path(f"/{fname}").exists():
            try:
                subprocess.run(["curl", "-sSL", 
                              f"https://raw.githubusercontent.com/matas-star/gsplat-worker/master/{fname}",
                              "-o", f"/{fname}"], check=True, timeout=60)
            except Exception:
                pass

    print(f"[bootstrap] Done (all_ok={deps_ok})")
    return deps_ok


# Run bootstrap at import time
_bootstrap_ok = bootstrap()
import runpod

OUTPUT_BASE = Path("/output")
OUTPUT_BASE.mkdir(exist_ok=True)


def upload_to_public(filepath: str) -> str | None:
    """Upload file to tmpfiles.org, return public download URL."""
    try:
        with open(filepath, 'rb') as f:
            r = requests.post(
                'https://tmpfiles.org/api/v1/upload',
                files={'file': (Path(filepath).name, f, 'application/octet-stream')},
                timeout=600,
            )
        if r.status_code in (200, 201):
            data = r.json()
            if data.get('status') == 'success':
                url = data['data']['url']
                return url.replace('https://tmpfiles.org/', 'https://tmpfiles.org/dl/')
    except Exception:
        pass

    # Fallback: transfer.sh
    try:
        with open(filepath, 'rb') as f:
            r = requests.put(
                f'https://transfer.sh/{Path(filepath).name}',
                data=f,
                headers={'Max-Downloads': '10', 'Max-Days': '3'},
                timeout=300,
            )
        if r.status_code in (200, 201):
            return r.text.strip()
    except Exception:
        pass

    return None


def run(cmd, **kwargs):
    """Run a command with logging."""
    print(f"[CMD] {' '.join(map(str, cmd))}")
    return subprocess.run(cmd, check=True, **kwargs)


def handler(event):
    """RunPod handler — receives event, returns result."""
    job_input = event.get('input', {})
    video_url = job_input.get('video_url', '')
    sampling_rate = int(job_input.get('sampling_rate', 24))
    max_iterations = int(job_input.get('max_iterations', 15000))
    fps = min(sampling_rate, 30)

    if not video_url:
        return {'error': 'No video_url provided'}

    job_id = str(uuid.uuid4())[:8]
    workdir = Path(tempfile.mkdtemp(prefix=f'gsplat_{job_id}_'))
    frames_dir = workdir / 'frames'
    colmap_dir = workdir / 'colmap'
    output_dir = OUTPUT_BASE / job_id

    print(f"[{job_id}] Starting. Video: {video_url}")
    print(f"[{job_id}] fps={fps}, iterations={max_iterations}")

    try:
        # 1. Download video
        print(f"[{job_id}] (1/4) Downloading video...")
        video_path = workdir / 'input.mp4'
        run(['wget', '-q', '-O', str(video_path), video_url], timeout=600)
        size_mb = video_path.stat().st_size / 1024 / 1024
        print(f"[{job_id}] Video: {size_mb:.1f} MB")

        # 2. Extract frames
        print(f"[{job_id}] (2/4) Extracting frames ({fps} fps)...")
        frames_dir.mkdir()
        run([
            'ffmpeg', '-i', str(video_path),
            '-vf', f'fps={fps},scale=1920:-1',
            '-q:v', '2', '-loglevel', 'error',
            str(frames_dir / 'frame_%04d.jpg'),
        ], timeout=300)

        frame_count = len(list(frames_dir.glob('*.jpg')))
        print(f"[{job_id}] {frame_count} frames")

        if frame_count < 5:
            return {'error': f'Too few frames: {frame_count} (need at least 5)'}

        # 3. COLMAP SfM
        print(f"[{job_id}] (3/4) COLMAP Structure-from-Motion...")
        colmap_dir.mkdir()
        database_path = colmap_dir / 'database.db'
        sparse_dir = colmap_dir / 'sparse'
        sparse_dir.mkdir()

        print(f"[{job_id}]   Feature extraction...")
        run([
            'colmap', 'feature_extractor',
            '--database_path', str(database_path),
            '--image_path', str(frames_dir),
            '--ImageReader.camera_model', 'SIMPLE_RADIAL',
            '--SiftExtraction.use_gpu', '1',
            '--SiftExtraction.max_image_size', '1920',
        ], timeout=600)

        print(f"[{job_id}]   Feature matching...")
        run([
            'colmap', 'sequential_matcher',
            '--database_path', str(database_path),
            '--SiftMatching.use_gpu', '1',
        ], timeout=600)

        print(f"[{job_id}]   Sparse reconstruction...")
        run([
            'colmap', 'mapper',
            '--database_path', str(database_path),
            '--image_path', str(frames_dir),
            '--output_path', str(sparse_dir),
        ], timeout=1200)

        # Export text format for gsplat
        text_dir = colmap_dir / 'text'
        text_dir.mkdir()
        run([
            'colmap', 'model_converter',
            '--input_path', str(sparse_dir / '0'),
            '--output_path', str(text_dir),
            '--output_type', 'TXT',
        ], timeout=120)

        print(f"[{job_id}]   COLMAP done")

        # 4. GSplat training
        print(f"[{job_id}] (4/4) GSplat training ({max_iterations} iterations)...")
        output_dir.mkdir(parents=True)

        train_script = Path('/app/train_gsplat.py')
        run([
            sys.executable, str(train_script),
            '--source_path', str(frames_dir),
            '--model_path', str(output_dir),
            '--colmap_path', str(text_dir),
            '--iterations', str(max_iterations),
            '--save_ply', '1',
        ], timeout=7200, cwd='/app')

        # 5. Upload result
        ply_file = None
        for f in output_dir.rglob('*.ply'):
            if f.stat().st_size > 1000:
                ply_file = f
                break

        if not ply_file:
            return {'error': 'No .ply file created'}

        print(f"[{job_id}] PLY: {ply_file} ({ply_file.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"[{job_id}] Uploading to public hosting...")
        ply_url = upload_to_public(str(ply_file))

        if not ply_url:
            return {'error': 'Failed to upload result'}

        print(f"[{job_id}] Done! PLY URL: {ply_url}")

        return {
            'status': 'completed',
            'ply_url': ply_url,
            'frame_count': frame_count,
            'iterations': max_iterations,
        }

    except subprocess.CalledProcessError as e:
        print(f"[{job_id}] Subprocess error: {e}")
        return {'error': f'Process failed: {e.returncode}'}
    except Exception as e:
        print(f"[{job_id}] Error: {e}")
        return {'error': str(e)[:500]}
    finally:
        try:
            shutil.rmtree(workdir)
        except Exception:
            pass


# ── RunPod entrypoint ──
if __name__ == '__main__':
    print("GSplat Worker ready")
    runpod.serverless.start({'handler': handler})
