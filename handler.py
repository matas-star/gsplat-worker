#!/usr/bin/env python3
"""
RunPod Serverless Handler — Video to 3D Gaussian Splatting
Registers worker FIRST, then bootstraps deps lazily on first job.
"""

import os, sys, uuid, shutil, subprocess, tempfile, time, threading
from pathlib import Path

# Import runpod immediately (installed by dockerStartCmd)
import runpod

print("[init] Worker starting", flush=True)

_bootstrap_done = False
_bootstrap_lock = threading.Lock()

def ensure(cmd, desc):
    """Run a command, log, return True on success."""
    print(f"[bootstrap] {desc}...", flush=True)
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        print(f"[bootstrap]   OK ({time.time()-t0:.0f}s)", flush=True)
        return True
    except Exception as e:
        print(f"[bootstrap]   WARN: {e}", flush=True)
        return False


def do_bootstrap():
    """Install all deps. Called once before first job."""
    global _bootstrap_done
    with _bootstrap_lock:
        if _bootstrap_done:
            return
        print("[bootstrap] Starting...", flush=True)

        for pkg in ["colmap", "ffmpeg"]:
            if shutil.which(pkg) is None:
                ensure(["apt-get", "update", "-qq"], "apt update")
                ensure(["apt-get", "install", "-y", "-qq", pkg], f"apt install {pkg}")

        for dep in ["jaxtyping", "gsplat", "nerfview", "viser", "opencv-python-headless", "plyfile"]:
            try:
                __import__(dep.replace("-", "_"))
            except ImportError:
                # Install without touching torch/cuda
                ensure([sys.executable, "-m", "pip", "install", "-q",
                        "--no-build-isolation", dep], f"pip install {dep}")

        if not Path("/app/gsplat/examples/simple_trainer.py").exists():
            ensure(["git", "clone", "--depth", "1",
                    "https://github.com/nerfstudio-project/gsplat.git",
                    "/app/gsplat"], "git clone gsplat")

        if not Path("/train_gsplat.py").exists():
            ensure(["curl", "-sSL",
                    "https://raw.githubusercontent.com/matas-star/gsplat-worker/master/train_gsplat.py",
                    "-o", "/train_gsplat.py"], "download train_gsplat.py")

        print("[bootstrap] Done", flush=True)
        _bootstrap_done = True


import requests

OUTPUT_BASE = Path("/output")
OUTPUT_BASE.mkdir(exist_ok=True)


def upload_to_public(filepath: str) -> str | None:
    """Upload file to catbox.moe, fallback to tmpfiles, transfer.sh."""
    # 1. catbox.moe
    try:
        with open(filepath, 'rb') as f:
            r = requests.post(
                'https://litterbox.catbox.moe/resources/internals/api.php',
                files={'fileToUpload': (Path(filepath).name, f, 'application/octet-stream')},
                data={'reqtype': 'fileupload', 'time': '72h'},
                timeout=600,
            )
        if r.status_code == 200:
            url = r.text.strip()
            if url.startswith('https://'):
                return url
    except Exception:
        pass

    # 2. tmpfiles.org
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
    print(f"[CMD] {' '.join(map(str, cmd))}", flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def handler(event):
    """RunPod handler — bootstrap on first call, then process."""
    try:
        return _handler_impl(event)
    except SystemExit:
        raise
    except BaseException as e:
        import traceback
        print(f"[FATAL] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return {'error': f'{type(e).__name__}: {e}'[:500]}


def _handler_impl(event):
    job_input = event.get('input', {})
    video_url = job_input.get('video_url', '')

    # Health check — respond immediately, no bootstrap
    if not video_url:
        return {'status': 'ok'}

    # Real job — bootstrap first
    try:
        do_bootstrap()
    except Exception as e:
        import traceback
        print(f"[bootstrap] FATAL: {traceback.format_exc()}", flush=True)
        return {'error': f'Bootstrap failed: {e}'[:500]}

    sampling_rate = int(job_input.get('sampling_rate', 24))
    max_iterations = int(job_input.get('max_iterations', 15000))
    fps = min(sampling_rate, 30)

    job_id = str(uuid.uuid4())[:8]
    workdir = Path(tempfile.mkdtemp(prefix=f'gsplat_{job_id}_'))
    frames_dir = workdir / 'frames'
    colmap_dir = workdir / 'colmap'
    output_dir = OUTPUT_BASE / job_id

    print(f"[{job_id}] Starting. fps={fps}, iters={max_iterations}", flush=True)

    try:
        # 1. Download video
        print(f"[{job_id}] (1/4) Downloading...", flush=True)
        video_path = workdir / 'input.mp4'
        run(['wget', '-q', '-O', str(video_path), video_url], timeout=600)

        # 2. Extract frames
        print(f"[{job_id}] (2/4) Extracting frames...", flush=True)
        frames_dir.mkdir()
        run([
            'ffmpeg', '-i', str(video_path),
            '-vf', f'fps={fps},scale=1920:-1',
            '-q:v', '2', '-loglevel', 'error',
            str(frames_dir / 'frame_%04d.jpg'),
        ], timeout=300)

        frame_count = len(list(frames_dir.glob('*.jpg')))
        if frame_count < 5:
            return {'error': f'Too few frames: {frame_count}'}

        # 3. COLMAP
        print(f"[{job_id}] (3/4) COLMAP...", flush=True)
        colmap_dir.mkdir()
        database_path = colmap_dir / 'database.db'
        sparse_dir = colmap_dir / 'sparse'
        sparse_dir.mkdir()

        run(['colmap', 'feature_extractor',
             '--database_path', str(database_path),
             '--image_path', str(frames_dir),
             '--ImageReader.camera_model', 'SIMPLE_RADIAL',
             '--SiftExtraction.use_gpu', '1'], timeout=600)

        run(['colmap', 'sequential_matcher',
             '--database_path', str(database_path),
             '--SiftMatching.use_gpu', '1'], timeout=600)

        run(['colmap', 'mapper',
             '--database_path', str(database_path),
             '--image_path', str(frames_dir),
             '--output_path', str(sparse_dir)], timeout=1200)

        text_dir = colmap_dir / 'text'
        text_dir.mkdir()
        run(['colmap', 'model_converter',
             '--input_path', str(sparse_dir / '0'),
             '--output_path', str(text_dir),
             '--output_type', 'TXT'], timeout=120)

        # 4. GSplat training
        print(f"[{job_id}] (4/4) GSplat training...", flush=True)
        output_dir.mkdir(parents=True)

        run([sys.executable, '/train_gsplat.py',
             '--source_path', str(frames_dir),
             '--model_path', str(output_dir),
             '--colmap_path', str(text_dir),
             '--iterations', str(max_iterations),
             '--save_ply', '1'], timeout=7200, cwd='/app')

        # 5. Upload
        ply_files = [f for f in output_dir.rglob('*.ply') if f.stat().st_size > 1000]
        if not ply_files:
            return {'error': 'No .ply created'}

        ply_url = upload_to_public(str(ply_files[0]))
        if not ply_url:
            return {'error': 'Upload failed'}

        return {
            'status': 'completed',
            'ply_url': ply_url,
            'frame_count': frame_count,
            'iterations': max_iterations,
        }

    except subprocess.CalledProcessError as e:
        return {'error': f'Process {e.returncode}: {e.cmd[:3]}'}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[{job_id}] ERROR: {tb}", flush=True)
        return {'error': f'{type(e).__name__}: {e} (see worker logs)'[:500]}
    finally:
        try:
            shutil.rmtree(workdir)
        except Exception:
            pass


if __name__ == '__main__':
    print("[init] Starting RunPod serverless...", flush=True)
    runpod.serverless.start({'handler': handler})
