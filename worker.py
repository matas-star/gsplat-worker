#!/usr/bin/env python3
"""
RunPod Serverless Worker — Video → 3D Gaussian Splatting

Priima video URL, grazina .ply / .splat URL.
Pipeline: ffmpeg frames → COLMAP SfM → Speedy-Splat → .ply
"""

import os
import sys
import json
import time
import uuid
import shutil
import subprocess
import tempfile
from pathlib import Path

import runpod
import requests

# ── Konfigūracija ─────────────────────────────────────────────
OUTPUT_BASE = Path("/output")
OUTPUT_BASE.mkdir(exist_ok=True)

# Upload į viešą hostingą (tmpfiles.org)
def upload_to_public(filepath: str) -> str | None:
    """Įkelia failą į tmpfiles.org, grąžina public download URL."""
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


def run(cmd: list[str], **kwargs):
    """Paleidžia komandą su log'inimu."""
    print(f"[CMD] {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def process_video(job: dict) -> dict:
    """Pagrindinė funkcija — apdoroja video į 3D Gaussian Splat."""
    job_input = job.get('input', {})
    video_url = job_input.get('video_url', '')
    sampling_rate = int(job_input.get('sampling_rate', 24))
    max_iterations = int(job_input.get('max_iterations', 15000))
    fps = min(sampling_rate, 30)

    if not video_url:
        return {'error': 'Nenurodytas video_url'}

    job_id = str(uuid.uuid4())[:8]
    workdir = Path(tempfile.mkdtemp(prefix=f'gsplat_{job_id}_'))
    frames_dir = workdir / 'frames'
    colmap_dir = workdir / 'colmap'
    output_dir = OUTPUT_BASE / job_id

    print(f"[{job_id}] Pradedama. Video: {video_url}")
    print(f"[{job_id}] fps={fps}, iterations={max_iterations}")
    print(f"[{job_id}] Workdir: {workdir}")

    try:
        # ── 1. Download video ──
        print(f"[{job_id}] (1/4) Atsisiunčiamas video...")
        video_path = workdir / 'input.mp4'
        run(['wget', '-q', '-O', str(video_path), video_url], timeout=600)
        size_mb = video_path.stat().st_size / 1024 / 1024
        print(f"[{job_id}] Video atsiųstas: {size_mb:.1f} MB")

        # ── 2. Extract frames ──
        print(f"[{job_id}] (2/4) Ištraukiami kadrai ({fps} fps)...")
        frames_dir.mkdir()
        run([
            'ffmpeg', '-i', str(video_path),
            '-vf', f'fps={fps},scale=1920:-1',
            '-q:v', '2',
            '-loglevel', 'error',
            str(frames_dir / 'frame_%04d.jpg'),
        ], timeout=300)

        frame_count = len(list(frames_dir.glob('*.jpg')))
        print(f"[{job_id}] Ištraukta {frame_count} kadrų")

        if frame_count < 5:
            return {'error': f'Per mažai kadrų: {frame_count} (reikia bent 5)'}

        # ── 3. COLMAP SfM ──
        print(f"[{job_id}] (3/4) COLMAP Structure-from-Motion...")
        colmap_dir.mkdir()
        database_path = colmap_dir / 'database.db'
        sparse_dir = colmap_dir / 'sparse'
        sparse_dir.mkdir()

        # Feature extraction
        print(f"[{job_id}]   COLMAP: feature extraction...")
        run([
            'colmap', 'feature_extractor',
            '--database_path', str(database_path),
            '--image_path', str(frames_dir),
            '--ImageReader.camera_model', 'SIMPLE_RADIAL',
            '--SiftExtraction.use_gpu', '1',
            '--SiftExtraction.max_image_size', '1920',
        ], timeout=600)

        # Feature matching
        print(f"[{job_id}]   COLMAP: feature matching...")
        run([
            'colmap', 'sequential_matcher',
            '--database_path', str(database_path),
            '--SiftMatching.use_gpu', '1',
        ], timeout=600)

        # Sparse reconstruction
        print(f"[{job_id}]   COLMAP: sparse reconstruction...")
        run([
            'colmap', 'mapper',
            '--database_path', str(database_path),
            '--image_path', str(frames_dir),
            '--output_path', str(sparse_dir),
        ], timeout=1200)

        # Export to text format (for gsplat)
        text_dir = colmap_dir / 'text'
        text_dir.mkdir()
        run([
            'colmap', 'model_converter',
            '--input_path', str(sparse_dir / '0'),
            '--output_path', str(text_dir),
            '--output_type', 'TXT',
        ], timeout=120)

        print(f"[{job_id}]   COLMAP baigtas")

        # ── 4. GSplat training ──
        print(f"[{job_id}] (4/4) 3D Gaussian Splatting ({max_iterations} iteracijų)...")
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

        # ── 5. Upload results ──
        ply_file = None
        for f in output_dir.rglob('*.ply'):
            if f.stat().st_size > 1000:
                ply_file = f
                break

        if not ply_file:
            return {'error': 'Nesukurtas .ply failas'}

        print(f"[{job_id}] PLY failas: {ply_file} ({ply_file.stat().st_size / 1024 / 1024:.1f} MB)")

        print(f"[{job_id}] Įkeliama į public hosting...")
        ply_url = upload_to_public(str(ply_file))

        if not ply_url:
            return {'error': 'Nepavyko įkelti rezultato'}

        print(f"[{job_id}] ✓ Baigta! Ply URL: {ply_url}")

        return {
            'status': 'completed',
            'ply_url': ply_url,
            'frame_count': frame_count,
            'iterations': max_iterations,
            'job_id': job_id,
        }

    except subprocess.CalledProcessError as e:
        print(f"[{job_id}] ✗ Klaida: {e}")
        return {'error': f'Subprocess klaida: {e.returncode} — {e.cmd[:5]}'}
    except Exception as e:
        print(f"[{job_id}] ✗ Klaida: {e}")
        return {'error': str(e)[:500]}
    finally:
        # Valymas
        try:
            shutil.rmtree(workdir)
        except Exception:
            pass


# ── RunPod entrypoint ─────────────────────────────────────────
if __name__ == '__main__':
    print("Gaussian Splatting Worker — pasiruošęs")
    runpod.serverless.start({
        'handler': process_video,
    })
