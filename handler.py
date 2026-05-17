#!/usr/bin/env python3
"""
RunPod Serverless Handler — Video → 3D Gaussian Splatting
Pipeline: ffmpeg frames → COLMAP SfM → GSplat training → .ply
All dependencies are installed by dockerStartCmd before this runs.
"""

import os
import sys
import uuid
import shutil
import subprocess
import tempfile
from pathlib import Path

import requests
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

    print(f"[{job_id}] Starting. fps={fps}, iterations={max_iterations}")

    try:
        # 1. Download video
        print(f"[{job_id}] (1/4) Downloading video...")
        video_path = workdir / 'input.mp4'
        run(['wget', '-q', '-O', str(video_path), video_url], timeout=600)

        # 2. Extract frames
        print(f"[{job_id}] (2/4) Extracting frames...")
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

        # 3. COLMAP SfM
        print(f"[{job_id}] (3/4) COLMAP...")
        colmap_dir.mkdir()
        database_path = colmap_dir / 'database.db'
        sparse_dir = colmap_dir / 'sparse'
        sparse_dir.mkdir()

        run([
            'colmap', 'feature_extractor',
            '--database_path', str(database_path),
            '--image_path', str(frames_dir),
            '--ImageReader.camera_model', 'SIMPLE_RADIAL',
            '--SiftExtraction.use_gpu', '1',
        ], timeout=600)

        run([
            'colmap', 'sequential_matcher',
            '--database_path', str(database_path),
            '--SiftMatching.use_gpu', '1',
        ], timeout=600)

        run([
            'colmap', 'mapper',
            '--database_path', str(database_path),
            '--image_path', str(frames_dir),
            '--output_path', str(sparse_dir),
        ], timeout=1200)

        text_dir = colmap_dir / 'text'
        text_dir.mkdir()
        run([
            'colmap', 'model_converter',
            '--input_path', str(sparse_dir / '0'),
            '--output_path', str(text_dir),
            '--output_type', 'TXT',
        ], timeout=120)

        # 4. GSplat training
        print(f"[{job_id}] (4/4) GSplat training...")
        output_dir.mkdir(parents=True)

        run([
            sys.executable, '/train_gsplat.py',
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

        ply_url = upload_to_public(str(ply_file))
        if not ply_url:
            return {'error': 'Failed to upload result'}

        print(f"[{job_id}] Done! {ply_url}")

        return {
            'status': 'completed',
            'ply_url': ply_url,
            'frame_count': frame_count,
            'iterations': max_iterations,
        }

    except subprocess.CalledProcessError as e:
        print(f"[{job_id}] Error: {e}")
        return {'error': f'Process failed: {e.returncode}'}
    except Exception as e:
        print(f"[{job_id}] Error: {e}")
        return {'error': str(e)[:500]}
    finally:
        try:
            shutil.rmtree(workdir)
        except Exception:
            pass


if __name__ == '__main__':
    print("GSplat Worker ready")
    runpod.serverless.start({'handler': handler})
