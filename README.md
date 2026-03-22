# KITTI 3D LiDAR Live Mapping Viewer

This project visualizes KITTI Velodyne LiDAR scans in 3D and can build a progressively growing map (SLAM-like playback).

## Features

- Real-time playback of KITTI `.bin` LiDAR frames
- Pose-based global accumulation using KITTI OXTS data
- Two accumulation modes:
- `global`: map grows over time (SLAM-like)
- `sliding`: only last `N` frames are shown
- Mouse + keyboard camera controls

## Requirements

- Python 3.10+
- OpenGL-capable GPU/driver
- KITTI raw-style folder structure:
- `data/velodyne_points/data/*.bin`
- `data/oxts/data/*.txt`

## Quick Start

```bash
cd /home/schmijul/fun/kitti
source slam_env/bin/activate
python3 viewer.py --mode global --delay 0.05 --max_dist 80 --max_points 120000 --global_max_points 1800000
```

## Controls

- `Mouse drag`: rotate view
- `Mouse wheel`: zoom
- `Arrow keys`: rotate camera
- `W / S`: move camera closer/farther
- `+ / -`: point size
- `ESC`: exit

## Useful Parameters

- `--mode {global,sliding}`
- `--delay 0.03` to `0.1` for playback speed
- `--max_dist` to control visible radius (meters)
- `--max_points` max points per frame after downsampling
- `--global_max_points` cap for full accumulated map
- `--window` number of recent frames in `sliding` mode

## Troubleshooting

### I only see a few points

Try this:

```bash
python3 viewer.py --mode global --max_dist 100 --max_points 150000 --global_max_points 2200000 --delay 0.03
```

Then in the viewer:

- Zoom in with mouse wheel
- Increase point size with `+`
- Rotate camera with mouse drag

### Performance is low

- Reduce `--global_max_points` (for example `800000`)
- Reduce `--max_points` (for example `70000`)
- Increase `--delay` (for example `0.08`)

## Notes

- The mapping is pose-accumulation based (using OXTS poses), not a full loop-closing SLAM backend.
- Visual quality and FPS depend heavily on GPU/driver support for OpenGL.
