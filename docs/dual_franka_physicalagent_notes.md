# Dual Franka PhysicalAgent Notes

## Lumos / SeerSense depth status

Current conclusion: the Lumos / XVisio hardware may expose depth or spatial
data through vendor tooling, but the current RLinf backend does not read it.

The repository implementation at
`rlinf/envs/realworld/common/camera/lumos_camera.py` uses OpenCV V4L2 to read a
YU12/I420 RGB stream. It explicitly rejects `camera_info.enable_depth=True`,
because this backend has no depth stream.

Do not infer from this code that the hardware lacks depth capability. Treat it
as a backend limitation:

- `RealSenseCamera` can provide RGBD through the existing `enable_depth` path.
- `LumosCamera` currently provides RGB only through V4L2.
- To use Lumos depth, add a dedicated vendor-SDK/ROS backend or extend
  `LumosCamera` after the real device is connected and the available streams
  are verified.

Recommended next step on hardware:

1. Inspect the Lumos device nodes and vendor SDK examples.
2. Confirm whether depth is exposed as a synchronized stream, a point cloud, or
   pose/map output.
3. Add a separate camera backend if the API is not V4L2-compatible.
4. Only then enable depth in `DualFrankaEnv`, ideally per camera slot so base
   RealSense and wrist Lumos can differ.

Current three-camera deployment assumption:

- `base_0_rgb` is RealSense. RLinf's `RealSenseCamera` already has an
  `enable_depth` path and can return RGBD if the camera/config enable it.
- `left_wrist_0_rgb` and `right_wrist_0_rgb` are Lumos/XVisio. The hardware may
  theoretically expose depth, but RLinf's current `LumosCamera` V4L2 backend is
  RGB-only and raises if `enable_depth=True`.
- The current dual-Franka VLA path uses RGB only:
  `main_images` plus two `extra_view_images`. Do not add depth tensors to the
  VLA request until the trained checkpoint and preprocessing are confirmed to
  expect them.

## Merge direction

This branch should focus on dual-arm Franka deployment. Preserve the dual-arm
TCP path as the main PhysicalAgent control target:

- Primary env: `DualFrankaTcpEnv-v1`
- Primary action layout:
  `[L_xyz, L_rot6d, L_grip, R_xyz, R_rot6d, R_grip]`
- Primary policy config: `pi05_dualfranka_tcp_rot6d`
- Keep the policy-facing `env.step(...)` path as absolute TCP-rot6d control
  for the trained dual-arm VLA.
- Use `DualFrankaTcpEnv.step_arm_delta(...)` and
  `DualFrankaTcpEnv.set_arm_gripper(...)` for PhysicalAgent rule-based
  primitives. These helpers select exactly one arm (`left` or `right`) and do
  not send TCP commands to the non-selected arm.
- Safety boxes should come from the existing dual-arm configs, especially
  `examples/embodiment/config/env/realworld_dual_franka_tcp_rot6d.yaml`. Do not
  replace them with the single-arm PhysicalAgent bounds or a left/right
  half-space split.

Single-arm PhysicalAgent support does not need to be preserved in this branch.
Useful single-arm changes should be extracted only when they serve the dual-arm
path, such as generic RGBD wrapping or camera metadata.

## Hardware config assumption before robot access

Until the dual Franka hardware is connected, assume the existing dual-arm RLinf
hardware configuration is valid and already matches the physical setup:

- Franka robot IPs
- RealSense serial number
- Lumos `/dev/v4l/by-id/...` device paths
- Robotiq `/dev/serial/by-id/...` adapter paths
- GELLO `/dev/serial/by-id/...` adapter paths
- Left/right controller node rank assignment
- Reset poses, TCP safety limits, and default target poses

Do not spend time re-deriving these values before hardware access. Treat all
basic smoke tests as passing for planning and software integration.

When the real setup is available, run only a minimal confirmation pass before
policy deployment:

1. Confirm `/dev/v4l/by-id/`, `/dev/serial/by-id/`, and foot-switch
   `/dev/input/eventXX` paths.
2. Confirm left/right arm mapping with low-risk Franky commands such as
   `getjoint`, `open`, and `close`.
3. Confirm left/right wrist camera semantics by viewing each stream.
4. Confirm GELLO left/right mapping before recording demonstrations.
5. Keep the current config unchanged if these mappings match.
