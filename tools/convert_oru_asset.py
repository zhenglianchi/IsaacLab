"""Rebuild the ORU collision asset used by Isaac-Oru-Direct-v0.

Two things this script does that the plain `scripts/tools/convert_mesh.py` does not:

1. `make_instanceable=False`. With make_instanceable=True the mesh ends up behind
   a USD instance prototype, and IsaacLab's spawn-time `collision_props`
   (schemas.modify_collision_properties -> apply_nested) cannot author onto an
   instance proxy - the override is silently dropped. That is why the ORU's
   contact_offset/rest_offset used to have no effect at all (see
   ORU_EXPERIMENT_PLAN.md 3.3). Non-instanceable keeps those knobs live, which the
   task now depends on: ORU_CFG sets rest_offset=-0.0005 (a 0.5 mm contact
   penetration allowance) and without it the pin stops ~6 mm short of the seat.

2. It (re)creates the empty `/ORU/base_link` anchor. oru_env._create_fixed_joints
   anchors the Bridge->SixForce->Gripper->ORU chain on "<ns>/ORU/base_link"; the
   MeshConverter output does not contain that prim, so the FixedJoint fails with
   "body relationship ... points to a non existent prim" and the ORU is never
   attached to the gripper.

Usage (Isaac Sim env, from the repo root):
    python tools/convert_oru_asset.py                    # -> assets/USD/oru
    python tools/convert_oru_asset.py --out assets/USD/oru_try
"""
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mesh", default="assets/ORU.STL", help="source mesh")
parser.add_argument("--out", default="assets/USD/oru", help="output asset directory")
parser.add_argument("--mass", type=float, default=2.0, help="rigid body mass (kg)")
parser.add_argument(
    "--collision-approximation",
    default="sdf",
    choices=["convexDecomposition", "convexHull", "triangleMesh", "meshSimplification",
             "sdf", "boundingCube", "boundingSphere"],
    help="collision approximation; 'sdf' matches the shipped asset",
)
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]
os.chdir(root)

from isaaclab.app import AppLauncher

launcher = AppLauncher(headless=True)

import numpy as np  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402,F401
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg  # noqa: E402
from isaaclab.sim.schemas import schemas_cfg  # noqa: E402

_APPROX = {
    "convexDecomposition": schemas_cfg.ConvexDecompositionPropertiesCfg,
    "convexHull": schemas_cfg.ConvexHullPropertiesCfg,
    "triangleMesh": schemas_cfg.TriangleMeshPropertiesCfg,
    "meshSimplification": schemas_cfg.TriangleMeshSimplificationPropertiesCfg,
    "sdf": schemas_cfg.SDFMeshPropertiesCfg,
    "boundingCube": schemas_cfg.BoundingCubePropertiesCfg,
    "boundingSphere": schemas_cfg.BoundingSpherePropertiesCfg,
}

out_dir = (root / args.out).resolve()
usd_path = out_dir / "ORU.usd"

cfg = MeshConverterCfg(
    asset_path=str((root / args.mesh).resolve()),
    force_usd_conversion=True,
    usd_dir=str(out_dir),
    usd_file_name="ORU.usd",
    make_instanceable=False,
    mass_props=schemas_cfg.MassPropertiesCfg(mass=args.mass),
    rigid_props=schemas_cfg.RigidBodyPropertiesCfg(),
    collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True),
    mesh_collision_props=_APPROX[args.collision_approximation](),
)
converter = MeshConverter(cfg)
print(f"[convert] {converter.usd_path}", flush=True)

# --- anchor for the FixedJoint chain -----------------------------------------
stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
anchor = "/ORU/base_link"
if stage.GetPrimAtPath(anchor).IsValid():
    print(f"[anchor ] {anchor} already present", flush=True)
else:
    UsdGeom.Xform.Define(stage, anchor)
    stage.GetRootLayer().Save()
    print(f"[anchor ] created {anchor}", flush=True)

stage = Usd.Stage.Open(str(usd_path))
n_mesh = sum(1 for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
print(f"[verify ] meshes={n_mesh}  base_link={stage.GetPrimAtPath(anchor).IsValid()}  "
      f"root={list(stage.GetPrimAtPath('/ORU').GetAppliedSchemas())}", flush=True)

launcher.app.close(wait_for_replicator=False)
