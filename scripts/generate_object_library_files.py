#!/usr/bin/env python3
"""
Script to generate object library files for MJCF models.
"""

import argparse
from pathlib import Path
import shutil
import yaml

import xml.etree.ElementTree as ET
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Generate object library files from object library path"
    )
    parser.add_argument(
        "--object-library-path",
        type=str,
        required=True,
        help="Path to the object library directory"
    )
    parser.add_argument(
        "--mesh-output-path",
        type=str,
        required=True,
        help="Path where mesh files will be output"
    )
    parser.add_argument(
        "--mjcf-output-path",
        type=str,
        required=True,
        help="Path where MJCF files will be output"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="google_",
    )
    parser.add_argument(
        "--max-volume",
        type=float,
        default=0.15 ** 3,
        help="Maximum bounding box volume for objects to be included (in cubic meters)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # TODO: Implement object library file generation logic
    print(f"Object library path: {args.object_library_path}")
    print(f"Mesh output path: {args.mesh_output_path}")
    print(f"MJCF output path: {args.mjcf_output_path}")
    object_library_path = Path(args.object_library_path)
    mesh_output_path = Path(args.mesh_output_path)
    mjcf_output_path = Path(args.mjcf_output_path)

    # Get list of objects
    library_dirs = [d for d in object_library_path.iterdir() if d.is_dir()]
    n_objects = len(library_dirs)
    n_included = 0
    for obj_dir in library_dirs:
        all_obj_files = list(obj_dir.glob("*.obj"))
        collision_meshes = [
            f for f in all_obj_files if "_decomp_" in f.name
        ]
        visual_meshes = [
            f for f in all_obj_files if "_decomp_" not in f.name
        ]
        texture_path = obj_dir / "texture.png"
        assert len(collision_meshes) > 0, f"No collision meshes found for {obj_dir}"
        assert len(visual_meshes) > 0, f"No visual mesh found for {obj_dir}"
        visual_mesh = visual_meshes[0]
        yaml_files = list(obj_dir.glob("*.yaml"))

        # Check if bounding box volume is above threshold (replace .obj with .yaml)
        # Remove file extension and add .yaml
        yaml_path = visual_mesh.with_suffix('.yaml')
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
            bbox = np.array(yaml_data["bbox"])
            volume = np.prod(bbox[1::2] - bbox[::2])
            if volume > args.max_volume:
                if args.verbose:
                    print(f"Skipping object {obj_dir} with volume {volume:.6f} m^3")
                continue
            else:
                n_included += 1

        # Get dir name
        obj_name = obj_dir.name
        # Strip prefix
        assert obj_name.startswith(args.prefix), f"Object name {obj_name} does not start with prefix {args.prefix}"
        obj_name = obj_name[len(args.prefix):]

        ############################
        # Copy mesh and texture files
        ############################

        # Create output directory for meshes
        mesh_dir = mesh_output_path / "library_objects" / obj_name
        mesh_dir.mkdir(parents=True, exist_ok=True)

        # Copy collision meshes
        for collision_mesh in collision_meshes:
            dst_path = mesh_dir / collision_mesh.name[len(args.prefix):]
            if not dst_path.exists():
                if args.verbose:
                    print(f"Copying collision mesh {collision_mesh} to {dst_path}")
                shutil.copy2(collision_mesh, dst_path)

        # Copy visual mesh
        visual_mesh_dst = mesh_dir / visual_mesh.name[len(args.prefix):]
        if not visual_mesh_dst.exists():
            if args.verbose:
                print(f"Copying visual mesh {visual_mesh} to {visual_mesh_dst}")
            shutil.copy2(visual_mesh, visual_mesh_dst)
        
        # Copy texture
        texture_dst = mesh_dir / "texture.png"
        if not texture_dst.exists():
            if args.verbose:
                print(f"Copying texture {texture_path} to {texture_dst}")
            shutil.copy2(texture_path, texture_dst)

        # Copy ymal files
        for yaml_file in yaml_files:
            dst_path = mesh_dir / yaml_file.name[len(args.prefix):]
            if not dst_path.exists():
                if args.verbose:
                    print(f"Copying yaml file {yaml_file} to {dst_path}")
                shutil.copy2(yaml_file, dst_path)

        ############################
        # Create MJCF file
        ############################

        # Transform paths into relative paths
        relative_mesh_dir = Path("library_objects") / obj_name
        collision_meshes_dst = [relative_mesh_dir / m.name[len(args.prefix):] for m in collision_meshes]
        visual_mesh_dst = relative_mesh_dir / visual_mesh.name[len(args.prefix):]
        texture_dst = relative_mesh_dir / "texture.png"

        # Create XML tree for dependency file (assets)
        root_dependencies = ET.Element("mujocoinclude")
        dependencies = ET.ElementTree(root_dependencies)
        assets = ET.SubElement(root_dependencies, "asset")
        # Texture
        ET.SubElement(
            assets,
            "texture",
            type="2d",
            name=f"{obj_name}_texture",
            file=str(texture_dst),
        )
        # Material
        ET.SubElement(
            assets,
            "material",
            name=f"{obj_name}_material",
            texture=f"{obj_name}_texture",
        )
        # Visual mesh
        ET.SubElement(
            assets,
            "mesh",
            name=f"{obj_name}",
            file=str(visual_mesh_dst),
        )
        # Collision meshes
        for j, collision_mesh in enumerate(collision_meshes_dst):
            ET.SubElement(
                assets,
                "mesh",
                name=f"{obj_name}_part_{j}",
                file=str(collision_mesh),
            )
        # Make directory for MJCF output if it doesn't exist
        library_object_dir = mjcf_output_path / "library_objects"
        library_object_dir.mkdir(parents=True, exist_ok=True)
        # Write MJCF dependency file
        mjcf_file_path = library_object_dir / f"{obj_name}_dependencies.xml"
        ET.indent(dependencies, space="  ")
        dependencies.write(mjcf_file_path, encoding="unicode")

        # Create XML tree for object MJCF file
        root_object = ET.Element("mujocoinclude")
        object_tree = ET.ElementTree(root_object)

        # Visual geom
        body = ET.SubElement(
            root_object,
            "geom",
            material=f"{obj_name}_material",
            mesh=f"{obj_name}",
            type="mesh",
            contype="0",
            conaffinity="0",
        )
        # Collision geoms
        for j in range(len(collision_meshes)):
            ET.SubElement(
                root_object,
                "geom",
                type="mesh",
                rgba="0.5 0.5 0.5 1",
                mesh=f"{obj_name}_part_{j}",
            )
        # Write MJCF object file
        object_mjcf_file_path = library_object_dir / f"{obj_name}.xml"
        ET.indent(object_tree, space="  ")
        object_tree.write(object_mjcf_file_path, encoding="unicode")

    print(f"Included {n_included} out of {n_objects} objects based on volume threshold.")


if __name__ == "__main__":
    main()
