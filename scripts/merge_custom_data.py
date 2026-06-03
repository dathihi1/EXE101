import argparse
import sys
from pathlib import Path

import numpy as np

# Set UTF-8 encoding for Vietnamese terminal output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

def main() -> None:
    workspace_dir = Path(__file__).parent.parent
    original_features_path = workspace_dir / "data/processed/features_vsl_mvp30_v2_sf8.npz"
    custom_dir = workspace_dir / "data/custom_recorded"
    output_features_path = workspace_dir / "data/processed/features_vsl_mvp30_v2_sf8_custom.npz"
    
    if not original_features_path.exists():
        print(f"Error: Original features file not found at {original_features_path}")
        return
        
    print(f"Loading original dataset: {original_features_path.name}")
    orig = np.load(original_features_path, allow_pickle=True)
    
    X = list(orig["X"])
    y = list(orig["y"])
    paths = list(orig["paths"])
    signers = list(orig["signers"])
    statuses = list(orig["statuses"])
    valid_frames = list(orig["valid_frames"])
    
    quality = list(orig["quality"])
    quality_keys = orig["quality_keys"].tolist()
    labels = orig["labels"].tolist()
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    
    # Locate all custom npz files
    if not custom_dir.exists():
        print(f"Directory {custom_dir} does not exist. No custom recordings to merge.")
        return
        
    custom_files = list(custom_dir.glob("**/*.npz"))
    if not custom_files:
        print("No custom recorded sample .npz files found to merge.")
        return
        
    print(f"Found {len(custom_files)} custom sample recordings. Starting merge...")
    
    merge_counts = {}
    for fp in custom_files:
        try:
            cdata = np.load(fp, allow_pickle=True)
            label = str(cdata["label"])
            
            if label not in label_to_id:
                print(f"[x] Warning: Label '{label}' from {fp.name} not in the 30 classes! Skipped.")
                continue
                
            label_id = label_to_id[label]
            
            # Align quality array with orig quality_keys
            c_quality_keys = cdata["quality_keys"].tolist()
            c_quality_vals = cdata["quality"].tolist()
            c_quality_map = dict(zip(c_quality_keys, c_quality_vals))
            
            aligned_quality = [c_quality_map.get(k, 0.0) for k in quality_keys]
            
            # Append features
            X.append(cdata["X"])
            y.append(label_id)
            paths.append(str(fp.relative_to(workspace_dir)))
            signers.append("custom")
            statuses.append("ok")
            valid_frames.append(int(cdata["valid_frames"][0]))
            quality.append(aligned_quality)
            
            merge_counts[label] = merge_counts.get(label, 0) + 1
        except Exception as e:
            print(f"[x] Error reading {fp.name}: {e}")
            
    print("\nMerge statistics:")
    for l, count in sorted(merge_counts.items()):
        print(f"  - '{l}': +{count} custom samples")
        
    if not merge_counts:
        print("No samples were merged.")
        return
        
    # Save the new merged npz
    print(f"\nSaving merged dataset to: {output_features_path}")
    
    schema_version = orig["schema_version"]
    schema_metadata = orig["schema_metadata"]
    
    np.savez_compressed(
        output_features_path,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        paths=np.asarray(paths),
        signers=np.asarray(signers),
        statuses=np.asarray(statuses),
        valid_frames=np.asarray(valid_frames, dtype=np.int32),
        labels=np.asarray(labels),
        feature_dim=orig["feature_dim"],
        sequence_length=orig["sequence_length"],
        schema_version=schema_version,
        quality_keys=np.asarray(quality_keys),
        quality=np.asarray(quality, dtype=np.float32),
        schema_metadata=schema_metadata,
    )
    
    print("=" * 60)
    print("SUCCESSFULLY MERGED!")
    print(f"New dataset has {len(X)} total samples (Original: {len(orig['X'])}, Added: {sum(merge_counts.values())}).")
    print("=" * 60)
    print("\nNext step: Retrain your model by running:")
    print("  python -m vsl_mvp.train --features data/processed/features_vsl_mvp30_v2_sf8_custom.npz --model lite_transformer --out-dir runs/vsl_mvp30_v2_lite_transformer")
    print("  python -m vsl_mvp.export_onnx --run-dir runs/vsl_mvp30_v2_lite_transformer")
    print("=" * 60)

if __name__ == "__main__":
    main()
