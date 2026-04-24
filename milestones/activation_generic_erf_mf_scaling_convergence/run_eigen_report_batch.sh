#!/bin/bash
# Script to run eigen_report_fcn3.py on all directories in P_scan_KAPPA0.1_FINALIZED
# except d150_P209_N1600_chi10_kappa0.1/

BASE_DIR="P_scan_KAPPA0.1_FINALIZED"
EXCLUDE_DIR="d150_P209_N1600_chi10_kappa0.1"

# Loop through all directories in P_scan_KAPPA0.1_FINALIZED
for dir in "$BASE_DIR"/*/ ; do
    # Get the directory name without the path
    dir_name=$(basename "$dir")
    
    # Skip the excluded directory
    if [ "$dir_name" = "$EXCLUDE_DIR" ]; then
        echo "Skipping $dir_name"
        continue
    fi
    
    # Run the python script with --force recompute
    echo "Processing $dir_name..."
    python eigen_report_fcn3.py --directory "$dir" --force recompute
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed $dir_name"
    else
        echo "✗ Error processing $dir_name"
    fi
    echo "---"
done

echo "Batch processing complete!"
