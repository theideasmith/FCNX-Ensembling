#!/bin/bash

# Requires yq (v4+) and bash

# Take PARAM_FILE as the first argument
PARAM_FILE="$1"

# Check if PARAM_FILE is provided
if [ -z "$PARAM_FILE" ]; then
  echo "Usage: $0 <path_to_params.yaml>"
  exit 1
fi

# Check if PARAM_FILE exists
if [ ! -f "$PARAM_FILE" ]; then
  echo "Error: Parameter file '$PARAM_FILE' not found."
  exit 1
fi

SCRIPT="net_einsum_parallel.py"

# Load YAML values
Ps=($(yq '.Ps[]' "$PARAM_FILE"))
Ns=($(yq '.Ns[]' "$PARAM_FILE"))
Ds=($(yq '.Ds[]' "$PARAM_FILE"))
ens=$(yq '.ens' "$PARAM_FILE")

chi_config=$(yq '.chi' "$PARAM_FILE")
nepochs_if_1=$(yq '.nepochs.if_chi_eq_1' "$PARAM_FILE")
nepochs_otherwise=$(yq '.nepochs.otherwise' "$PARAM_FILE")
ondata=$(yq '.ondata' "$PARAM_FILE")
save_prefix=$(yq '.save_prefix' "$PARAM_FILE")

echo "Ps: ${Ps[@]}"
echo "Ns: ${Ns[@]}"
echo "Ds: ${Ds[@]}"
echo "ens: $ens"
echo "chi setting: $chi_config"
echo "ondata: $ondata"
echo "save_prefix: $save_prefix"

# Timestamp and save path
timestamp=$(date +%Y%m%d_%H%M%S)
SAVEPATH="/home/akiva/gpnettrain/${save_prefix}_${timestamp}"
echo "Save to: $SAVEPATH"
echo "Starting hyperparameter sweep..."
echo "--------------------------------------------------"

for p in "${Ps[@]}"; do
  for n in "${Ns[@]}"; do
    for d in "${Ds[@]}"; do

      # Determine chi
      if [[ "$chi_config" == "use_N_as_chi" ]]; then
        chi="$n"
      else
        chi="$chi_config"
      fi

      # Determine nepochs
      if [[ "$chi" -eq 1 ]]; then
        nepochs="$nepochs_if_1"
      else
        nepochs="$nepochs_otherwise"
      fi

      echo "Running training for P=$p, N=$n, D=$d, ens=$ens"
      echo "$p, $n, $d, $chi, nepochs=$nepochs"

      # Build command
      cmd=(python "$SCRIPT"
        --P "$p"
        --N "$n"
        --D "$d"
        --chi "$chi"
        --epochs "$nepochs"
        --to "$SAVEPATH"
        --ens "$ens"
      )

      # Include --off_data only if ondata is false
      if [[ "$ondata" == "false" ]]; then
        cmd+=(--off_data)
      fi

      # Execute and capture output
      "${cmd[@]}" > out.log 2> err.log
      status=$?

      if [[ $status -eq 0 ]]; then
        echo "✅ Successfully completed P=$p, N=$n, D=$d"
        echo "--- Output from net.py ---"
        cat out.log
        if [ -s err.log ]; then
          echo "--- Errors/Warnings from net.py ---"
          cat err.log
        fi
      else
        echo "❌ Error occurred during training for P=$p, N=$n, D=$d"
        echo "--- Stdout ---"
        cat out.log
        echo "--- Stderr ---"
        cat err.log
      fi

      echo "--------------------------------------------------"
    done
  done
done

echo "Hyperparameter sweep finished."
