#!/bin/bash

# Requires yq (v4+) and bash

# Check for --parallel flag
PARALLEL=false
PARAM_FILE=""
for arg in "$@"; do
  if [[ "$arg" == "--parallel" ]]; then
    PARALLEL=true
    echo "Running in parallel mode"
  else
    PARAM_FILE="$arg"
  fi
  # Only use the first non-flag argument as PARAM_FILE
  if [[ -n "$PARAM_FILE" && "$PARAM_FILE" != "--parallel" ]]; then
    break
  fi
done

# Check if PARAM_FILE is provided
if [ -z "$PARAM_FILE" ]; then
  echo "Usage: $0 <path_to_params.yaml> [--parallel]"
  exit 1
fi

# Check if PARAM_FILE exists
if [ ! -f "$PARAM_FILE" ]; then
  echo "Error: Parameter file '$PARAM_FILE' not found."
  exit 1
fi

SCRIPT=$(yq '.script' "$PARAM_FILE")

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

PIDS=()

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
        --to "${SAVEPATH}_epochs_${nepochs}"
        --ens "$ens"
      )

      echo "Running command: ${cmd[@]}"

      # Include --off_data only if ondata is false
      if [[ "$ondata" == "false" ]]; then
        cmd+=(--off_data)
      fi

      if $PARALLEL; then
        # Run in background, redirect output
        ("${cmd[@]}" > out_${p}_${n}_${d}.log 2> err_${p}_${n}_${d}.log &)
        PIDS+=("$!")
      else
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
      fi
    done
  done
done

if $PARALLEL; then
  echo "Waiting for all parallel jobs to finish..."
  wait
  echo "All parallel jobs finished."
fi

echo "Hyperparameter sweep finished."
