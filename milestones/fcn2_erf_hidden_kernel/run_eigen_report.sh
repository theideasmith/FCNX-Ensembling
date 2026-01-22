#!/bin/bash

model_dirs=(
"/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P3000_N800_chi_10.0_lr_1e-05_T_16.0_seed_0_eps_0.03"
"/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P3000_N800_chi_10.0_lr_1e-05_T_16.0_seed_2_eps_0.03"
"/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P3000_N800_chi_10.0_lr_1e-05_T_16.0_seed_1_eps_0.03"
"/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P3000_N800_chi_10.0_lr_1e-05_T_16.0_seed_3_eps_0.03"
)



# Save the list of model_dirs to a JSON file in the output directory
json_list="$outdir/model_dirs.json"
printf '{\n  "model_dirs": [\n' > "$json_list"
for i in "${!model_dirs[@]}"; do
    printf '    "%s"' "${model_dirs[$i]}" >> "$json_list"
    if [ "$i" -lt "$((${#model_dirs[@]}-1))" ]; then
        printf ',\n' >> "$json_list"
    else
        printf '\n' >> "$json_list"
    fi
done
printf '  ]\n}\n' >> "$json_list"

# Call the eigen_report.py script with the model_dirs and output directory
python3 ../../lib/eigen_report.py "${model_dirs[@]}"
