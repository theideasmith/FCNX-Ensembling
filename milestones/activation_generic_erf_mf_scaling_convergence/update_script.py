import sys

file_path = '/home/akiva/FCNX-Ensembling/milestones/activation_generic_erf_mf_scaling_convergence/analysis_script.py'

with open(file_path, 'r') as f:
    lines = f.readlines()

definitions = [
    "    legend_emp_color = 'gray' if multi_d_mode else color_empirical\n",
    "    legend_theo_color = 'gray' if multi_d_mode else color_theory\n",
    "    legend_nngp_color = 'gray' if multi_d_mode else color_nngp\n"
]

patterns = ['emp_color', 'theo_color', 'nngp_color']
new_lines = []
i = 0

while i < len(lines):
    line = lines[i]
    # Check if lines should be modified based on current position
    # The line indices here are 0-indexed, so 980 corresponds to line 981
    in_range = (980 <= i <= 1010) or (1040 <= i <= 1080) or (1090 <= i <= 1170) or (1180 <= i <= 1220)
    
    if in_range and any(p in line for p in patterns) and '.plot(' in line:
        # Check if already added recently
        already_added = False
        for j in range(max(0, len(new_lines)-15), len(new_lines)):
            if "legend_emp_color =" in new_lines[j]:
                already_added = True
                break
        
        if not already_added:
            new_lines.extend(definitions)
            
        # Replace occurrences in the current line
        line = line.replace('emp_color', 'legend_emp_color')
        line = line.replace('theo_color', 'legend_theo_color')
        line = line.replace('nngp_color', 'legend_nngp_color')
    
    new_lines.append(line)
    i += 1

with open(file_path, 'w') as f:
    f.writelines(new_lines)
