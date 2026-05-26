import sys

def fix_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 1. Detection of 'legend_emp_color = ...' assignments at module level that should be indented
        # and checking if subsequent lines are also under-indented.
        
        # Check if line is one of the mis-indented assignments (4 spaces instead of 8+)
        if (line.strip().startswith("legend_emp_color =") or 
            line.strip().startswith("legend_theo_color =") or 
            line.strip().startswith("legend_nngp_color =")) and line.startswith("    ") and not line.startswith("        "):
            
            # Look ahead: if the following lines are indented more (e.g. 8 spaces '        '), 
            # maybe these assignments should be too.
            # In the observed snippets, these assignments are at 4 spaces but appear inside a 'for' loop logic.
            
            # Specific pattern for the first block (around 1050)
            # The lines AFTER it like 'plt.plot' have 8 spaces.
            pass

        # Manual Fixes based on the snippet provided:
        
        # Block 1 (around 1050)
        if "legend_emp_color = \"gray\" if multi_d_mode else color_empirical" in line and i < 1100:
             # Check if we are in the first problematic block
             if i > 1040 and i < 1060:
                 # It seems there are duplicate definitions and mis-indented plt.plots
                 # We want to indent these to 8 spaces and remove duplicates if they are right next to each other
                 pass

        fixed_lines.append(line)
        i += 1

    # Alternative: use a more robust replacement strategy for the known broken segments.
    
    full_content = "".join(lines)
    
    # Segment 1 fix:
    # From:
    #     legend_emp_color = "gray" if multi_d_mode else color_empirical
    #     legend_theo_color = "gray" if multi_d_mode else color_theory
    #     legend_nngp_color = "gray" if multi_d_mode else color_nngp
    #     legend_emp_color = 'gray' if multi_d_mode else color_empirical
    #     legend_theo_color = 'gray' if multi_d_mode else color_theory
    #     legend_nngp_color = 'gray' if multi_d_mode else color_nngp
    #         plt.plot(unique_p, mean_theo_h3, ...
    
    # This is clearly garbled. Let's fix the specific blocks.
    
    # This logic is a bit complex for a regex. Let's just rewrite the problematic areas.
    
    new_content = full_content
    
    # Fix the first block (Mean-Field scaling vs P)
    old_block1 = """    legend_emp_color = "gray" if multi_d_mode else color_empirical
    legend_theo_color = "gray" if multi_d_mode else color_theory
    legend_nngp_color = "gray" if multi_d_mode else color_nngp
    legend_emp_color = 'gray' if multi_d_mode else color_empirical
    legend_theo_color = 'gray' if multi_d_mode else color_theory
    legend_nngp_color = 'gray' if multi_d_mode else color_nngp
        plt.plot(unique_p, mean_theo_h3, '--', color=legend_theo_color, linewidth=3,
             marker=theo_marker, markersize=8, alpha=0.8)
        plt.plot(unique_p, mean_nngp_h3, ':', color=legend_nngp_color, linewidth=3,
             marker=theo_marker, markersize=6, alpha=0.7)"""

    new_block1 = """        legend_emp_color = "gray" if multi_d_mode else color_empirical
        legend_theo_color = "gray" if multi_d_mode else color_theory
        legend_nngp_color = "gray" if multi_d_mode else color_nngp
        plt.plot(unique_p, mean_theo_h3, '--', color=legend_theo_color, linewidth=3,
                 marker=theo_marker, markersize=8, alpha=0.8)
        plt.plot(unique_p, mean_nngp_h3, ':', color=legend_nngp_color, linewidth=3,
                 marker=theo_marker, markersize=6, alpha=0.7)"""

    new_content = new_content.replace(old_block1, new_block1)

    # Fix the second block (Mean-Field scaling vs alpha)
    old_block2 = """    legend_emp_color = "gray" if multi_d_mode else color_empirical
    legend_theo_color = "gray" if multi_d_mode else color_theory
    legend_nngp_color = "gray" if multi_d_mode else color_nngp
        unique_alpha = sorted(alpha_to_emp_h3.keys())
        mean_emp_h3 = [np.mean(alpha_to_emp_h3[a]) for a in unique_alpha]
        mean_theo_h3 = [np.mean(alpha_to_theo_h3[a]) for a in unique_alpha]
        mean_nngp_h3 = [np.mean(alpha_to_nngp_h3[a]) for a in unique_alpha]
        draw_top_errorbar(plt, unique_alpha, mean_emp_h3, yerr=[float(np.std(alpha_to_emp_h3[a], ddof=1) / np.sqrt(len(alpha_to_emp_h3[a]))) if len(alpha_to_emp_h3[a]) > 1 else 0.0 for a in unique_alpha], color=legend_emp_color, marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
    legend_emp_color = 'gray' if multi_d_mode else color_empirical
    legend_theo_color = 'gray' if multi_d_mode else color_theory
    legend_nngp_color = 'gray' if multi_d_mode else color_nngp
        plt.plot(unique_alpha, mean_theo_h3, '--', color=legend_theo_color, linewidth=3,
             marker=theo_marker, markersize=8, alpha=0.8)
        plt.plot(unique_alpha, mean_nngp_h3, ':', color=legend_nngp_color, linewidth=3,
             marker=theo_marker, markersize=6, alpha=0.7)"""

    new_block2 = """        legend_emp_color = "gray" if multi_d_mode else color_empirical
        legend_theo_color = "gray" if multi_d_mode else color_theory
        legend_nngp_color = "gray" if multi_d_mode else color_nngp
        unique_alpha = sorted(alpha_to_emp_h3.keys())
        mean_emp_h3 = [np.mean(alpha_to_emp_h3[a]) for a in unique_alpha]
        mean_theo_h3 = [np.mean(alpha_to_theo_h3[a]) for a in unique_alpha]
        mean_nngp_h3 = [np.mean(alpha_to_nngp_h3[a]) for a in unique_alpha]
        draw_top_errorbar(plt, unique_alpha, mean_emp_h3, yerr=[float(np.std(alpha_to_emp_h3[a], ddof=1) / np.sqrt(len(alpha_to_emp_h3[a]))) if len(alpha_to_emp_h3[a]) > 1 else 0.0 for a in unique_alpha], color=legend_emp_color, marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
        plt.plot(unique_alpha, mean_theo_h3, '--', color=legend_theo_color, linewidth=3,
                 marker=theo_marker, markersize=8, alpha=0.8)
        plt.plot(unique_alpha, mean_nngp_h3, ':', color=legend_nngp_color, linewidth=3,
                 marker=theo_marker, markersize=6, alpha=0.7)"""
    
    new_content = new_content.replace(old_block2, new_block2)

    with open(filename, 'w') as f:
        f.write(new_content)

if __name__ == "__main__":
    fix_file(sys.argv[1])
