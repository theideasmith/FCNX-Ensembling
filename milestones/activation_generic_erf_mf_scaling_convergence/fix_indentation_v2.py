import sys

def fix_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    full_content = "".join(lines)
    
    # Block 3 fix (around 1170)
    old_block3 = """    legend_emp_color = 'gray' if multi_d_mode else color_empirical
    legend_theo_color = 'gray' if multi_d_mode else color_theory
    legend_nngp_color = 'gray' if multi_d_mode else color_nngp
        plt.plot(unique_alpha, mean_theo_w, '--', color=legend_theo_color, linewidth=3, marker=theo_marker, markersize=8, alpha=0.8)
        plt.plot(unique_alpha, mean_nngp_w, ':', color=legend_nngp_color, linewidth=3, marker=theo_marker, markersize=6, alpha=0.7)"""

    new_block3 = """        legend_emp_color = 'gray' if multi_d_mode else color_empirical
        legend_theo_color = 'gray' if multi_d_mode else color_theory
        legend_nngp_color = 'gray' if multi_d_mode else color_nngp
        plt.plot(unique_alpha, mean_theo_w, '--', color=legend_theo_color, linewidth=3, marker=theo_marker, markersize=8, alpha=0.8)
        plt.plot(unique_alpha, mean_nngp_w, ':', color=legend_nngp_color, linewidth=3, marker=theo_marker, markersize=6, alpha=0.7)"""
    
    full_content = full_content.replace(old_block3, new_block3)

    # Let's also check if there are other similar blocks.
    # Looking for: 4 spaces legend_{emp,theo,nngp}_color then 8 spaces plt.plot
    # Actually, let's look for any 4 space legend assignment followed by 8 space unique_ or plt.plot
    # But specifically in the loops.
    
    with open(filename, 'w') as f:
        f.write(full_content)

if __name__ == "__main__":
    fix_file(sys.argv[1])
