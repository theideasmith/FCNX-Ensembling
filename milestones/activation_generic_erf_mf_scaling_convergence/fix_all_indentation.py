import sys

def fix_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    fixed_lines = []
    i = 0
    in_loop = False
    while i < len(lines):
        line = lines[i]
        
        # Detect start of 'for i, (val, res) in enumerate(groups.items()):' or similar loops
        if "for i, (val, res) in enumerate(groups.items()):" in line or "for r in res:" in line or "for mode in" in line:
            in_loop = True # Simple tracking, not perfect but helps context

        # The pattern of the error is:
        # A block of code that SHOULD be in the loop (8+ spaces) is preceded by
        # assignments that are at 4 spaces.
        
        # If we see:
        #     legend_emp_color = ... (4 spaces)
        # FOLLOWED by:
        #         something (8 spaces)
        # It's almost certain the legend assignment should be at 8 spaces.
        
        if (i < len(lines) - 1 and 
            line.startswith("    ") and not line.startswith("        ") and
            (line.strip().startswith("legend_emp_color =") or 
             line.strip().startswith("legend_theo_color =") or 
             line.strip().startswith("legend_nngp_color =")) and
            lines[i+1].startswith("        ")):
            
            # Indent this line!
            fixed_lines.append("    " + line)
            i += 1
            continue

        # Also handle multiple such lines in a row
        if (i < len(lines) - 2 and 
            line.startswith("    ") and not line.startswith("        ") and
            line.strip().startswith("legend_") and
            lines[i+1].startswith("    ") and not lines[i+1].startswith("        ") and
            lines[i+1].strip().startswith("legend_") and
            lines[i+2].startswith("        ")):

                fixed_lines.append("    " + line)
                i += 1
                continue

        # Specific case: plt.plot(unique_alpha, mean_nngp_h, ...) around 1220
        # It is at 8 spaces but the loop ended or the previous lines were at 4.
        # Actually in the snippet:
        # 1219:     legend_nngp_color = ... (4 spaces)
        # 1220:         plt.plot(...) (8 spaces)
        # This confirms my logic above.
        
        fixed_lines.append(line)
        i += 1

    # Second pass for duplicates or misplaced 8-space lines
    final_lines = []
    for j, line in enumerate(fixed_lines):
        # If a line is 8-space but the block around it is 4-space and it's not in a loop...
        # But wait, my previous script already fixed some.
        final_lines.append(line)

    with open(filename, 'w') as f:
        f.writelines(final_lines)

if __name__ == "__main__":
    fix_file(sys.argv[1])
