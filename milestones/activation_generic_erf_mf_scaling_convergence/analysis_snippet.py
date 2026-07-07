    kappa = final_data[0]["kappa"] if final_data else 0
    kappa_eff = final_data[0]["kappa_eff"] if final_data else 0
    N = final_data[0]["N"] if final_data else 0
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_d{d}_N{N}_kappa_bare{kappa:.2g}_kappa_eff{kappa_eff:.2g}.png", dpi=300)

    # Separate plots
    # Lambda H
    fig_h = plt.figure(figsize=(10, 6))
    for i, (val, res) in enumerate(groups.items()):
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        p_vals = [r["P"] for r in res]
        if color_by == "chi":
            label = r"$\chi = $" + str(val)
        elif color_by == "d":
            label = f"d={val}"
        else:
            label = f"{color_by}={val}"
        plt.scatter(p_vals, [r["emp_h"] for r in res], color=color_empirical, label=label if i == 0 else "", alpha=0.7, s=50, marker=exp_marker)
        # Mean lines
        p_to_emp_h = defaultdict(list)
        p_to_theo_h = defaultdict(list)
        p_to_nngp_h = defaultdict(list)
        for r in res:
            p_to_emp_h[r["P"]].append(r["emp_h"])
            p_to_theo_h[r["P"]].append(r["theo_h"])
            p_to_nngp_h[r["P"]].append(r.get("theo_h_nngp", np.nan))
        unique_p = sorted(p_to_emp_h.keys())
        mean_emp_h = [np.mean(p_to_emp_h[p]) for p in unique_p]
        mean_theo_h = [np.mean(p_to_theo_h[p]) for p in unique_p]
        mean_nngp_h = [np.mean(p_to_nngp_h[p]) for p in unique_p]
        # Use grouped color when multi_d_mode, otherwise use fixed colors
        emp_color = c if multi_d_mode else color_empirical
        theo_color = c if multi_d_mode else color_theory
        nngp_color = c if multi_d_mode else color_nngp
        draw_top_errorbar(plt, unique_p, mean_emp_h, yerr=[float(np.std(p_to_emp_h[p], ddof=1) / np.sqrt(len(p_to_emp_h[p]))) if len(p_to_emp_h[p]) > 1 else 0.0 for p in unique_p], color=emp_color, marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
        plt.plot(unique_p, mean_theo_h, '--', color=theo_color, linewidth=3, marker=theo_marker, markersize=8, alpha=0.8)
        plt.plot(unique_p, mean_nngp_h, ':', color=nngp_color, linewidth=3, marker=theo_marker, markersize=6, alpha=0.7)
    plt.plot([], [], '-', color=color_empirical, linewidth=3, marker=exp_marker, markersize=6, label="Experiment")
    plt.plot([], [], '--', color=color_theory, linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
    plt.plot([], [], ':', color=color_nngp, linewidth=3, marker=theo_marker, markersize=6, label="NNGP")
    plt.axvline(d, color='gray', linestyle='--', alpha=0.5, linewidth=2, label="P=d")
    plt.title(r"Linear Target Eigenvalues ($\lambda_H^*$)" + f" d={d}, $\\kappa_{{eff}}={kappa_eff:.2g}$, N={N}")
    plt.xlabel("P (dataset size)"); plt.legend(); plt.grid(True, alpha=0.3); plt.xscale('log')
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_H_d{d}_N{N}_kappa_bare{kappa:.2g}_kappa_eff{kappa_eff:.2g}.png", dpi=300)

    # Lambda W
    fig_w = plt.figure(figsize=(10, 10))
    for i, (val, res) in enumerate(groups.items()):
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        p_vals = [r["P"] for r in res]
        if color_by == "chi":
            label = r"$\chi = $" + str(val)
        elif color_by == "d":
            label = f"d={val}"
        else:
            label = f"{color_by}={val}"
        plt.scatter(p_vals, [r["emp_w0"] for r in res], color=color_empirical, label=label if i == 0 else "", alpha=0.7, s=50, marker=exp_marker)
        # Mean lines
        p_to_emp_w = defaultdict(list)
        p_to_theo_w = defaultdict(list)
        p_to_nngp_w = defaultdict(list)
        for r in res:
            p_to_emp_w[r["P"]].append(r["emp_w0"])
            p_to_theo_w[r["P"]].append(r["theo_w"])
            p_to_nngp_w[r["P"]].append(r.get("theo_w_nngp", np.nan))
        unique_p = sorted(p_to_emp_w.keys())
        mean_emp_w = [np.mean(p_to_emp_w[p]) for p in unique_p]
        mean_theo_w = [np.mean(p_to_theo_w[p]) for p in unique_p]
        mean_nngp_w = [np.mean([v for v in p_to_nngp_w[p] if np.isfinite(v)]) if any(np.isfinite(p_to_nngp_w[p])) else np.nan for p in unique_p]
        # Use grouped color when multi_d_mode, otherwise use fixed colors
        emp_color = c if multi_d_mode else color_empirical
        theo_color = c if multi_d_mode else color_theory
        nngp_color = c if multi_d_mode else color_nngp
        draw_top_errorbar(plt, unique_p, mean_emp_w, yerr=[float(np.std(p_to_emp_w[p], ddof=1) / np.sqrt(len(p_to_emp_w[p]))) if len(p_to_emp_w[p]) > 1 else 0.0 for p in unique_p], color=emp_color, marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
        plt.plot(unique_p, mean_theo_w, '--', color=theo_color, linewidth=3, marker=theo_marker, markersize=8, alpha=0.8)
        plt.plot(unique_p, mean_nngp_w, ':', color=nngp_color, linewidth=3, marker=theo_marker, markersize=6, alpha=0.7)
    plt.plot([], [], '-', color=color_empirical, linewidth=3, marker=exp_marker, markersize=6, label="Experiment")
    plt.plot([], [], '--', color=color_theory, linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
    plt.plot([], [], ':', color=color_nngp, linewidth=3, marker=theo_marker, markersize=6, label="NNGP")
    plt.axvline(d, color='gray', linestyle='--', alpha=0.5, linewidth=2, label="P=d")
    plt.title(r"Linear Target Eigenvalues ($\lambda_W^*$)" + f" d={d}, $\\kappa_{{eff}}={kappa_eff:.2g}$, N={N}")
    plt.ylabel(r"$\lambda_W^* = v^T \Sigma_w v$");
    plt.xlabel("P (dataset size)"); plt.legend(); plt.grid(True, alpha=0.3); plt.xscale('log')
    plt.ylim(0,None)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_W_d{d}_N{N}_kappa_bare{kappa:.2g}_kappa_eff{kappa_eff:.2g}.png", dpi=300)

    # Learnability
    for mode in ["h1", "h3"]:
        plt.figure(figsize=(14, 8))
        for i, (val, res) in enumerate(groups.items()):
            res = collapse_rows_to_seed_means(res)
            c = series_color(i)
            p_vals = [r["P"] for r in res]
            label = r"$\chi = $" + str(val) if color_by == "chi" else f"{color_by}={val}"
            plt.scatter(p_vals, [r[f"{mode}_emp"] for r in res], color=color_empirical, label=label if i == 0 else "", alpha=0.7, s=50, marker=exp_marker)
            d = res[0]["d"] if res else 0
            plt.axvline(d, color='gray', linestyle='--', alpha=0.5, linewidth=2, label="P=d" if i == 0 else None)

            # Mean lines
            p_to_emp = defaultdict(list)
            p_to_theo = defaultdict(list)
            p_to_nngp = defaultdict(list)
            for r in res:
                p_to_emp[r["P"]].append(r[f"{mode}_emp"])
                p_to_theo[r["P"]].append(r[f"{mode}_theory"])
                p_to_nngp[r["P"]].append(r.get(f"{mode}_nngp_theory", np.nan))
            unique_p = sorted(p_to_emp.keys())
            mean_emp = [np.mean(p_to_emp[p]) for p in unique_p]
            mean_theo = [np.mean(p_to_theo[p]) for p in unique_p]
            mean_nngp = [np.mean(p_to_nngp[p]) for p in unique_p]
            draw_top_errorbar(plt, unique_p, mean_emp, yerr=[float(np.std(p_to_emp[p], ddof=1) / np.sqrt(len(p_to_emp[p]))) if len(p_to_emp[p]) > 1 else 0.0 for p in unique_p], color=color_empirical, marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
            plt.plot(unique_p, mean_theo, '--', color=color_theory, linewidth=3, marker=theo_marker, markersize=8, alpha=0.8)
            plt.plot(unique_p, mean_nngp, ':', color=color_nngp, linewidth=3, marker=theo_marker, markersize=6, alpha=0.7)
        # Dummy plot for legend
        plt.plot([], [], '-', color=color_empirical, linewidth=3, marker=exp_marker, markersize=6, label="Experiment")
        plt.plot([], [], '--', color=color_theory, linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
        plt.plot([], [], ':', color=color_nngp, linewidth=3, marker=theo_marker, markersize=6, label="NNGP")
        d = res[0]["d"] if res else 0
        kappa = res[0]['kappa'] if res else 0
        kappa_eff = res[0]['kappa_eff'] if res else 0
        N = res[0]['N'] if res else 0

        title_mode = "Hermite-1" if mode == "h1" else "Hermite-3"
        plt.title(rf"{title_mode} Learnability $\eta_{{He1}}$" + f", $d={d}, N={N}, \\kappa_{{eff}}={kappa_eff:.2g}$"); plt.xlabel("P (dataset size)"); plt.legend(); plt.grid(True, alpha=0.3); plt.xscale('log')
        plt.ylabel(r"$\eta_{He1} = \frac{\langle f \mid He_1 \rangle}{y_{He1}}$ (Learnability)")
        plt.tight_layout(); plt.savefig(RESULTS_DIR / f"learnability_{mode}_d{d}_N{N}_kappa_bare{kappa:.2g}_kappa_eff{kappa_eff:.2g}.png", dpi=300)
        plt.ylim(0, None)
    # --- Additional plots with alpha on x-axis (linear scale) ---
    # Eigenvalues with alpha
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    for i, (val, res) in enumerate(groups.items()):
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        if color_by == "chi":
            label = r"$\chi = $" + str(val)
        elif color_by == "d":
            label = f"d={val}"
        else:
            label = f"{color_by}={val}"
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        alpha_vals = [np.log(r["P"]) / np.log(r["d"]) for r in res]
        label = r"$\chi = $" + str(val) if color_by == "chi" else f"{color_by}={val}"
        ax1.scatter(alpha_vals, [r["emp_h"] for r in res], color=color_empirical, label=label if i == 0 else "", alpha=0.7, s=50, marker=exp_marker)
        ax2.scatter(alpha_vals, [r["emp_w0"] for r in res], color=color_empirical, label=label if i == 0 else "", alpha=0.7, s=50, marker=exp_marker)
        
        # Mean lines
        alpha_to_emp_h = defaultdict(list)
        alpha_to_theo_h = defaultdict(list)
        alpha_to_emp_w = defaultdict(list)
        alpha_to_theo_w = defaultdict(list)
        alpha_to_nngp_w = defaultdict(list)
        for r in res:
            alpha = np.log(r["P"]) / np.log(r["d"])
            alpha_to_emp_h[alpha].append(r["emp_h"])
            alpha_to_theo_h[alpha].append(r["theo_h"])
            alpha_to_emp_w[alpha].append(r["emp_w0"])
            alpha_to_theo_w[alpha].append(r["theo_w"])
            alpha_to_nngp_w[alpha].append(r.get("theo_w_nngp", np.nan))
        unique_alpha = sorted(alpha_to_emp_h.keys())
        mean_emp_h = [np.mean(alpha_to_emp_h[a]) for a in unique_alpha]
        mean_theo_h = [np.mean(alpha_to_theo_h[a]) for a in unique_alpha]
        mean_emp_w = [np.mean(alpha_to_emp_w[a]) for a in unique_alpha]
        mean_theo_w = [np.mean(alpha_to_theo_w[a]) for a in unique_alpha]
        mean_nngp_w = [np.mean([v for v in alpha_to_nngp_w[a] if np.isfinite(v)]) if any(np.isfinite(alpha_to_nngp_w[a])) else np.nan for a in unique_alpha]
        draw_top_errorbar(ax1, unique_alpha, mean_emp_h, yerr=[float(np.std(alpha_to_emp_h[a], ddof=1) / np.sqrt(len(alpha_to_emp_h[a]))) if len(alpha_to_emp_h[a]) > 1 else 0.0 for a in unique_alpha], color=color_empirical, marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
        ax1.plot(unique_alpha, mean_theo_h, '--', color=color_theory, linewidth=3, marker=theo_marker, markersize=8, alpha=0.8)
        draw_top_errorbar(ax2, unique_alpha, mean_emp_w, yerr=[float(np.std(alpha_to_emp_w[a], ddof=1) / np.sqrt(len(alpha_to_emp_w[a]))) if len(alpha_to_emp_w[a]) > 1 else 0.0 for a in unique_alpha], color=color_empirical, marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
        ax2.plot(unique_alpha, mean_theo_w, '--', color=color_theory, linewidth=3, marker=theo_marker, markersize=8, alpha=0.8)
        ax2.plot(unique_alpha, mean_nngp_w, ':', color=color_nngp, linewidth=3, marker=theo_marker, markersize=6, alpha=0.7)
        ax1.set_ylim(0, None); ax2.set_ylim(0, None)
    # Dummy plot for legend
    ax1.plot([], [], '-', color=color_empirical, linewidth=3, marker=exp_marker, markersize=6, label="Experiment")
    ax1.plot([], [], '--', color=color_theory, linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
    ax2.plot([], [], '-', color=color_empirical, linewidth=3, marker=exp_marker, markersize=6, label="Experiment")
    ax2.plot([], [], '--', color=color_theory, linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
    ax2.plot([], [], ':', color=color_nngp, linewidth=3, marker=theo_marker, markersize=6, label="NNGP")
    ax1.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=r"$\alpha=1$")
    ax2.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=r"$\alpha=1$")
    ax1.set_title(r"$\lambda_H$ Eigenvalue"); ax2.set_title(r"$\lambda_W$ Eigenvalue")
    for ax in [ax1, ax2]: ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlabel(r"$\alpha$")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_alpha_linear_d{d}_N{N}_kappa_bare{kappa:.2g}_kappa_eff{kappa_eff:.2g}.png", dpi=300)

    # Separate alpha plots
    # Lambda H alpha
    fig_h = plt.figure(figsize=(8, 8))
    for i, (val, res) in enumerate(groups.items()):
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        if color_by == "chi":
            label = r"$\chi = $" + str(val)
        elif color_by == "d":
            label = f"d={val}"
        else:
            label = f"{color_by}={val}"
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        alpha_vals = [np.log(r["P"]) / np.log(r["d"]) for r in res]
        label = r"$\chi = $" + str(val) if color_by == "chi" else f"{color_by}={val}"
        plt.scatter(alpha_vals, [r["emp_h"] for r in res], color=color_empirical, label=label if i == 0 else "", alpha=0.7, s=50, marker=exp_marker)
        # Mean lines
        alpha_to_emp_h = defaultdict(list)
        alpha_to_theo_h = defaultdict(list)
        alpha_to_nngp_h = defaultdict(list)
        for r in res:
            alpha = np.log(r["P"]) / np.log(r["d"])
            alpha_to_emp_h[alpha].append(r["emp_h"])
            alpha_to_theo_h[alpha].append(r["theo_h"])
            alpha_to_nngp_h[alpha].append(r.get("theo_h_nngp", np.nan))
        unique_alpha = sorted(alpha_to_emp_h.keys())
        mean_emp_h = [np.mean(alpha_to_emp_h[a]) for a in unique_alpha]
        mean_theo_h = [np.mean(alpha_to_theo_h[a]) for a in unique_alpha]
        mean_nngp_h = [np.mean(alpha_to_nngp_h[a]) for a in unique_alpha]
        draw_top_errorbar(plt, unique_alpha, mean_emp_h, yerr=[float(np.std(alpha_to_emp_h[a], ddof=1) / np.sqrt(len(alpha_to_emp_h[a]))) if len(alpha_to_emp_h[a]) > 1 else 0.0 for a in unique_alpha], color=color_empirical, marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
        plt.plot(unique_alpha, mean_theo_h, '--', color=color_theory, linewidth=3, marker=theo_marker, markersize=8, alpha=0.8)
        plt.plot(unique_alpha, mean_nngp_h, ':', color=color_nngp, linewidth=3, marker=theo_marker, markersize=6, alpha=0.7)
    plt.plot([], [], '-', color=color_empirical, linewidth=3, marker=exp_marker, markersize=6, label="Experiment")
    plt.plot([], [], '--', color=color_theory, linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
    plt.plot([], [], ':', color=color_nngp, linewidth=3, marker=theo_marker, markersize=6, label="NNGP")
    plt.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=r"$\alpha=1$")
    plt.title("Preactivation Target Eigenvalues "+ r"$\lambda^{H,He1}_*$"  + "\n" + f" d={d}, $\\kappa_{{eff}}={kappa_eff:.2g}$, N={N}")
    plt.xlabel(r"$\alpha$"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.ylim(0, None)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_H_alpha_linear_d{d}_N{N}_kappa_bare{kappa:.2g}_kappa_eff{kappa_eff:.2g}.png", dpi=300)
    # He3 Lambda H (target) vs P
    fig_h3 = plt.figure(figsize=(10, 6))
    for i, (val, res) in enumerate(groups.items()):
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        if color_by == "chi":
            label = r"$\chi = $" + str(val)
        elif color_by == "d":
            label = f"d={val}"
        else:
            label = f"{color_by}={val}"
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        p_vals = [r["P"] for r in res]
        label = r"$\chi = $" + str(val) if color_by == "chi" else f"{color_by}={val}"
        
        # scatter of empirical He3 target eigenvalues
        plt.scatter(p_vals, [r["h3_target_eig"] for r in res],
                    color=color_empirical, label=label if i == 0 else "", alpha=0.7, s=50, marker=exp_marker)

        # mean lines over seeds
        p_to_emp_h3 = defaultdict(list)
        p_to_theo_h3 = defaultdict(list)
        p_to_nngp_h3 = defaultdict(list)
        for r in res:
            p_to_emp_h3[r["P"]].append(r["h3_target_eig"])
            p_to_theo_h3[r["P"]].append(r["theo_h3"])
            p_to_nngp_h3[r["P"]].append(r.get("theo_h3_nngp", np.nan))
        unique_p = sorted(p_to_emp_h3.keys())
        mean_emp_h3 = [np.mean(p_to_emp_h3[p]) for p in unique_p]
        mean_theo_h3 = [np.mean(p_to_theo_h3[p]) for p in unique_p]
        mean_nngp_h3 = [np.mean(p_to_nngp_h3[p]) for p in unique_p]
        draw_top_errorbar(plt, unique_p, mean_emp_h3, yerr=[float(np.std(p_to_emp_h3[p], ddof=1) / np.sqrt(len(p_to_emp_h3[p]))) if len(p_to_emp_h3[p]) > 1 else 0.0 for p in unique_p], color=color_empirical, marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
        plt.plot(unique_p, mean_theo_h3, '--', color=color_theory, linewidth=3,
             marker=theo_marker, markersize=8, alpha=0.8)
        plt.plot(unique_p, mean_nngp_h3, ':', color=color_nngp, linewidth=3,
             marker=theo_marker, markersize=6, alpha=0.7)

    # dummy handles for legend
    plt.plot([], [], '-', color=color_empirical, linewidth=3, marker=exp_marker, markersize=6, label="Experiment")
    plt.plot([], [], '--', color=color_theory, linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
    plt.plot([], [], ':', color=color_nngp, linewidth=3, marker=theo_marker, markersize=6, label="NNGP")
    plt.axvline(d, color='gray', linestyle='--', alpha=0.5, linewidth=2, label="P=d")

    plt.title(r"He3 Target Eigenvalues ($\lambda^{H,He3}_*$)" +
            f" d={d}, $\\kappa_{{eff}}={kappa_eff:.2g}$, N={N}")
    plt.xlabel("P (dataset size)")
    plt.ylabel(r"$\lambda^{H,He3}_*$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.ylim(0, None)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"eigenvalues_He3_d{d}_N{N}_kappa_bare{kappa:.2g}_kappa_eff{kappa_eff:.2g}.png", dpi=300)

    # He3 Lambda H (target) vs alpha
    fig_h3_alpha = plt.figure(figsize=(8, 8))
    for i, (val, res) in enumerate(groups.items()):
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        if color_by == "chi":
            label = r"$\chi = $" + str(val)
        elif color_by == "d":
            label = f"d={val}"
        else:
            label = f"{color_by}={val}"
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        alpha_vals = [np.log(r["P"]) / np.log(r["d"]) for r in res]
        label = r"$\chi = $" + str(val) if color_by == "chi" else f"{color_by}={val}"
        
        # scatter of empirical He3 target eigenvalues
        plt.scatter(alpha_vals, [r["h3_target_eig"] for r in res],
                    color=color_empirical, label=label if i == 0 else "", alpha=0.7, s=50, marker=exp_marker)

        # mean lines over seeds
        alpha_to_emp_h3 = defaultdict(list)
        alpha_to_theo_h3 = defaultdict(list)
        alpha_to_nngp_h3 = defaultdict(list)
        for r in res:
            alpha = np.log(r["P"]) / np.log(r["d"])
            alpha_to_emp_h3[alpha].append(r["h3_target_eig"])
            alpha_to_theo_h3[alpha].append(r["theo_h3"])
            alpha_to_nngp_h3[alpha].append(r.get("theo_h3_nngp", np.nan))
        unique_alpha = sorted(alpha_to_emp_h3.keys())
        mean_emp_h3 = [np.mean(alpha_to_emp_h3[a]) for a in unique_alpha]
        mean_theo_h3 = [np.mean(alpha_to_theo_h3[a]) for a in unique_alpha]
        mean_nngp_h3 = [np.mean(alpha_to_nngp_h3[a]) for a in unique_alpha]
        draw_top_errorbar(plt, unique_alpha, mean_emp_h3, yerr=[float(np.std(alpha_to_emp_h3[a], ddof=1) / np.sqrt(len(alpha_to_emp_h3[a]))) if len(alpha_to_emp_h3[a]) > 1 else 0.0 for a in unique_alpha], color=color_empirical, marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
        plt.plot(unique_alpha, mean_theo_h3, '--', color=color_theory, linewidth=3,
             marker=theo_marker, markersize=8, alpha=0.8)
        plt.plot(unique_alpha, mean_nngp_h3, ':', color=color_nngp, linewidth=3,
             marker=theo_marker, markersize=6, alpha=0.7)

    # dummy handles for legend
    plt.plot([], [], '-', color=color_empirical, linewidth=3, marker=exp_marker, markersize=6, label="Experiment")
    plt.plot([], [], '--', color=color_theory, linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
    plt.plot([], [], ':', color=color_nngp, linewidth=3, marker=theo_marker, markersize=6, label="NNGP")
    plt.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=r"$\alpha=1$")

    plt.title(r"He3 Target Eigenvalues ($\lambda^{H,He3}_*$)" +
            f" d={d}, $\\kappa_{{eff}}={kappa_eff:.2g}$, N={N}")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\lambda^{H,He3}_*$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, None)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"eigenvalues_He3_alpha_linear_d{d}_N{N}_kappa_bare{kappa:.2g}_kappa_eff{kappa_eff:.2g}.png", dpi=300)

    # Lambda W alpha
    fig_w = plt.figure(figsize=(8, 8))
    for i, (val, res) in enumerate(groups.items()):
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        if color_by == "chi":
            label = r"$\chi = $" + str(val)
        elif color_by == "d":
            label = f"d={val}"
        else:
            label = f"{color_by}={val}"
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        alpha_vals = [np.log(r["P"]) / np.log(r["d"]) for r in res]
        label = r"$\chi = $" + str(val) if color_by == "chi" else f"{color_by}={val}"
        plt.scatter(alpha_vals, [r["emp_w0"] for r in res], color=color_empirical, label=label if i == 0 else "", alpha=0.7, s=50, marker=exp_marker)
        # Mean lines
        alpha_to_emp_w = defaultdict(list)
        alpha_to_theo_w = defaultdict(list)
        alpha_to_nngp_w = defaultdict(list)
        for r in res:
            alpha = np.log(r["P"]) / np.log(r["d"])
            alpha_to_emp_w[alpha].append(r["emp_w0"])
            alpha_to_theo_w[alpha].append(r["theo_w"])
            alpha_to_nngp_w[alpha].append(r.get("theo_w_nngp", np.nan))
        unique_alpha = sorted(alpha_to_emp_w.keys())
        mean_emp_w = [np.mean(alpha_to_emp_w[a]) for a in unique_alpha]
        mean_theo_w = [np.mean(alpha_to_theo_w[a]) for a in unique_alpha]
        mean_nngp_w = [np.mean([v for v in alpha_to_nngp_w[a] if np.isfinite(v)]) if any(np.isfinite(alpha_to_nngp_w[a])) else np.nan for a in unique_alpha]
        draw_top_errorbar(plt, unique_alpha, mean_emp_w, yerr=[float(np.std(alpha_to_emp_w[a], ddof=1) / np.sqrt(len(alpha_to_emp_w[a]))) if len(alpha_to_emp_w[a]) > 1 else 0.0 for a in unique_alpha], color=color_empirical, marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
        plt.plot(unique_alpha, mean_theo_w, '--', color=color_theory, linewidth=3, marker=theo_marker, markersize=8, alpha=0.8)
        plt.plot(unique_alpha, mean_nngp_w, ':', color=color_nngp, linewidth=3, marker=theo_marker, markersize=6, alpha=0.7)
    # Y minimum is 0
    plt.ylim(0, None)
    plt.plot([], [], '-', color=color_empirical, linewidth=3, marker=exp_marker, markersize=6, label="Experiment")
    plt.plot([], [], '--', color=color_theory, linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
    plt.plot([], [], ':', color=color_nngp, linewidth=3, marker=theo_marker, markersize=6, label="NNGP")
    plt.ylabel(r"$\lambda_W^* = v^T \Sigma_w v$");
    plt.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=r"$\alpha=1$")
    plt.title(r"$\lambda_W$ Eigenvalue" + f" d={d}, $\\kappa_{{eff}}={kappa_eff:.2g}$, N={N}")
    plt.xlabel(r"$\alpha$"); plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_W_alpha_linear_d{d}_N{N}_kappa_bare{kappa:.2g}_kappa_eff{kappa_eff:.2g}.png", dpi=300)

    # NNGP Eigenvalues with alpha
    # Lambda H (NNGP alpha)
    fig_nngp_h_alpha = plt.figure(figsize=(8, 8))
    for i, (val, res) in enumerate(groups.items()):
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        if color_by == "chi":
            label = r"$\chi = $" + str(val)
        elif color_by == "d":
            label = f"d={val}"
        else:
            label = f"{color_by}={val}"
        alpha_vals = [np.log(r["P"]) / np.log(r["d"]) for r in res]
        plt.scatter(alpha_vals, [r["emp_h"] for r in res], color=color_empirical, label=label if i == 0 else "", alpha=0.7, s=50, marker=exp_marker)
        # Mean lines
        alpha_to_emp_h = defaultdict(list)
        alpha_to_nngp_h = defaultdict(list)
        for r in res:
            alpha = np.log(r["P"]) / np.log(r["d"])
            alpha_to_emp_h[alpha].append(r["emp_h"])
            alpha_to_nngp_h[alpha].append(r.get("theo_h_nngp", np.nan))
        unique_alpha = sorted(alpha_to_emp_h.keys())
        mean_emp_h = [np.mean(alpha_to_emp_h[a]) for a in unique_alpha]
        mean_nngp_h = [np.mean(alpha_to_nngp_h[a]) for a in unique_alpha]
        draw_top_errorbar(plt, unique_alpha, mean_emp_h, yerr=[float(np.std(alpha_to_emp_h[a], ddof=1) / np.sqrt(len(alpha_to_emp_h[a]))) if len(alpha_to_emp_h[a]) > 1 else 0.0 for a in unique_alpha], color=color_empirical, marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
        plt.plot(unique_alpha, mean_nngp_h, '--', color=color_nngp, linewidth=3, marker=theo_marker, markersize=8, alpha=0.8)
    plt.plot([], [], '-', color=color_empirical, linewidth=3, marker=exp_marker, markersize=6, label="Experiment")
    plt.plot([], [], '--', color=color_nngp, linewidth=3, marker=theo_marker, markersize=8, label="NNGP")
    plt.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=r"$\alpha=1$")
    plt.title(r"NNGP Hidden Kernel Eigenvalues ($\lambda_H^{NNGP}$)" + f" d={d}, $\\kappa_{{eff}}={kappa_eff:.2g}$, N={N}")
    plt.xlabel(r"$\alpha$"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.ylim(0, None)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_H_NNGP_alpha_linear_d{d}_N{N}_kappa_bare{kappa:.2g}_kappa_eff{kappa_eff:.2g}.png", dpi=300)

    # Learnability with alpha
    for mode in ["h1", "h3"]:
        plt.figure(figsize=(8, 8))
        for i, (val, res) in enumerate(groups.items()):
            res = collapse_rows_to_seed_means(res)
            c = series_color(i)
            alpha_vals = [np.log(r["P"]) / np.log(r["d"]) for r in res]
            if color_by == "chi":
                label = r"$\chi = $" + str(val)
            elif color_by == "d":
                label = f"d={val}"
            else:
                label = f"{color_by}={val}"
            plt.scatter(alpha_vals, [r[f"{mode}_emp"] for r in res], color=color_empirical, label=label if i == 0 else "", alpha=0.7, s=50, marker=exp_marker)
            plt.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=r"$\alpha=1$" if i == 0 else None)

            # Mean lines
            alpha_to_emp = defaultdict(list)
            alpha_to_theo = defaultdict(list)
            alpha_to_nngp = defaultdict(list)
            for r in res:
                alpha = np.log(r["P"]) / np.log(r["d"])
                alpha_to_emp[alpha].append(r[f"{mode}_emp"])
                alpha_to_theo[alpha].append(r[f"{mode}_theory"])
                alpha_to_nngp[alpha].append(r.get(f"{mode}_nngp_theory", np.nan))
            unique_alpha = sorted(alpha_to_emp.keys())
            mean_emp = [np.mean(alpha_to_emp[a]) for a in unique_alpha]
            mean_theo = [np.mean(alpha_to_theo[a]) for a in unique_alpha]
            mean_nngp = [np.mean(alpha_to_nngp[a]) for a in unique_alpha]
            draw_top_errorbar(plt, unique_alpha, mean_emp, yerr=[float(np.std(alpha_to_emp[a], ddof=1) / np.sqrt(len(alpha_to_emp[a]))) if len(alpha_to_emp[a]) > 1 else 0.0 for a in unique_alpha], color=color_empirical, marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
            plt.plot(unique_alpha, mean_theo, '--', color=color_theory, linewidth=3, marker=theo_marker, markersize=8, alpha=0.8)
            plt.plot(unique_alpha, mean_nngp, ':', color=color_nngp, linewidth=3, marker=theo_marker, markersize=6, alpha=0.7)
        # Dummy plot for legend
        plt.plot([], [], '-', color=color_empirical, linewidth=3, marker=exp_marker, markersize=6, label="Experiment")
        plt.plot([], [], '--', color=color_theory, linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
        plt.plot([], [], ':', color=color_nngp, linewidth=3, marker=theo_marker, markersize=6, label="NNGP")
        d = res[0]["d"] if res else 0
        kappa = res[0]['kappa'] if res else 0
        kappa_eff = res[0]['kappa_eff'] if res else 0
        N = res[0]['N'] if res else 0
        title_mode = "Hermite-1" if mode == "h1" else "Hermite-3"
        plt.title(rf"{title_mode} Learnability $\eta_{{He1}}$" + f"\n$d={d}, N={N}, \\kappa_{{eff}}={kappa_eff:.2g}$"); plt.xlabel(r"$\alpha$"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.ylabel(r"$\eta_{He1} = \frac{\langle f \mid He_1 \rangle}{y_{He1}}$ (Learnability)")
        plt.ylim(0, None)

        plt.tight_layout(); plt.savefig(RESULTS_DIR / f"learnability_{mode}_alpha_linear_d{d}_N{N}_kappa_bare{kappa:.2g}_kappa_eff{kappa_eff:.2g}.png", dpi=300)
    # plt.show()
