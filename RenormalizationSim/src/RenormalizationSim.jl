module RenormalizationSim

using Plots
using QuadGK
using Printf

export calculate_sigma, run_full_simulation

# ----------------------------------------------------------------------------------
# 物理計算のコア関数
# ----------------------------------------------------------------------------------

"""
スカラー場の1ループ自己エネルギー Σ(p^2) を計算します。
この関数は、運動量カットオフ Λ を用いた紫外正則化に基づいています。
物理的には、仮想粒子ループによる質量補正の主要項を計算しています。
簡単化のため、このモデルでは自己エネルギーの運動量依存性を無視しています。
Σ = (λ / (16π^2)) * ∫[0,Λ] dk k^3 / (k^2 + m_phys^2)
"""
function calculate_sigma(m_squared_phys::Float64, lambda_coupling::Float64, cutoff_Lambda::Float64)
    prefactor = lambda_coupling / (16.0 * π^2)
    integrand(k) = k^3 / (k^2 + m_squared_phys)
    integral_val, _ = quadgk(integrand, 0, cutoff_Lambda)
    return prefactor * integral_val
end

"""
自己エネルギーの運動量 p^2 に関する微分 dΣ/dp^2 を計算します。
これは波動関数の繰り込み定数 Z を決定するために必要です。
オンシェル条件では、Z⁻¹ = 1 - dΣ/dp^2|_{p^2=m_phys^2} となります。
この微分項は通常、紫外発散しません。
"""
function calculate_sigma_derivative(m_squared_phys::Float64, lambda_coupling::Float64, cutoff_Lambda::Float64)
    # これは1ループ計算の有限部分をモデル化したものです。
    # 完全な計算はより複雑ですが、このモデルはZが1からずれる効果を捉えます。
    prefactor = -lambda_coupling / (16.0 * π^2)
    integrand(k) = k^3 / (6.0 * (k^2 + m_squared_phys)^2) # モデル化された被積分関数
    integral_val, _ = quadgk(integrand, 0, cutoff_Lambda)
    return prefactor * integral_val
end


# ----------------------------------------------------------------------------------
# シミュレーションの実行とプロット
# ----------------------------------------------------------------------------------

"""
シミュレーション全体を実行し、結果をプロット。
1. 繰り込みフロー（裸の質量とZ因子）を計算
2. 繰り込み済み伝播関数を計算
3. 結果をプロットし、画像として保存
"""
function run_full_simulation()
    # --- 物理パラメータの設定 ---
    m_phys = 7.0                # 物理質量 (GeV)
    m_squared_phys = m_phys^2   # 物理質量の2乗 (GeV^2)
    lambda_coupling = 20.0      # 結合定数
    epsilon = 1e-6              # iε処方のための微小量

    # --- グラフ1 & 2: 繰り込みフローを計算 ---
    println("Graph 1 & 2: Calculating renormalization flow for m₀²(Λ) and Z(Λ)...")
    cutoffs_for_running = 1.0:1.0:1000.0
    bare_mass_sq_values = []
    sigma_values = []
    z_factor_values = []

    for Lambda in cutoffs_for_running
        # オンシェル条件に基づき、裸の質量を計算
        sigma_at_mass_pole = calculate_sigma(m_squared_phys, lambda_coupling, Lambda)
        bare_mass_sq = m_squared_phys - sigma_at_mass_pole
        push!(sigma_values, sigma_at_mass_pole)
        push!(bare_mass_sq_values, bare_mass_sq)

        # 波動関数の繰り込み定数 Z を計算
        sigma_deriv = calculate_sigma_derivative(m_squared_phys, lambda_coupling, Lambda)
        z_factor = 1.0 / (1.0 - sigma_deriv)
        push!(z_factor_values, z_factor)
    end
    println("Calculation complete.")

    # --- グラフ3: 伝播関数の安定性を計算 ---
    println("\nGraph 3: Calculating propagator stability...")
    p_squared_minkowski_range = 0.0:0.02:100.0
    p_squared_euclidean_range = -p_squared_minkowski_range
    low_cutoff = 10.0
    high_cutoff = 1000.0
    
    # 低いカットオフでのパラメータ
    sigma_low = calculate_sigma(m_squared_phys, lambda_coupling, low_cutoff)
    m0_sq_low = m_squared_phys - sigma_low
    sigma_deriv_low = calculate_sigma_derivative(m_squared_phys, lambda_coupling, low_cutoff)
    z_low = 1.0 / (1.0 - sigma_deriv_low)
    @printf("Low cutoff (Λ=%.1f): m₀²=%.4f, Z=%.4f\n", low_cutoff, m0_sq_low, z_low)
    
    # 高いカットオフでのパラメータ
    sigma_high = calculate_sigma(m_squared_phys, lambda_coupling, high_cutoff)
    m0_sq_high = m_squared_phys - sigma_high
    sigma_deriv_high = calculate_sigma_derivative(m_squared_phys, lambda_coupling, high_cutoff)
    z_high = 1.0 / (1.0 - sigma_deriv_high)
    @printf("High cutoff (Λ=%.1f): m₀²=%.4f, Z=%.4f\n", high_cutoff, m0_sq_high, z_high)

    propagator_free = []
    propagator_low_cutoff = []
    propagator_high_cutoff = []

    for (i, p_sq_minkowski) in enumerate(p_squared_minkowski_range)
        p_sq_euclidean = p_squared_euclidean_range[i]

        # 1. 自由な伝播関数 (Z=1)
        D_free_inv = p_sq_minkowski - m_squared_phys + im * epsilon
        push!(propagator_free, abs2(1.0 / D_free_inv))

        # 2. 低いカットオフを持つ繰り込み済み伝播関数
        sigma_p_low = calculate_sigma(m_squared_phys, lambda_coupling, low_cutoff) # p依存性を無視したモデル
        D_low_inv = p_sq_minkowski - m0_sq_low - sigma_p_low + im * epsilon
        push!(propagator_low_cutoff, abs2(z_low / D_low_inv)) # Z因子の効果を追加

        # 3. 高いカットオフを持つ繰り込み済み伝播関数
        sigma_p_high = calculate_sigma(m_squared_phys, lambda_coupling, high_cutoff) # p依存性を無視したモデル
        D_high_inv = p_sq_minkowski - m0_sq_high - sigma_p_high + im * epsilon
        push!(propagator_high_cutoff, abs2(z_high / D_high_inv)) # Z因子の効果を追加
    end
    println("Calculation complete.")

    
    plot1 = plot(
        cutoffs_for_running,
        [sigma_values, bare_mass_sq_values],
        title="Graph 1: Renormalization Flow of Mass",
        xlabel="Momentum Cutoff Λ (GeV)",
        ylabel="Mass Squared (GeV²)",
        label=["Self-Energy Σ(Λ)" "Bare Mass m₀²(Λ)"],
        linewidth=2,
        legend=:bottomleft
    )
    hline!(plot1, [m_squared_phys], linestyle=:dash, color=:black, label="Physical Mass m_phys²")

    plot2 = plot(
        cutoffs_for_running,
        z_factor_values,
        title="Graph 2: Renormalization Flow of Z-factor",
        xlabel="Momentum Cutoff Λ (GeV)",
        ylabel="Field Strength Renormalization Z(Λ)",
        label="Z(Λ)",
        linewidth=2,
        legend=:bottomright
    )
    hline!(plot2, [1.0], linestyle=:dash, color=:black, label="Z = 1 (Free Theory)")

    plot3 = plot(
        p_squared_minkowski_range,
        [propagator_free, propagator_low_cutoff, propagator_high_cutoff],
        title="Graph 3: Dressed Propagator",
        xlabel="Momentum Squared p² (GeV²)",
        ylabel="Propagator |D'(p)|²",
        label=["Free Propagator" "Dressed Propagator (Low Λ)" "Dressed Propagator (High Λ)"],
        linewidth=[2 2 2],
        linestyle=[:dot :solid :dash],
        legend=:topleft,
        yaxis=:log
    )
    vline!(plot3, [m_squared_phys], linestyle=:dash, color=:black, label="Physical Mass Pole")

# 書き加え始め--- グラフ4: 三次元プロパゲータプロット ---
    p_vals = range(0.0, stop=100.0, length=50)
    Λ_vals = range(10.0, stop=1000.0, length=50)
    propagator_surface = [begin
        sigma = calculate_sigma(m_squared_phys, lambda_coupling, Λ)
        sigma_deriv = calculate_sigma_derivative(m_squared_phys, lambda_coupling, Λ)
        m0_sq = m_squared_phys - sigma
        Z = 1.0 / (1.0 - sigma_deriv)
        D_inv = p² - m0_sq - sigma + im * epsilon
        abs2(Z / D_inv)
    end for Λ in Λ_vals, p² in p_vals]

plot4 = surface(
        p_vals, Λ_vals, propagator_surface,
        xlabel="Momentum Squared p² (GeV²)",
        ylabel="Cutoff Energy Λ (GeV)",
        zlabel="|D'(p)|²",
        title="Graph 4: 3D Dressed Propagator",
        color=:viridis
    )
#書き加え終わり

    final_plot = plot(plot1, plot2, plot3,plot4, layout=(4, 1), size=(800, 1200))
    savefig(final_plot, "renormalization_plots_advanced.png")
    println("\nPlots saved to 'renormalization_plots_advanced.png'")
end

end
