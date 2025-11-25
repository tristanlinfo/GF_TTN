using Random
using Plots
using Logging

include(joinpath(@__DIR__, "..", "src", "treetci_src.jl"))
using .SampleTreeTCI

include(joinpath(@__DIR__, "..", "src", "topologies.jl"))

function main()
    R = 16
    nstep = 4
    maxbonddim_step = 5
    nsamples = 1000

    n = 30
    ϵ = abs.(randn(n)) * 3.0
    λ = abs.(randn(n)) * 10.0

    results = SampleTreeTCI.run_experiment_gf(R, nstep, maxbonddim_step, nsamples, Info,
        Dict(
            "BTTN" => BTTN(R),
            "CTTN" => CTTN(R),
            "QTT_Block" => QTT_Block(R),
            "QTT_Alt" => QTT_Alt(R),
            "BTTN_Alt" => BTTN_Alt(R),
            "CTTN_Alt" => CTTN_Alt(R),
        ), n, ϵ, λ
    )
    for (topology, res) in results
        println("Topology: ", topology)
        println("  errors: ", res[:errors])
        println("  sampled L1: ", res[:sampled])
    end
    keys_list = collect(keys(results))
    if !isempty(keys_list)
        # x-axis: max bond dimensions used at each step
        x = [i * maxbonddim_step for i in 1:nstep]
        p = plot()
        for k in keys_list
            plot!(p, x, results[k][:sampled], label=string(k), marker=:o, yscale=:log10)
        end
        xlabel!(p, "max bond dim")
        ylabel!(p, "sampled L1 error")
        title!(p, "TreeTCI sampled L1 error")
        display(p)
        # savefig(p, "treetci_sampled_l1_error.png")  
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end