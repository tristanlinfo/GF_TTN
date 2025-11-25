using Random
using Plots
using Logging

include(joinpath(@__DIR__, "..", "src", "tci_src.jl"))
using .SampleTCI

using Random
using Plots
using Logging

include(joinpath(@__DIR__, "..", "src", "tci_src.jl"))
using .SampleTCI

# Minimal runner that uses the convenience wrappers exported by SampleTCI.
function main()

    mode = "gf" # or "pw"

    if mode == "gf"
        println("Running fictitious-GF experiment (seq + int + cttn)...")
        results = SampleTCI.run_gf_experiment()
    else
        println("Running plane-wave experiment (seq + int)...")
        results = SampleTCI.run_plane_experiment()
    end

    for (topology, res) in results
        println("Topology: ", topology)
        println("  errors: ", res[:errors])
        println("  sampled L1: ", res[:sampled])
    end

    keys_list = collect(keys(results))
    if !isempty(keys_list)
        x = results[keys_list[1]][:ranks]
        p = plot()
        for k in keys_list
            plot!(p, x, results[k][:sampled], label=string(k), marker=:o, yscale=:log10)
        end
        xlabel!(p, "max bond dim")
        ylabel!(p, "sampled L1 error")
        title!(p, "TCI sampled L1 error")
        display(p)
        # savefig(p, "tci_sampled_l1_error.png")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end