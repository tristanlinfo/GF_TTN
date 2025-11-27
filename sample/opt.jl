using Random
using Plots
include(joinpath(@__DIR__, "..", "src", "topologies.jl"))
include(joinpath(@__DIR__, "..", "src", "utils.jl"))
using .TTNUtils: bits2decimal, fused_preparations, compress_indexset
using TreeTCI

function sampled_error(f, ttn, nsamples, bits, d)
    """ Compute sampled errors between function f and ttn approximation over nsamples random inputs of length 2*bits."""
    eval_ttn = if ttn isa TreeTCI.SimpleTCI
        sitetensors = TreeTCI.fillsitetensors(ttn, f)
        TreeTCI.TreeTensorNetwork(ttn.g, sitetensors)
    else
        ttn
    end
    error_l1 = 0.0
    for _ in 1:nsamples
        # Generate a random 3R sequence of 1s and 2s
        x = rand(1:2, d * bits)
        # Evaluate the concrete TreeTensorNetwork (it provides evaluate/call)
        approx = TreeTCI.evaluate(eval_ttn, x)
        err = abs(f(x) - approx)
        error_l1 += err
    end
    return error_l1 / nsamples
end

global n = 30
global ϵ = abs.(randn(n)) * 3.0
global λ = abs.(randn(n)) * 10.0

function gf(v)
    """ Fictious Green's function in 2D with 30 different wavevectors k_j """
    x = bits2decimal(v[1:div(length(v), 2)])
    y = bits2decimal(v[div(length(v), 2)+1:end])
    # Use the module-level globals `n`, `ϵ`, `λ` (avoid shadowing them with a local `n = n`)
    n_local = n
    eps_local = ϵ
    lambda_local = λ
    g = zero(ComplexF64)
    for j in 1:n_local
        g += exp(-1im * j * eps_local[j] * (x - y)) * exp(-lambda_local[j] * abs(x - y))
    end
    return 1im * g
end

function init(topo::Dict{String,NamedGraph{Int64}}, f; R::Int, d::Int, localdims::Vector{Int})
    newtopo = Dict()
    kwargs = (
        maxiter=5,
        sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer(),
        tolerance=1e-4,
    )
    println("---- Starting structural optimization runs ----")
    for (name, graph) in pairs(topo)
        println("Topology ", name, ":")
        ttn, ranks, errors = TreeTCI.crossinterpolate(ComplexF64, f, localdims, graph; kwargs...)
        errl1 = sampled_error(f, ttn, 1000, R, d)
        println("Sampled L1 error: ", errl1)
        println("Maxbonddim: ", ranks[end])
        center_vertex = 1
        # structural optimization can sometimes fail if the entanglement vector is empty
        g_tmp, original_entanglements, entanglements = TreeTCI.ttnopt(ttn; ortho_vertex=center_vertex, max_degree=1, T0=1.0)
        ttn_struct, ranks_struct, errors_struct = TreeTCI.crossinterpolate(ComplexF64, f, localdims, g_tmp; center_vertex=center_vertex, kwargs...)
        errl1_struct = sampled_error(f, ttn_struct, 1000, R, d)
        println("Sampled L1 error after structural optimization: ", errl1_struct)
        println("Maxbonddim after structural optimization: ", ranks_struct[end])
        newtopo[string(name, "_struct")] = g_tmp
        g_tmp, original_entanglements, entanglements = TreeTCI.ttnopt(ttn; ortho_vertex=center_vertex, max_degree=2, T0=1.0)
        ttn_struct, ranks_struct, errors_struct = TreeTCI.crossinterpolate(ComplexF64, f, localdims, g_tmp; center_vertex=center_vertex, kwargs...)
        errl1_struct = sampled_error(f, ttn_struct, 1000, R, d)
        newtopo[string(name, "_struct2")] = g_tmp
    end
    return newtopo
end

function main()
    # Parameters for TCI
    R = 16 # number of bits per dimension
    d = 2 # spatial dimension
    localdims = fill(2, d * R) # local dimensions for d dimensions with R bits each

    topo = Dict(
        #"BTTN" => BTTN(R, d),
        "QTT_Int" => QTT_Alt(R, d),
        "QTT_Seq" => QTT_Block(R, d),
        "CTTN" => CTTN(R, d)
    )

    newtopo = init(topo, gf; R=R, d=d, localdims=localdims)

    fulltopo = merge(topo, newtopo)

    ntopos = length(fulltopo)
    nsteps = 10
    step = 3
    maxit = 5
    nsamples = 1000

    # Output storage
    error_l1 = zeros(ntopos, nsteps)
    error_pivot = zeros(ntopos, nsteps)
    rankendlist = zeros(ntopos, nsteps)
    ranklist = zeros(ntopos, nsteps)

    for i in 1:nsteps
        maxbd = step * i
        println("Max bond dimension: $maxbd")
        # -----------------------------
        # Construct initial TCIs
        # -----------------------------
        for (j, (toponame, topology)) in enumerate(fulltopo)
            println("Topology: $toponame")
            # Build Simple TCIs so each run starts from identical initial pivots
            ttn = TreeTCI.SimpleTCI{ComplexF64}(gf, localdims, topology)
            #TreeTCI.addglobalpivots!(ttn,global_pivots)
            # Optimize TCI
            ranks, errors = TreeTCI.optimize!(ttn, gf; tolerance=1e-16, maxiter=maxit, maxbonddim=maxbd, sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer())

            # Compute sampled L1 error
            errl1 = sampled_error(gf, ttn, 1000, R, d)
            # Store results
            error_l1[j, i] = errl1
            error_pivot[j, i] = errors[end]
            rankendlist[j, i] = ranks[end]
            ranklist[j, i] = maxbd
        end

    end

    plt = plot()
    x = [i * step for i in 1:nsteps]
    for (j, (toponame, topology)) in enumerate(fulltopo)
        plot!(plt, x, error_l1[j, :], label=toponame, marker=:o, yscale=:log10)
    end
    xlabel!(plt, "max bond dim")
    ylabel!(plt, "sampled L1 error")
    title!(plt, "Fictitious GF sampled L1 error")
    savefig(plt, "SVG/fictitious_gf_sampled_l1_error.svg")
    display(plt)
end