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

global n = 15
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
        g += exp(-1im * j * eps_local[j] * (x - y)) #* exp(-lambda_local[j] * abs(x - y))
    end
    return 1im * g
end

function main()
    # Parameters for TCI
    R = 16 # number of bits per dimension
    d = 2 # spatial dimension
    localdims = fill(2, d * R) # local dimensions for d dimensions with R bits each


    groups = [[i, R + i] for i in 1:R]
    fused_g, fused_localdims, f_fused = fused_preparations(gf, groups, localdims)
    topo = Dict(
        "BTTN" => BTTN(R, d),
        "QTT_Seq" => QTT_Block(R, d),
        "QTT_Int" => QTT_Alt(R, d),
        "CTTN" => CTTN(R, d),
        "Fused_BTTN" => fused_g
    )

    ntopos = length(topo)
    nsteps = 6
    step = 3
    maxit = 5
    nsamples = 1000

    # Output storage
    error_l1 = zeros(ntopos, nsteps)
    error_pivot = zeros(ntopos, nsteps)
    rankendlist = zeros(ntopos, nsteps)
    ranklist = zeros(ntopos, nsteps)

    # generate 10 random global pivot of length 3R
    global_pivots = Vector{Vector{Int}}()
    for _ in 1:10
        push!(global_pivots, rand(1:2, length(localdims)))
    end

    #-----------------------------
    # Main loop over maxbonddim
    #-----------------------------
    for i in 1:nsteps
        maxbd = step * i
        println("Max bond dimension: $maxbd")
        tstart = @elapsed begin

            # -----------------------------
            # Construct initial TCIs
            # -----------------------------
            for (j, (toponame, topology)) in enumerate(topo)
                println("Topology: $toponame")
                if toponame == "Fused_BTTN"
                    f = f_fused
                    localdims_use = fused_localdims
                    # Build Simple TCIs so each run starts from identical initial pivots
                    ttn = TreeTCI.SimpleTCI{ComplexF64}(f, localdims_use, topology)
                    #TreeTCI.addglobalpivots!(ttn,global_pivots)
                    # Optimize TCI
                    ranks, errors = TreeTCI.optimize!(ttn, f; maxiter=maxit, maxbonddim=maxbd, sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer())
                    errl1_fused = 0.0
                    for k in 1:nsamples
                        orig = rand(1:2, 2R)
                        fused_idx = compress_indexset(orig, groups, localdims)
                        eval_ttn = if ttn isa TreeTCI.SimpleTCI
                            sitetensors = TreeTCI.fillsitetensors(ttn, f)
                            TreeTCI.TreeTensorNetwork(ttn.g, sitetensors)
                        else
                            ttn
                        end
                        approx = TreeTCI.evaluate(eval_ttn, fused_idx)
                        errl1_fused += abs(f(orig) - approx)
                    end
                    error_l1[j, i] = errl1_fused / nsamples
                    error_pivot[j, i] = errors[end]
                    rankendlist[j, i] = ranks[end]
                    ranklist[j, i] = maxbd
                else
                    ttn = TreeTCI.SimpleTCI{ComplexF64}(gf, localdims, topology)
                    #TreeTCI.addglobalpivots!(ttn,global_pivots)
                    # Optimize TCI
                    ranks, errors = TreeTCI.optimize!(ttn, gf; maxiter=maxit, maxbonddim=maxbd, sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer())
                    # Compute sampled L1 error
                    errl1 = sampled_error(gf, ttn, nsamples, R, d)
                    # Store results
                    error_l1[j, i] = errl1
                    error_pivot[j, i] = errors[end]
                    rankendlist[j, i] = ranks[end]
                    ranklist[j, i] = maxbd
                end
            end

        end # tstart elapsed
        println("Elapsed time for step $step: $tstart seconds")
    end

    p1 = plot(xlabel="Max bond dimension", ylabel="Sampled L1 Error", yscale=:log10)
    topo_names = collect(keys(topo))
    for j in 1:ntopos
        plot!(p1, step * collect(1:nsteps), error_l1[j, :], label=topo_names[j], marker=:o)
    end
    savefig(p1, "gf_tci_sampled_l1_error.svg")
    display(p1)
end
