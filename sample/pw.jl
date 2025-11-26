using Random
using Plots
include(joinpath(@__DIR__, "..", "src", "topologies.jl"))
include(joinpath(@__DIR__, "..", "src", "utils.jl"))
using .TTNUtils: bits2decimal
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

global k = randn(3, 30)
global n = 30

function pw(v)
    """ Plane wave function in 3D with 30 different randomly (normal distribution) generated wavevectors k_j """
    r = [bits2decimal(v[1:div(length(v), 3)]), bits2decimal(v[div(length(v), 3)+1:2*div(length(v), 3)]), bits2decimal(v[2*div(length(v), 3)+1:end])]
    sum = 0.0
    for i in 1:n
        sum += cos(i * r' * k[:, i])
    end
    return sum
end

function main()
    # Parameters for TCI
    R = 16 # number of bits per dimension
    d = 3 # spatial dimension
    localdims = fill(2, d * R) # local dimensions for d dimensions with R bits each

    topo = Dict(
        "BTTN" => BTTN(R, d),
        "QTT_Seq" => QTT_Block(R, d),
        "QTT_Int" => QTT_Alt(R, d),
        "CTTN" => CTTN(R, d)
    )

    ntopos = length(topo)
    nsteps = 16
    step = 3
    maxit = 5

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
                # Build Simple TCIs so each run starts from identical initial pivots
                ttn = TreeTCI.SimpleTCI{Float64}(pw, localdims, topology)
                #TreeTCI.addglobalpivots!(ttn,global_pivots)
                # Optimize TCI
                ranks, errors = TreeTCI.optimize!(ttn, pw; maxiter=maxit, maxbonddim=maxbd, sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer())

                # Compute sampled L1 error
                errl1 = sampled_error(pw, ttn, 1000, R, d)


                # Store results
                error_l1[j, i] = errl1
                error_pivot[j, i] = errors[end]
                rankendlist[j, i] = ranks[end]
                ranklist[j, i] = maxbd
            end

        end # tstart elapsed
        println("Elapsed time for step $step: $tstart seconds")
    end

    p1 = plot(xlabel="Max bond dimension", ylabel="Sampled L1 Error", yscale=:log10)
    topo_names = collect(keys(topo))
    for j in 1:ntopos
        plot!(p1, step * collect(1:nsteps), error_l1[j, :], label=topo_names[j], marker=:o)
    end
    savefig(p1, "pw_tci_sampled_l1_error.svg")
    display(p1)
end
