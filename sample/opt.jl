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
        g += exp(-1im * j * eps_local[j] * (x - y)) #* exp(-lambda_local[j] * abs(x - y))
    end
    return 1im * g
end

function main()
    # Parameters for TCI
    R = 16 # number of bits per dimension
    d = 2 # spatial dimension
    localdims = fill(2, d * R) # local dimensions for d dimensions with R bits each


end