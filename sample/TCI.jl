import TensorCrossInterpolation as TCI
using Random
using Plots
include("../src/utils.jl")
import .TTNUtils: bits2decimal, sampled_errors

k = randn(3, 30)

function f(v)
    """ Plane wave function in 3D with 30 different randomly (normal distribution) generated wavevectors k_j """
    r = [bits2decimal(v[1:div(length(v), 3)]), bits2decimal(v[div(length(v), 3)+1:2*div(length(v), 3)]), bits2decimal(v[2*div(length(v), 3)+1:end])]
    sum = 0.0
    for i in 1:30
        sum += cos(i * r' * k[:, i])
    end
    return sum
end

function f_INT(v)
    """ Plane wave function in 3D with 30 different randomly (normal distribution) generated wavevectors k_j with interleaved bits """
    r = [bits2decimal(v[1:3:length(v)]), bits2decimal(v[2:3:length(v)]), bits2decimal(v[3:3:length(v)])]
    sum = 0.0
    for i in 1:30
        sum += cos(i * r' * k[:, i])
    end
    return sum
end


function L1_error(f, ttn, nsamples::Int, bits::Int)
    """ Compute L1 error between function f and ttn approximation over nsamples random inputs of length 3*bits."""
    error_l1 = 0.0
    for i in 1:nsamples
        # Generate a random 3R sequence of 1s and 2s
        x = rand(1:2, 3 * bits)
        approx = TCI.evaluate(ttn, x)
        err = abs(f(x) - approx)
        error_l1 += err
    end
    return error_l1 / nsamples
end


function main()

    nsteps = 4
    errorlist = zeros(2, nsteps)
    samplederrorlist = zeros(2, nsteps)
    ranklist = zeros(2, nsteps)
    rankendlist = zeros(2, nsteps)

    # Parameters for TCI
    R = 16 # number of bits per dimension
    localdims = fill(2, 3R) # quantics representation so the local dimensions are 2

    ttn_seq = TCI.TensorCI2{Float64}(f, localdims)
    ttn_int = TCI.TensorCI2{Float64}(f_INT, localdims)

    for step in 1:nsteps
        kwargs = (
            maxbonddim=5 * step,
            tolerance=1e-15,
            verbosity=0,
        )

        # Sequential QTT
        ranks, errors = TCI.optimize!(ttn_seq, f; kwargs...)
        errorlist[1, step] = errors[end]
        errl1 = L1_error(f, ttn_seq, 1000, R)
        samplederrorlist[1, step] = errl1
        ranklist[1, step] = 5 * step
        rankendlist[1, step] = ranks[end]
        println("Error for maxbonddim=$(5*step): $(errors[end]), Sampled L1 error: $errl1")
        # Interleaved QTT
        ranks_int, errors_int = TCI.optimize!(ttn_int, f_INT; kwargs...)
        errorlist[2, step] = errors_int[end]
        errl1_int = L1_error(f_INT, ttn_int, 1000, R)
        samplederrorlist[2, step] = errl1_int
        ranklist[2, step] = 5 * step
        rankendlist[2, step] = ranks_int[end]
        println("Error for maxbonddim=$(5*step) (interleaved): $(errors_int[end]), Sampled L1 error: $errl1_int")
        println("Step $step completed.")

    end
end