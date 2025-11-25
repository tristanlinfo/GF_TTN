module SampleTreeTCI

using Random
using Logging
using TreeTCI
using NamedGraphs: NamedGraph, add_edge!, vertices, NamedEdge, has_edge

include(joinpath(@__DIR__, "utils.jl"))
import .TTNUtils: bits2decimal

export make_pw, make_gf, l1_error, run_experiment, run_plane_experiment, run_gf_experiment, set_gf_params, set_k

## Module-level storage for configurable parameters
const GF_N_DEFAULT = 30
const PW_K_DEFAULT = zeros(3, GF_N_DEFAULT)

mutable struct _GFParams
    n::Int
    eps::Vector{Float64}
    lambda::Vector{Float64}
end

# default params
const GF_PARAMS = Ref(_GFParams(GF_N_DEFAULT, abs.(randn(GF_N_DEFAULT)), 10 .* abs.(randn(GF_N_DEFAULT))))

# plane-wave wavevectors
const PW_K = Ref(PW_K_DEFAULT)

"""Set fictitious-GF parameters used by `make_gf`.
   `n` must match length of `eps` and `lambda`.
"""
function set_gf_params(n::Integer, eps::AbstractVector{<:Real}, lambda::AbstractVector{<:Real})
    @assert length(eps) == n "length(eps) must equal n"
    @assert length(lambda) == n "length(lambda) must equal n"
    GF_PARAMS[] = _GFParams(Int(n), Float64.(eps), Float64.(lambda))
    return nothing
end

"""Set plane-wave wavevectors `k` (d × M matrix)."""
function set_k(k::AbstractMatrix{<:Real})
    PW_K[] = Array{Float64}(k)
    return nothing
end

function make_pw(v)
    """ Plane wave function in 3D with 30 different randomly (normal distribution) generated wavevectors k_j """
    r = [bits2decimal(v[1:div(length(v), 3)]), bits2decimal(v[div(length(v), 3)+1:2*div(length(v), 3)]), bits2decimal(v[2*div(length(v), 3)+1:end])]
    sum = 0.0
    k = PW_K[]
    nvec = size(k, 2)
    for i in 1:nvec
        sum += cos(i * r' * k[:, i])
    end
    return sum
end

function make_gf(v)
    """ Fictious Green's function in 3D with 30 different wavevectors k_j """
    x = bits2decimal(v[1:div(length(v), 2)])
    y = bits2decimal(v[div(length(v), 2)+1:end])
    params = GF_PARAMS[]
    n = params.n
    eps = params.eps
    lambda = params.lambda
    g = zero(ComplexF64)
    for j in 1:n
        g += exp(-1im * eps[j] * (x - y)) * exp(-lambda[j] * abs(x - y))
    end
    return 1im * g
end

function l1_error(f, ttn, nsamples, bits)
    """ Compute sampled errors between function f and ttn approximation over nsamples random inputs of length 2*bits."""
    eval_ttn = if ttn isa TreeTCI.SimpleTCI
        sitetensors = TreeTCI.fillsitetensors(ttn, f)
        TreeTCI.TreeTensorNetwork(ttn.g, sitetensors)
    else
        ttn
    end

    error_inf = 0.0
    error_l1 = 0.0
    for _ in 1:nsamples
        # Generate a random 3R sequence of 1s and 2s
        x = rand(1:2, 2 * bits)
        # Evaluate the concrete TreeTensorNetwork (it provides evaluate/call)
        approx = TreeTCI.evaluate(eval_ttn, x)
        err = abs(f(x) - approx)
        error_inf = max(error_inf, err)
        error_l1 += err
    end
    return error_inf, error_l1 / nsamples
end

function run_experiment_gf(R::Int, nsteps::Int, maxbonddim_step::Int,
    nsamples::Int, verbosity::Logging.LogLevel, topo::Dict{String,NamedGraph{Int64}}, n::Int, eps::AbstractVector{<:Real}, lambda::AbstractVector{<:Real})
    """ Run fictitious-GF experiments.
    `topo` should be a Dict mapping topology name => `NamedGraph`, e.g.
    `topos = Dict("BTTN" => BTTN_Alt(R), "CTTN" => CTTN_Alt(R), ...)`.
    Returns a dictionary of results.
    """
    logger = ConsoleLogger(stderr, verbosity)
    global_logger(logger)

    # set the module-level GF parameters
    set_gf_params(n, eps, lambda)

    results = Dict{String,Any}()
    for step in 1:nsteps
        maxbonddim = step * maxbonddim_step
        println("---- ---- Running step $step with max bond dim $maxbonddim ---- ----")
        kwargs = (
            maxbonddim=maxbonddim,
            tolerance=1e-14,
            sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer(),
        )
        for (topology, g) in topo
            println("Running topology: ", topology)
            localdims = fill(2, 2 * R)
            # use module-level `make_gf` as the target function (expects globals n, ϵ, λ to be set)
            elapsed = @elapsed ttn, ranks, errors = TreeTCI.crossinterpolate(ComplexF64, make_gf, localdims, g; kwargs...)
            println("Elapsed time: $elapsed seconds")
            println("Final ranks: $ranks")
            println("Final errors: $errors")

            # Compute sampled errors
            err_inf, err_l1 = l1_error(make_gf, ttn, nsamples, R)
            println("Sampled errors - Inf: $err_inf, L1: $err_l1")

            if !haskey(results, topology)
                results[topology] = Dict(
                    :ranks => Any[],
                    :errors => Any[],
                    :sampled => Float64[],
                )
            end
            push!(results[topology][:ranks], ranks)
            push!(results[topology][:errors], errors)
            push!(results[topology][:sampled], err_l1)
        end
    end
    return results
end

function run_experiment_pw(R::Int, nsteps::Int, maxbonddim_step::Int,
    nsamples::Int, verbosity::Logging.LogLevel, topo::Dict{String,NamedGraph{Int64}})
    """ Run plane-wave experiments.
    `topo` should be a Dict mapping topology name => `NamedGraph`.
    Returns a dictionary of results.
    """
    logger = ConsoleLogger(stderr, verbosity)
    global_logger(logger)

    results = Dict{String,Any}()
    for step in 1:nsteps
        maxbonddim = step * maxbonddim_step
        println("Running step $step with max bond dim $maxbonddim")
        kwargs = (
            maxbonddim=maxbonddim,
            tolerance=1e-14,
            sweepstrategy=TreeTCI.LocalAdjacentSweep2sitePathProposer(),
        )
        for (topology, g) in topo
            println("Running topology: ", topology)
            localdims = fill(2, length(TreeTCI.vertices(g)))
            # use module-level `make_pw` as the target function (expects global `k` to be set)
            elapsed = @elapsed ttn, ranks, errors = TreeTCI.crossinterpolate(Float64, make_pw, localdims, g; kwargs...)
            println("Elapsed time: $elapsed seconds")
            println("Final ranks: $ranks")
            println("Final errors: $errors")

            # Compute sampled errors
            err_inf, err_l1 = l1_error(make_pw, ttn, nsamples, R)
            println("Sampled errors - Inf: $err_inf, L1: $err_l1")

            if !haskey(results, topology)
                results[topology] = Dict(
                    :ranks => Any[],
                    :errors => Any[],
                    :sampled => Float64[],
                )
            end
            push!(results[topology][:ranks], ranks)
            push!(results[topology][:errors], errors)
            push!(results[topology][:sampled], err_l1)
        end
    end
    return results
end

"""Convenience wrapper: run plane-wave experiments."""
function run_plane_experiment(args...)
    return run_experiment_pw(args...)
end

"""Convenience wrapper: run fictitious-GF experiments."""
function run_gf_experiment(args...)
    return run_experiment_gf(args...)
end

"""Dispatching runner: `run_experiment(:pw, ...)` or `run_experiment(:gf, ...)`."""
function run_experiment(mode::Symbol, args...)
    if mode == :pw || mode == :plane || mode == :plane_wave
        return run_experiment_pw(args...)
    elseif mode == :gf || mode == :fictitious || mode == :fictitious_gf
        return run_experiment_gf(args...)
    else
        throw(ArgumentError("Unknown mode: $mode"))
    end
end

end