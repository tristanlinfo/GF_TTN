module SampleTCI

using Random
using Logging
import TensorCrossInterpolation as TCI


include(joinpath(@__DIR__, "utils.jl"))
import .TTNUtils: bits2decimal, sampled_errors

export make_plane_wave, make_interleaved_plane_wave, l1_error, run_experiment, run_plane_experiment, run_gf_experiment

"""
    make_plane_wave(k::Matrix{Float64})

Return a function `f(v::AbstractVector{Int})` representing a plane-wave sum
constructed from columns of `k`. `k` is 3×M where M is the number of wavevectors.
"""
function make_plane_wave(k::Matrix{Float64})
    # k is (d × M) where d is the spatial dimension and M the number of vectors
    d = size(k, 1)
    nvec = size(k, 2)
    function f(v::AbstractVector{Int})
        L = length(v)
        R = div(L, d)
        r = zeros(Float64, d)
        for j in 1:d
            r[j] = bits2decimal(view(v, (j-1)*R+1:j*R))
        end
        s = 0.0
        for i in 1:nvec
            s += cos(i * r' * k[:, i])
        end
        return s
    end
    return f
end

"""Convenience wrapper: run the plane-wave experiments (seq + interleaved)
with simple keyword arguments. Returns the same `results` dict as
`run_experiment`."""
function run_plane_experiment(; kwargs...)
    return run_experiment(; kwargs..., mode=:plane)
end

"""Convenience wrapper: run fictitious-GF experiments (seq + int + cttn).
Pass `gf_params` as a Dict keyword argument if you want to customize `:n`,
`:eps`, or `:lambda`. Returns the `results` dict from `run_experiment`."""
function run_gf_experiment(; kwargs...)
    return run_experiment(; kwargs..., mode=:fictitious)
end

"""
    make_interleaved_plane_wave(k::Matrix{Float64})

Return an interleaved-bit variant.
"""
function make_interleaved_plane_wave(k::Matrix{Float64})
    # k is (d × M)
    d = size(k, 1)
    nvec = size(k, 2)
    function f_INT(v::AbstractVector{Int})
        L = length(v)
        r = zeros(Float64, d)
        for j in 1:d
            r[j] = bits2decimal(v[j:d:L])
        end
        s = 0.0
        for i in 1:nvec
            s += cos(i * r' * k[:, i])
        end
        return s
    end
    return f_INT
end

"""
Factory functions to build fictitious Green's-function-like callables.

Each factory returns a function `f(v::AbstractVector{Int})` that accepts a
bit-vector and computes a complex-valued Green's-function sum using the
provided parameters `n`, `eps`, and `lambda`. The grouping of bits differs
between sequential, interleaved and cttn variants.
"""
function make_fictitious_gf_seq(n::Int, eps::AbstractVector{<:Real}, lambda::AbstractVector{<:Real})
    function f(v::AbstractVector{Int})
        L = length(v)
        half = div(L, 2)
        x = bits2decimal(view(v, 1:half))
        y = bits2decimal(view(v, half+1:L))
        g = zero(ComplexF64)
        for j in 1:n
            g += exp(-1im * eps[j] * (x - y)) * exp(-lambda[j] * abs(x - y))
        end
        return 1im * g
    end
    return f
end

function make_fictitious_gf_int(n::Int, eps::AbstractVector{<:Real}, lambda::AbstractVector{<:Real})
    function f(v::AbstractVector{Int})
        x = bits2decimal(v[1:2:end])
        y = bits2decimal(v[2:2:end])
        g = zero(ComplexF64)
        for j in 1:n
            g += exp(-1im * eps[j] * (x - y)) * exp(-lambda[j] * abs(x - y))
        end
        return 1im * g
    end
    return f
end

function make_fictitious_gf_cttn(n::Int, eps::AbstractVector{<:Real}, lambda::AbstractVector{<:Real})
    function f(v::AbstractVector{Int})
        L = length(v)
        half = div(L, 2)
        x = bits2decimal(reverse(view(v, 1:half)))
        y = bits2decimal(view(v, half+1:L))
        g = zero(ComplexF64)
        for j in 1:n
            g += exp(-1im * eps[j] * (x - y)) * exp(-lambda[j] * abs(x - y))
        end
        return 1im * g
    end
    return f
end

"""
    l1_error(f, ttn, nsamples::Int, bits::Int; rng=GLOBAL_RNG, threads=false)

Estimate the mean absolute error (L1) between `f` and `TCI.evaluate(ttn, x)`
using `nsamples` random samples of bit-length `3*bits`. Accepts `rng` and an
option to use `Threads.@threads`.
"""
function l1_error(f, ttn, nsamples::Int, bits::Int, d::Int; rng=Random.GLOBAL_RNG, threads::Bool=false)
    total = Ref(0.0)
    sample_len = d * bits
    if threads
        Threads.@threads for i in 1:nsamples
            x = rand(rng, 1:2, sample_len)
            total[] += abs(f(x) - TCI.evaluate(ttn, x))
        end
    else
        for i in 1:nsamples
            x = rand(rng, 1:2, sample_len)
            total[] += abs(f(x) - TCI.evaluate(ttn, x))
        end
    end
    return total[] / nsamples
end

"""
    run_experiment(; rng_seed=1234, R=16, nsteps=4, maxbonddim_step=5, nsamples=1000, verbosity=Info)

constructs TCI.TensorCI2 objects, runs `TCI.optimize!`, collects errors
and sampled L1 errors, and returns a dictionary of results.
"""
function run_experiment(; rng_seed::Int=1234, R::Int=16, nsteps::Int=10, maxbonddim_step::Int=5,
    nsamples::Int=1000, verbosity::Logging.LogLevel=Info, d::Int=2, num_vecs::Int=30,
    mode::Symbol=:plane, gf_kind::Symbol=:seq, gf_params=Dict())
    Random.seed!(rng_seed)
    logger = ConsoleLogger(stderr, verbosity)
    global_logger(logger)

    # Create random k: d × num_vecs
    k = randn(d, num_vecs)

    # Choose which callable(s) to use depending on the run mode.
    funcs = Dict{Symbol,Any}()
    eltype_tt = Float64
    if mode == :plane
        f_seq = make_plane_wave(k)
        f_int = make_interleaved_plane_wave(k)
        funcs[:seq] = f_seq
        funcs[:int] = f_int
        eltype_tt = Float64
    elseif mode == :fictitious
        # Extract GF parameters (with sensible defaults)
        n = get(gf_params, :n, min(num_vecs, 30))
        eps = get(gf_params, :eps, randn(n))
        lambda = get(gf_params, :lambda, rand(n))
        # build all three topologies for fictitious GF
        f_seq = make_fictitious_gf_seq(n, eps, lambda)
        f_int = make_fictitious_gf_int(n, eps, lambda)
        f_cttn = make_fictitious_gf_cttn(n, eps, lambda)
        funcs[:seq] = f_seq
        funcs[:int] = f_int
        funcs[:cttn] = f_cttn
        eltype_tt = ComplexF64
    else
        error("Unknown mode: $mode")
    end

    localdims = fill(2, d * R)

    # build TTN objects for each selected topology
    ttns = Dict{Symbol,Any}()
    for (name, func) in funcs
        ttns[name] = TCI.TensorCI2{eltype_tt}(func, localdims)
    end

    # prepare results container per topology
    results = Dict{Symbol,Any}()
    for name in keys(funcs)
        results[name] = Dict(:errors => Float64[], :sampled => Float64[], :ranks => Int[], :rank_end => Int[])
    end

    for step in 1:nsteps
        maxbd = maxbonddim_step * step
        kwargs = (maxbonddim=maxbd, tolerance=1e-15, verbosity=0)

        for name in keys(ttns)
            func = funcs[name]
            ttn = ttns[name]
            @info "Optimizing" topology = name step = step maxbonddim = maxbd
            ranks, errors = TCI.optimize!(ttn, func; kwargs...)
            push!(results[name][:errors], errors[end])
            push!(results[name][:sampled], l1_error(func, ttn, nsamples, R, d))
            push!(results[name][:ranks], maxbd)
            push!(results[name][:rank_end], ranks[end])
        end
    end

    return results
end

end # module