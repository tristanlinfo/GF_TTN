module TTNUtils

"""
Utilities for TTN project — fused-topology helpers and index mappings.

Conventions:
- `localdims` is a Vector{Int} giving the physical dimension for each original site (1-based indexing).
- `groups` is a Vector of vectors of integer site indices (1-based) describing which original sites are fused together.
- For mixed-radix packing we treat the first site in each group as the least-significant digit.

Exported functions:
- `fused_localdims(localdims, groups)` -> Vector of fused dims (product per group).
- `compress_indexset(orig, groups, localdims)` -> compress a full original index vector into fused indices per group.
- `expand_fused_indexset(fused, groups, localdims)` -> inverse mapping returning the original full index vector.
- `fuse_graph_pairs(g, groups)` -> fused graph where each group is a vertex.
- `make_fused_wrapper(f_orig, groups, localdims)` -> returns a function accepting fused indices and calling `f_orig` with expanded indices.
- `bits2decimal(v)` -> convert a vector of bits (1/2) to a decimal number between 0 and 1.
- `seed_pivots!(tci)` -> seed initial pivots for SimpleTCI based on binary representations.
- `sampled_errors(f, ttn, nsamples, bits)` -> compute sampled errors between function f and ttn approximation over nsamples random inputs of length 2*bits.
"""

export fused_localdims, compress_indexset, expand_fused_indexset, fuse_graph_pairs, make_fused_wrapper, bits2decimal, seed_pivots!, sampled_errors

using Graphs
using TreeTCI

function fused_localdims(localdims::AbstractVector{<:Integer}, groups::AbstractVector{<:AbstractVector{<:Integer}})
    return [prod(localdims[g]) for g in groups]
end

"""
compress_indexset(orig, groups, localdims)
- `orig` : Vector{Int} of length N (N = total number of original sites), where orig[i] in 1:localdims[i]
- returns: Vector{Int} of length `length(groups)`, each entry in 1:fused_dim

The packing uses mixed-radix with the first site in the group as least-significant.
"""
function compress_indexset(orig::AbstractVector{<:Integer}, groups::AbstractVector{<:AbstractVector{<:Integer}}, localdims::AbstractVector{<:Integer})
    fused = Vector{Int}(undef, length(groups))
    for (i, g) in enumerate(groups)
        dims = localdims[g]
        idxs = orig[g]
        multiplier = 1
        val = 0
        for j in eachindex(dims)
            val += (idxs[j] - 1) * multiplier
            multiplier *= dims[j]
        end
        fused[i] = val + 1
    end
    return fused
end

"""
expand_fused_indexset(fused, groups, localdims)
- `fused` : Vector{Int} of length n_groups (each in 1:fused_dim)
- returns: Vector{Int} of length N (original sites), in the same ordering as the concatenation of `groups`.
"""
function expand_fused_indexset(fused::AbstractVector{<:Integer}, groups::AbstractVector{<:AbstractVector{<:Integer}}, localdims::AbstractVector{<:Integer})
    N = sum(length.(groups))
    out = Vector{Int}(undef, N)
    offset = 1
    for (i, g) in enumerate(groups)
        dims = localdims[g]
        v = fused[i] - 1
        for j in 1:length(g)
            d = dims[j]
            out[offset] = (v % d) + 1
            v ÷= d
            offset += 1
        end
    end
    return out
end

"""
fuse_graph_pairs(g, groups)
- `g` : a Graphs graph (e.g., `SimpleGraph`) indexing original sites with integers 1..N
- `groups` : grouping of original vertices
- returns: new `SimpleGraph` with `length(groups)` vertices. An edge exists between fused vertices if any cross-edge exists.
"""
function fuse_graph_pairs(g::Graphs.AbstractGraph, groups::AbstractVector{<:AbstractVector{<:Integer}})
    ng = length(groups)
    G = SimpleGraph(ng)
    v2group = Dict{Int,Int}()
    for (gi, grp) in enumerate(groups)
        for v in grp
            v2group[v] = gi
        end
    end
    for e in edges(g)
        s = src(e)
        d = dst(e)
        gs = v2group[s]
        gd = v2group[d]
        if gs != gd && !has_edge(G, gs, gd)
            add_edge!(G, gs, gd)
        end
    end
    return G
end

"""
make_fused_wrapper(f_orig, groups, localdims)
- returns a function `f_fused(fused_idxs)` that expands `fused_idxs` and calls `f_orig(expanded)`.
- `f_orig` is expected to accept a vector of original indices.
"""
function make_fused_wrapper(f_orig::Function, groups, localdims)
    return function (fused_idxs)
        expanded = expand_fused_indexset(fused_idxs, groups, localdims)
        return f_orig(expanded)
    end
end


function bits2decimal(v::AbstractVector{<:Integer})
    """Convert a vector of bits (1/2) to a decimal number between 0 and 1"""
    sum = 0.0
    for i in 1:length(v)
        sum += (v[i] - 1) * 2.0^(-i)
    end
    return sum
end

function seed_pivots!(tci)
    """ Seed initial pivots for the SimpleTCI tci based on binary representations. """

    for i in 1:10
        # Generate a random input of length equal to number of vertices
        x = rand(1:2, length(vertices(tci.g)))
        TreeTCI.addglobalpivots!(tci, [x])
    end
end

function sampled_errors(f, ttn, nsamples::Int, bits::Int)
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

end # module
