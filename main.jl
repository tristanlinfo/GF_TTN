# main.jl
import NamedGraphs: NamedGraph, NamedEdge, add_edge!, vertices, edges, has_edge
using TreeTCI
include("src/Topologies.jl")


function main()
    R = 30
    g = QTT_Block_Alt(R)

    localdims = fill(2, length(vertices(g)))
    f(v) = 1 / (1 + v' * v)
    kwargs = (
        maxbonddim=50,
        tolerance=1e-14,
        pivotstrategy=TreeTCI.DefaultPivotCandidateProposer(),
    )
    elapsed = @elapsed ttn, ranks, errors = TreeTCI.crossinterpolate(Float64, f, localdims, g; kwargs...)
    println("Elapsed time: $elapsed seconds")
    println("Final ranks: $ranks")
    println("Final errors: $errors")
end