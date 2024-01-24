module RBPmotifsCore

# Write your package code here.
const float_type = Float32
using Random, StatsBase, LinearAlgebra, CUDA, SeqShuffle
using Flux, Zygote
using Zygote: @ignore

include("loadfasta/helpers.jl")
include("loadfasta/fasta.jl")
include("model.jl")
include("opt.jl")


end
