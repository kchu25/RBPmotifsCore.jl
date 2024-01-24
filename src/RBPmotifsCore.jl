module RBPmotifsCore

# Write your package code here.
const float_type = Float32
using Random, StatsBase, LinearAlgebra, CUDA, SeqShuffle
using Flux, Zygote
using Zygote: @ignore
const convolution = Flux.NNlib.conv;

include("loadfasta/helpers.jl")
include("loadfasta/fasta.jl")
include("helpers.jl")
include("model.jl")
include("opt.jl")
include("train.jl")
include("code_retrieval.jl")



function discover_motifs(datapath::String; num_epochs=25)
    data = FASTA_DNA{RBPmotifsCore.float_type}(datapath);
    this_bg = get_data_bg(data)
    cdl, hp, len, projs = train(data; num_epochs=num_epochs);
    stored_code_components = RBPmotifsCore.code_retrieval(data, cdl, hp, len, projs);
    return stored_code_components, data, hp, this_bg
end




end
