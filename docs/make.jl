using RBPmotifsCore
using Documenter

DocMeta.setdocmeta!(RBPmotifsCore, :DocTestSetup, :(using RBPmotifsCore); recursive=true)

makedocs(;
    modules=[RBPmotifsCore],
    authors="Shane Kuei-Hsien Chu (skchu@wustl.edu)",
    sitename="RBPmotifsCore.jl",
    format=Documenter.HTML(;
        canonical="https://kchu25.github.io/RBPmotifsCore.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kchu25/RBPmotifsCore.jl",
    devbranch="main",
)
