#!/usr/bin/env julia
#
#

using Pkg
using Manifolds, ManifoldsBase, Documenter
using ManifoldsGPU
using DocumenterCitations, DocumenterInterLinks

# (e) add CONTRIBUTING.md and NEWS.md to docs

function add_links(line::String, url::String = "https://github.com/JuliaManifolds/ManifoldsGPU.jl")
    # replace issues (#XXXX) -> ([#XXXX](url/issue/XXXX))
    while (m = match(r"\(\#([0-9]+)\)", line)) !== nothing
        id = m.captures[1]
        line = replace(line, m.match => "([#$id]($url/issues/$id))")
    end
    # replace ## [X.Y.Z] -> with a link to the release [X.Y.Z](url/releases/tag/vX.Y.Z)
    while (m = match(r"\#\# \[([0-9]+.[0-9]+.[0-9]+)\] (.*)", line)) !== nothing
        tag = m.captures[1]
        date = m.captures[2]
        line = replace(line, m.match => "## [$tag]($url/releases/tag/v$tag) ($date)")
    end
    return line
end

generated_path = joinpath(@__DIR__, "src", "misc")
base_url = "https://github.com/JuliaManifolds/ManifoldsGPU.jl/blob/main/"
isdir(generated_path) || mkdir(generated_path)
for fname in ["NEWS.md"]
    open(joinpath(generated_path, fname), "w") do io
        # Point to source license file
        println(
            io,
            """
            ```@meta
            EditURL = "$(base_url)$(fname)"
            ```
            """,
        )
        # Write the contents out below the meta block
        for line in eachline(joinpath(dirname(@__DIR__), fname))
            println(io, add_links(line))
        end
    end
end

# (f) final step: render the docs
bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"); style = :alpha)
links = InterLinks(
    "ManifoldsBase" => ("https://juliamanifolds.github.io/ManifoldsBase.jl/stable/"),
    "Manifolds" => ("https://juliamanifolds.github.io/Manifolds.jl/stable/"),
)

# We'd like to build docs on CI, so we can't load the CUDA extension
# The rendered docs need to be in the main module
modules = [
    ManifoldsGPU,
]

if modules isa Vector{Union{Nothing, Module}}
    error("At least one module has not been properly loaded: ", modules)
end

makedocs(;
    format = Documenter.HTML(
        prettyurls = (get(ENV, "CI", nothing) == "true") || ("--prettyurls" ∈ ARGS),
        assets = ["assets/favicon.ico", "assets/citations.css", "assets/link-icons.css"],
        search_size_threshold_warn = 1000 * 2^10, # raise slightly from 500 to 1 MiB
        size_threshold_warn = 200 * 2^10, # raise slightly from 100 to 200 KiB
        size_threshold = 300 * 2^10,      # raise slightly 200 to 300 KiB
    ),
    modules = modules,
    authors = "Mateusz Baran, Shiwen An, and contributors.",
    sitename = "ManifoldsGPU.jl",
    pages = [
        "Home" => "index.md",
        "Miscellanea" => [
            "Changelog" => "misc/NEWS.md",
            "Internals" => "misc/internals.md",
            "References" => "misc/references.md",
        ],
    ],
    plugins = [bib, links],
)
deploydocs(repo = "github.com/JuliaManifolds/ManifoldsGPU.jl.git", push_preview = true)
#back to main env
Pkg.activate()
