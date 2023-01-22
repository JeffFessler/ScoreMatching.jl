#=
# [Score Matching overview](@id 01-overview)

This page illustrates the Julia package
[`ScoreMatching`](https://github.com/JeffFessler/ScoreMatching.jl).

This page was generated from a single Julia file:
[01-overview.jl](@__REPO_ROOT_URL__/01-overview.jl).
=#

#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](https://nbviewer.org/) here:
#md # [`01-overview.ipynb`](@__NBVIEWER_ROOT_URL__/01-overview.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`01-overview.ipynb`](@__BINDER_ROOT_URL__/01-overview.ipynb).


# ### Setup

# Packages needed here.

using ScoreMatching
using MIRTjim: jim, prompt
using Plots: plot, plot!, scatter, default
using InteractiveUtils: versioninfo
default(label="", markerstrokecolor=:auto)

# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);


#=
## Overview

Given ``T``
[IID](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)
training data samples
``\mathbf{x}_1, …, \mathbf{x}_T ∈ \mathbb{R}^N``,
we often want to find the parameters
``\mathrm{θ}``
of a model distribution
``p(\mathrm{x}; \mathrm{θ})``
that "best fit" the data.

Maximum-likelihood estimation
is impractical for complicated models
where the normalizing constant is intractable.

[Hyväarinen 2005](http://jmlr.org/papers/v6/hyvarinen05a.html)
proposed an alternative called
_score matching_
that circumvents
needing to find the normalizing constant.

The idea of the score matching approach
is 
```math
\hat{\mathbf{θ}} = \arg \min_{\mathbf{θ}}
\frac{1}{T} ∑_{t=1}^T
\| \mathbf{s}(\mathbf{x}_t; \mathbf{θ}) - \mathbf{s}(\mathbf{x}_t) \|^2
```
where
``
\mathbf{s}(\mathbf{x}; \mathbf{θ}) =
\nabla_{\mathbf{x}} \log p(\mathbf{x}; \mathbf{θ})
``
is the _score function_
of the model distribution,
and
``
\mathbf{s}(\mathbf{x}) =
\nabla_{\mathbf{x}} \log p(\mathbf{x})
``
is the _score function_
of the (typically unknown) data distribution.


## Illustration

For didactic purposes,
we illustrate this basic version of score matching
by fitting samples from a
[Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution)
to a mixture of gaussians.

=#

using Distributions: Gamma, Normal, MixtureModel, logpdf, pdf
#using Flux:
import ForwardDiff
using LaTeXStrings
#import ReverseDiff

T = 100
data_dis = Gamma(8, 1.0)

data_logpdf = x -> logpdf(data_dis, x)
data_score = x -> ForwardDiff.derivative(data_logpdf, x)

data = rand(data_dis, T)

scatter(data, zeros(T))

function make_mix(θ)
    (mu1, mu2, sig1, sig2, π1) = θ # model parameters
    mix = MixtureModel(Normal, [(mu1, sig1), (mu2, sig2)], [π1, 1-π1])
    return mix
end

function make_model_score(θ)
    mix = make_mix(θ)
    model_logpdf = x -> logpdf(mix, x)
    return x -> ForwardDiff.derivative(model_logpdf, x)
end

function fit2(x::AbstractVector{<:Real}, θ)
    model_score = make_model_score(θ)
    return sum(abs2, model_score.(x) - data_score.(x)) / T
end

fit1 = (θ) -> fit2(data, θ) # minimize this

# Initial guess of mixture model parameters
θ0 = (4, 9, 2, 2, 0.3)
tmp = make_model_score(θ0)
tmp = make_mix(θ0)

data_pdf = x -> pdf(data_dis, x)

#function plot_fit(θ; label="Initial")
    pf = plot(data_pdf; xlims = (-1, 25), label="Gamma pdf",
     color = :green,
     xlabel = L"x",
     ylabel = L"p(x) \ \mathrm{ and } \ p(x;θ)",
    )
    tmp = make_mix(θ0)
    plot!(pf, x -> pdf(tmp, x), label = "Initial Gaussian Mixture", color=:blue)
    #return p
#end


# function score

function gd(θ; niter=300, step=3e-2)
    θ = collect(θ0)
    for _ in 1:niter
        θ -= step * ForwardDiff.gradient(fit1, θ)
        @show fit1(θ)
    end
    return θ
end

θh = gd(θ0)

    tmp = make_mix(θh)
    plot!(pf, x -> pdf(tmp, x), label = "Final Gaussian Mixture", color=:magenta)

#=

# ### Reproducibility

# This page was generated with the following version of Julia:

io = IOBuffer(); versioninfo(io); split(String(take!(io)), '\n')


# And with the following package versions

import Pkg; Pkg.status()
=#
