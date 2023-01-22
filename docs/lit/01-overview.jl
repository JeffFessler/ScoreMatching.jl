#=
# [Score Matching overview](@id 01-overview)

This page introduces the Julia package
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
using Distributions: Distribution, Gamma, Normal, MixtureModel, logpdf, pdf
import Distributions: logpdf, pdf
import ForwardDiff
using LaTeXStrings
using Random: seed!; seed!(0)
using Optim: optimize, BFGS
import Optim: minimizer
#using Flux:
#import ReverseDiff
using Plots: plot, plot!, scatter, default, gui
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

The idea behind the score matching approach
to model fitting is
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

# Some convenience methods
logpdf(d::Distribution) = x -> logpdf(d, x)
pdf(d::Distribution) = x -> pdf(d, x)
derivative(f::Function) = x -> ForwardDiff.derivative(f, x)
gradient(f::Function) = x -> ForwardDiff.gradient(f, x)
score(d::Distribution) = derivative(logpdf(d))

# Generate training data
T = 100
data_dis = Gamma(8, 1.0)
data_score = derivative(logpdf(data_dis))
data = rand(data_dis, T)
scatter(data, zeros(T))


#=
To perform unconstrained minimization
of a ``D``-component mixture,
the following mapping from ``\mathbb{R}^{D-1}``
to the ``D``-dimensional simplex is helpful.
It is the inverse of the
[Additive logratio transform](https://en.wikipedia.org/wiki/Compositional_data#Additive_logratio_transform).
=#

function map_r_s(y::AbstractVector)
    p = exp.([y; 1])
    return p ./ sum(p)
end
map_r_s(y::Real...) = map_r_s([y...])

y1 = range(-1,1,101) * 9
y2 = range(-1,1,101) * 9
tmp = map_r_s.(y1, y2')
jim(y1, y2, tmp; title="Simplex parameterization", nrow=1)


# Define model distribution

nmix = 3 # how many gaussians in the mixture model
function make_mix(θ ; σmin::Real=1e-2)
    mu = θ[1:nmix]
    sig = σmin .+ exp.(θ[nmix .+ (1:nmix)]) # ensure σ > 0
    p = map_r_s(θ[2nmix .+ (1:(nmix-1))])
#@show p
    tmp = [(μ,σ) for (μ,σ) in zip(mu, sig)]
    mix = MixtureModel(Normal, tmp, p)
    return mix
end

function make_model_score(θ)
    mix = make_mix(θ)
    return score(mix)
end

function fit2(x::AbstractVector{<:Real}, θ)
    model_score = make_model_score(θ)
    return sum(abs2, model_score.(x) - data_score.(x)) / T
end

fit1 = (θ) -> fit2(data, θ) # minimize this

# Initial crude guess of mixture model parameters
θ0 = [4, 7, 9, 0.1, 0.1, 0.5, 0, 0]

# Plot data pdf and initial model pdf
pf = plot(pdf(data_dis); xlims = (-1, 25), label="Gamma pdf",
    color = :green,
    xlabel = L"x",
    ylabel = L"p(x) \ \mathrm{ and } \ p(x;θ)",
)
tmp = make_mix(θ0)
plot!(pf, pdf(tmp), label = "Initial Gaussian Mixture", color=:blue)

#prompt()
#gui(); throw()


opt_sm = optimize(fit1, θ0, BFGS(); autodiff = :forward)
θsm = minimizer(opt_sm)
#gui(); throw()

#=
#todo cut 
function gd(θ; niter=100, step=1e-0)
    θ = collect(θ0)
    for _ in 1:niter
#       θ -= step * ForwardDiff.gradient(fit1, θ)
        θ -= step * gradient(fit1)(θ)
#       @show fit1(θ)
    end
    return θ
end

θh = gd(θ0)

tmp = make_mix(θh)
=#

tmp = make_mix(θsm)
plot!(pf, pdf(tmp), label = "Final Gaussian Mixture", color=:red)


#=
Plot the data score and model score functions
to see how well they match.
The largest mismatch is in the tails of the distribution
where there are few (if any) data points.
=#
ps = plot(data_score; xlims=(1,20), label = "Data score function",
    xticks=[1,20], xlabel=L"x")
tmp = make_model_score(θh)
plot!(ps, tmp; label = "Model score function", color=:red)

#=
## Maximum-likelihood estimation

This toy example is simple enough
that we can apply ML estimation to it directly.

As expected,
ML estimation leads to a lower negative log-likelihood.
=#

negloglike(θ) = (-1/T) * sum(logpdf(make_mix(θ)), data)
opt_ml = optimize(negloglike, θsm, BFGS(); autodiff = :forward)
θml = minimizer(opt_ml)
negloglike.([θml, θsm, θh])

#=
Bafflingly, ML estimation leads to much worse fits to the pdf,
even though we initialized the ML optimizer
with the score-matching parameters.
=#
plot!(pf, pdf(make_mix(θml)), label = "ML Gaussian Mixture", color=:magenta)
plot!(ps, make_model_score(θml), label = "ML score function", color=:magenta)
plot(pf, ps)


#=

# ### Reproducibility

# This page was generated with the following version of Julia:

io = IOBuffer(); versioninfo(io); split(String(take!(io)), '\n')


# And with the following package versions

import Pkg; Pkg.status()
=#
