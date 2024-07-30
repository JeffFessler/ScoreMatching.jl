#=
# [Sampling: Generalized Gaussian](@id 03-gen-gauss)

WIP on sampling using a score model
from Generalized Gaussian prior.
=#

#srcURL

# ### Setup

# Packages needed here.

#src using ScoreMatching
using MIRT: diffl_map
using MIRTjim: jim, prompt
#using Distributions: Distribution, Normal, MixtureModel, logpdf, pdf
#import Distributions # var, mean
#import Distributions: logpdf, pdf
#import ForwardDiff # derivative, gradient
using LaTeXStrings
using Printf: @sprintf
using Random: shuffle, seed!; seed!(0)
using StatsBase: mean, std
using Plots: Plot, plot, plot!, scatter, scatter!, histogram, quiver!
using Plots: @animate, gif
import Plots # Animation
using Plots: default, gui, savefig
using Plots.PlotMeasures: px
default(label="", markerstrokecolor=:auto, linewidth=2,
 labelfontsize = 14, tickfontsize = 12, legendfontsize = 14,
)


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:prompt);


#=
## Overview

Given a score function
``
\bm{s}(\bm{x}; \bm{θ}) =
\nabla_{\bm{x}} \log p(\bm{x}; \bm{θ}),
``
one can use Langevin dynamics
to draw samples from
``p(\bm{x}; \bm{θ}).``
=#


#=
## Illustration

Sampling from a generalized gaussian image prior.
``
p(\bm{x}) ∝ \exp(- ‖ \bm{Δ} \bm{x} ‖_p^p)
= \exp(- ∑_n |[\bm{Δ} \bm{x}]_n|^p)
= \exp(- \bm{1}' ϕ.(\bm{Δ} \bm{x}))
``
where
``p > 1``,
``ϕ(t) = |t|^p``,
and
``\bm{Δ}``
denotes a finite-difference matrix.
The corresponding score function is
``
\bm{s}(\bm{x}) = - \bm{Δ}' \dot{ϕ}.(\bm{Δ} \bm{x})
``

=#

N = (128, 120)
Δ = diffl_map(N, [1,2]) # ; T = ComplexF32)
p = 1.5
ϕ(t) = abs(t)^p
dϕ(t) = p * abs(t)^(p-1) * sign(t)

# Some convenience methods
#logpdf(d::Distribution) = x -> logpdf(d, x)
#pdf(d::Distribution) = x -> pdf(d, x)
#derivative(f::Function) = x -> ForwardDiff.derivative(f, x)
#gradient(f::Function) = x -> ForwardDiff.gradient(f, x)
## hessian(f::Function) = x -> ForwardDiff.hessian(f, x)
#score(d::Distribution) = derivative(logpdf(d))
#score_deriv(d::Distribution) = derivative(score(d)); # scalar x only
score(x::AbstractArray) = - (Δ' * dϕ.(Δ * x))


function sampler( ;
    score::Function = score,
    N::Dims = N,
    T::Int = 600,
    α0::Real = 1,
    p0::Real = 0.99,
    alpha::AbstractVector = (@. α0 * (p0 ^ (1:T))^2),
    beta::Real = 1,
    seed::Int = 0,
    init::Any = zeros(N),
)

    seed!(seed)
    xrun = Array{Float32}(undef, N..., T)
    xrun[:,:,1] = init

    for it in 1:(T-1)
        old = xrun[:,:,it]
        αt = alpha[it]
        xrun[:,:,it+1] = old + αt * score(old) + sqrt(2*beta*αt) * randn(N)
    end
    return xrun
end;


if !@isdefined(xrun) || true
    T = 600
    xrun = sampler(; T)
end;

#=

#
prompt()
## todo savefig(psl, "gmm-prior-trace-$ntrace.pdf")


function gmm_hist(it::Int)
    ph = histogram(xrun[:,it];
        bins = -12:0.5:36, xaxis,
        label = "$ntrial generated samples", normalize = true,
        yaxis = (L"p(x)", (0, 0.17), 0:0.1:0.2),
        annotate = (-3, 0.14, "t = $it", :left),
    )
    plot!(ph, x -> pdf(mix)(x);
         linewidth=3, color=:black, label="GMM Distribution",
    )
    return ph
end

# Animate sampling process over time
if isinteractive()
    ph = gmm_hist(T)
else
    anim = @animate for it in [1:10; 20:10:100; 200:100:T]
        ph = gmm_hist(it)
    ## tmp = @sprintf("%03d", it)
    ## savefig(ph, "gmm-prior-sample-$ntrial,$tmp.pdf")
    end
    gif(anim, "gmm-hist.gif", fps = 6)
end

#
prompt()
## savefig(ph, "gmm-prior-sample-$ntrial.pdf")


# Kernel density estimate and its score function
ntrain = 200
train_data = rand(mix, ntrain)
gsig = 0.9

kde = MixtureModel(Normal, [(x, gsig) for x in train_data])
pkd = deepcopy(pmp)
plot!(pkd, pdf(kde); xaxis, label="KDE, σ=$gsig", widen=true, color=:green)
scatter!(pkd, train_data, zeros(ntrain), label="data, N=$ntrain", color=:black)

#
prompt()
## savefig(pkd, "gmm-kde-pdf-$ntrain.pdf")

pks = deepcopy(ps1)
plot!(pks, score(kde), label="KDE score, σ=$gsig", color=:green)

#
prompt()
## savefig(pks, "gmm-kde-score-$ntrain.pdf")

## plot(pkd, pks; layout=(2,1))
=#
