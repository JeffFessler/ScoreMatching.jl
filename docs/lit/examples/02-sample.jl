#=
# [Sampling](@id 02-sample)

WIP on sampling using a score model.
=#

#srcURL

# ### Setup

# Packages needed here.

#src using ScoreMatching
using MIRTjim: jim, prompt
using Distributions: Distribution, Normal, MixtureModel, logpdf, pdf
import Distributions # var, mean
import Distributions: logpdf, pdf
import ForwardDiff # derivative, gradient
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

Sampling from a gaussian mixture distribution.

=#

# Some convenience methods
logpdf(d::Distribution) = x -> logpdf(d, x)
pdf(d::Distribution) = x -> pdf(d, x)
derivative(f::Function) = x -> ForwardDiff.derivative(f, x)
gradient(f::Function) = x -> ForwardDiff.gradient(f, x)
## hessian(f::Function) = x -> ForwardDiff.hessian(f, x)
score(d::Distribution) = derivative(logpdf(d))
score_deriv(d::Distribution) = derivative(score(d)); # scalar x only


mix = MixtureModel(Normal, [(3,1), (13,3)], [0.4, 0.6])

left_margin = 20px; bottom_margin = 10px
xaxis = (L"x", (-4,24), [0, 3, 13, 20])
pmp = plot(pdf(mix); label="Gaussian mixture pdf", color = :blue,
 left_margin, bottom_margin, xaxis, size=(600,300),
 yaxis = (L"p(x)", (0, 0.17), (0:3)*0.05),
)

score1 = score(mix)

ylabel_score1 = L"s(x) = \frac{\mathrm{d}}{\mathrm{d}x} \, \log \ p(x)"
ps1 = plot(score1; xaxis, color=:magenta,
 size=(600,300), label = "GMM score function",
 yaxis = (ylabel_score1, (-5,5), -4:2:4), left_margin, bottom_margin,
)

pps = plot(pmp, ps1, layout=(2,1))

#
prompt()


function sampler( ;
    score::Function = score1,
    T::Int = 600,
    α0::Real = 1,
    p0::Real = 0.99,
    alpha::AbstractVector = (@. α0 * (p0 ^ (1:T))^2),
    ntrial::Int = 1000, # for ph
    beta::Real = 1,
    seed::Int = 0,
    init_mean::Real = Distributions.mean(mix), # todo: cheating?
    init_std::Real = sqrt(Distributions.var(mix)),
)

    seed!(seed)
    xrun = Matrix{Float32}(undef, ntrial, T+1)
    xrun[:,1] = init_mean .+ init_std * randn(ntrial)

    for it in 1:T
        old = xrun[:,it]
        αt = alpha[it]
        xrun[:,it+1] = old + αt * score.(old) + sqrt(2*beta*αt) * randn(ntrial)
    end
    return xrun
end;


if !@isdefined(xrun) || true
    T = 600
    ntrial = 5000
    xrun = sampler(; T, ntrial)
end;

ntrace = 50
psl = plot(xrun[1:ntrace,:]', xlabel="Iteration (t)",
 xticks = 0:100:T,
 yaxis = (L"x_t", (-4,26), [0, 3, 13, 20]),
 annotate = (T, -2, "$ntrace generated samples", :right),
)

#
prompt()
## savefig(psl, "gmm-prior-trace-$ntrace.pdf")


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
