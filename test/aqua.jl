using ScoreMatching: ScoreMatching
import Aqua
using Test: @testset

@testset "aqua" begin
    Aqua.test_all(ScoreMatching)
end
