# using ConvolutionOperators
# using Test
using TestItemRunner

@testitem "size(bandedconvop)" begin
    include("setup.jl")
    @test size(cv) == (2,2,ln)
    @test axes(cv) == (Base.OneTo(2), Base.OneTo(2), Base.OneTo(ln))
end

@testitem "matrices" begin
    include("setup.jl")

    Z = map(k -> zeros(T,2,2), axes(cv,3))
    for k in axes(cv,3)
        ConvolutionOperators.timeslice!(Z[k], cv, k)
    end
    
    @test Z[1] == [111 0; 0 122]
    @test Z[2] == [1011 212; 0 222]
    @test Z[3] == [1011 312; 321 1022]
    @test Z[4] == [1011 1012; 1021 1022]
    @test Z[5] == Z[4]

    std = rand(2,7)
    STD = cumsum(std, dims=2)

    ln = 12
    cv = ConvolutionOperators.ConvOp(data, k0, k1, tail, ln)
    Z = map(k -> zeros(T,2,2), 1:12)
    for k in axes(cv,3)
        ConvolutionOperators.timeslice!(Z[k], cv, k)
    end

    for j in 1:7
        for kstart in 1:ln
            for kstop in kstart:ln
                y1 = zeros(2);
                ConvolutionOperators.convolve!(y1, cv, std, STD, j, kstart, kstop)
                y2 = zeros(2);
                for k in kstart:kstop
                    j - k + 1 > 0 || continue
                    y2 += Z[k]*std[:,j-k+1]
                end
                @test y1 ≈ y2
            end
        end
    end
end

@testitem "linearcombinations" begin
    include("setup.jl")
    lc = cv + 3*cv

    std = rand(2,7)
    STD = cumsum(std, dims=2)
    y1 = zeros(2)
    y2 = zeros(2)

    z1 = ConvolutionOperators.convolve!(y1, lc, std, STD, 3)
    z2 = 4*ConvolutionOperators.convolve!(y2, cv, std, STD, 3)

    @test z1 ≈ z2
end

@testitem "leftrightmult" begin
    include("setup.jl")

    A = rand(2,2)
    B = rand(2,2)
    lrc = A * cv * B

    std = rand(2,7)
    STD = cumsum(std, dims=2)
    y1 = zeros(2)
    y2 = zeros(2)

    z1 = ConvolutionOperators.convolve!(y1, lrc, std, STD, 3)
    z2 = A*ConvolutionOperators.convolve!(y2, cv, B*std, B*STD, 3)

    @test z1 ≈ z2
end

@testitem "liftedconvop" begin
    include("setup.jl")

    rowaxis = Base.OneTo(3)
    colaxis = Base.OneTo(4)

    rowblock = 1:2
    colblock = 3:4

    lifted = ConvolutionOperators.LiftedConvOp(cv, rowaxis, colaxis, rowblock, colblock)

    std1 = rand(2,7)
    std2 = zeros(4,7)
    std2[colblock,:] = std1

    STD1 = cumsum(std1, dims=2)
    STD2 = cumsum(std2, dims=2)

    y1 = zeros(2)
    y2 = zeros(3)

    y1 = ConvolutionOperators.convolve!(y1, cv, std1, STD1, 3)
    y2 = ConvolutionOperators.convolve!(y2, lifted, std2, STD2, 3)

    @test y1 ≈ y2[rowblock]
end

@testitem "truncatedconvop" begin
    include("setup.jl")
    @test size(cv) == (2,2,5)
    @test ConvolutionOperators.hastail(cv) == true
    tcv = ConvolutionOperators.truncate(cv,4)
    @test size(tcv) == (2,2,5)
    @test ConvolutionOperators.hastail(tcv) == false

    std = rand(2,12)
    STD = cumsum(std, dims=2)

    y1 = zeros(2); ConvolutionOperators.convolve!(y1, cv, std, STD, 12, 1, 4)
    y2 = zeros(2); ConvolutionOperators.convolve!(y2, cv, std, STD, 12, 1, 8)
    @show y1 ≈ y2

    y1 = zeros(2); ConvolutionOperators.convolve!(y1, cv, std, STD, 12, 2, 4)
    y2 = zeros(2); ConvolutionOperators.convolve!(y2, cv, std, STD, 12, 2, 8)
    @show y1 ≈ y2

    y1 = zeros(2); ConvolutionOperators.convolve!(y1, cv, std, STD, 12, 3, 4)
    y2 = zeros(2); ConvolutionOperators.convolve!(y2, cv, std, STD, 12, 3, 8)
    @show y1 ≈ y2

    y1 = zeros(2); ConvolutionOperators.convolve!(y1, cv, std, STD, 12, 4, 4)
    y2 = zeros(2); ConvolutionOperators.convolve!(y2, cv, std, STD, 12, 4, 8)
    @show y1 ≈ y2

end


@run_package_tests
