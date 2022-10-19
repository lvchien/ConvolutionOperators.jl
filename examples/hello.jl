using ConvolutionOperators
using Test

k0 = [
    1 2
    3 1
]

k1 = [
    1 3
    3 2
]

T = Float64
tail = T[
    1011 1012
    1021 1022
]

bandwidth = maximum(k1 .- k0 .+ 1)
data = zeros(T, 2, 2, bandwidth)
data[:,1,1] .= T[111, 0]
data[:,1,2] .= T[212, 312]
data[:,2,1] .= T[321, 0]
data[:,2,2] .= T[122, 222]

ln = 5
cv = ConvolutionOperators.ConvOp(data, k0, k1, tail, ln)

@test size(cv) == (2,2,ln)

import Base: OneTo
@test axes(cv) == (OneTo(2), OneTo(2), OneTo(ln))

Z1 = [cv[m,n,1] for m in 1:size(cv)[1], n in 1:size(cv)[2]]
Z2 = [cv[m,n,2] for m in 1:size(cv)[1], n in 1:size(cv)[2]]
Z3 = [cv[m,n,3] for m in 1:size(cv)[1], n in 1:size(cv)[2]]
Z4 = [cv[m,n,4] for m in 1:size(cv)[1], n in 1:size(cv)[2]]
Z5 = [cv[m,n,5] for m in 1:size(cv)[1], n in 1:size(cv)[2]]

@test Z1 == [111 0; 0 122]
@test Z2 == [1011 212; 0 222]
@test Z3 == [1011 312; 321 1022]
@test Z4 == [1011 1012; 1021 1022]
@test Z5 == Z4

lc = cv + 3*cv

A = rand(2,2)
B = rand(2,2)

lrc = A * cv * B

std = rand(2,7)
STD = cumsum(std, dims=2)
y1 = zeros(2)
y2 = zeros(2)
y3 = zeros(2)
y4 = zeros(2)

ConvolutionOperators.convolve!(y1, lrc, std, STD, 3)
A*ConvolutionOperators.convolve!(y2, cv, B*std, B*STD, 3)
ConvolutionOperators.convolve!(y3, lc, std, STD, 3)
4*ConvolutionOperators.convolve!(y4, cv, std, STD, 3)

rowaxis = Base.OneTo(3)
colaxis = Base.OneTo(4)

rowblock = 1:2
colblock = 3:4

lifted = ConvolutionOperators.LiftedConvOp(cv, rowaxis, colaxis, rowblock, colblock)
std2 = zeros(4,7)
std2[colblock,:] = std
STD2 = cumsum(std2, dims=2)
y5 = zeros(3)

ConvolutionOperators.convolve!(y5, lifted, std2, STD2, 3)