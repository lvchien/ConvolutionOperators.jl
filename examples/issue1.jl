using ConvolutionOperators
# using Test

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

using LinearMaps
A = LinearMap(rand(2,2))
B = LinearMap(rand(2,2))

ABcv = A * B * cv
Z0 = zeros(2,2)
ConvolutionOperators.timeslice!(Z0, ABcv, 1)