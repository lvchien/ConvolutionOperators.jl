function companion(Z)

    T = eltype(Z)
    K = size(Z,3)
    @assert K > 1

    M, N = size(Z)[1:2]
    C = similar(Z, M*(K-1), N*(K-1))
    fill!(C,0)

    @assert M == N
    Id = Matrix{T}(I, M, N)
    for m in 2:K-1
        n = m-1
        C[(m-1)*M+1:m*M, (n-1)*N+1:n*N] = Id
    end

    W = -inv(Z[:,:,1])
    for n in 1:K-1
        m = 1
        C[(m-1)*M+1:m*M, (n-1)*N+1:n*N] = W*Z[:,:,n+1]
    end

    return C
end


function timeslice(Z,k)
    T = eltype(Z)
    Zk = zeros(T, size(Z)[1:2])
    ConvolutionOperators.timeslice!(Zk, Z, k)
end


function materializeconvop(Z)
    M = size(Z,1)
    T = eltype(Z)
    
    if hastail(Z)
        kmax = ConvolutionOperators.tailindex(Z)
        Q = zeros(T, 2M, 2M, kmax+1)
        for k in 1:kmax
            Q[1:M,1:M,k] .= timeslice(Z,k)
        end
        Id = Matrix{T}(LinearAlgebra.I,M,M)
        Q[M+1:2M,1:M,1] .= -Id
        Q[M+1:2M,M+1:2M,1] .= Id
        Q[M+1:2M,M+1:2M,2] .= -Id
        Q[1:M,M+1:2M,kmax+1] .= timeslice(Z,kmax+1)
        # return eigvals(companion(Q))
    else
        Q = zeros(T, M, M, size(Z,3))
        for k in 1:size(Z,3)
            Q[:,:,k] = timeslice(Z,k)
        end
    end
    return Q
end

function polyvals(Z)
    Q = materializeconvop(Z)
    C = companion(Q)
    @show size(C)
    return eigvals(C)
end

function polyeig(Z)
    Q = materializeconvop(Z)
    C = companion(Q)
    @show size(C)
    w, V = eigen(C)
    M = size(Z,1)
    return w, V[1:M,:]
end