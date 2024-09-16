# TODO: remove the nullspace of the tail

function companion(Z)
    M, N = size(Z)[1:2]
    T = eltype(Z)
    @assert M == N

    if hastail(Z)
        K = ConvolutionOperators.tailindex(Z) 
    else
        K = size(Z,3) - 1
    end
    @assert K > 1

    C = zeros(T, M*(K-1), N*(K-1))
    Id = Matrix{T}(I, M, N)

    for m in 2:K-1
        n = m-1
        C[(m-1)*M+1:m*M, (n-1)*N+1:n*N] = Id
    end

    W = -inv(timeslice(Z,1))
    for n in 1:K-1
        m = 1
        C[(m-1)*M+1:m*M, (n-1)*N+1:n*N] = W*timeslice(Z,n+1)
    end
    
    if hastail(Z)
        m, n = K-1, K-1
        C[(m-1)*M+1:m*M, (n-1)*N+1:n*N] = Id      
    end

    return C
end

function polyvals(Z)
    C = companion(Z)
    @show size(C)
    return eigvals(C)
end

function polyeig(Z)
    C = companion(Q)
    @show size(C)
    w, V = eigen(C)
    M = size(Z,1)
    return w, V[1:M,:]
end