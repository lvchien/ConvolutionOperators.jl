using LinearAlgebra

```
    Build the companion matrix for Z that has no tail
```
function companion_no_tail(Z)
    M, N = size(Z)[1:2]
    T = eltype(Z)
    @assert M == N

    K = size(Z,3) - 1
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
    
    return C
end

```
    Build the companion matrix for Z that has a tail. The `ranktail` specifies the dimension of the range of tail. SVD does not require the tail to be symmetric
```
function companion_with_tail(Z; ranktail)
    M, N = size(Z)[1:2]
    T = eltype(Z)
    @assert M == N
    @assert ranktail <= M
    Mt = min(ranktail, M)

    K = ConvolutionOperators.tailindex(Z) 
    @assert K > 1

    C = zeros(T, M*(K-2)+Mt, N*(K-2)+Mt)

    Id = Matrix{T}(I, M, N)
    Ztail = timeslice(Z, K)
    if Mt < M
        SVD = svd(Ztail)
        S = zeros(T, M, Mt)
        for i in 1:Mt
            S[i, i] = SVD.S[i]
        end
        LD = SVD.U*S
        RD = SVD.Vt[1:Mt, :]
    else
        LD = Ztail
        RD = Id
    end

    for m in 2:K-2
        n = m-1
        C[(m-1)*M+1:m*M, (n-1)*N+1:n*N] = Id
    end
    C[(K-2)*M+1:(K-2)*M+Mt, (K-3)*N+1:(K-2)*N] = RD

    W = -inv(timeslice(Z,1))
    for n in 1:K-2
        m = 1
        C[(m-1)*M+1:m*M, (n-1)*N+1:n*N] = W*timeslice(Z,n+1)
    end
    C[1:M, (K-2)*N+1:(K-2)*N+Mt] = W*LD

    Idt = Matrix{T}(I, Mt, Mt)
    C[(K-2)*M+1:(K-2)*M+Mt, (K-2)*N+1:(K-2)*N+Mt] = Idt      

    return C
end

function polyvals(Z; ranktail=size(Z,1))
    if hastail(Z)
        C = companion_with_tail(Z; ranktail)
    else
        C = companion_no_tail(Z)
    end
    @show size(C)
    return eigvals(C)
end

function polyeig(Z; ranktail=size(Z,1))
    if hastail(Z)
        C = companion_with_tail(Z; ranktail)
    else
        C = companion_no_tail(Z)
    end
    @show size(C)
    w, V = eigen(C)
    M = size(Z,1)
    return w, V[1:M,:]
end