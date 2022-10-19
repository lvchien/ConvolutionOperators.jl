macro linearspace(T)

    F = :(Any)
    LC = Symbol(T, :(_LC))
    TI = Symbol(T, :(_TI))
    xp = quote
        struct $(LC)
            a::Vector{$F}
            v::Vector{$T}
        end

        Base.size(x::$LC) = size(first(x.v))
        Base.size(x::$LC,i) = size(x)[i]

        import Base: convert, promote_rule, +, *
        #convert(::Type{$LC}, u::$T) = $LC(ones($F,1),[u])
        convert(::Type{$LC}, u::$T) = $LC($F[1],[u])
        promote_rule(::Type{$T}, ::Type{$LC}) = $LC
        # Base.:+(x::Union{$LC,$T}, y::Union{$LC,$T}) = Base.:+(promote(x,y)...)
        Base.:+(x::$LC, y::$T) = Base.:+(x, convert($LC, y))
        Base.:+(x::$T, y::$LC) = Base.:+(convert($LC, x), y)
        Base.:+(x::$T, y::$T) = Base.:+(convert($LC,x), convert($LC,y))
        Base.:+(x::$LC, y::$LC) = $LC([x.a; y.a], [x.v; y.v])
        Base.:*(a::Number, x::$LC) = $LC(a*x.a, x.v)
        Base.:*(a::Number, x::$T) = a*convert($LC,x)

        struct $TI
            lc::$LC
        end

        start(ti::$TI) = 1
        next(ti::$TI, st) = ((ti.lc.a[st],ti.lc.v[st]),st+1)
        done(ti::$TI,st) = (length(ti.lc.a) < st)
        Base.length(ti::$TI) = length(ti.lc.a)
        Base.iterate(ti::$TI, st=start(ti)) = done(ti, st) ? nothing : next(ti, st) 

        terms(lc::$LC) = ($TI)(lc)
        terms(x::$T) = ($TI)(convert($LC,x))

        lctype(lc::Type{$T}) = $LC
    end

    return esc(xp)
end

@linearspace AbstractConvOp

function convolve!(y, Z::AbstractConvOp_LC, x, X, j, k_start=1, k_stop=size(Z,3))

    
    for (a,cvo) in terms(Z)
        
        w = similar(y)
        fill!(w,0)

        convolve!(w, cvo, x, X, j, k_start, k_stop)
        y .+= a*w
    end

    return y
end
