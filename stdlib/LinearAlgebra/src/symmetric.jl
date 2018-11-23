# This file is a part of Julia. License is MIT: https://julialang.org/license

const UPPER = Val(:U)
const LOWER = Val(:L)

# Symmetric and Hermitian matrices
struct Symmetric{T,S<:AbstractMatrix{<:T},U} <: AbstractMatrix{T}
    data::S

    function Symmetric{T,S,U}(data) where {T,S<:AbstractMatrix{<:T},U}
        @assert !has_offset_axes(data)
        new{T,S,U}(data)
    end
end
"""
    Symmetric(A, uplo={UPPER|LOWER})

Construct a `Symmetric` view of the upper (if `uplo = UPPER`) or lower (if `uplo = LOWER`)
triangle of the matrix `A`.

# Examples
```jldoctest
julia> A = [1 0 2 0 3; 0 4 0 5 0; 6 0 7 0 8; 0 9 0 1 0; 2 0 3 0 4]
5×5 Array{Int64,2}:
 1  0  2  0  3
 0  4  0  5  0
 6  0  7  0  8
 0  9  0  1  0
 2  0  3  0  4

julia> Supper = Symmetric(A)
5×5 Symmetric{Int64,Array{Int64,2}}:
 1  0  2  0  3
 0  4  0  5  0
 2  0  7  0  8
 0  5  0  1  0
 3  0  8  0  4

julia> Slower = Symmetric(A, LOWER)
5×5 Symmetric{Int64,Array{Int64,2}}:
 1  0  6  0  2
 0  4  0  9  0
 6  0  7  0  3
 0  9  0  1  0
 2  0  3  0  4
```

Note that `Supper` will not be equal to `Slower` unless `A` is itself symmetric (e.g. if `A == transpose(A)`).
"""
Symmetric(A::AbstractMatrix, uplo::Symbol) = Symmetric(A, Val(uplo))
function Symmetric(A::AbstractMatrix, uplo::Val=UPPER)
    checksquare(A)
    return symmetric_type(typeof(A), uplo)(A)
end

"""
    symmetric(A, uplo={UPPER|LOWER})

Construct a symmetric view of `A`. If `A` is a matrix, `uplo` controls whether the upper
(if `uplo = UPPER`) or lower (if `uplo = LOWER`) triangle of `A` is used to implicitly
fill the other one. If `A` is a `Number`, it is returned as is.

If a symmetric view of a matrix is to be constructed of which the elements are neither
matrices nor numbers, an appropriate method of `symmetric` has to be implemented. In that
case, `symmetric_type` has to be implemented, too.
"""
symmetric(A::AbstractMatrix, uplo::Val=UPPER) = Symmetric(A, uplo)
symmetric(A::Number, ::Val) = A

"""
    symmetric_type(T::Type, uplo)

The type of the object returned by `symmetric(::T, ::Val)`. For matrices, this is an
appropriately typed `Symmetric`, for `Number`s, it is the original type. If `symmetric` is
implemented for a custom type, so should be `symmetric_t ype`, and vice versa.
"""
function symmetric_type(::Type{T}, uplo::Val{U}) where {S, T<:AbstractMatrix{S},U}
    return Symmetric{Union{S, promote_op(transpose, S), symmetric_type(S, uplo)}, T, U}
end
symmetric_type(::Type{T}, ::Val) where {T<:Number} = T

struct Hermitian{T,S<:AbstractMatrix{<:T},U} <: AbstractMatrix{T}
    data::S

    function Hermitian{T,S,U}(data) where {T,S<:AbstractMatrix{<:T},U}
        @assert !has_offset_axes(data)
        new{T,S,U}(data)
    end
end
"""
    Hermitian(A, uplo=UPPER)

Construct a `Hermitian` view of the upper (if `uplo = :U`) or lower (if `uplo = :L`)
triangle of the matrix `A`.

# Examples
```jldoctest
julia> A = [1 0 2+2im 0 3-3im; 0 4 0 5 0; 6-6im 0 7 0 8+8im; 0 9 0 1 0; 2+2im 0 3-3im 0 4];

julia> Hupper = Hermitian(A)
5×5 Hermitian{Complex{Int64},Array{Complex{Int64},2}}:
 1+0im  0+0im  2+2im  0+0im  3-3im
 0+0im  4+0im  0+0im  5+0im  0+0im
 2-2im  0+0im  7+0im  0+0im  8+8im
 0+0im  5+0im  0+0im  1+0im  0+0im
 3+3im  0+0im  8-8im  0+0im  4+0im

julia> Hlower = Hermitian(A, LOWER)
5×5 Hermitian{Complex{Int64},Array{Complex{Int64},2}}:
 1+0im  0+0im  6+6im  0+0im  2-2im
 0+0im  4+0im  0+0im  9+0im  0+0im
 6-6im  0+0im  7+0im  0+0im  3+3im
 0+0im  9+0im  0+0im  1+0im  0+0im
 2+2im  0+0im  3-3im  0+0im  4+0im
```

Note that `Hupper` will not be equal to `Hlower` unless `A` is itself Hermitian (e.g. if `A == adjoint(A)`).

All non-real parts of the diagonal will be ignored.

```julia
Hermitian(fill(complex(1,1), 1, 1)) == fill(1, 1, 1)
```
"""
Hermitian(A::AbstractMatrix, uplo::Symbol) = Hermitian(A, Val(uplo))
function Hermitian(A::AbstractMatrix, uplo::Val=UPPER)
    n = checksquare(A)
    return hermitian_type(typeof(A), uplo)(A)
end

"""
    hermitian(A, uplo={UPPER|LOWER})

Construct a hermitian view of `A`. If `A` is a matrix, `uplo` controls whether the upper
(if `uplo = UPPER`) or lower (if `uplo = LOWER`) triangle of `A` is used to implicitly fill
the other one. If `A` is a `Number`, its real part is returned converted back to the input
type.

If a hermitian view of a matrix is to be constructed of which the elements are neither
matrices nor numbers, an appropriate method of `hermitian` has to be implemented. In that
case, `hermitian_type` has to be implemented, too.
"""
hermitian(A::AbstractMatrix, uplo::Val=UPPER) = Hermitian(A, uplo)
hermitian(A::Number, ::Val) = convert(typeof(A), real(A))

"""
    hermitian_type(T::Type, uplo)

The type of the object returned by `hermitian(::T, ::Symbol)`. For matrices, this is an
appropriately typed `Hermitian`, for `Number`s, it is the original type. If `hermitian` is
implemented for a custom type, so should be `hermitian_type`, and vice versa.
"""
function hermitian_type(::Type{T}, uplo::Val{U}) where {S,T<:AbstractMatrix{S},U}
    return Hermitian{Union{S, promote_op(adjoint, S), hermitian_type(S, uplo)}, T, U}
end
hermitian_type(::Type{T},::Val) where {T<:Number} = T

for (S, H) in ((:Symmetric, :Hermitian), (:Hermitian, :Symmetric))
    @eval begin
        $S(A::$S) = A
        function $S(A::$S{<:Any,<:Any,U}, ::Val{V}) where {U,V}
            if U == V
                return A
            else
                throw(ArgumentError("Cannot construct $($S); uplo doesn't match"))
            end
        end
        $S(A::$H{<:Any,<:Any,U}) where U = $S(A.data, U)
        function $S(A::$H{<:Any,<:Any,U}, ::Val{V}) where {U,V}
            if U == V
                return $S(A.data, Val(U))
            else
                throw(ArgumentError("Cannot construct $($S); uplo doesn't match"))
            end
        end
    end
end

convert(T::Type{<:Symmetric}, m::Union{Symmetric,Hermitian}) = m isa T ? m : T(m)
convert(T::Type{<:Hermitian}, m::Union{Symmetric,Hermitian}) = m isa T ? m : T(m)

const HermOrSym{T,S,U} = Union{Hermitian{T,S,U}, Symmetric{T,S,U}}
const RealHermSymComplexHerm{T<:Real,S,U} = Union{Hermitian{T,S,U}, Symmetric{T,S,U}, Hermitian{Complex{T},S,U}}
const RealHermSymComplexSym{T<:Real,S,U} = Union{Hermitian{T,S,U}, Symmetric{T,S,U}, Symmetric{Complex{T},S,U}}

Base.getproperty(A::HermOrSym{<:Any,<:Any,U}, f::Symbol) where U = f == :uplo ? char_uplo(U) : getfield(A, f)
Base.propertynames(A::HermOrSym) = (fieldnames(typeof(A))..., :uplo)

size(A::HermOrSym, d) = size(A.data, d)
size(A::HermOrSym) = size(A.data)
@inline function getindex(A::Symmetric{<:Any,<:Any,U}, i::Integer, j::Integer) where U
    @boundscheck checkbounds(A, i, j)
    @inbounds if i == j
        return symmetric(A.data[i, j], Val(U))::symmetric_type(eltype(A.data), Val(U))
    elseif (U == :U) == (i < j)
        return A.data[i, j]
    else
        return transpose(A.data[j, i])
    end
end
@inline function getindex(A::Hermitian{<:Any,<:Any,U}, i::Integer, j::Integer) where U
    @boundscheck checkbounds(A, i, j)
    @inbounds if i == j
        return hermitian(A.data[i, j], Val(U))::hermitian_type(eltype(A.data), Val(U))
    elseif (U == :U) == (i < j)
        return A.data[i, j]
    else
        return adjoint(A.data[j, i])
    end
end

function setindex!(A::Symmetric, v, i::Integer, j::Integer)
    i == j || throw(ArgumentError("Cannot set a non-diagonal index in a symmetric matrix"))
    setindex!(A.data, v, i, j)
end

function setindex!(A::Hermitian, v, i::Integer, j::Integer)
    if i != j
        throw(ArgumentError("Cannot set a non-diagonal index in a Hermitian matrix"))
    elseif !isreal(v)
        throw(ArgumentError("Cannot set a diagonal entry in a Hermitian matrix to a nonreal value"))
    else
        setindex!(A.data, v, i, j)
    end
end

# For A<:Union{Symmetric,Hermitian}, similar(A[, neweltype]) should yield a matrix with the same
# symmetry type, uplo flag, and underlying storage type as A. The following methods cover these cases.
similar(A::Symmetric{<:Any,<:Any,U}, ::Type{T}) where {T,U} = Symmetric(similar(parent(A), T), U)
# If the the Hermitian constructor's check ascertaining that the wrapped matrix's
# diagonal is strictly real is removed, the following method can be simplified.
function similar(A::Hermitian{<:Any,<:Any,U}, ::Type{T}) where {T,U}
    B = similar(parent(A), T)
    for i in 1:size(B, 1) B[i, i] = 0 end
    return Hermitian(B, U)
end
# On the other hand, similar(A, [neweltype,] shape...) should yield a matrix of the underlying
# storage type of A (not wrapped in a symmetry type). The following method covers these cases.
similar(A::Union{Symmetric,Hermitian}, ::Type{T}, dims::Dims{N}) where {T,N} = similar(parent(A), T, dims)

# Conversion
function Matrix(A::Symmetric{<:Any,<:Any,U}) where U
    B = copytri!(convert(Matrix, copy(A.data)), char_uplo(U))
    for i = 1:size(A, 1)
        B[i,i] = symmetric(B[i,i], Val(U))::symmetric_type(eltype(A.data), Val(U))
    end
    return B
end
function Matrix(A::Hermitian{<:Any,<:Any,U}) where U
    B = copytri!(convert(Matrix, copy(A.data)), char_uplo(U), true)
    for i = 1:size(A, 1)
        B[i,i] = hermitian(B[i,i], Val(U))::hermitian_type(eltype(A.data), Val(U))
    end
    return B
end
Array(A::Union{Symmetric,Hermitian}) = convert(Matrix, A)

parent(A::HermOrSym) = A.data
Symmetric{T,S,U}(A::Symmetric{T,S,U}) where {T,S<:AbstractMatrix{<:T},U} = A
Symmetric{T,S}(A::Symmetric{<:Any,<:Any,U}) where {T,S<:AbstractMatrix,U} = Symmetric{T,S,U}(convert(S,A.data))
AbstractMatrix{T}(A::Symmetric{<:Any,<:Any,U}) where {T,U} = Symmetric(convert(AbstractMatrix{T}, A.data), Val(U))
Hermitian{T,S,U}(A::Hermitian{T,S,U}) where {T,S<:AbstractMatrix{<:T},U} = A
Hermitian{T,S}(A::Hermitian{T,S}) where {T,S<:AbstractMatrix} = A
Hermitian{T,S}(A::Hermitian{<:Any,<:Any,U}) where {T,S<:AbstractMatrix,U} = Hermitian{T,S,U}(convert(S,A.data))
AbstractMatrix{T}(A::Hermitian{<:Any,<:Any,U}) where {T,U} = Hermitian(convert(AbstractMatrix{T}, A.data), Val(U))

copy(A::Symmetric{T,S,U}) where {T,S,U} = (B = copy(A.data); Symmetric{T,typeof(B),U}(B))
copy(A::Hermitian{T,S,U}) where {T,S,U} = (B = copy(A.data); Hermitian{T,typeof(B),U}(B))

function copyto!(dest::Symmetric{<:Any,<:Any,U}, src::Symmetric{<:Any,<:Any,U}) where U
        copyto!(dest.data, src.data)
        return dest
end
function copyto!(dest::Symmetric{<:Any,<:Any,U}, src::Symmetric{<:Any,<:Any,V}) where {U,V}
    transpose!(dest.data, src.data)
    return dest
end

function copyto!(dest::Hermitian{<:Any,<:Any,U}, src::Hermitian{<:Any,<:Any,U}) where U
        copyto!(dest.data, src.data)
        return dest
end
function copyto!(dest::Hermitian{<:Any,<:Any,U}, src::Hermitian{<:Any,<:Any,V}) where {U,V}
    transpose!(dest.data, src.data)
    return dest
end

# fill[stored]!
fill!(A::HermOrSym, x) = fillstored!(A, x)
function fillstored!(A::HermOrSym{T,<:Any,U}, x) where {T,U}
    xT = convert(T, x)
    if isa(A, Hermitian)
        isreal(xT) || throw(ArgumentError("cannot fill Hermitian matrix with a nonreal value"))
    end
    if U == :U
        fillband!(A.data, xT, 0, size(A,2)-1)
    else # U == :L
        fillband!(A.data, xT, 1-size(A,1), 0)
    end
    return A
end

function Base.isreal(A::HermOrSym{<:Any,<:Any,:U})
    n = size(A, 1)
    @inbounds for j in 1:n
        for i in 1:(j - (A isa Hermitian))
            if !isreal(A.data[i,j])
                return false
            end
        end
    end
    return true
end
function Base.isreal(A::HermOrSym{<:Any,<:Any,:L})
    n = size(A, 1)
    @inbounds for j in 1:n
        for i in (j + (A isa Hermitian)):n
            if !isreal(A.data[i,j])
                return false
            end
        end
    end
    return true
end

ishermitian(A::Hermitian) = true
ishermitian(A::Symmetric{<:Real}) = true
ishermitian(A::Symmetric{<:Complex}) = isreal(A)
issymmetric(A::Hermitian{<:Real}) = true
issymmetric(A::Hermitian{<:Complex}) = isreal(A)
issymmetric(A::Symmetric) = true

adjoint(A::Hermitian) = A
transpose(A::Symmetric) = A
adjoint(A::Symmetric{<:Real}) = A
transpose(A::Hermitian{<:Real}) = A
adjoint(A::Symmetric) = Adjoint(A)
transpose(A::Hermitian) = Transpose(A)

Base.copy(A::Adjoint{<:Any,<:Hermitian}) = copy(A.parent)
Base.copy(A::Transpose{<:Any,<:Symmetric}) = copy(A.parent)
Base.copy(A::Adjoint{<:Any,<:Symmetric{<:Any,<:Any,U}}) where U =
    Symmetric(copy(adjoint(A.parent.data)), U)
Base.collect(A::Transpose{<:Any,<:Hermitian{<:Any,<:Any,U}}) where U =
    Hermitian(copy(transpose(A.parent.data)), U)

tr(A::Hermitian) = real(tr(A.data))

Base.conj(A::HermOrSym) = typeof(A)(conj(A.data))
Base.conj!(A::HermOrSym) = typeof(A)(conj!(A.data))

# tril/triu
function tril(A::Hermitian{<:Any,<:Any,:U}, k::Integer=0)
    if k <= 0
        return tril!(copy(A.data'),k)
    else
        return tril!(copy(A.data'),-1) + tril!(triu(A.data),k)
    end
end
function tril(A::Hermitian{<:Any,<:Any,:L}, k::Integer=0)
    if k <= 0
        return tril(A.data,k)
    else
        return tril(A.data,-1) + tril!(triu!(copy(A.data')),k)
    end
end

function tril(A::Symmetric{<:Any,<:Any,:U}, k::Integer=0)
    if k <= 0
        return tril!(copy(transpose(A.data)),k)
    else
        return tril!(copy(transpose(A.data)),-1) + tril!(triu(A.data),k)
    end
end
function tril(A::Symmetric{<:Any,<:Any,:L}, k::Integer=0)
    if k <= 0
        return tril(A.data,k)
    else
        return tril(A.data,-1) + tril!(triu!(copy(transpose(A.data))),k)
    end
end

function triu(A::Hermitian{<:Any,<:Any,:U}, k::Integer=0)
    if k >= 0
        return triu(A.data,k)
    else
        return triu(A.data,1) + triu!(tril!(copy(A.data')),k)
    end
end
function triu(A::Hermitian{<:Any,<:Any,:L}, k::Integer=0)
    if k >= 0
        return triu!(copy(A.data'),k)
    else
        return triu!(copy(A.data'),1) + triu!(tril(A.data),k)
    end
end

function triu(A::Symmetric{<:Any,<:Any,:U}, k::Integer=0)
    if k >= 0
        return triu(A.data,k)
    else
        return triu(A.data,1) + triu!(tril!(copy(transpose(A.data))),k)
    end
end
function triu(A::Symmetric{<:Any,<:Any,:L}, k::Integer=0)
    if k >= 0
        return triu!(copy(transpose(A.data)),k)
    else
        return triu!(copy(transpose(A.data)),1) + triu!(tril(A.data),k)
    end
end

(-)(A::Symmetric{Tv,S,U}) where {Tv,S,U} = Symmetric{Tv,S,U}(-A.data)
(-)(A::Hermitian{Tv,S,U}) where {Tv,S,U} = Hermitian{Tv,S,U}(-A.data)

## Matvec
mul!(y::StridedVector{T}, A::Symmetric{T,<:StridedMatrix,U}, x::StridedVector{T}) where {T<:BlasFloat,U} =
    BLAS.symv!(char_uplo(U), one(T), A.data, x, zero(T), y)
mul!(y::StridedVector{T}, A::Hermitian{T,<:StridedMatrix,U}, x::StridedVector{T}) where {T<:BlasReal,U} =
    BLAS.symv!(char_uplo(U), one(T), A.data, x, zero(T), y)
mul!(y::StridedVector{T}, A::Hermitian{T,<:StridedMatrix,U}, x::StridedVector{T}) where {T<:BlasComplex,U} =
    BLAS.hemv!(char_uplo(U), one(T), A.data, x, zero(T), y)
## Matmat
mul!(C::StridedMatrix{T}, A::Symmetric{T,<:StridedMatrix,U}, B::StridedMatrix{T}) where {T<:BlasFloat,U} =
    BLAS.symm!('L', char_uplo(U), one(T), A.data, B, zero(T), C)
mul!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::Symmetric{T,<:StridedMatrix,U}) where {T<:BlasFloat,U} =
    BLAS.symm!('R', char_uplo(U), one(T), B.data, A, zero(T), C)
mul!(C::StridedMatrix{T}, A::Hermitian{T,<:StridedMatrix,U}, B::StridedMatrix{T}) where {T<:BlasReal,U} =
    BLAS.symm!('L', char_uplo(U), one(T), A.data, B, zero(T), C)
mul!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::Hermitian{T,<:StridedMatrix,U}) where {T<:BlasReal,U} =
    BLAS.symm!('R', char_uplo(U), one(T), B.data, A, zero(T), C)
mul!(C::StridedMatrix{T}, A::Hermitian{T,<:StridedMatrix,U}, B::StridedMatrix{T}) where {T<:BlasComplex,U} =
    BLAS.hemm!('L', char_uplo(U), one(T), A.data, B, zero(T), C)
mul!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::Hermitian{T,<:StridedMatrix,U}) where {T<:BlasComplex,U} =
    BLAS.hemm!('R', char_uplo(U), one(T), B.data, A, zero(T), C)

*(A::HermOrSym, B::HermOrSym) = A * copyto!(similar(parent(B)), B)

# Fallbacks to avoid generic_matvecmul!/generic_matmatmul!
## Symmetric{<:Number} and Hermitian{<:Real} are invariant to transpose; peel off the t
*(transA::Transpose{<:Any,<:RealHermSymComplexSym}, B::AbstractVector) = transA.parent * B
*(transA::Transpose{<:Any,<:RealHermSymComplexSym}, B::AbstractMatrix) = transA.parent * B
*(A::AbstractMatrix, transB::Transpose{<:Any,<:RealHermSymComplexSym}) = A * transB.parent
## Hermitian{<:Number} and Symmetric{<:Real} are invariant to adjoint; peel off the c
*(adjA::Adjoint{<:Any,<:RealHermSymComplexHerm}, B::AbstractVector) = adjA.parent * B
*(adjA::Adjoint{<:Any,<:RealHermSymComplexHerm}, B::AbstractMatrix) = adjA.parent * B
*(A::AbstractMatrix, adjB::Adjoint{<:Any,<:RealHermSymComplexHerm}) = A * adjB.parent

# ambiguities with transposed AbstractMatrix methods in linalg/matmul.jl
*(transA::Transpose{<:Any,<:RealHermSymComplexSym}, transB::Transpose{<:Any,<:RealHermSymComplexSym}) = transA.parent * transB.parent
*(transA::Transpose{<:Any,<:RealHermSymComplexSym}, transB::Transpose{<:Any,<:RealHermSymComplexHerm}) = transA.parent * transB
*(transA::Transpose{<:Any,<:RealHermSymComplexHerm}, transB::Transpose{<:Any,<:RealHermSymComplexSym}) = transA * transB.parent
*(adjA::Adjoint{<:Any,<:RealHermSymComplexHerm}, adjB::Adjoint{<:Any,<:RealHermSymComplexHerm}) = adjA.parent * adjB.parent
*(adjA::Adjoint{<:Any,<:RealHermSymComplexSym}, adjB::Adjoint{<:Any,<:RealHermSymComplexHerm}) = adjA * adjB.parent
*(adjA::Adjoint{<:Any,<:RealHermSymComplexHerm}, adjB::Adjoint{<:Any,<:RealHermSymComplexSym}) = adjA.parent * adjB

# ambiguities with AbstractTriangular
*(transA::Transpose{<:Any,<:RealHermSymComplexSym}, B::AbstractTriangular) = transA.parent * B
*(A::AbstractTriangular, transB::Transpose{<:Any,<:RealHermSymComplexSym}) = A * transB.parent
*(adjA::Adjoint{<:Any,<:RealHermSymComplexHerm}, B::AbstractTriangular) = adjA.parent * B
*(A::AbstractTriangular, adjB::Adjoint{<:Any,<:RealHermSymComplexHerm}) = A * adjB.parent

# Scaling with Number
*(A::Symmetric{<:Any,<:Any,U}, x::Number) where U = Symmetric(A.data*x, U)
*(x::Number, A::Symmetric{<:Any,<:Any,U}) where U = Symmetric(x*A.data, U)
*(A::Hermitian{<:Any,<:Any,U}, x::Real) where U = Hermitian(A.data*x, U)
*(x::Real, A::Hermitian{<:Any,<:Any,U}) where U = Hermitian(x*A.data, U)
/(A::Symmetric{<:Any,<:Any,U}, x::Number) where U = Symmetric(A.data/x, U)
/(A::Hermitian{<:Any,<:Any,U}, x::Real) where U= Hermitian(A.data/x, U)

function factorize(A::HermOrSym{T}) where T
    TT = typeof(sqrt(oneunit(T)))
    if TT <: BlasFloat
        return bunchkaufman(A)
    else # fallback
        return lu(A)
    end
end

det(A::RealHermSymComplexHerm) = real(det(factorize(A)))
det(A::Symmetric{<:Real}) = det(factorize(A))
det(A::Symmetric) = det(factorize(A))

\(A::HermOrSym{<:Any,<:StridedMatrix}, B::AbstractVector) = \(factorize(A), B)
# Bunch-Kaufman solves can not utilize BLAS-3 for multiple right hand sides
# so using LU is faster for AbstractMatrix right hand side
\(A::HermOrSym{<:Any,<:StridedMatrix}, B::AbstractMatrix) = \(lu(A), B)

function _inv(A::HermOrSym{<:Any,<:Any,U}) where U
    n = checksquare(A)
    B = inv!(lu(A))
    conjugate = isa(A, Hermitian)
    # symmetrize
    if U == :U # add to upper triangle
        @inbounds for i = 1:n, j = i:n
            B[i,j] = conjugate ? (B[i,j] + conj(B[j,i])) / 2 : (B[i,j] + B[j,i]) / 2
        end
    else # U == :L, add to lower triangle
        @inbounds for i = 1:n, j = i:n
            B[j,i] = conjugate ? (B[j,i] + conj(B[i,j])) / 2 : (B[j,i] + B[i,j]) / 2
        end
    end
    B
end
inv(A::Hermitian{<:Any,<:StridedMatrix,U}) where U = Hermitian(_inv(A), U)
inv(A::Symmetric{<:Any,<:StridedMatrix,U}) where U = Symmetric(_inv(A), U)

eigen!(A::RealHermSymComplexHerm{<:BlasReal,<:StridedMatrix,U}) where U =
    Eigen(LAPACK.syevr!('V', 'A', char_uplo(U), A.data, 0.0, 0.0, 0, 0, -1.0)...)

function eigen(A::RealHermSymComplexHerm)
    T = eltype(A)
    S = eigtype(T)
    eigen!(S != T ? convert(AbstractMatrix{S}, A) : copy(A))
end

eigen!(A::RealHermSymComplexHerm{<:BlasReal,<:StridedMatrix,U}, irange::UnitRange) where U =
    Eigen(LAPACK.syevr!('V', 'I', char_uplo(U), A.data, 0.0, 0.0, irange.start, irange.stop, -1.0)...)

"""
    eigen(A::Union{SymTridiagonal, Hermitian, Symmetric}, irange::UnitRange) -> Eigen

Computes the eigenvalue decomposition of `A`, returning an `Eigen` factorization object `F`
which contains the eigenvalues in `F.values` and the eigenvectors in the columns of the
matrix `F.vectors`. (The `k`th eigenvector can be obtained from the slice `F.vectors[:, k]`.)

Iterating the decomposition produces the components `F.values` and `F.vectors`.

The following functions are available for `Eigen` objects: [`inv`](@ref), [`det`](@ref), and [`isposdef`](@ref).

The `UnitRange` `irange` specifies indices of the sorted eigenvalues to search for.

!!! note
    If `irange` is not `1:n`, where `n` is the dimension of `A`, then the returned factorization
    will be a *truncated* factorization.
"""
function eigen(A::RealHermSymComplexHerm, irange::UnitRange)
    T = eltype(A)
    S = eigtype(T)
    eigen!(S != T ? convert(AbstractMatrix{S}, A) : copy(A), irange)
end

eigen!(A::RealHermSymComplexHerm{T,<:StridedMatrix,U}, vl::Real, vh::Real) where {T<:BlasReal,U} =
    Eigen(LAPACK.syevr!('V', 'V', char_uplo(U), A.data, convert(T, vl), convert(T, vh), 0, 0, -1.0)...)

"""
    eigen(A::Union{SymTridiagonal, Hermitian, Symmetric}, vl::Real, vu::Real) -> Eigen

Computes the eigenvalue decomposition of `A`, returning an `Eigen` factorization object `F`
which contains the eigenvalues in `F.values` and the eigenvectors in the columns of the
matrix `F.vectors`. (The `k`th eigenvector can be obtained from the slice `F.vectors[:, k]`.)

Iterating the decomposition produces the components `F.values` and `F.vectors`.

The following functions are available for `Eigen` objects: [`inv`](@ref), [`det`](@ref), and [`isposdef`](@ref).

`vl` is the lower bound of the window of eigenvalues to search for, and `vu` is the upper bound.

!!! note
    If [`vl`, `vu`] does not contain all eigenvalues of `A`, then the returned factorization
    will be a *truncated* factorization.
"""
function eigen(A::RealHermSymComplexHerm, vl::Real, vh::Real)
    T = eltype(A)
    S = eigtype(T)
    eigen!(S != T ? convert(AbstractMatrix{S}, A) : copy(A), vl, vh)
end

eigvals!(A::RealHermSymComplexHerm{<:BlasReal,<:StridedMatrix,U}) where U =
    LAPACK.syevr!('N', 'A', char_uplo(U), A.data, 0.0, 0.0, 0, 0, -1.0)[1]

function eigvals(A::RealHermSymComplexHerm)
    T = eltype(A)
    S = eigtype(T)
    eigvals!(S != T ? convert(AbstractMatrix{S}, A) : copy(A))
end

"""
    eigvals!(A::Union{SymTridiagonal, Hermitian, Symmetric}, irange::UnitRange) -> values

Same as [`eigvals`](@ref), but saves space by overwriting the input `A`, instead of creating a copy.
`irange` is a range of eigenvalue *indices* to search for - for instance, the 2nd to 8th eigenvalues.
"""
eigvals!(A::RealHermSymComplexHerm{<:BlasReal,<:StridedMatrix,U}, irange::UnitRange) where U =
    LAPACK.syevr!('N', 'I', char_uplo(U), A.data, 0.0, 0.0, irange.start, irange.stop, -1.0)[1]

"""
    eigvals(A::Union{SymTridiagonal, Hermitian, Symmetric}, irange::UnitRange) -> values

Returns the eigenvalues of `A`. It is possible to calculate only a subset of the
eigenvalues by specifying a `UnitRange` `irange` covering indices of the sorted eigenvalues,
e.g. the 2nd to 8th eigenvalues.

```jldoctest
julia> A = SymTridiagonal([1.; 2.; 1.], [2.; 3.])
3×3 SymTridiagonal{Float64,Array{Float64,1}}:
 1.0  2.0   ⋅
 2.0  2.0  3.0
  ⋅   3.0  1.0

julia> eigvals(A, 2:2)
1-element Array{Float64,1}:
 0.9999999999999996

julia> eigvals(A)
3-element Array{Float64,1}:
 -2.1400549446402604
  1.0000000000000002
  5.140054944640259
```
"""
function eigvals(A::RealHermSymComplexHerm, irange::UnitRange)
    T = eltype(A)
    S = eigtype(T)
    eigvals!(S != T ? convert(AbstractMatrix{S}, A) : copy(A), irange)
end

"""
    eigvals!(A::Union{SymTridiagonal, Hermitian, Symmetric}, vl::Real, vu::Real) -> values

Same as [`eigvals`](@ref), but saves space by overwriting the input `A`, instead of creating a copy.
`vl` is the lower bound of the interval to search for eigenvalues, and `vu` is the upper bound.
"""
eigvals!(A::RealHermSymComplexHerm{T,<:StridedMatrix,U}, vl::Real, vh::Real) where {T<:BlasReal,U} =
    LAPACK.syevr!('N', 'V', char_uplo(U), A.data, convert(T, vl), convert(T, vh), 0, 0, -1.0)[1]

"""
    eigvals(A::Union{SymTridiagonal, Hermitian, Symmetric}, vl::Real, vu::Real) -> values

Returns the eigenvalues of `A`. It is possible to calculate only a subset of the eigenvalues
by specifying a pair `vl` and `vu` for the lower and upper boundaries of the eigenvalues.

```jldoctest
julia> A = SymTridiagonal([1.; 2.; 1.], [2.; 3.])
3×3 SymTridiagonal{Float64,Array{Float64,1}}:
 1.0  2.0   ⋅
 2.0  2.0  3.0
  ⋅   3.0  1.0

julia> eigvals(A, -1, 2)
1-element Array{Float64,1}:
 1.0000000000000009

julia> eigvals(A)
3-element Array{Float64,1}:
 -2.1400549446402604
  1.0000000000000002
  5.140054944640259
```
"""
function eigvals(A::RealHermSymComplexHerm, vl::Real, vh::Real)
    T = eltype(A)
    S = eigtype(T)
    eigvals!(S != T ? convert(AbstractMatrix{S}, A) : copy(A), vl, vh)
end

eigmax(A::RealHermSymComplexHerm{<:Real,<:StridedMatrix}) = eigvals(A, size(A, 1):size(A, 1))[1]
eigmin(A::RealHermSymComplexHerm{<:Real,<:StridedMatrix}) = eigvals(A, 1:1)[1]

function eigen!(A::HermOrSym{T,S,U}, B::HermOrSym{T,S,U}) where {T<:BlasReal,S<:StridedMatrix,U}
    vals, vecs, _ = LAPACK.sygvd!(1, 'V', char_uplo(U), A.data, B.data)
    GeneralizedEigen(vals, vecs)
end
function eigen!(A::HermOrSym{T,S,U}, B::HermOrSym{T,S,V}) where {T<:BlasReal,S<:StridedMatrix,U,V}
    vals, vecs, _ = LAPACK.sygvd!(1, 'V', char_uplo(U), A.data, copy(B.data'))
    GeneralizedEigen(vals, vecs)
end
function eigen!(A::Hermitian{T,S,U}, B::Hermitian{T,S,U}) where {T<:BlasComplex,S<:StridedMatrix,U}
    vals, vecs, _ = LAPACK.sygvd!(1, 'V', char_uplo(U), A.data, B.data)
    GeneralizedEigen(vals, vecs)
end
function eigen!(A::Hermitian{T,S,U}, B::Hermitian{T,S,V}) where {T<:BlasComplex,S<:StridedMatrix,U,V}
    vals, vecs, _ = LAPACK.sygvd!(1, 'V', char_uplo(U), A.data, copy(B.data'))
    GeneralizedEigen(vals, vecs)
end

eigvals!(A::HermOrSym{T,S,U}, B::HermOrSym{T,S,U}) where {T<:BlasReal,S<:StridedMatrix,U} =
    LAPACK.sygvd!(1, 'N', char_uplo(U), A.data, B.data)[1]
eigvals!(A::HermOrSym{T,S,U}, B::HermOrSym{T,S,V}) where {T<:BlasReal,S<:StridedMatrix,U,V} =
    LAPACK.sygvd!(1, 'N', char_uplo(U), A.data, copy(B.data'))[1]
eigvals!(A::Hermitian{T,S,U}, B::Hermitian{T,S,U}) where {T<:BlasComplex,S<:StridedMatrix,U} =
    LAPACK.sygvd!(1, 'N', char_uplo(U), A.data, B.data)[1]
eigvals!(A::Hermitian{T,S,U}, B::Hermitian{T,S,V}) where {T<:BlasComplex,S<:StridedMatrix,U,V} =
    LAPACK.sygvd!(1, 'N', char_uplo(U), A.data, copy(B.data'))[1]

eigvecs(A::HermOrSym) = eigvecs(eigen(A))

function svdvals!(A::RealHermSymComplexHerm)
    vals = eigvals!(A)
    for i = 1:length(vals)
        vals[i] = abs(vals[i])
    end
    return sort!(vals, rev = true)
end

# Matrix functions
^(A::Symmetric{<:Real}, p::Integer) = sympow(A, p)
^(A::Symmetric{<:Complex}, p::Integer) = sympow(A, p)
function sympow(A::Symmetric, p::Integer)
    if p < 0
        return Symmetric(Base.power_by_squaring(inv(A), -p))
    else
        return Symmetric(Base.power_by_squaring(A, p))
    end
end
function ^(A::Symmetric{<:Real}, p::Real)
    isinteger(p) && return integerpow(A, p)
    F = eigen(A)
    if all(λ -> λ ≥ 0, F.values)
        return Symmetric((F.vectors * Diagonal((F.values).^p)) * F.vectors')
    else
        return Symmetric((F.vectors * Diagonal((complex(F.values)).^p)) * F.vectors')
    end
end
function ^(A::Symmetric{<:Complex}, p::Real)
    isinteger(p) && return integerpow(A, p)
    return Symmetric(schurpow(A, p))
end
function ^(A::Hermitian, p::Integer)
    if p < 0
        retmat = Base.power_by_squaring(inv(A), -p)
    else
        retmat = Base.power_by_squaring(A, p)
    end
    for i = 1:size(A,1)
        retmat[i,i] = real(retmat[i,i])
    end
    return Hermitian(retmat)
end
function ^(A::Hermitian{T}, p::Real) where T
    isinteger(p) && return integerpow(A, p)
    F = eigen(A)
    if all(λ -> λ ≥ 0, F.values)
        retmat = (F.vectors * Diagonal((F.values).^p)) * F.vectors'
        if T <: Real
            return Hermitian(retmat)
        else
            for i = 1:size(A,1)
                retmat[i,i] = real(retmat[i,i])
            end
            return Hermitian(retmat)
        end
    else
        return (F.vectors * Diagonal((complex(F.values).^p))) * F.vectors'
    end
end

for func in (:exp, :cos, :sin, :tan, :cosh, :sinh, :tanh, :atan, :asinh, :atanh)
    @eval begin
        function ($func)(A::HermOrSym{<:Real})
            F = eigen(A)
            return Symmetric((F.vectors * Diagonal(($func).(F.values))) * F.vectors')
        end
        function ($func)(A::Hermitian{<:Complex})
            n = checksquare(A)
            F = eigen(A)
            retmat = (F.vectors * Diagonal(($func).(F.values))) * F.vectors'
            for i = 1:n
                retmat[i,i] = real(retmat[i,i])
            end
            return Hermitian(retmat)
        end
    end
end

for func in (:acos, :asin)
    @eval begin
        function ($func)(A::HermOrSym{<:Real})
            F = eigen(A)
            if all(λ -> -1 ≤ λ ≤ 1, F.values)
                retmat = (F.vectors * Diagonal(($func).(F.values))) * F.vectors'
            else
                retmat = (F.vectors * Diagonal(($func).(complex.(F.values)))) * F.vectors'
            end
            return Symmetric(retmat)
        end
        function ($func)(A::Hermitian{<:Complex})
            n = checksquare(A)
            F = eigen(A)
            if all(λ -> -1 ≤ λ ≤ 1, F.values)
                retmat = (F.vectors * Diagonal(($func).(F.values))) * F.vectors'
                for i = 1:n
                    retmat[i,i] = real(retmat[i,i])
                end
                return Hermitian(retmat)
            else
                return (F.vectors * Diagonal(($func).(complex.(F.values)))) * F.vectors'
            end
        end
    end
end

function acosh(A::HermOrSym{<:Real})
    F = eigen(A)
    if all(λ -> λ ≥ 1, F.values)
        retmat = (F.vectors * Diagonal(acosh.(F.values))) * F.vectors'
    else
        retmat = (F.vectors * Diagonal(acosh.(complex.(F.values)))) * F.vectors'
    end
    return Symmetric(retmat)
end
function acosh(A::Hermitian{<:Complex})
    n = checksquare(A)
    F = eigen(A)
    if all(λ -> λ ≥ 1, F.values)
        retmat = (F.vectors * Diagonal(acosh.(F.values))) * F.vectors'
        for i = 1:n
            retmat[i,i] = real(retmat[i,i])
        end
        return Hermitian(retmat)
    else
        return (F.vectors * Diagonal(acosh.(complex.(F.values)))) * F.vectors'
    end
end

function sincos(A::HermOrSym{<:Real})
    n = checksquare(A)
    F = eigen(A)
    S, C = Diagonal(similar(A, (n,))), Diagonal(similar(A, (n,)))
    for i in 1:n
        S.diag[i], C.diag[i] = sincos(F.values[i])
    end
    return Symmetric((F.vectors * S) * F.vectors'), Symmetric((F.vectors * C) * F.vectors')
end
function sincos(A::Hermitian{<:Complex})
    n = checksquare(A)
    F = eigen(A)
    S, C = Diagonal(similar(A, (n,))), Diagonal(similar(A, (n,)))
    for i in 1:n
        S.diag[i], C.diag[i] = sincos(F.values[i])
    end
    retmatS, retmatC = (F.vectors * S) * F.vectors', (F.vectors * C) * F.vectors'
    for i = 1:n
        retmatS[i,i] = real(retmatS[i,i])
        retmatC[i,i] = real(retmatC[i,i])
    end
    return Hermitian(retmatS), Hermitian(retmatC)
end


for func in (:log, :sqrt)
    @eval begin
        function ($func)(A::HermOrSym{<:Real})
            F = eigen(A)
            if all(λ -> λ ≥ 0, F.values)
                retmat = (F.vectors * Diagonal(($func).(F.values))) * F.vectors'
            else
                retmat = (F.vectors * Diagonal(($func).(complex.(F.values)))) * F.vectors'
            end
            return Symmetric(retmat)
        end

        function ($func)(A::Hermitian{<:Complex})
            n = checksquare(A)
            F = eigen(A)
            if all(λ -> λ ≥ 0, F.values)
                retmat = (F.vectors * Diagonal(($func).(F.values))) * F.vectors'
                for i = 1:n
                    retmat[i,i] = real(retmat[i,i])
                end
                return Hermitian(retmat)
            else
                retmat = (F.vectors * Diagonal(($func).(complex(F.values)))) * F.vectors'
                return retmat
            end
        end
    end
end

# disambiguation methods: *(Adj of RealHermSymComplexHerm, Trans of RealHermSymComplexSym) and symmetric partner
*(A::Adjoint{<:Any,<:RealHermSymComplexHerm}, B::Transpose{<:Any,<:RealHermSymComplexSym}) = A.parent * B.parent
*(A::Transpose{<:Any,<:RealHermSymComplexSym}, B::Adjoint{<:Any,<:RealHermSymComplexHerm}) = A.parent * B.parent
# disambiguation methods: *(Adj/Trans of AbsVec/AbsMat, Adj/Trans of RealHermSymComplex{Herm|Sym})
*(A::Adjoint{<:Any,<:AbstractVector}, B::Adjoint{<:Any,<:RealHermSymComplexHerm}) = A * B.parent
*(A::Adjoint{<:Any,<:AbstractMatrix}, B::Adjoint{<:Any,<:RealHermSymComplexHerm}) = A * B.parent
*(A::Adjoint{<:Any,<:AbstractVector}, B::Transpose{<:Any,<:RealHermSymComplexSym}) = A * B.parent
*(A::Adjoint{<:Any,<:AbstractMatrix}, B::Transpose{<:Any,<:RealHermSymComplexSym}) = A * B.parent
*(A::Transpose{<:Any,<:AbstractVector}, B::Adjoint{<:Any,<:RealHermSymComplexHerm}) = A * B.parent
*(A::Transpose{<:Any,<:AbstractMatrix}, B::Adjoint{<:Any,<:RealHermSymComplexHerm}) = A * B.parent
*(A::Transpose{<:Any,<:AbstractVector}, B::Transpose{<:Any,<:RealHermSymComplexSym}) = A * B.parent
*(A::Transpose{<:Any,<:AbstractMatrix}, B::Transpose{<:Any,<:RealHermSymComplexSym}) = A * B.parent
# disambiguation methods: *(Adj/Trans of RealHermSymComplex{Herm|Sym}, Adj/Trans of AbsVec/AbsMat)
*(A::Adjoint{<:Any,<:RealHermSymComplexHerm}, B::Adjoint{<:Any,<:AbstractVector}) = A.parent * B
*(A::Adjoint{<:Any,<:RealHermSymComplexHerm}, B::Adjoint{<:Any,<:AbstractMatrix}) = A.parent * B
*(A::Adjoint{<:Any,<:RealHermSymComplexHerm}, B::Transpose{<:Any,<:AbstractVector}) = A.parent * B
*(A::Adjoint{<:Any,<:RealHermSymComplexHerm}, B::Transpose{<:Any,<:AbstractMatrix}) = A.parent * B
*(A::Transpose{<:Any,<:RealHermSymComplexSym}, B::Adjoint{<:Any,<:AbstractVector}) = A.parent * B
*(A::Transpose{<:Any,<:RealHermSymComplexSym}, B::Adjoint{<:Any,<:AbstractMatrix}) = A.parent * B
*(A::Transpose{<:Any,<:RealHermSymComplexSym}, B::Transpose{<:Any,<:AbstractVector}) = A.parent * B
*(A::Transpose{<:Any,<:RealHermSymComplexSym}, B::Transpose{<:Any,<:AbstractMatrix}) = A.parent * B

# disambiguation methods: *(Adj/Trans of AbsTri or RealHermSymComplex{Herm|Sym}, Adj/Trans of other)
*(A::Adjoint{<:Any,<:AbstractTriangular}, B::Adjoint{<:Any,<:RealHermSymComplexHerm}) = A * B.parent
*(A::Adjoint{<:Any,<:AbstractTriangular}, B::Transpose{<:Any,<:RealHermSymComplexSym}) = A * B.parent
*(A::Transpose{<:Any,<:AbstractTriangular}, B::Adjoint{<:Any,<:RealHermSymComplexHerm}) = A * B.parent
*(A::Transpose{<:Any,<:AbstractTriangular}, B::Transpose{<:Any,<:RealHermSymComplexSym}) = A * B.parent
*(A::Adjoint{<:Any,<:RealHermSymComplexHerm}, B::Adjoint{<:Any,<:AbstractTriangular}) = A.parent * B
*(A::Adjoint{<:Any,<:RealHermSymComplexHerm}, B::Transpose{<:Any,<:AbstractTriangular}) = A.parent * B
*(A::Transpose{<:Any,<:RealHermSymComplexSym}, B::Adjoint{<:Any,<:AbstractTriangular}) = A.parent * B
*(A::Transpose{<:Any,<:RealHermSymComplexSym}, B::Transpose{<:Any,<:AbstractTriangular}) = A.parent * B
