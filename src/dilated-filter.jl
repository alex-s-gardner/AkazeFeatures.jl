import ImageFiltering
using StaticArrays: SVector

"""
    DilatedSeparable(k1, k2, r)

A separable 2d kernel whose factors are each 3-tap but *dilated*: the taps sit at
offsets `-r`, `0`, `+r` rather than `-1, 0, 1`. `k1` applies along the first
(row) dimension and `k2` along the second (column) dimension.

A dense length-`2r+1` factor with only three nonzero entries costs a general
convolution `2r+1` multiply-adds per pixel where three suffice; this type skips
the zeros. Filter with it through `ImageFiltering.imfilter!`.
"""
struct DilatedSeparable{T<:Real}
    k1::SVector{3,T}
    k2::SVector{3,T}
    r::Int
end

"""
    imfilter!(dst, src, kern::DilatedSeparable; scratch = similar(src))

Filter `src` by `kern` into `dst`, using `scratch` (same axes as `src`) to hold
the intermediate pass. Borders replicate the edge value. Pass a reused `scratch`
on hot paths to keep the call allocation-free.

`dst`, `src`, and `scratch` must be three distinct arrays: both passes read
neighbors of the element they are writing, so aliasing any two of them smears
already-written values into the result. The row factor is applied first, then the
column factor, matching the dense-kernel `ImageFiltering.imfilter!` ordering so
the two agree bit-for-bit.
"""
function ImageFiltering.imfilter!(dst::AbstractMatrix, src::AbstractMatrix,
                                  kern::DilatedSeparable; scratch = similar(src))
    Base.require_one_based_indexing(dst, src, scratch)
    axes(dst) == axes(src) ||
        throw(DimensionMismatch("dst must have the same axes as src: $(axes(dst)) vs $(axes(src))"))
    axes(scratch) == axes(src) ||
        throw(DimensionMismatch("scratch must have the same axes as src: $(axes(scratch)) vs $(axes(src))"))
    (dst === src || dst === scratch || scratch === src) &&
        throw(ArgumentError("dst, src, and scratch must be three distinct arrays"))
    dilated_filter_dim1!(scratch, src, kern.k1, kern.r)
    dilated_filter_dim2!(dst, scratch, kern.k2, kern.r)
end

function dilated_filter_dim1!(dst, src, taps, r) @inbounds begin
    rows = size(src, 1)
    tm, t0, tp = taps
    ## The border rows are peeled off so the interior loop needs no clamping and
    ## can vectorize as three contiguous streams; clamping per element here
    ## (unlike in dim2!, where it hoists out of the inner loop) costs ~3x.
    tap(j, k) = (tm * src[max(j-r, 1), k] + t0 * src[j, k]) + tp * src[min(j+r, rows), k]
    for k in axes(src, 2)
        for j in 1:min(r, rows)
            dst[j, k] = tap(j, k)
        end
        @simd for j in r+1:rows-r
            dst[j, k] = (tm * src[j-r, k] + t0 * src[j, k]) + tp * src[j+r, k]
        end
        for j in max(r+1, rows-r+1):rows
            dst[j, k] = tap(j, k)
        end
    end
    dst
end end

function dilated_filter_dim2!(dst, src, taps, r) @inbounds begin
    cols = size(src, 2)
    tm, t0, tp = taps
    for k in axes(src, 2)
        km = ifelse(k > r, k - r, 1)
        kp = ifelse(k + r <= cols, k + r, cols)
        @simd for j in axes(src, 1)
            dst[j, k] = (tm * src[j, km] + t0 * src[j, k]) + tp * src[j, kp]
        end
    end
    dst
end end
