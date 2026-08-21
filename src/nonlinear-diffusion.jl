using ImageFiltering: Kernel, imfilter
using StaticArrays: SVector


# function nld_step_scalar(Ld, c, stepsize)
#     dx = 0.5 * stepsize * (c[1:end, 1:end-1] + c[1:end, 2:end]) .* (Ld[1:end, 2:end] - Ld[1:end, 1:end-1])
#     dy = 0.5 * stepsize * (c[1:end-1, 1:end] + c[2:end, 1:end]) .* (Ld[2:end, 1:end] - Ld[1:end-1, 1:end])

#     Ld[1:end, 1:end-1] .+= dx
#     Ld[1:end, 2:end] .-= dx
#     Ld[1:end-1, 1:end] .+= dy
#     Ld[2:end, 1:end] .-= dy
# end

function nld_step_scalar(Ld, c, stepsize, dx, dy) @inbounds begin
    rows::Int64=size(Ld,1)
    cols::Int64=size(Ld,2)

    # for k in 1:cols-1
    #     for j in 1:rows
    #         dx[j,k+1] = 0.5 * stepsize * (c[j,k] + c[j,k+1]) .* (Ld[j,k+1] - Ld[j,k])
    #     end
    # end
    # for k in 1:cols
    #     for j in 1:rows-1
    #         dy[j+1,k] = 0.5 * stepsize * (c[j,k] + c[j+1, k]) .* (Ld[j+1, k] - Ld[j,k])
    #     end
    # end

    for k in 1:cols-1
        for j in 1:rows-1
            cjk=c[j,k]
            Ldjk = Ld[j,k]
            dx[j,k+1] = 0.5 * stepsize * (cjk + c[j,k+1]) * (Ld[j,k+1] - Ldjk)
            dy[j+1,k] = 0.5 * stepsize * (cjk + c[j+1, k]) * (Ld[j+1, k] - Ldjk)
        end
    end
    j=rows
    for k in 1:cols-1
        dx[j,k+1] = 0.5 * stepsize * (c[j,k] + c[j,k+1]) .* (Ld[j,k+1] - Ld[j,k])
    end
    k=cols
    for j in 1:rows-1
        dy[j+1,k] = 0.5 * stepsize * (c[j,k] + c[j+1, k]) .* (Ld[j+1, k] - Ld[j,k])
    end

    for k in 1:cols
        for j in 1:rows
            Ld[j,k] += dx[j,k+1] - dx[j,k] + dy[j+1,k] - dy[j,k]
        end
    end
end end

pm_g1_diffusivity!(dst, Lx, Ly, k) =
    calculate_diffusivity′!(dst, Lx, Ly, k) do dL
        exp(-dL)
    end

pm_g2_diffusivity!(dst, Lx, Ly, k) =
    calculate_diffusivity′!(dst, Lx, Ly, k) do dL
        1.0 / (1.0 + dL)
    end

weickert_diffusivity!(dst, Lx, Ly, k) =
    calculate_diffusivity′!(dst, Lx, Ly, k) do dL
        1.0 - exp(-3.315 / (dL * dL * dL * dL))
    end

charbonnier_diffusivity!(dst, Lx, Ly, k) =
    calculate_diffusivity′!(dst, Lx, Ly, k) do dL
        1.0 / sqrt(1.0 + dL)
    end

pm_g1_diffusivity(Lx, Ly, k) = pm_g1_diffusivity!(similar(Lx), Lx, Ly, k)
pm_g2_diffusivity(Lx, Ly, k) = pm_g2_diffusivity!(similar(Lx), Lx, Ly, k)
weickert_diffusivity(Lx, Ly, k) = weickert_diffusivity!(similar(Lx), Lx, Ly, k)
charbonnier_diffusivity(Lx, Ly, k) = charbonnier_diffusivity!(similar(Lx), Lx, Ly, k)

function calculate_diffusivity′!(func, dst, Lx, Ly, k)
    invk2 = 1.0 / (k * k)
    @inbounds begin
        for col in axes(Lx, 2)
            @simd for row in axes(Lx, 1)
                lx, ly = Lx[row, col], Ly[row, col]
                dL = (lx * lx + ly * ly) * invk2
                dst[row, col] = func(dL)
            end
        end
    end
    dst
end

function compute_k_percentile(img, perc; gscale = 1.0, nbins = 300)
    Lsmooth = imfilter(img, Kernel.gaussian(gscale))
    ## Scharr gradients scaled by 32, over the interior only — the border ring is
    ## exactly what the original trimmed away, so it never enters the histogram.
    ## Applying the stencil inline rather than through `imfilter` is what makes
    ## this fast; the magnitudes themselves are still buffered, because the
    ## histogram needs the maximum before it can pick its bin scale.
    compute_k_percentile_scharr′(Lsmooth, perc, nbins)
end

function compute_k_percentile_scharr′(L, perc, nbins) @inbounds begin
    rows, cols = size(L)

    hist = zeros(Int32, nbins)

    ## Gradient magnitude over the interior, keeping the largest so the bin
    ## scale is known. Buffered rather than recomputed: the 8-tap stencil costs
    ## more than a store plus a reload.
    modg = Matrix{Float64}(undef, rows-2, cols-2)
    hmax = 0.0
    for j in 2:cols-1
        @simd for i in 2:rows-1
            lx, ly = scharr32(L, i, j)
            m = sqrt(lx^2 + ly^2)
            modg[i-1, j-1] = m
            hmax = max(hmax, m)
        end
    end
    indexscale = nbins / hmax

    for m in modg
        nbin = ceil(Int, m * indexscale)
        if nbin > 0 && m > 1e-10
            hist[nbin] += 1
        end
    end

    nthreshold = floor(Int, sum(hist) * perc)
    k = findfirst(hx -> hx > nthreshold, cumsum(hist))

    return if isnothing(k)
        0.03
    else
        k / indexscale
    end
end end

"""
    scharr32(L, i, j)

The 3×3 Scharr x- and y-gradients of `L` at `(i, j)`, scaled by 32 — the same
operator as [`scharr32_x`](@ref)/[`scharr32_y`](@ref), but with both passes fused
into one stencil evaluation so no intermediate image is needed. Requires `(i, j)`
to be an interior pixel.
"""
@inline function scharr32(L, i, j) @inbounds begin
    sm, s0, sp = scharr32_smooth
    dm, _, dp = scharr32_diff
    a = L[i-1, j-1]; b = L[i-1, j]; c = L[i-1, j+1]
    d = L[i,   j-1];                f = L[i,   j+1]
    g = L[i+1, j-1]; h = L[i+1, j]; k = L[i+1, j+1]
    ## The separable kernels below, expanded: the smoothing factor weights the
    ## three lines across the derivative direction, the difference factor the two
    ## outer lines along it.
    lx = (sm*dp)*(c - a) + (s0*dp)*(f - d) + (sp*dp)*(k - g)
    ly = (sm*dp)*(g - a) + (s0*dp)*(h - b) + (sp*dp)*(k - c)
    (lx, ly)
end end


function demo_diffusivity_functions()
    imshow(Lsmooth)
    aa = pm_g1_diffusivity(Lx, Ly, 0.01)
    imshow(aa)
    aa = pm_g2_diffusivity(Lx, Ly, 0.01)
    imshow(aa)
    aa = weickert_diffusivity(Lx, Ly, 0.01)
    imshow(aa)
    aa = charbonnier_diffusivity(Lx, Ly, 0.01)
    imshow(aa)
end

function demo_k_percentile()
    gthresh = compute_k_percentile(img, 0.95)
    imshow(sqrt.(Lx .^ 2 + Ly .^ 2))
    imshow(sqrt.(Lx .^ 2 + Ly .^ 2) .< gthresh)
end

function demo_nld()
    Lflow = charbonnier_diffusivity(Lx, Ly, 0.01)

    imshow(copy(Lt))
    for _ = 1:3
        for _ = 1:100
            nld_step_scalar(Lt, Lflow, 0.05)
        end
        imshow(copy(Lt))
    end
end

"""
    DilatedSeparable(k1, k2, r)

A separable 2d kernel whose factors are each 3-tap but *dilated*: the taps sit at
offsets `-r`, `0`, `+r` rather than `-1, 0, 1`. `k1` applies along the first
(row) dimension and `k2` along the second (column) dimension.

The AKAZE multiscale derivative kernels have this shape — a length `2r+1` factor
with only three nonzero entries — so representing them densely and handing them
to a general convolution makes it do `2r+1` multiply-adds per pixel where three
suffice.
"""
struct DilatedSeparable{T<:Real}
    k1::SVector{3,T}
    k2::SVector{3,T}
    r::Int
end

## Scharr factors scaled by 32, the single source of truth for the gradient
## operator. `scharr32_x`/`scharr32_y` filter whole images; `scharr32` evaluates
## the same operator one pixel at a time.
const scharr32_smooth = SVector(3.0, 10.0, 3.0) * 2
const scharr32_diff = SVector(-1.0, 0.0, 1.0) / 2
const scharr32_x = DilatedSeparable(scharr32_smooth, scharr32_diff, 1)
const scharr32_y = DilatedSeparable(scharr32_diff, scharr32_smooth, 1)

"""
    dilated_imfilter!(dst, src, kern::DilatedSeparable, scratch = similar(src))

Filter `src` by `kern` into `dst`, using `scratch` (same axes as `src`) to hold
the intermediate pass. Borders replicate the edge value. Pass a reused `scratch`
on hot paths to keep the call allocation-free.

`dst` may alias `scratch` but not `src`. The row factor is applied first, then
the column factor, matching `ImageFiltering.imfilter!`'s ordering so results
agree bit-for-bit.
"""
function dilated_imfilter!(dst, src, kern::DilatedSeparable, scratch = similar(src))
    axes(scratch) == axes(src) ||
        throw(DimensionMismatch("scratch must have the same axes as src"))
    dst === src && throw(ArgumentError("dst must not alias src"))
    dilated_filter_dim1!(scratch, src, kern.k1, kern.r)
    dilated_filter_dim2!(dst, scratch, kern.k2, kern.r)
end

function dilated_filter_dim1!(dst, src, k, r) @inbounds begin
    rows = size(src, 1)
    km, k0, kp = k
    ## The border rows are peeled off so the interior loop needs no clamping and
    ## can vectorize as three contiguous streams; clamping per element here
    ## (unlike in dim2!, where it hoists out of the inner loop) costs ~3x.
    tap(i, j) = (km * src[max(i-r, 1), j] + k0 * src[i, j]) + kp * src[min(i+r, rows), j]
    for j in axes(src, 2)
        for i in 1:min(r, rows)
            dst[i, j] = tap(i, j)
        end
        @simd for i in r+1:rows-r
            dst[i, j] = (km * src[i-r, j] + k0 * src[i, j]) + kp * src[i+r, j]
        end
        for i in max(r+1, rows-r+1):rows
            dst[i, j] = tap(i, j)
        end
    end
    dst
end end

function dilated_filter_dim2!(dst, src, k, r) @inbounds begin
    cols = size(src, 2)
    km, k0, kp = k
    for j in axes(src, 2)
        jm = ifelse(j > r, j - r, 1)
        jp = ifelse(j + r <= cols, j + r, cols)
        @simd for i in axes(src, 1)
            dst[i, j] = (km * src[i, jm] + k0 * src[i, j]) + kp * src[i, jp]
        end
    end
    dst
end end

function compute_derivative_kernels(scale)

    ## The dense kernel would be length 3 + 2*(scale-1) with its outer taps at
    ## ±(scale), so the dilation radius simply is the scale.
    w = 10.0/3.0
    norm = 1.0/(2.0*scale*(w+2.0))

    smooth = SVector(norm, w*norm, norm)
    central = SVector(-1.0, 0.0, 1.0)

    (DilatedSeparable(smooth, central, scale), DilatedSeparable(central, smooth, scale))
end
