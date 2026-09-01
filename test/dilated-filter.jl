using Random: MersenneTwister

const DS = AkazeFeatures.DilatedSeparable

"""
    dense(taps, r)

The length `2r+1` dense vector whose only nonzero entries are `taps`, at offsets
`-r`, `0`, `+r`.
"""
function dense(taps, r)
    v = zeros(eltype(taps), 2r + 1)
    v[1], v[r+1], v[2r+1] = taps
    v
end

@testset "DilatedSeparable imfilter!" begin
    rng = MersenneTwister(20)

    ## Asymmetric factors, so a k1/k2 swap or a sign flip cannot pass.
    k1 = SVector(0.3, 0.7, -0.2)
    k2 = SVector(-0.5, 0.25, 0.5)

    @testset "matches the dense kernel (r=$r, size=$sz)" for
            r in (1, 2, 4, 7), sz in ((16, 16), (5, 9), (9, 5), (3, 3), (2, 2), (1, 1))
        src = rand(rng, sz...)
        got = imfilter!(similar(src), src, DS(k1, k2, r), scratch = similar(src))
        want = imfilter(src, kernelfactors((centered(dense(k1, r)),
                                            centered(dense(k2, r)))), "replicate")
        ## Both apply the row factor then the column factor with the same
        ## association, so agreement is exact rather than approximate. `r` larger
        ## than a dimension is covered by the smaller sizes above.
        @test got == want
    end

    @testset "the AKAZE Scharr kernels match their dense form" begin
        src = rand(rng, 12, 15)
        for (kern, taps) in ((AkazeFeatures.scharr32_x,
                              (AkazeFeatures.scharr32_smooth, AkazeFeatures.scharr32_diff)),
                             (AkazeFeatures.scharr32_y,
                              (AkazeFeatures.scharr32_diff, AkazeFeatures.scharr32_smooth)))
            got = imfilter!(similar(src), src, kern, scratch = similar(src))
            want = imfilter(src, kernelfactors((centered(dense(taps[1], 1)),
                                                centered(dense(taps[2], 1)))), "replicate")
            @test got == want
        end
    end

    @testset "scharr32 stencil matches the two-pass filter" begin
        src = rand(rng, 10, 11)
        Lx = imfilter!(similar(src), src, AkazeFeatures.scharr32_x, scratch = similar(src))
        Ly = imfilter!(similar(src), src, AkazeFeatures.scharr32_y, scratch = similar(src))
        for k in 2:size(src, 2)-1, j in 2:size(src, 1)-1
            lx, ly = AkazeFeatures.scharr32(src, j, k)
            @test lx ≈ Lx[j, k]
            @test ly ≈ Ly[j, k]
        end
    end

    @testset "rejects aliased and mismatched buffers" begin
        src = rand(rng, 6, 6)
        kern = DS(k1, k2, 2)
        @test_throws "three distinct arrays" imfilter!(src, src, kern, scratch = similar(src))
        out = similar(src)
        @test_throws "three distinct arrays" imfilter!(out, src, kern, scratch = out)
        @test_throws "three distinct arrays" imfilter!(out, src, kern, scratch = src)
        @test_throws "same axes as src" imfilter!(out, src, kern, scratch = similar(src, 7, 6))
        @test_throws "same axes as src" imfilter!(similar(src, 7, 6), src, kern,
                                                 scratch = similar(src))
    end
end

@testset "compute_k_percentile_scharr′" begin
    ## A constant image has no interior gradient, so there is no percentile to
    ## take and the bin scale would be infinite.
    @test AkazeFeatures.compute_k_percentile_scharr′(zeros(16, 16), 0.7, 300) == 0.03
    @test AkazeFeatures.compute_k_percentile_scharr′(fill(0.5, 16, 16), 0.7, 300) == 0.03

    ## The largest magnitude scales to bin `nbins`, which floating point can push
    ## to `nbins + 1` before the clamp.
    rng = MersenneTwister(1)
    for _ in 1:200
        k = AkazeFeatures.compute_k_percentile_scharr′(rand(rng, 9, 9), 0.7, 300)
        @test isfinite(k)
        @test k > 0
    end
end
