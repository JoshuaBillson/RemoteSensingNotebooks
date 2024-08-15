### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 5f498a2a-3da1-11ef-182e-c71861b0e27d
# ╠═╡ show_logs = false
begin
	import Pkg
	Pkg.add(url="https://github.com/JoshuaBillson/Apollo.jl")
	Pkg.add(["Rasters", "ArchGDAL", "Pipe", "Flux", "JLD2"])
	Pkg.add(["Images",  "Statistics", "Random"])

	import Apollo, Rasters, ArchGDAL, Flux, JLD2, Images
	using Random, Statistics
	using Pipe: @pipe

	# Uncomment to Add GPU Support
	Pkg.add(["CUDA", "cuDNN"])
	import CUDA, cuDNN
end

# ╔═╡ d3b012c0-4ba6-4a61-85a2-f28ae68aa930
const A = Apollo;

# ╔═╡ a3d6e7c9-dc7e-4ae8-8557-3be384048455
md"""
# Load Data
"""

# ╔═╡ ff82080c-57e6-4657-96d0-bcaf7060d9ad
s2_10m = Rasters.Raster("data/RiceSC_South_S2/2018-01-05_10m.tif") .* 0.0001f0;

# ╔═╡ 38a34002-bd1c-4f33-987d-077f8d272b59
s2_20m = Rasters.Raster("data/RiceSC_South_S2/2018-01-05_20m.tif") .* 0.0001f0;

# ╔═╡ 379f31aa-149e-44ca-abac-6c1e1c17abe0
md"""
# Train Model
"""

# ╔═╡ 8150e927-1728-46c5-9dc8-d3d26d831f3d
model = Apollo.SSC_CNN() |> Flux.gpu;

# ╔═╡ cd5f5b98-c116-4243-b542-967dfcc45c95
let
	# Sample Tiles
	sampler_10m = Apollo.TileSampler(s2_10m, 128)
	sampler_20m = Apollo.TileSampler(s2_20m, 64)
	data = A.zipobs(sampler_20m, sampler_10m)

	# Define Train Pipeline
	train_pipe = A.mapobs(data) do (lr, hr)
		y = A.tensor(lr)
		lr = @pipe Flux.meanpool(y, (2,2)) |> Flux.upsample_bilinear(_, (2,2))
		hr = @pipe A.tensor(hr) |> Flux.meanpool(_, (2,2))
		return (lr, hr, y)
	end

	# Construct Data Loader
	traindata = Flux.DataLoader(train_pipe, batchsize=8, shuffle=true) |> Flux.gpu

	# Prepare Optimiser
	opt_state = Flux.setup(Flux.Optimise.Adam(), model)

	# Train Model
	for epoch in 1:5

		# Iterate Over Data
		total_loss = 0.0f0
 		for (x1, x2, y) in traindata
			grads = Flux.gradient(m -> Flux.mae(m(x1, x2), y), model)
 			Flux.update!(opt_state, model, grads[1])
			total_loss += Flux.mae(model(x1, x2), y) |> Flux.cpu
 		end

		# End of Epoch Performance
 		@info "Epoch $epoch Loss: $(total_loss / length(traindata))"
	end
end;

# ╔═╡ d3eed672-6f30-4481-89e4-eaff4e774cd9
md"""
# Generate Sharpened Bands
"""

# ╔═╡ 10b3fc50-827e-4a04-b2be-15e07f6fae1d
function merge_tiles(tiles...; to=first(tiles))
	dims = Rasters.dims(to)
	n = Rasters.Raster(zeros(Float32, size(dims)), dims)
	dst = Rasters.Raster(zeros(Float32, size(dims)), dims)
	for tile in tiles
		extended = Rasters.extend(tile, to=dims, missingval=0.0f0)
		n .+= ifelse.(extended .== 0.0f0, 0.0f0, 1.0f0)
		dst .+= extended
	end
	return dst ./ clamp!(n, 1.0f0, Inf32)
end;

# ╔═╡ 60846068-4b3f-4465-99a5-716f7f74602e
sharpened = let
	# Create Samplers
	s2_20m_up = Rasters.resample(s2_20m, res=10, method=:bilinear)
	sampler_20m = Apollo.TileSampler(s2_20m_up, 1024)
	sampler_10m = Apollo.TileSampler(s2_10m, 1024)
	
	# Sharpen Bands
	preds = map(A.zipobs(sampler_20m, sampler_10m)) do (lr, hr)
		x1 = A.tensor(lr) |> Flux.gpu
		x2 = A.tensor(hr) |> Flux.gpu
		pred = model(x1, x2) |> Flux.cpu
		return Rasters.Raster(pred[:,:,:,1], Rasters.dims(lr))
	end

	# Merge Sharpened Tiles
	merge_tiles(preds..., to=s2_20m_up)
end;

# ╔═╡ f7cd8eec-2629-4c21-85f9-6c8adb80ec89
let
	#dims1 = LinRange(610000.0, 611270.0, 128)
	#dims2 = LinRange(610000.0, 661190.0, 5120)
	#r1 = Rasters.Raster(rand(128, 128), (Rasters.X(dims1), Rasters.Y(dims1)))
	#r2 = Rasters.Raster(rand(5120, 5120), (Rasters.X(dims2), Rasters.Y(dims2)))
	#Rasters.extend(r1, to=r2)
end

# ╔═╡ d42dce3e-81f0-478d-aa0f-91f36fd2f844
md"""
# Qualitative Analysis
"""

# ╔═╡ 17336d28-1a3b-4ee0-b4c2-2a7e850bc6f9
ub_20m = Apollo.folddims(x->quantile(x, 0.98), s2_20m, dims=Rasters.Band);

# ╔═╡ 26a5611e-43d9-4e01-84db-ab0e69b8935d
lb_20m = Apollo.folddims(x->quantile(x, 0.02), s2_20m, dims=Rasters.Band);

# ╔═╡ 5d13feb8-11c9-43af-8c2b-1d8ecdcef42b
ub_10m = Apollo.folddims(x->quantile(x, 0.98), s2_10m, dims=Rasters.Band);

# ╔═╡ 002e003c-4ef1-495f-9488-3fa02fa99e1c
lb_10m = Apollo.folddims(x->quantile(x, 0.02), s2_10m, dims=Rasters.Band);

# ╔═╡ c927cd70-5678-4131-a882-bf0bdeed1ed8
let
	# Sample Tiles
	s2_20m = Rasters.resample(s2_20m, res=10, method=:near)
	sampler_10m = Apollo.TileSampler(s2_10m, 256)
	sampler_20m = Apollo.TileSampler(s2_20m, 256)
	sampler_preds = Apollo.TileSampler(sharpened, 256)
	pipeline = Apollo.zipobs(sampler_10m, sampler_20m, sampler_preds)

	# Visualize Samples
	plots = map(A.sampleobs(pipeline, 3)) do (hr, lr, pred)
		rgbimg = Apollo.rgb(hr, lb_10m, ub_10m, bands=[3, 2, 1])
		ximg = Apollo.rgb(lr, lb_20m, ub_20m, bands=[1, 2, 3])
		yimg = Apollo.rgb(pred, lb_20m, ub_20m, bands=[1, 2, 3])
		Apollo.mosaicview((1,3), rgbimg, ximg, yimg)
	end

	# Mosaic
	Apollo.mosaicview((3,1), plots...)
end

# ╔═╡ 03ce88f1-2d78-45b1-a218-95070d4189a8
md"""
# Save Sharpened Bands
"""

# ╔═╡ 350fe341-cfce-49f7-99a3-024c1af9df70
@pipe sharpened .* 10000 |>
round.(UInt16, _) |>
Rasters.write("data/RiceSC_South_S2/2018-01-05_20m_sharpened.tif", _; force=true);

# ╔═╡ Cell order:
# ╠═5f498a2a-3da1-11ef-182e-c71861b0e27d
# ╠═d3b012c0-4ba6-4a61-85a2-f28ae68aa930
# ╟─a3d6e7c9-dc7e-4ae8-8557-3be384048455
# ╠═ff82080c-57e6-4657-96d0-bcaf7060d9ad
# ╠═38a34002-bd1c-4f33-987d-077f8d272b59
# ╟─379f31aa-149e-44ca-abac-6c1e1c17abe0
# ╠═8150e927-1728-46c5-9dc8-d3d26d831f3d
# ╠═cd5f5b98-c116-4243-b542-967dfcc45c95
# ╟─d3eed672-6f30-4481-89e4-eaff4e774cd9
# ╠═10b3fc50-827e-4a04-b2be-15e07f6fae1d
# ╠═60846068-4b3f-4465-99a5-716f7f74602e
# ╟─f7cd8eec-2629-4c21-85f9-6c8adb80ec89
# ╟─d42dce3e-81f0-478d-aa0f-91f36fd2f844
# ╠═17336d28-1a3b-4ee0-b4c2-2a7e850bc6f9
# ╠═26a5611e-43d9-4e01-84db-ab0e69b8935d
# ╠═5d13feb8-11c9-43af-8c2b-1d8ecdcef42b
# ╠═002e003c-4ef1-495f-9488-3fa02fa99e1c
# ╠═c927cd70-5678-4131-a882-bf0bdeed1ed8
# ╟─03ce88f1-2d78-45b1-a218-95070d4189a8
# ╠═350fe341-cfce-49f7-99a3-024c1af9df70
