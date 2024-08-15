### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 9126822e-52b8-11ef-0c62-d1a026b84d66
begin
	import Pkg
	Pkg.add(url="https://github.com/JoshuaBillson/Apollo.jl")
	Pkg.add(["Rasters", "ArchGDAL", "Flux", "JLD2", "Images"])
	Pkg.add(["MultivariateStats", "Pipe",  "Statistics", "Random"])

	import Apollo, Rasters, ArchGDAL, Flux, JLD2, Images, MultivariateStats
	using Random, Statistics
	using Pipe: @pipe

	# Uncomment to Add GPU Support
	Pkg.add(["CUDA", "cuDNN"])
	import CUDA, cuDNN
end

# ╔═╡ f9018cc2-261a-4dd2-9a69-3f0a3b7bdaca
const MVS = MultivariateStats;

# ╔═╡ 1f2c5ee2-0822-4967-a278-036f6e8f608b
labels = Rasters.Raster("data/rice/labels_2018_2019_roi.tif", lazy=false);

# ╔═╡ 7682ee39-e82a-4ddc-9e6e-cff9ff893510
features = Rasters.Raster("data/rice/2018-01-05_10m.tif", lazy=false);

# ╔═╡ de35adb0-6b65-4c2e-b4a2-85d99815a30f
μ = Apollo.folddims(mean, Float64.(features), dims=Rasters.Band);

# ╔═╡ 9c377fc8-9196-4459-8561-2ea10ae88df6
σ = Apollo.folddims(std, Float64.(features), dims=Rasters.Band);

# ╔═╡ 336e4104-acbc-4416-b482-43d00838149b
model = let
	model_state = JLD2.load("models/rice10m_unet.jld2", "model_state")
	model = Apollo.UNet(channels=4)
	Flux.loadmodel!(model, model_state) |> Flux.gpu
end;

# ╔═╡ c9b9d56d-da73-4390-afe4-8beac058259c
md"""
# Generate Features
"""

# ╔═╡ da1be246-efd9-42e7-aa69-63656618a88c
function merge_tiles(to, tiles...)
	n = Rasters.Raster(zeros(Float32, size(to)), to)
	dst = Rasters.Raster(zeros(Float32, size(to)), to)
	for tile in tiles
		extended = Rasters.extend(tile, to=to, missingval=0.0f0)
		n .+= ifelse.(extended .== 0.0f0, 0.0f0, 1.0f0)
		dst .+= extended
	end
	return dst ./ clamp!(n, 1.0f0, Inf32)
end;

# ╔═╡ 563b3567-4175-49b5-95cd-0ddabc215d54
preds = map(Apollo.TileSampler(features, 1024)) do x
		@pipe Apollo.tensor(x) |> 
		Apollo.normalize(_, μ, σ) |>
		Flux.gpu |>
		Apollo.features(model, _) |>
		Flux.cpu |>
		Rasters.Raster(_[:,:,:,1], (Rasters.dims(x)[1:2]..., Rasters.Band))
end;

# ╔═╡ 9263b0e4-e502-41e8-99b9-deffabaf7ac1
feature_sample = let
	samples = map(preds) do pred
		features = @pipe reshape(pred.data, (:, 64)) |> permutedims(_, (2, 1))
		sample = Random.randperm(size(features, 2))[1:8000]
		return features[:,sample]
	end
	cat(samples..., dims=2)
end

# ╔═╡ e2842b95-b3a9-47e4-beab-16ddba1572cf
feature_tiles = let
	M = MVS.fit(MVS.PCA, feature_sample; maxoutdim=3)

	map(preds) do pred
		@pipe pred.data |>
		reshape(_, (:, 64)) |>
		permutedims(_, (2, 1)) |>
		MVS.predict(M, _) |>
		permutedims(_, (2, 1)) |>
		reshape(_, (1024, 1024, 3)) |>
		Rasters.Raster(_, (Rasters.dims(pred)[1:2]..., Rasters.Band))
	end
end

# ╔═╡ e3bd0f32-22cf-4541-914d-d6b60d21e094
mosaic = merge_tiles((Rasters.dims(features)[1:2]..., Rasters.Band(1:3)), feature_tiles...)

# ╔═╡ 2ec18160-b8e3-4ea2-81cd-73d9e9e611c9
let
	lb = Apollo.folddims(x -> quantile(x, 0.02), mosaic, dims=Rasters.Band)
	ub = Apollo.folddims(x -> quantile(x, 0.98), mosaic, dims=Rasters.Band)
	rgb = Apollo.rgb(mosaic, lb, ub)
end

# ╔═╡ Cell order:
# ╠═9126822e-52b8-11ef-0c62-d1a026b84d66
# ╠═f9018cc2-261a-4dd2-9a69-3f0a3b7bdaca
# ╠═1f2c5ee2-0822-4967-a278-036f6e8f608b
# ╠═7682ee39-e82a-4ddc-9e6e-cff9ff893510
# ╠═de35adb0-6b65-4c2e-b4a2-85d99815a30f
# ╠═9c377fc8-9196-4459-8561-2ea10ae88df6
# ╠═336e4104-acbc-4416-b482-43d00838149b
# ╟─c9b9d56d-da73-4390-afe4-8beac058259c
# ╠═da1be246-efd9-42e7-aa69-63656618a88c
# ╠═563b3567-4175-49b5-95cd-0ddabc215d54
# ╠═9263b0e4-e502-41e8-99b9-deffabaf7ac1
# ╠═e2842b95-b3a9-47e4-beab-16ddba1572cf
# ╠═e3bd0f32-22cf-4541-914d-d6b60d21e094
# ╠═2ec18160-b8e3-4ea2-81cd-73d9e9e611c9
