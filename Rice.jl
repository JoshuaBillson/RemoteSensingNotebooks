### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 2937b07c-342a-11ef-1042-1d855c9401dd
# ╠═╡ show_logs = false
begin
	import Pkg
	Pkg.add(url="https://github.com/JoshuaBillson/Apollo.jl")
	Pkg.add(["Rasters", "ArchGDAL", "Flux", "JLD2", "ImageShow"])
	Pkg.add(["Pipe",  "Statistics", "Random"])

	import Apollo, Rasters, ArchGDAL, Flux, JLD2, ImageShow
	using Random, Statistics
	using Pipe: @pipe

	# Uncomment to Add GPU Support
	Pkg.add(["CUDA", "cuDNN"])
	import CUDA, cuDNN
end

# ╔═╡ d9df1d6f-69e8-4ce5-94f8-5b03d8623d13
md"""
# Import Packages
"""

# ╔═╡ c40df519-d8d9-4710-8ede-0b3093850da2
md"""
# Load Data
"""

# ╔═╡ fc337c18-e1ec-4ca9-8c81-35c05b7c3607
labels = Rasters.Raster("data/rice/labels_2018_2019_roi.tif");

# ╔═╡ 69833c13-1bb5-499c-a25c-613e9cb61644
s2_10m = Rasters.Raster("data/RiceSC_South_S2/2018-01-05_10m.tif");

# ╔═╡ 08a2bbc8-44a3-450b-bea6-d37b9efde2e6
s2_20m = Rasters.Raster("data/RiceSC_South_S2/2018-01-05_20m_sharpened.tif");

# ╔═╡ a27c4708-3b23-4616-9e24-21cf6f2aeff7
features = cat(s2_10m, s2_20m, dims=Rasters.Band) .* 0.0001f0;

# ╔═╡ 58f2e552-8f24-4fae-80c6-fdaae235d3bb
md"""
# Define Data Pipeline
"""

# ╔═╡ 6a1b5d3a-c3f0-4bae-9087-504008031972
μ = Apollo.folddims(mean, features, dims=Rasters.Band);

# ╔═╡ 46da0b07-8216-4f37-94e2-8a61309e1e03
σ = Apollo.folddims(std, features, dims=Rasters.Band);

# ╔═╡ 1fd08553-8cfa-4e4f-bae5-8d958de559ad
begin
	# Sample 100 512x512 Tiles
	xsampler = Apollo.TileView(features, 512)
	ysampler = Apollo.TileView(labels, 512)
	data = Apollo.zipobs(xsampler, ysampler)

	# Split Train and Test
	train, test = Apollo.splitobs(data, at=0.7)

	# Define Data Pipeline
	dtypes = (Apollo.Image(), Apollo.Mask())
	test_transform = Apollo.Tensor() |> Apollo.Normalize(μ, σ)
	train_transform = test_transform |> Apollo.RandomCrop((256,256))

	# Apply Transforms
	train_pipe = Apollo.transform(train_transform, dtypes, train)
	test_pipe = Apollo.transform(test_transform, dtypes, test)

	# Create DataLoader
	traindata = Flux.DataLoader(train_pipe, batchsize=8, shuffle=true)
	testdata = Flux.DataLoader(test_pipe, batchsize=8)
end;

# ╔═╡ 96be9145-86f4-489c-8472-ec5be68514c3
md"""
# Visualize Data
"""

# ╔═╡ d7eabd7f-f919-40fe-9e00-5659ac238e88
lb = Apollo.folddims(x -> quantile(x, 0.02), features, dims=Rasters.Band);

# ╔═╡ 46c021ea-90d5-41a3-8f3c-be70feb6ef92
ub = Apollo.folddims(x -> quantile(x, 0.98), features, dims=Rasters.Band);

# ╔═╡ 57626792-b4a2-44c0-8d3b-738440ace66d
let
	plots = map(Apollo.sampleobs(train_pipe, 9)) do (x, y)
		rgb = Apollo.rgb(Apollo.denormalize(x, μ, σ), lb, ub, bands=[3,2,1])
		mask = Apollo.binmask(y)
		Apollo.mosaicview((1,2), rgb, mask)
	end
	Apollo.mosaicview((3,3), plots...)
end

# ╔═╡ 707fedd1-bce7-4e16-9082-f13ca890b92a
md"""
# Train Model
"""

# ╔═╡ 6dfbd392-96c3-4193-8eb0-4eebc1932ef0
loss = Apollo.BinaryCrossEntropy();

# ╔═╡ b840198e-8ed4-42f3-872a-2b385199ee1b
tracker = Apollo.Tracker(
	"train_loss" => Apollo.Loss(loss), 
	"train_MIoU" => Apollo.MIoU([0,1]), 
	"test_loss" => Apollo.Loss(loss), 
	"test_MIoU" => Apollo.MIoU([0,1]),
);

# ╔═╡ 8181878a-50d3-4aff-bc18-04cc76923301
model = let
	# Build Model
	input = Apollo.Single(channels=10)
	encoder = Apollo.ResNet18(weights=:ImageNet)
	model = Apollo.UNet(input=input, encoder=encoder, activation=Flux.sigmoid)

	# Move Model to GPU
	model = Flux.gpu(model)
	
	# Prepare Optimiser
	opt = Flux.Optimisers.Adam(5e-4)
	opt_state = Flux.Optimisers.setup(opt, model)
	Flux.Optimisers.freeze!(opt_state.encoder)

	# Train for 15 Epochs
 	for epoch in 1:30

		# Update Model
		Apollo.train!(loss, model, opt_state, Flux.gpu(traindata))

		# Evaluate Train Performance
		for (x, y) in traindata
			ŷ = Flux.gpu(x) |> model |> Flux.cpu
			Apollo.step!(tracker, r"train_", ŷ, y)
		end

		# Evaluate Validation Performance
		for (x, y) in testdata
			ŷ = Flux.gpu(x) |> model |> Flux.cpu
			Apollo.step!(tracker, r"test_", ŷ, y)
		end

		# End Epoch
		@info Apollo.printscores(tracker, metrics=r"_loss")
		Apollo.epoch!(tracker)
	end

	# Return Best Model
	model
end;

# ╔═╡ edaa7503-3f44-4ae2-b8e3-4b7da945fd04
metrics = [
	Apollo.Loss(loss), 
	Apollo.MIoU([0,1]), 
	Apollo.Accuracy()
];

# ╔═╡ 181232d2-6d62-4e6c-a7fe-a0e50068c28b
train_metrics = Apollo.evaluate(traindata, metrics...) do (x, y)
	ŷ = Flux.gpu(x) |> model |> Flux.cpu
	return (ŷ, y)
end;

# ╔═╡ 4c93bc93-5106-4aa8-b702-2f7d00a38cba
test_metrics = Apollo.evaluate(testdata, metrics...) do (x, y)
	ŷ = Flux.gpu(x) |> model |> Flux.cpu
	return (ŷ, y)
end;

# ╔═╡ 0a7c6896-9038-428d-b375-24182a32736a
md"""
# Quantitative Analysis

**Train Accuracy:** $(round(train_metrics.accuracy, digits=4))

**Test Accuracy:** $(round(test_metrics.accuracy, digits=4))

**Train MIoU:** $(round(Float64(train_metrics.MIoU), digits=4))

**Test MIoU:** $(round(Float64(test_metrics.MIoU), digits=4))

**Train Loss:** $(round(Float64(train_metrics.loss), digits=4))

**Test Loss:** $(round(Float64(test_metrics.loss), digits=4))
"""

# ╔═╡ 224ca017-8998-4474-89cd-ffab071d4b50
md"""
# Qualitative Analysis
"""

# ╔═╡ 0e837215-f43b-4521-9b01-77606dc4d0c7
let
	# Select Tiles With At Least 10% Rice
	test_sample = Apollo.filterobs(test_pipe) do (x, y)
		return (sum(y) / length(y)) >= 0.01
	end

	# Plot Label/Prediction Pairs
	plots = map(Apollo.sampleobs(test_sample, 3)) do (x, y)
		# Plot RGB
		rgb = Apollo.rgb(Apollo.denormalize(x, μ, σ), lb, ub, bands=[3,2,1])

		# Plot label
		label = Apollo.binmask(y)

		# Plot Prediction
		ŷ = @pipe Flux.gpu(x) |> model |> Flux.cpu |> round.(Int, _)
		pred = Apollo.binmask(ŷ)
		
		# Combine RGB, Label, and Prediction
		Apollo.mosaicview((1,3), rgb, label, pred)
	end

	# Plot Mosaic
	Apollo.mosaicview((3,1), plots...)
end

# ╔═╡ 64198967-17e4-465f-8ecc-81ac92826a02
md"""
# Save Model
"""

# ╔═╡ 789d8002-1d49-45ac-b13f-749e5cc6a218
false && let
	model_state = model |> Flux.cpu |> Flux.state
	JLD2.jldsave("models/rice10m_unet.jld2"; model_state)
end;

# ╔═╡ Cell order:
# ╟─d9df1d6f-69e8-4ce5-94f8-5b03d8623d13
# ╠═2937b07c-342a-11ef-1042-1d855c9401dd
# ╟─c40df519-d8d9-4710-8ede-0b3093850da2
# ╠═fc337c18-e1ec-4ca9-8c81-35c05b7c3607
# ╠═69833c13-1bb5-499c-a25c-613e9cb61644
# ╠═08a2bbc8-44a3-450b-bea6-d37b9efde2e6
# ╠═a27c4708-3b23-4616-9e24-21cf6f2aeff7
# ╟─58f2e552-8f24-4fae-80c6-fdaae235d3bb
# ╠═6a1b5d3a-c3f0-4bae-9087-504008031972
# ╠═46da0b07-8216-4f37-94e2-8a61309e1e03
# ╠═1fd08553-8cfa-4e4f-bae5-8d958de559ad
# ╟─96be9145-86f4-489c-8472-ec5be68514c3
# ╠═d7eabd7f-f919-40fe-9e00-5659ac238e88
# ╠═46c021ea-90d5-41a3-8f3c-be70feb6ef92
# ╠═57626792-b4a2-44c0-8d3b-738440ace66d
# ╟─707fedd1-bce7-4e16-9082-f13ca890b92a
# ╠═6dfbd392-96c3-4193-8eb0-4eebc1932ef0
# ╠═b840198e-8ed4-42f3-872a-2b385199ee1b
# ╠═8181878a-50d3-4aff-bc18-04cc76923301
# ╟─0a7c6896-9038-428d-b375-24182a32736a
# ╠═edaa7503-3f44-4ae2-b8e3-4b7da945fd04
# ╠═181232d2-6d62-4e6c-a7fe-a0e50068c28b
# ╠═4c93bc93-5106-4aa8-b702-2f7d00a38cba
# ╟─224ca017-8998-4474-89cd-ffab071d4b50
# ╠═0e837215-f43b-4521-9b01-77606dc4d0c7
# ╟─64198967-17e4-465f-8ecc-81ac92826a02
# ╠═789d8002-1d49-45ac-b13f-749e5cc6a218
