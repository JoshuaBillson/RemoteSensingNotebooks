### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 2937b07c-342a-11ef-1042-1d855c9401dd
# ╠═╡ show_logs = false
begin
	import Pkg
	Pkg.add(url="https://github.com/JoshuaBillson/Apollo.jl")
	Pkg.add(["Rasters", "ArchGDAL", "Pipe", "MLUtils", "Flux", "JLD2"])
	Pkg.add(["Images",  "Statistics", "Random"])

	import Apollo, Rasters, ArchGDAL, MLUtils, Flux, JLD2, Images
	using Random, Statistics
	using Pipe: @pipe
end

# ╔═╡ d9df1d6f-69e8-4ce5-94f8-5b03d8623d13
md"""
# Import Packages
"""

# ╔═╡ c14619c0-54a5-4be3-826c-d5689ee9a127
md"""
# Script Settings
"""

# ╔═╡ ae852cbc-9da6-46da-9cc8-37c833425dc7
const TRAIN_MODEL = true;

# ╔═╡ 88de5222-4bd4-406b-8202-cc402c251837
const LOAD_MODEL = false;

# ╔═╡ d972c90d-0f86-4d6a-bdd6-e493a17a39e2
const SAVE_MODEL = false;

# ╔═╡ c40df519-d8d9-4710-8ede-0b3093850da2
md"""
# Load Data
"""

# ╔═╡ 7c543af1-c07b-433b-9d4a-d4f87f3fe611
const R = Rasters;

# ╔═╡ fc337c18-e1ec-4ca9-8c81-35c05b7c3607
labels = Rasters.Raster("labels_2018_2019_roi.tif", lazy=true);

# ╔═╡ 69833c13-1bb5-499c-a25c-613e9cb61644
features = Rasters.Raster("2018-01-05_10m.tif", lazy=true);

# ╔═╡ 58f2e552-8f24-4fae-80c6-fdaae235d3bb
md"""
# Define Data Pipeline
"""

# ╔═╡ 631079ac-00d8-4fcd-ab4b-b4c60a932a5a
μ, σ = Apollo.stats(features);

# ╔═╡ f2ab25ee-9574-4289-b3f5-1f52c02475f3
xsampler = Apollo.TileSampler(Apollo.WHCN, features, 256);

# ╔═╡ 1519060a-7e9b-4fdb-b066-eeb1213b0f5c
xdata = MLUtils.mapobs(x -> Apollo.normalize(x, μ, σ), xsampler);

# ╔═╡ 08daaef7-c75f-4d95-968c-c5b8e8401572
ydata = Apollo.TileSampler(Apollo.WHCN, labels, 256);

# ╔═╡ 0e095f57-440f-4ee0-8fdb-3d2f6b934e61
sample = @pipe map(x -> sum(x) / length(x), ydata) |> findall(>=(0.05), _);

# ╔═╡ 32ac80ac-b8e6-4801-aab3-78afe6078856
x, y = MLUtils.obsview((xdata, ydata), sample);

# ╔═╡ 3eb12501-92e7-4235-8cd0-37864bb299b7
data = MLUtils.DataLoader((xdata, ydata), batchsize=4, shuffle=true, partial=true);

# ╔═╡ 96be9145-86f4-489c-8472-ec5be68514c3
md"""
# Visualize Data
"""

# ╔═╡ cc80e937-24c5-40ab-8c57-78ef291741d3
bounds = map(1:4) do i
	data = features[R.Band(i)] |> vec |> sort!
	lb = quantile(data, 0.02, sorted=true)
	ub = quantile(data, 0.98, sorted=true)
	return (lb, ub)
end;

# ╔═╡ 5de7b13a-dd7d-4a96-ac7c-e08fb0fe9b68
function linear_stretch(x::AbstractArray{<:Real,2}, lb, ub)
	clamp.((x .- lb) ./ (ub .- lb), 0.0f0, 1.0f0)
end;

# ╔═╡ 9c76bf2e-ce6d-4508-a8ec-1946d1849d26
function show_rgb(r, g, b, r_bounds, g_bounds, b_bounds)
	red = linear_stretch(r, r_bounds...)
	green = linear_stretch(g, g_bounds...)
	blue = linear_stretch(b, b_bounds...)
	@pipe cat(red, green, blue, dims=3) |> 
	permutedims(_, (3, 2, 1)) |>
	Images.N0f8.(_) |>
	Images.colorview(Images.RGB, _)
end;

# ╔═╡ 359d7124-5224-4c5b-ad04-b73f617b2ff5
function show_rgb(x::AbstractArray{<:Real,4}, bounds)
	show_rgb(x[:,:,3,1], x[:,:,2,1], x[:,:,1,1], bounds[3], bounds[2], bounds[1])
end;

# ╔═╡ c0bd9c49-5394-4176-99a7-737c5e9bb3b2
function show_mask(x::AbstractArray{<:Real,2})
	@pipe permutedims(x, (2, 1)) |> 
	Images.N0f8.(_) |> 
	Images.colorview(Images.Gray, _)
end;

# ╔═╡ 64dbfcc1-a686-4d17-97dc-ee1f5e186f8d
function show_mask(x::AbstractArray{<:Real,4})
	show_mask(x[:,:,1,1])
end;

# ╔═╡ 7509e099-55fc-423d-b9a3-d7fa18682bbf
let
	sample = rand(1:400, 9)
	imgs = [show_rgb(xsampler[i], bounds) for i in sample]
	masks = [show_mask(ydata[i]) for i in sample]
	m1 = Images.mosaicview(imgs..., nrow=3, ncol=3, npad=5, fillvalue=1.0)
	m2 = Images.mosaicview(masks..., nrow=3, ncol=3, npad=5, fillvalue=1.0)
	Images.mosaicview(m1, m2, nrow=1, ncol=2, npad=5, fillvalue=1.0)
end

# ╔═╡ 707fedd1-bce7-4e16-9082-f13ca890b92a
md"""
# Train Model
"""

# ╔═╡ 2e21bce8-99f2-4d5b-94f7-23d1530ce012
model = Apollo.UNet(4, 1, batch_norm=false);

# ╔═╡ 4319dc71-db9d-4341-a872-d7472be04ce1
model

# ╔═╡ ef63e89c-06e3-4ed4-8ff6-32e80a4d1c2e
opt = Flux.Optimisers.OptimiserChain(
	#Flux.Optimisers.AccumGrad(4),
	Flux.Optimisers.ClipGrad(0.5),
	Flux.Optimisers.Adam() );

# ╔═╡ 23e40dea-24e5-4e37-93eb-181f39375cca
#opt = Flux.Optimisers.Adam();

# ╔═╡ e15a7a09-fb5b-4176-987e-14af9a7919e2
opt_state = Flux.setup(opt, model);

# ╔═╡ 8181878a-50d3-4aff-bc18-04cc76923301
TRAIN_MODEL && let
 	for epoch in 1:15
		@info "Epoch: $epoch"
 
 		total_loss = 0.0f0
 		for (i, (x, y)) in enumerate(data)
 			
			l, grads = Flux.withgradient(model) do m
 				return Flux.logitbinarycrossentropy(m(x), y)
 			end
			
 			Flux.update!(opt_state, model, grads[1])
			
 			total_loss += l
			
 			@info "Loss $(i): $(total_loss / i)"
 		end
	end
end;

# ╔═╡ 4c93bc93-5106-4aa8-b702-2f7d00a38cba
# ╠═╡ disabled = true
#=╠═╡
let
	correct = 0
	total = 0
	for (x, y) in data
		ŷ = Flux.sigmoid.(model(x)) .>= 0.5
		correct += sum(ŷ .== y)
		total += length(y)
	end
	println(correct / total * 100)
end
  ╠═╡ =#

# ╔═╡ b6f908d5-28de-4b55-a688-27ec31b3cb59
md"""
# Save Model
"""

# ╔═╡ 8d2944c7-4924-4ce4-8326-210273a96552
SAVE_MODEL && let
	model_state = Flux.state(model)
	JLD2.jldsave("rice_s2_unet.jld2"; model_state)
end;

# ╔═╡ 0e837215-f43b-4521-9b01-77606dc4d0c7
let
	n = rand(1:400)
	y = ydata[n]
	#println(sum(y))
	ŷ = @pipe xdata[n] |> model(_) |> Flux.sigmoid.(_) |> (_ .>= 0.5)
	Images.mosaicview(show_mask(y), show_mask(ŷ), nrow=1, ncol=2, npad=5, fillvalue=0.8)
end

# ╔═╡ Cell order:
# ╟─d9df1d6f-69e8-4ce5-94f8-5b03d8623d13
# ╠═2937b07c-342a-11ef-1042-1d855c9401dd
# ╟─c14619c0-54a5-4be3-826c-d5689ee9a127
# ╠═ae852cbc-9da6-46da-9cc8-37c833425dc7
# ╠═88de5222-4bd4-406b-8202-cc402c251837
# ╠═d972c90d-0f86-4d6a-bdd6-e493a17a39e2
# ╟─c40df519-d8d9-4710-8ede-0b3093850da2
# ╠═7c543af1-c07b-433b-9d4a-d4f87f3fe611
# ╠═fc337c18-e1ec-4ca9-8c81-35c05b7c3607
# ╠═69833c13-1bb5-499c-a25c-613e9cb61644
# ╟─58f2e552-8f24-4fae-80c6-fdaae235d3bb
# ╠═631079ac-00d8-4fcd-ab4b-b4c60a932a5a
# ╠═f2ab25ee-9574-4289-b3f5-1f52c02475f3
# ╠═1519060a-7e9b-4fdb-b066-eeb1213b0f5c
# ╠═08daaef7-c75f-4d95-968c-c5b8e8401572
# ╠═0e095f57-440f-4ee0-8fdb-3d2f6b934e61
# ╠═32ac80ac-b8e6-4801-aab3-78afe6078856
# ╠═3eb12501-92e7-4235-8cd0-37864bb299b7
# ╟─96be9145-86f4-489c-8472-ec5be68514c3
# ╠═cc80e937-24c5-40ab-8c57-78ef291741d3
# ╠═5de7b13a-dd7d-4a96-ac7c-e08fb0fe9b68
# ╠═9c76bf2e-ce6d-4508-a8ec-1946d1849d26
# ╠═359d7124-5224-4c5b-ad04-b73f617b2ff5
# ╠═c0bd9c49-5394-4176-99a7-737c5e9bb3b2
# ╠═64dbfcc1-a686-4d17-97dc-ee1f5e186f8d
# ╠═7509e099-55fc-423d-b9a3-d7fa18682bbf
# ╟─707fedd1-bce7-4e16-9082-f13ca890b92a
# ╠═2e21bce8-99f2-4d5b-94f7-23d1530ce012
# ╠═4319dc71-db9d-4341-a872-d7472be04ce1
# ╠═ef63e89c-06e3-4ed4-8ff6-32e80a4d1c2e
# ╠═23e40dea-24e5-4e37-93eb-181f39375cca
# ╠═e15a7a09-fb5b-4176-987e-14af9a7919e2
# ╠═8181878a-50d3-4aff-bc18-04cc76923301
# ╠═4c93bc93-5106-4aa8-b702-2f7d00a38cba
# ╟─b6f908d5-28de-4b55-a688-27ec31b3cb59
# ╠═8d2944c7-4924-4ce4-8326-210273a96552
# ╠═0e837215-f43b-4521-9b01-77606dc4d0c7
