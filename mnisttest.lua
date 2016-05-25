require 'torch'
require 'acdc'
require 'nn'
local mnist = require 'mnist'

mlp = nn.Sequential();
inputs = 784; outputs = 10; HUs=200;
mlp:add(nn.Reshape(784))
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(acdc.ACDC(HUs))
mlp:add(nn.Tanh())
mlp:add(acdc.ACDC(HUs))
mlp:add(nn.Tanh())
mlp:add(acdc.ACDC(HUs))
mlp:add(nn.Tanh())
mlp:add(acdc.ACDC(HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, outputs))
mlp:add(nn.LogSoftMax())
crit = nn.ClassNLLCriterion()
local trainset = mnist.traindataset()
local testset = mnist.testdataset()

for i = 1,50000 do
  local input = trainset[i].x
  local output= trainset[i].y

  -- feed it to the neural network and the criterion
  crit:forward(mlp:forward(input), output)

  -- train over this example in 3 steps
  -- (1) zero the accumulation of the gradients
  mlp:zeroGradParameters()
  -- (2) accumulate gradients
  mlp:backward(input, crit:backward(mlp.output, output))
  -- (3) update parameters with a 0.01 learning rate
  mlp:updateParameters(0.01)
end

local total_acc = 0.0;
for j = 1,5000 do
  local curr_in = testset[j].x
  local curr_out = testset[j].y
  local res = mlp:forward(curr_in)
  if res == curr_out then
    total_acc = total_acc + 1
  end
end
print(total_acc)
