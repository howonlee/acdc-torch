require 'torch'
require 'acdc'
require 'nn'
require 'math'
local mnist = require 'mnist'

mlp = nn.Sequential();
inputs = 784; outputs = 10; HUs=200;
mlp:add(nn.Reshape(784))
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
--mlp:add(nn.Linear(HUs, HUs))
mlp:add(nn.Linear(HUs, outputs))
mlp:add(nn.LogSoftMax())
crit = nn.ClassNLLCriterion()
local trainset = mnist.traindataset()
local testset = mnist.testdataset()

for i = 1,5000 do
  if math.fmod(i, 100) == 0 then
    print(i)
  end
  local input = trainset[i].x:double()
  local output = trainset[i].y + 1 -- I am pretty sure this is not how it's supposed to work
  -- local output = torch.Tensor(10):double()

  crit:forward(mlp:forward(input), output)

  -- train over this example in 3 steps
  -- (1) zero the accumulation of the gradients
  mlp:zeroGradParameters()
  -- (2) accumulate gradients
  mlp:backward(input, crit:backward(mlp.output, output))
  -- (3) update parameters with a 0.01 learning rate
  mlp:updateParameters(0.01)
end

print('start the eval')
local total_acc = 0.0;
for j = 1,5000 do
  local curr_in = testset[j].x:double()
  local curr_out = testset[j].y + 1
  _, res = torch.max(mlp:forward(curr_in), 1)
  if res[1] == curr_out then
    total_acc = total_acc + 1
  end
end
print(total_acc)
