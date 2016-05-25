require 'torch'
require 'acdc'
require 'nn'
require 'mnist'

mlp = nn.Sequential();
inputs = 784; outputs = 10; HUs=200;
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
crit = nn.CrossEntropyCriterion()

for i = 1,2500 do
  -- random sample
  local input = mnist.something;
  -- local input= torch.randn(2);     -- normally distributed example in 2d
  local output= mnist.something;

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

x = torch.Tensor(2) -- get the accuracy
