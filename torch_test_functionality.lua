require 'nn'
require 'cunn'
mlp=nn.Sequential();
c=nn.Parallel(1,2)
for i=1,10 do
 local t=nn.Sequential()
 t:add(nn.Linear(3,2))
 t:add(nn.Reshape(2,1))
 c:add(t)
end
mlp:add(c)

pred=mlp:forward(torch.randn(10,3))
print(pred)

for i=1,10000 do     -- Train for a few iterations
 x=torch.randn(10,3);
 y=torch.ones(2,10);
 pred=mlp:forward(x)

 criterion= nn.MSECriterion()
 local err=criterion:forward(pred,y)
 local gradCriterion = criterion:backward(pred,y);
 mlp:zeroGradParameters();
 mlp:backward(x, gradCriterion); 
 mlp:updateParameters(0.01);
 print(err)
end

mlp=nn.Concat(1);
mlp:add(nn.Linear(5,3))
mlp:add(nn.Linear(5,7))
print(mlp:forward(torch.randn(5)))