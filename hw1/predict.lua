require 'nn'
require 'optim'
require 'cunn'


function predict(model_name)
local mnist = require 'mnist';

local trainData = mnist.traindataset().data:float();
local trainLabels = mnist.traindataset().label:add(1);
data = mnist.testdataset().data:float();
labels = mnist.testdataset().label:add(1);

local mean = trainData:mean()
local std = trainData:std()
data:add(-mean):div(std);
model = torch.load(model_name)
local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
local x=data:narrow(1,1,data:size(1)):cuda()

data=data:cuda()
labels=labels:cuda()
local y = model:forward(data)
confusion:batchAdd(y,labels)        
 
confusion:updateValids()
    local avgError = 1 - confusion.totalValid

    return  avgError
end

a=predict('hw1.t7')
print (a)