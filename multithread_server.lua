require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

require 'paths'
require 'nn'
require 'cutorch'
require 'cunn'

local gm = require 'graphicsmagick'   -- used to convert image string to Torch tensor
local async = require 'async'   -- used to restart server
local waffle = require 'waffle'
local threads = require 'threads'
threads.serialization('threads.sharedserialize')

-- define server address address
local host = '127.0.0.1'   -- switch to '0.0.0.0' for external use
local port = 8080

-- define number of GPUs available
local num_GPUs = 4

-- define number of neural nets and depth of nets
local num_nets = 10
local num_layers = 18

-- define the number of nets that will reside on multiple GPUs (all others will reside on a single GPU)
local num_popular_nets = 2

------------------------------------------------------------------------------
-- create a set of convolutional neural networks and move them to the GPUs

-- the following completely arbitrary nets are created for purpose of example only
print('Creating ' .. num_nets .. ' convolutional nets')
mem_nets = {}
for net_idx = 1, num_nets do
	local net = nn.Sequential()
	net:add( nn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1) )
	net:add( nn.ReLU() )
	for layer = 1, num_layers do
		net:add( nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1) )
		net:add( nn.ReLU() )
	end
	net:add( nn.View() )
	net:add( nn.Mean() )
	net:add( nn.Sigmoid() )
	mem_nets[net_idx] = net
end
print(num_nets .. ' nets created')
print('')


-- initiate a table for nets on the GPUs; there will be one sub-table for each GPU
local gpu_nets = {}

-- initiate sub-tables of gpu_nets and check memory consumption on each GPU
for gpu_idx = 1, num_GPUs do
	gpu_nets[gpu_idx] = {}
	print('GPU ' .. gpu_idx .. ' memory usage',cutorch.getMemoryUsage(gpu_idx))
end
print('')


-- move nets to GPUs; "popular nets" which will reside on all GPUs
print('Moving nets to GPU...')
for net_idx = 1, num_nets do
	if net_idx <= num_popular_nets then
		cutorch.setDevice(1)
		gpu_nets[1]['net'..net_idx] = mem_nets[net_idx]:cuda()
		for gpu_idx = 1, num_GPUs do
			cutorch.setDevice(gpu_idx)
			gpu_nets[gpu_idx]['net'..net_idx] = gpu_nets[1]['net'..net_idx]:clone()
		end
	else
		local gpu_idx = (net_idx - num_popular_nets - 1) % num_GPUs + 1
		cutorch.setDevice(gpu_idx)
		gpu_nets[gpu_idx]['net'..net_idx] = mem_nets[net_idx]:cuda()
	end
end

-- print information about each GPU
for gpu_idx = 1, num_GPUs do
	local loaded_nets_list = ''
	for loaded_net, _ in pairs(gpu_nets[gpu_idx]) do
		loaded_nets_list = loaded_nets_list .. loaded_net .. ' '
	end
	print('GPU ' .. gpu_idx .. ' : ' .. loaded_nets_list , 'memory usage', cutorch.getMemoryUsage(ne_idx))
end

print('')

------------------------------------------------------------------------------
-- create thread pool

local num_threads = num_GPUs
local pool = threads.Threads(

	-- create one thread for each GPU
	num_threads,
	
	-- initiate each thread with needed packages; set GPU for thread
	function(thread_ID)
		require 'nn'
		require 'cutorch'
		require 'cunn'
		cutorch.setDevice(thread_ID)
	end
)

-- thread will be chosen and specified dynamically
pool:specific(true)

print('threads created')

------------------------------------------------------------------------------
-- define GET functions for image analysis form and server termination

waffle.get('/image', function(req, res)
	
	res.send(html { body { form {
		action = '/m',
		method = 'POST',
		enctype = 'multipart/form-data',
		p { input {
			type = 'text',
			name = 'net',
			placeholder = 'choose net'
		}},
		p { input {
			type = 'file',
			name = 'file' 
		}},
		p { input {
			type = 'submit', 'Upload'
		}}
	}}})
   
end)

-- GET function for remote termination of web server
-- this provides a clean termination of the program for profiling purposes
waffle.get('/kill', function(req, res)
	res.send('Goodbye!\n')
	print('Server terminated by client\n')
	pool:terminate()
	os.exit()
end)

------------------------------------------------------------------------------
-- Dynamically choose which GPU to assign each job to

-- initially each GPU has no jobs in queue
local num_jobs = torch.Tensor(num_GPUs):fill(0)

-- function to determine the least-busy, eligble GPU to process a given request
local function choose_GPU(net_ID)
	
	local chosen_GPU = -1
	local min_num_jobs = math.huge
	for gpu_idx = 1, num_GPUs do
		if gpu_nets[gpu_idx]['net'..net_ID] and num_jobs[gpu_idx] < min_num_jobs then
			min_num_jobs = num_jobs[gpu_idx]
			chosen_GPU = gpu_idx
		end
	end
	
	return chosen_GPU
end

------------------------------------------------------------------------------
-- define POST function for image analysis

waffle.post('/image', function(req, res)
	
	-- convert image string to tensor
	local img = req.form.file:toImage()
	
	-- store image file name
	local img_name = req.form.file.filename
	
	-- net_ID is a number between 1 and 10 which identifies the requested neural net
	local net_ID = req.form.net
	
	-- determine the best GPU/thread to process request
	local thread_ID = choose_GPU(net_ID)
	
	-- increment num_jobs counter for current GPU
	num_jobs[thread_ID] = num_jobs[thread_ID] + 1
	
	-- create reference to the requested net on chosen GPU
	local net = gpu_nets[thread_ID]['net'..net_ID]
	
	-- if requested net exists, then push new job into queue of chosen thread
	if thread_ID ~= -1 then
		
		-- the job will be serialized and executed later via Threads:dojob
		pool:addjob(
			
			-- specify thread associated to chosen GPU
			thread_ID,
			
			-- this function will be processed by thread thread_ID
			function()
				
				-- forward input through neural net, then clear net buffers to save space on GPU
				local output = net:forward( img:cuda() ):float()[1]
				net:clearState()
				
				-- return output to master thread
				return output
				
			end,
			
			-- this function will be processed by the master thread
			function(output)
				
				-- decrement num_jobs counter for current GPU
				num_jobs[thread_ID] = num_jobs[thread_ID] - 1
				
				-- return result to client (WARNING: do not use res.send)
				-- req persists beyond the completion of the current route handler function (as does the seralized job)
				req:finish( tostring(output) )
				
			end
		)
		
	else
		
		-- decrement num_jobs counter for current GPU (job was canceled)
		num_jobs[thread_ID] = num_jobs[thread_ID] - 1
		
		-- return warning to client
		res.send('INVALID NET REQUESTED : CHOOSE INTERGER IN THE SET [1,' .. num_nets .. ']')
		
	end
	
end)

------------------------------------------------------------------------------
-- crash-proof server

-- run_jobs runs periodically to clear job queue
local function run_jobs()
	while pool:hasjob() do
		pool:dojob()
	end
end

-- run_jobs is called once per jobs_interval milliseconds
local jobs_interval = 50

-- restart crashed server
local function restart_server()
	print('restarting server ...')
	local okay, err = xpcall(async.go, debug.traceback, run_jobs, jobs_interval)
	print(err)
	restart_server()
end

-- start listening
local okay, err = xpcall(waffle.listen, debug.traceback, {host=host, port=port}, run_jobs, jobs_interval)
print(err)
restart_server()
