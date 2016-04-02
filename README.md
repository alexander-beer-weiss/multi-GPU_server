# Multi-GPU server

This is a minimal working example of a multi-threaded, multi-GPU Waffle web server.

Waffle is a fast, asynchronous, express-inspired web framework for Lua/Torch built on top of ASyNC:
https://github.com/benglard/waffle .


## Setup

You must have the following up-to-date packages installed:
torch, nn, cutorch, cunn, paths, threads, async, waffle, graphicsmagick

Torch is here:
http://torch.ch/docs/getting-started.html#installing-torch .

After Torch is installed, most of the other packages can be installed in the following manner:
```bash
	luarocks install cutorch
```

Waffle (and its depedency htmlua) can be found here:
https://github.com/benglard/waffle#installation .

Before using luarocks to install the GraphicsMagick wrapper, you must have GraphicsMagick installed.
On an Ubuntu/Debian system, this is simply:
```bash
	sudo apt-get install graphicsmagick
```
On Mac OSX:
```bash
	brew install graphicsmagick
```


## Use

To turn on the webserver:
```bash
	th multithread_server.lua
```


## Testing/Profiling

The script called multithread_posts.sh provides a test of multithread_server.lua by sending large numbers of POST requests in parallel.
For example, to send 500 POST requests in batches of 50 parallel requests:
```bash
	./multithread_posts.sh 50 500 random_image.jpg
```

To create a CPU/GPU timeline, first create an executable bash script called start_server.sh to run the server:
```bash
	#!/bin/bash
	/path/to/th /path/to/multithread_server.lua
```
Then, from the command line, run the following:
```bash
	nvprof --profile-child-processes -o timeline_%p /path/to/start_server.sh
```
This will create the timeline.  To view it, start up nvvp and import the timeline.
