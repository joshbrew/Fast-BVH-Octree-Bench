# Fast-BVH-Octree-Bench
Fast multithreaded BVH and Octree (dynamic forced cubes) generation in plain JS, ~10ms for 10,000 entities with leaf indexing for traversal! Traversal in our test is on the order of nanoseconds.

Improvements would come in the form of SharedArrayBuffers (requires hosted context) or fully offloading to a compute shader (tough as nuts! https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/). Nvidia's 2012 benchmark here is 0.53ms for 12.5K objects, we are still off by 20X without GPU on a single 2020-era cpu thread. 

### [Try it](https://codepen.io/mootytootyfrooty/pen/ogXBzwE)

On the right, that's 6-12 microseconds per collision step for a single particle, without offloading render and UI thread. Pretty good! We could do better. (e.g. https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/) Their tree traversal for 12.5K objects took 0.25ms, but this is on a 2012 gpu and also should work parallel for the number of cores the gpu supports, so we are cooked with our cpu implementation except for more minor tasks.

![image](https://github.com/user-attachments/assets/c74981a5-0dab-447d-84d6-ed4035ad789d)
![image](https://github.com/user-attachments/assets/8d45f303-3bc4-4c27-8bcc-663e41e5a803)
![image](https://github.com/user-attachments/assets/aa6f325f-babc-431c-acb3-140da3b700f3)

