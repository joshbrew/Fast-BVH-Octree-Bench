# Fast-BVH-Octree-Bench
Fast multithreaded BVH and Octree (dynamic forced cubes) generation in plain JS, ~4ms for 10,000 entities!

Improvements would come in the form of SharedArrayBuffers (requires hosted context) or fully offloading to a compute shader (tough as nuts!)

Note this sample doesn't have the tree traversal logic for resolving collisions. 

### [Try it](https://codepen.io/mootytootyfrooty/pen/ogXBzwE)

![image](https://github.com/user-attachments/assets/79cb3cdf-03fb-44e3-879c-def8e3a8d56b)
![image](https://github.com/user-attachments/assets/ff8d2d12-c4ad-48b0-bb0c-fa1e1e8fde74)
