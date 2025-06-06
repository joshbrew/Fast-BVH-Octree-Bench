/**
 * A small, highly‐optimized TypeScript BVH/Octree builder utility.
 * 
 * Provides:
 *  - buildBVH(...)  → constructs an axis‐aligned bounding‐box hierarchy (BVH)
 *  - buildOctree(...) → constructs a (non‐pointer) octree
 * 
 * Both return typed arrays for node bounds, leaf flags, child pointers, and leaf‐index lists.
 */

const EPS = Number.EPSILON;

/** Return type for buildBVH */
export interface BVHResult {
  /** Float32Array of length 6 * Nnodes: [cx, cy, cz, hx, hy, hz] for each node */
  data: Float32Array;
  /** Uint8Array of length Nnodes: 1 for leaf, 0 for internal */
  leafFlags: Uint8Array;
  /** Int32Array of length Nnodes: index of left child or -1 */
  childL: Int32Array;
  /** Int32Array of length Nnodes: index of right child or -1 */
  childR: Int32Array;
  /** Array of Uint32Array, one per leaf, listing the point‐indices in that leaf */
  leafIndices: Uint32Array[];
}

/** Return type for buildOctree */
export interface OctreeResult {
  /** Float32Array of length 6 * Nnodes: [cx, cy, cz, hx, hy, hz] for each node */
  data: Float32Array;
  /** Uint8Array of length Nnodes: 1 for leaf, 0 for internal */
  leafFlags: Uint8Array;
  /** Int32Array of length Nnodes: always all -1 (no child pointers) */
  childL: Int32Array;
  /** Int32Array of length Nnodes: always all -1 (no child pointers) */
  childR: Int32Array;
  /** Array of Uint32Array, one per leaf, listing the point‐indices in that leaf */
  leafIndices: Uint32Array[];
}

/**
 * Compute the AABB of a subset of points (with optional scale and radius).
 * @param p     Float32Array of length 3*Npoints: [x0,y0,z0, x1,y1,z1, …]
 * @param s     Float32Array of length 3*Npoints: scale factors per axis for each point
 * @param r     Float32Array of length Npoints: radius per point
 * @param idx   Uint32Array of point‐indices (into p, s, and r)
 * @param len   number of indices in idx to consider
 * @returns     AABB as [xmin, ymin, zmin, xmax, ymax, zmax]
 */
function computeAABB(
  p: Float32Array,
  s: Float32Array,
  r: Float32Array,
  idx: Uint32Array,
  len: number
): [number, number, number, number, number, number] {
  // initialize to very large/small
  let xmin = 1e9, ymin = 1e9, zmin = 1e9;
  let xmax = -1e9, ymax = -1e9, zmax = -1e9;

  for (let i = 0; i < len; i++) {
    const vi = idx[i] * 3;
    const x = p[vi], y = p[vi + 1], z = p[vi + 2];
    const sx = s[vi], sy = s[vi + 1], sz = s[vi + 2];
    // if radius is zero or undefined, use EPS
    const rad = r[idx[i]] || EPS;
    const minx = x - rad * sx, maxx = x + rad * sx;
    const miny = y - rad * sy, maxy = y + rad * sy;
    const minz = z - rad * sz, maxz = z + rad * sz;

    if (minx < xmin) xmin = minx;
    if (maxx > xmax) xmax = maxx;
    if (miny < ymin) ymin = miny;
    if (maxy > ymax) ymax = maxy;
    if (minz < zmin) zmin = minz;
    if (maxz > zmax) zmax = maxz;
  }

  return [xmin, ymin, zmin, xmax, ymax, zmax];
}

/**
 * Build a highly‐optimized BVH (binary bounding‐volume hierarchy) over points.
 *
 * @param p       Float32Array of length 3*Npoints: [x0,y0,z0, x1,y1,z1, …]
 * @param s       Float32Array of length 3*Npoints: per‐axis scale factors for each point
 * @param r       Float32Array of length Npoints: per‐point radius
 * @param minLeaf minimum number of points per leaf; if a partition yields both child sets ≥ minLeaf, split; otherwise make leaf
 * @returns       BVHResult with node bounds, leaf flags, child pointers, and leaf‐index lists
 */
export function buildBVH(
  p: Float32Array,
  s: Float32Array,
  r: Float32Array,
  minLeaf: number
): BVHResult {
  const N = p.length / 3;
  if (N === 0) {
    return {
      data: new Float32Array(0),
      leafFlags: new Uint8Array(0),
      childL: new Int32Array(0),
      childR: new Int32Array(0),
      leafIndices: []
    };
  }

  // 1) Initialize a single indices buffer: [0,1,2,…,N-1]
  const indices = new Uint32Array(N);
  for (let i = 0; i < N; i++) {
    indices[i] = i;
  }

  /** Job descriptor for iterative stack */
  interface Job {
    start: number;       // start index into `indices`
    len: number;         // length of this subset
    parentIndex: number; // index into jobList of parent node, or -1 for root
    isLeft: boolean;     // whether this node is left‐child (true) or right‐child (false) of its parent
    leaf: boolean;       // will be marked true if leaf
    nodeIndex?: number;  // assigned after jobList complete
  }

  // 2) Iteratively build jobList, partitioning in place
  const jobList: Job[] = [];
  const stack: Job[] = [];
  stack.push({ start: 0, len: N, parentIndex: -1, isLeft: false, leaf: false });

  while (stack.length > 0) {
    const job = stack.pop() as Job;
    jobList.push(job);
    const currIdx = job.start;
    const currLen = job.len;

    // Compute bounding box for this job (used to decide splitting axis)
    const box = computeAABB(
      p,
      s,
      r,
      indices.subarray(currIdx, currIdx + currLen),
      currLen
    );
    // compute center and half‐sizes (not saved now; recomputed later)
    const cx = 0.5 * (box[0] + box[3]);
    const cy = 0.5 * (box[1] + box[4]);
    const cz = 0.5 * (box[2] + box[5]);
    const hx = 0.5 * (box[3] - box[0]);
    const hy = 0.5 * (box[4] - box[1]);
    const hz = 0.5 * (box[5] - box[2]);

    let isLeaf = true;
    if (currLen > minLeaf) {
      // pick split axis = dimension with largest span
      let axis = hx * 2 >= hy * 2 ? 0 : 1;
      if (hz * 2 > (axis === 0 ? hx * 2 : hy * 2)) {
        axis = 2;
      }
      const midVal = axis === 0 ? cx : axis === 1 ? cy : cz;

      // partition in place: all points ≤ midVal on left, > midVal on right
      let leftWrite = currIdx;
      for (let i = currIdx; i < currIdx + currLen; i++) {
        const v = p[indices[i] * 3 + axis];
        if (v <= midVal) {
          const tmp = indices[i];
          indices[i] = indices[leftWrite];
          indices[leftWrite] = tmp;
          leftWrite++;
        }
      }
      const leftCount = leftWrite - currIdx;
      const rightCount = currLen - leftCount;

      // Only split if both children ≥ minLeaf
      if (leftCount >= minLeaf && rightCount >= minLeaf) {
        isLeaf = false;
        // Create right child job
        stack.push({
          start: currIdx + leftCount,
          len: rightCount,
          parentIndex: jobList.length - 1,
          isLeft: false,
          leaf: false
        });
        // Create left child job
        stack.push({
          start: currIdx,
          len: leftCount,
          parentIndex: jobList.length - 1,
          isLeft: true,
          leaf: false
        });
      }
    }

    job.leaf = isLeaf;
  }

  // 3) We now know how many nodes (jobList.length). Assign nodeIndex to each job.
  const Nnodes = jobList.length;
  for (let i = 0; i < Nnodes; i++) {
    jobList[i].nodeIndex = i;
  }

  // 4) Allocate output arrays
  const data = new Float32Array(Nnodes * 6);    // [cx, cy, cz, hx, hy, hz] × Nnodes
  const leafFlags = new Uint8Array(Nnodes);     // 1 if leaf, else 0
  const childL = new Int32Array(Nnodes);        // left child index or -1
  const childR = new Int32Array(Nnodes);        // right child index or -1
  const leafIndices: Uint32Array[] = [];

  // Initialize child pointers to -1
  for (let i = 0; i < Nnodes; i++) {
    childL[i] = -1;
    childR[i] = -1;
  }

  // 5) Second pass: fill node bounds, leaf flags, collect leafIndices, and wire up child pointers
  for (let i = 0; i < Nnodes; i++) {
    const job = jobList[i];
    const idxStart = job.start;
    const idxLen = job.len;
    const subIdx = indices.subarray(idxStart, idxStart + idxLen);

    // Compute BBox for this node
    const box = computeAABB(p, s, r, subIdx, idxLen);
    const cx = 0.5 * (box[0] + box[3]);
    const cy = 0.5 * (box[1] + box[4]);
    const cz = 0.5 * (box[2] + box[5]);
    const hx = 0.5 * (box[3] - box[0]);
    const hy = 0.5 * (box[4] - box[1]);
    const hz = 0.5 * (box[5] - box[2]);

    const offset = i * 6;
    data[offset]     = cx;
    data[offset + 1] = cy;
    data[offset + 2] = cz;
    data[offset + 3] = hx;
    data[offset + 4] = hy;
    data[offset + 5] = hz;

    leafFlags[i] = job.leaf ? 1 : 0;

    if (job.leaf) {
      // copy leaf indices into a fresh Uint32Array
      leafIndices.push(new Uint32Array(subIdx));
    }

    // Link to parent: assign this node as left or right child of its parent
    if (job.parentIndex >= 0) {
      const parentNodeIndex = jobList[job.parentIndex].nodeIndex as number;
      if (job.isLeft) {
        childL[parentNodeIndex] = job.nodeIndex as number;
      } else {
        childR[parentNodeIndex] = job.nodeIndex as number;
      }
    }
  }

  return {
    data,
    leafFlags,
    childL,
    childR,
    leafIndices
  };
}


/**
 * Build a (non‐pointer) Octree over points.
 *
 * Uses a simple iterative stack, no per‐node child pointers (all will remain -1).
 * Splits space into up to 8 octants as long as more than one octant contains points
 * and the node has > minLeaf points. If cubic=true, force the root bounding box to be a cube.
 *
 * @param p       Float32Array of length 3*Npoints
 * @param s       Float32Array of length 3*Npoints
 * @param r       Float32Array of length Npoints
 * @param minLeaf minimum number of points per leaf
 * @param cubic   if true, root box is expanded to a cube
 * @returns       OctreeResult with node bounds, leaf flags, child pointers, and leaf‐index lists
 */
export function buildOctree(
  p: Float32Array,
  s: Float32Array,
  r: Float32Array,
  minLeaf: number,
  cubic: boolean
): OctreeResult {
  const N = p.length / 3;
  if (N === 0) {
    return {
      data: new Float32Array(0),
      leafFlags: new Uint8Array(0),
      childL: new Int32Array(0),
      childR: new Int32Array(0),
      leafIndices: []
    };
  }

  // initial full‐set indices
  const indices = new Uint32Array(N);
  for (let i = 0; i < N; i++) {
    indices[i] = i;
  }

  // Compute root AABB
  let rootBox = computeAABB(p, s, r, indices, N);
  if (cubic) {
    const sizeX = rootBox[3] - rootBox[0];
    const sizeY = rootBox[4] - rootBox[1];
    const sizeZ = rootBox[5] - rootBox[2];
    const size = Math.max(sizeX, sizeY, sizeZ);
    const cx = 0.5 * (rootBox[0] + rootBox[3]);
    const cy = 0.5 * (rootBox[1] + rootBox[4]);
    const cz = 0.5 * (rootBox[2] + rootBox[5]);
    rootBox = [
      cx - 0.5 * size, cy - 0.5 * size, cz - 0.5 * size,
      cx + 0.5 * size, cy + 0.5 * size, cz + 0.5 * size
    ];
  }

  /** Octree job, storing index‐subset and bounding box */
  interface OTJob {
    start: number;        // start index into `indices`
    len: number;          // number of points
    box: [number, number, number, number, number, number]; // bounding box
  }

  const stack: OTJob[] = [];
  const outBoxes: number[] = [];      // flattened [cx,cy,cz,hx,hy,hz,...]
  const leafFlagsArr: number[] = [];  // parallel array of 1/0
  const leafIndices: Uint32Array[] = [];

  stack.push({ start: 0, len: N, box: rootBox });
  const counts = new Uint32Array(8);

  // iterative loop
  while (stack.length > 0) {
    const job = stack.pop() as OTJob;
    const [minX, minY, minZ, maxX, maxY, maxZ] = job.box;
    const cx = 0.5 * (minX + maxX);
    const cy = 0.5 * (minY + maxY);
    const cz = 0.5 * (minZ + maxZ);
    const hx = 0.5 * (maxX - minX);
    const hy = 0.5 * (maxY - minY);
    const hz = 0.5 * (maxZ - minZ);

    // default: mark as leaf
    let isLeaf = true;

    if (job.len > minLeaf) {
      counts.fill(0);
      // count how many points go to each octant
      for (let i = 0; i < job.len; i++) {
        const idx = indices[job.start + i] * 3;
        let oct = 0;
        if (p[idx] > cx) oct |= 1;
        if (p[idx + 1] > cy) oct |= 2;
        if (p[idx + 2] > cz) oct |= 4;
        counts[oct]++;
      }
      let nonEmpty = 0;
      for (let o = 0; o < 8; o++) {
        if (counts[o] > 0) nonEmpty++;
      }

      if (nonEmpty > 1) {
        // can split into octants
        isLeaf = false;
        // compute offsets
        const offs = new Uint32Array(8);
        for (let o = 1; o < 8; o++) {
          offs[o] = offs[o - 1] + counts[o - 1];
        }
        const tmp = new Uint32Array(counts);

        // pack into an intermediate array
        const packed = new Uint32Array(job.len);
        for (let i = 0; i < job.len; i++) {
          const idx = indices[job.start + i] * 3;
          let oct = 0;
          if (p[idx] > cx) oct |= 1;
          if (p[idx + 1] > cy) oct |= 2;
          if (p[idx + 2] > cz) oct |= 4;
          packed[offs[oct] + (tmp[oct] - 1)] = indices[job.start + i];
          tmp[oct]--;
        }
        // overwrite the region in indices
        for (let i = 0; i < job.len; i++) {
          indices[job.start + i] = packed[i];
        }

        // spawn child jobs for each nonempty octant
        for (let o = 0; o < 8; o++) {
          const cnt = counts[o];
          if (cnt === 0) continue;
          let childBox: [number, number, number, number, number, number];
          if (cubic) {
            const minCx = minX + ((o & 1) ? hx : 0);
            const minCy = minY + ((o & 2) ? hy : 0);
            const minCz = minZ + ((o & 4) ? hz : 0);
            const maxCx = minX + ((o & 1) ? hx * 2 : hx);
            const maxCy = minY + ((o & 2) ? hy * 2 : hy);
            const maxCz = minZ + ((o & 4) ? hz * 2 : hz);
            childBox = [minCx, minCy, minCz, maxCx, maxCy, maxCz];
          } else {
            const childStart = job.start + offs[o];
            const subIdx = indices.subarray(childStart, childStart + cnt);
            childBox = computeAABB(p, s, r, subIdx, cnt);
          }
          stack.push({ start: job.start + offs[o], len: cnt, box: childBox });
        }
      }
    }

    outBoxes.push(cx, cy, cz, hx, hy, hz);
    leafFlagsArr.push(isLeaf ? 1 : 0);

    if (isLeaf) {
      // copy leaf indices into a fresh Uint32Array
      const subIdx = indices.subarray(job.start, job.start + job.len);
      leafIndices.push(new Uint32Array(subIdx));
    }
  }

  // flatten results into typed arrays
  const Nnodes = outBoxes.length / 6;
  const data = new Float32Array(Nnodes * 6);
  for (let i = 0; i < Nnodes * 6; i++) {
    data[i] = outBoxes[i];
  }
  const leafFlags = new Uint8Array(Nnodes);
  for (let i = 0; i < Nnodes; i++) {
    leafFlags[i] = leafFlagsArr[i];
  }
  // child pointers are not used in this simple octree
  const childL = new Int32Array(Nnodes);
  const childR = new Int32Array(Nnodes);
  for (let i = 0; i < Nnodes; i++) {
    childL[i] = -1;
    childR[i] = -1;
  }

  return {
    data,
    leafFlags,
    childL,
    childR,
    leafIndices
  };
}


/*

import { buildBVH, buildOctree } from "./bvhOctreeUtil";

// Suppose you have N points with:
//   pos  = Float32Array.of(x0, y0, z0,  x1, y1, z1,  …);
//   scl  = Float32Array.of(sx0, sy0, sz0,  sx1, sy1, sz1,  …);
//   rad  = Float32Array.of(r0, r1, r2, …);
// Choose a minLeaf, e.g. 8:

const minLeaf = 8;
const bvhResult = buildBVH(pos, scl, rad, minLeaf);
// bvhResult.data    → Float32Array([cx0, cy0, cz0, hx0, hy0, hz0,  cx1, …])
// bvhResult.leafFlags → Uint8Array([0, 0, 1, 1, …])
// bvhResult.childL   → Int32Array([1, 2, -1, -1, …])
// bvhResult.childR   → Int32Array([3, 4, -1, -1, …])
// bvhResult.leafIndices → e.g. [Uint32Array([idxA, idxB,…]), …]

const cubic = true;
const octResult = buildOctree(pos, scl, rad, minLeaf, cubic);
// octResult.data       → Float32Array([cx0, cy0, cz0, hx0, hy0, hz0,  cx1, …])
// octResult.leafFlags  → Uint8Array([0, 1, 1, 0, …])
// octResult.childL/R   → all −1 (no pointers)
// octResult.leafIndices → Array<Uint32Array> with one entry per octree leaf

*/