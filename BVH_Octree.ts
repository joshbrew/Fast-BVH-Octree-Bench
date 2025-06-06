/**
 * A small, highly‐optimized TypeScript BVH/Octree builder utility.
 *
 * Provides:
 *  - buildBVH(...)    → constructs an axis‐aligned bounding‐box hierarchy (BVH)
 *  - buildOctree(...) → constructs a (pointer‐free) octree, returning explicit children‐lists
 *
 * In addition, we expose:
 *  - findLeafBVH(...)         → descend BVH to find a leaf index for a query point
 *  - collectHierarchyBVH(...) → return full root‐to‐leaf node‐index path in BVH
 *  - findLeafOctree(...)         → descend Octree to find a leaf index for a query point
 *  - collectHierarchyOctree(...) → return full root‐to‐leaf node‐index path in Octree
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
  /** Int32Array of length Nnodes: always all -1 (no binary child pointers) */
  childL: Int32Array;
  /** Int32Array of length Nnodes: always all -1 (no binary child pointers) */
  childR: Int32Array;
  /** Array of Uint32Array, one per leaf, listing the point‐indices in that leaf */
  leafIndices: Uint32Array[];
  /**
   * Array of Uint32Array, one per node, listing up to 8 child‐node indices.  
   * If a node is a leaf, childrenIdxs[nodeIndex] is an empty Uint32Array.
   */
  childrenIdxs: Uint32Array[];
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
  let xmin = 1e9, ymin = 1e9, zmin = 1e9;
  let xmax = -1e9, ymax = -1e9, zmax = -1e9;

  for (let i = 0; i < len; i++) {
    const vi = idx[i] * 3;
    const x = p[vi], y = p[vi + 1], z = p[vi + 2];
    const sx = s[vi], sy = s[vi + 1], sz = s[vi + 2];
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
 * Build a (non‐pointer) Octree over points, but also record up to 8 child pointers per node.
 *
 * Uses a two‐pass approach (identical in style to buildBVH):
 *   - Pass 1: gather all jobs into jobList[], each with parentIndex
 *   - Pass 2: assign nodeIndex, build childrenIdxs[], then flatten out box/leaf/leafIndices
 *
 * @param p       Float32Array of length 3*Npoints
 * @param s       Float32Array of length 3*Npoints
 * @param r       Float32Array of length Npoints
 * @param minLeaf minimum number of points per leaf
 * @param cubic   if true, root box is expanded to a cube
 * @returns       OctreeResult with node bounds, leaf flags, binary child pointers (all -1), 
 *                plus leafIndices and childrenIdxs
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
      leafIndices: [],
      childrenIdxs: []
    };
  }

  // 1) Initial indices: [0,1,2,…,N-1]
  const indices = new Uint32Array(N);
  for (let i = 0; i < N; i++) {
    indices[i] = i;
  }

  // Compute root bounding box
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

  /** Octree job (for Pass 1) */
  interface OTJob {
    start: number;  // start index into `indices`
    len: number;    // number of points in this node
    box: [number, number, number, number, number, number]; // bounding box
    parentIndex: number; // index in jobList[] of this job's parent, or -1 for root
    nodeIndex?: number;  // assigned in Pass 2
    leaf: boolean;       // assigned in Pass 1
  }

  const jobList: OTJob[] = [];
  const stack: OTJob[] = [];
  stack.push({ start: 0, len: N, box: rootBox, parentIndex: -1, leaf: false });

  // Temporary array to count octant populations
  const counts = new Uint32Array(8);

  // 2) Pass 1: gather all OTJob entries, partition in place, record parentIndex
  while (stack.length > 0) {
    const job = stack.pop() as OTJob;
    jobList.push(job);

    const [minX, minY, minZ, maxX, maxY, maxZ] = job.box;
    const cx = 0.5 * (minX + maxX);
    const cy = 0.5 * (minY + maxY);
    const cz = 0.5 * (minZ + maxZ);
    const hx = 0.5 * (maxX - minX);
    const hy = 0.5 * (maxY - minY);
    const hz = 0.5 * (maxZ - minZ);

    let isLeaf = true;
    if (job.len > minLeaf) {
      counts.fill(0);
      // count how many points land in each octant
      for (let i = 0; i < job.len; i++) {
        const vi = indices[job.start + i] * 3;
        let oct = 0;
        if (p[vi]     > cx) oct |= 1;
        if (p[vi + 1] > cy) oct |= 2;
        if (p[vi + 2] > cz) oct |= 4;
        counts[oct]++;
      }
      let nonEmpty = 0;
      for (let o = 0; o < 8; o++) {
        if (counts[o] > 0) nonEmpty++;
      }
      if (nonEmpty > 1) {
        isLeaf = false;
        // compute prefix sums → offsets
        const offs = new Uint32Array(8);
        for (let o = 1; o < 8; o++) offs[o] = offs[o - 1] + counts[o - 1];
        const tmp = counts.slice();

        // pack into a temporary array
        const packed = new Uint32Array(job.len);
        for (let i = 0; i < job.len; i++) {
          const vi = indices[job.start + i] * 3;
          let oct = 0;
          if (p[vi]     > cx) oct |= 1;
          if (p[vi + 1] > cy) oct |= 2;
          if (p[vi + 2] > cz) oct |= 4;
          const pos = offs[oct] + (tmp[oct] - 1);
          packed[pos] = indices[job.start + i];
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
            // always subdivide equally
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
          stack.push({
            start: job.start + offs[o],
            len: cnt,
            box: childBox,
            parentIndex: jobList.length - 1,
            leaf: false
          });
        }
      }
    }

    job.leaf = isLeaf;
  }

  // 3) Assign nodeIndex to every job
  const Nnodes = jobList.length;
  for (let i = 0; i < Nnodes; i++) {
    jobList[i].nodeIndex = i;
  }

  // 4) Build a childrenIndices array of length Nnodes
  const childrenIndices: number[][] = Array.from({ length: Nnodes }, () => []);
  for (let i = 0; i < Nnodes; i++) {
    const job = jobList[i];
    if (job.parentIndex >= 0) {
      const pi = job.parentIndex;
      childrenIndices[pi].push(job.nodeIndex as number);
    }
  }
  // convert each childrenIndices[i] to a Uint32Array
  const childrenIdxs: Uint32Array[] = childrenIndices.map(arr => new Uint32Array(arr));

  // 5) Now build output arrays: data, leafFlags, leafIndices
  const outBoxes: number[] = [];      // flattened [cx,cy,cz,hx,hy,hz,...]
  const leafFlagsArr: number[] = [];  // 1 or 0
  const leafIndices: Uint32Array[] = [];

  // childL/childR remain all -1
  const childL = new Int32Array(Nnodes);
  const childR = new Int32Array(Nnodes);
  for (let i = 0; i < Nnodes; i++) {
    childL[i] = -1;
    childR[i] = -1;
  }

  // 6) Pass 2: fill outBoxes, leafFlagsArr, leafIndices
  while (outBoxes.length < Nnodes * 6) {
    const i = outBoxes.length / 6;
    const job = jobList[i];
    const [minX, minY, minZ, maxX, maxY, maxZ] = job.box;
    const cx = 0.5 * (minX + maxX);
    const cy = 0.5 * (minY + maxY);
    const cz = 0.5 * (minZ + maxZ);
    const hx = 0.5 * (maxX - minX);
    const hy = 0.5 * (maxY - minY);
    const hz = 0.5 * (maxZ - minZ);

    outBoxes.push(cx, cy, cz, hx, hy, hz);
    leafFlagsArr.push(job.leaf ? 1 : 0);

    if (job.leaf) {
      const subIdx = indices.subarray(job.start, job.start + job.len);
      leafIndices.push(new Uint32Array(subIdx));
    }
  }

  // flatten outBoxes into Float32Array
  const data = new Float32Array(Nnodes * 6);
  for (let i = 0; i < Nnodes * 6; i++) {
    data[i] = outBoxes[i];
  }
  const leafFlags = new Uint8Array(Nnodes);
  for (let i = 0; i < Nnodes; i++) {
    leafFlags[i] = leafFlagsArr[i];
  }

  return {
    data,
    leafFlags,
    childL,
    childR,
    leafIndices,
    childrenIdxs
  };
}


/* ─────────────────────────────────────────────────────────────────────────────
   “Tree‐Traversal” helpers
──────────────────────────────────────────────────────────────────────────── */

/**
 * Descend a BVH to find the leaf node index that contains (x,y,z).
 * Returns -1 if point falls outside the root AABB (i.e. no containment).
 *
 * @param data      Float32Array of length 6*Nnodes: [cx, cy, cz, hx, hy, hz] per node
 * @param leafFlags Uint8Array of length Nnodes: 1 if leaf, 0 if internal
 * @param childL    Int32Array of length Nnodes: left‐child index or -1
 * @param childR    Int32Array of length Nnodes: right‐child index or -1
 * @param x         point x‐coordinate
 * @param y         point y‐coordinate
 * @param z         point z‐coordinate
 * @returns         index of leaf node containing (x,y,z), or -1 if none
 */
export function findLeafBVH(
  data: Float32Array,
  leafFlags: Uint8Array,
  childL: Int32Array,
  childR: Int32Array,
  x: number, y: number, z: number
): number {
  // First check root AABB
  const cx0 = data[0], cy0 = data[1], cz0 = data[2];
  const hx0 = data[3], hy0 = data[4], hz0 = data[5];
  if (
    x < cx0 - hx0 || x > cx0 + hx0 ||
    y < cy0 - hy0 || y > cy0 + hy0 ||
    z < cz0 - hz0 || z > cz0 + hz0
  ) {
    return -1;
  }

  let nodeIdx = 0;
  while (true) {
    if (leafFlags[nodeIdx] === 1) {
      return nodeIdx;
    }

    // Check left child
    const L = childL[nodeIdx];
    if (L >= 0) {
      const ld = L * 6;
      const lcX = data[ld],   lcY = data[ld + 1], lcZ = data[ld + 2];
      const lhX = data[ld + 3], lhY = data[ld + 4], lhZ = data[ld + 5];
      if (
        x >= lcX - lhX && x <= lcX + lhX &&
        y >= lcY - lhY && y <= lcY + lhY &&
        z >= lcZ - lhZ && z <= lcZ + lhZ
      ) {
        nodeIdx = L;
        continue;
      }
    }

    // Otherwise, check right child
    const R = childR[nodeIdx];
    if (R >= 0) {
      const rd = R * 6;
      const rcX = data[rd],   rcY = data[rd + 1], rcZ = data[rd + 2];
      const rhX = data[rd + 3], rhY = data[rd + 4], rhZ = data[rd + 5];
      if (
        x >= rcX - rhX && x <= rcX + rhX &&
        y >= rcY - rhY && y <= rcY + rhY &&
        z >= rcZ - rhZ && z <= rcZ + rhZ
      ) {
        nodeIdx = R;
        continue;
      }
    }

    // If neither child contains the point, we are outside
    return -1;
  }
}

/**
 * Return the full root‐to‐leaf node‐index path in a BVH for (x,y,z).
 * The returned array always begins with 0 (the root index) and ends with the leaf index.
 * If the point is not contained in the root, returns an empty array.
 *
 * @param data      Float32Array of length 6*Nnodes
 * @param leafFlags Uint8Array of length Nnodes
 * @param childL    Int32Array of length Nnodes
 * @param childR    Int32Array of length Nnodes
 * @param x         point x‐coordinate
 * @param y         point y‐coordinate
 * @param z         point z‐coordinate
 * @returns         array of node‐indices along the descent; [] if
 *                  (x,y,z) is not in the BVH root.
 */
export function collectHierarchyBVH(
  data: Float32Array,
  leafFlags: Uint8Array,
  childL: Int32Array,
  childR: Int32Array,
  x: number, y: number, z: number
): number[] {
  const path: number[] = [];

  // Check root AABB
  const cx0 = data[0], cy0 = data[1], cz0 = data[2];
  const hx0 = data[3], hy0 = data[4], hz0 = data[5];
  if (
    x < cx0 - hx0 || x > cx0 + hx0 ||
    y < cy0 - hy0 || y > cy0 + hy0 ||
    z < cz0 - hz0 || z > cz0 + hz0
  ) {
    return [];
  }

  let nodeIdx = 0;
  while (true) {
    path.push(nodeIdx);
    if (leafFlags[nodeIdx] === 1) {
      break;
    }

    // Try left child
    const L = childL[nodeIdx];
    if (L >= 0) {
      const ld = L * 6;
      const lcX = data[ld],   lcY = data[ld + 1], lcZ = data[ld + 2];
      const lhX = data[ld + 3], lhY = data[ld + 4], lhZ = data[ld + 5];
      if (
        x >= lcX - lhX && x <= lcX + lhX &&
        y >= lcY - lhY && y <= lcY + lhY &&
        z >= lcZ - lhZ && z <= lcZ + lhZ
      ) {
        nodeIdx = L;
        continue;
      }
    }

    // Otherwise, try right child
    const R = childR[nodeIdx];
    if (R >= 0) {
      const rd = R * 6;
      const rcX = data[rd],   rcY = data[rd + 1], rcZ = data[rd + 2];
      const rhX = data[rd + 3], rhY = data[rd + 4], rhZ = data[rd + 5];
      if (
        x >= rcX - rhX && x <= rcX + rhX &&
        y >= rcY - rhY && y <= rcY + rhY &&
        z >= rcZ - rhZ && z <= rcZ + rhZ
      ) {
        nodeIdx = R;
        continue;
      }
    }

    // If neither child contains the point, we stop
    break;
  }

  return path;
}


/**
 * Descend an Octree to find the leaf node index containing (x,y,z).
 * Returns -1 if point is not contained in the root or a child chain.
 *
 * @param data        Float32Array of length 6*Nnodes: [cx, cy, cz, hx, hy, hz] per node
 * @param leafFlags   Uint8Array of length Nnodes: 1 if leaf, 0 if internal
 * @param childrenIdxs Array of Uint32Array, length Nnodes: list of child node indices per internal node
 * @param x           point x‐coordinate
 * @param y           point y‐coordinate
 * @param z           point z‐coordinate
 * @returns           index of leaf node containing (x,y,z), or -1 if none
 */
export function findLeafOctree(
  data: Float32Array,
  leafFlags: Uint8Array,
  childrenIdxs: Uint32Array[],
  x: number, y: number, z: number
): number {
  // Check root AABB
  const cx0 = data[0], cy0 = data[1], cz0 = data[2];
  const hx0 = data[3], hy0 = data[4], hz0 = data[5];
  if (
    x < cx0 - hx0 || x > cx0 + hx0 ||
    y < cy0 - hy0 || y > cy0 + hy0 ||
    z < cz0 - hz0 || z > cz0 + hz0
  ) {
    return -1;
  }

  let nodeIdx = 0;
  while (true) {
    if (leafFlags[nodeIdx] === 1) {
      return nodeIdx;
    }
    const children = childrenIdxs[nodeIdx];
    let moved = false;
    for (let i = 0, len = children.length; i < len; i++) {
      const c = children[i];
      const cd = c * 6;
      const cx = data[cd], cy = data[cd + 1], cz = data[cd + 2];
      const hx = data[cd + 3], hy = data[cd + 4], hz = data[cd + 5];
      if (
        x >= cx - hx && x <= cx + hx &&
        y >= cy - hy && y <= cy + hy &&
        z >= cz - hz && z <= cz + hz
      ) {
        nodeIdx = c;
        moved = true;
        break;
      }
    }
    if (!moved) {
      return -1;
    }
  }
}

/**
 * Return the full root‐to‐leaf node‐index path in an Octree for (x,y,z).
 * The returned array begins at 0 (root) and ends at the leaf index.
 * If (x,y,z) is not contained in the root, returns an empty array.
 *
 * @param data         Float32Array of length 6*Nnodes
 * @param leafFlags    Uint8Array of length Nnodes
 * @param childrenIdxs Array of Uint32Array, length Nnodes
 * @param x            point x‐coordinate
 * @param y            point y‐coordinate
 * @param z            point z‐coordinate
 * @returns            array of node‐indices along the descent; [] if no containment
 */
export function collectHierarchyOctree(
  data: Float32Array,
  leafFlags: Uint8Array,
  childrenIdxs: Uint32Array[],
  x: number, y: number, z: number
): number[] {
  const path: number[] = [];

  // Check root AABB
  const cx0 = data[0], cy0 = data[1], cz0 = data[2];
  const hx0 = data[3], hy0 = data[4], hz0 = data[5];
  if (
    x < cx0 - hx0 || x > cx0 + hx0 ||
    y < cy0 - hy0 || y > cy0 + hy0 ||
    z < cz0 - hz0 || z > cz0 + hz0
  ) {
    return [];
  }

  let nodeIdx = 0;
  while (true) {
    path.push(nodeIdx);
    if (leafFlags[nodeIdx] === 1) {
      break;
    }
    const children = childrenIdxs[nodeIdx];
    let moved = false;
    for (let i = 0, len = children.length; i < len; i++) {
      const c = children[i];
      const cd = c * 6;
      const cx = data[cd], cy = data[cd + 1], cz = data[cd + 2];
      const hx = data[cd + 3], hy = data[cd + 4], hz = data[cd + 5];
      if (
        x >= cx - hx && x <= cx + hx &&
        y >= cy - hy && y <= cy + hy &&
        z >= cz - hz && z <= cz + hz
      ) {
        nodeIdx = c;
        moved = true;
        break;
      }
    }
    if (!moved) {
      break;
    }
  }

  return path;
}


/*
Example Usage:

import { buildBVH, buildOctree, findLeafBVH, collectHierarchyBVH, findLeafOctree, collectHierarchyOctree } from "./bvhOctreeUtil";

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

const leafIndex = findLeafBVH(
  bvhResult.data,
  bvhResult.leafFlags,
  bvhResult.childL,
  bvhResult.childR,
  queryX, queryY, queryZ
);
// leafIndex is the index of the leaf in which (queryX,queryY,queryZ) lies (or -1)

const pathBVH = collectHierarchyBVH(
  bvhResult.data,
  bvhResult.leafFlags,
  bvhResult.childL,
  bvhResult.childR,
  queryX, queryY, queryZ
);
// pathBVH is an array of node indices from root (0) through the leaf

const cubic = true;
const octResult = buildOctree(pos, scl, rad, minLeaf, cubic);
// octResult.data       → Float32Array([cx0, cy0, cz0, hx0, hy0, hz0,  cx1, …])
// octResult.leafFlags  → Uint8Array([0, 1, 1, 0, …])
// octResult.childL/R   → all −1 (no pointers)
// octResult.leafIndices → Array<Uint32Array> with one entry per octree leaf
// octResult.childrenIdxs → Array<Uint32Array> giving up to 8 children per node

const leafIndexOct = findLeafOctree(
  octResult.data,
  octResult.leafFlags,
  octResult.childrenIdxs,
  queryX, queryY, queryZ
);

const pathOct = collectHierarchyOctree(
  octResult.data,
  octResult.leafFlags,
  octResult.childrenIdxs,
  queryX, queryY, queryZ
);
*/
