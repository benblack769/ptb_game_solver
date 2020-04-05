import numpy as np

def coord_mapper(xsize,ysize):
    mapper = {}
    idx = 1
    for y in range(ysize):
        for x in range(xsize):
            mapper[(x,y)] = idx
            idx += 1
    return mapper

def grid_to_jumps(xsize,ysize,mapper):
    tot_size = xsize*ysize + 1
    jumps = [[] for _ in range(tot_size)]
    for x in range(xsize):
        for y in range(ysize):
            cur_loc = mapper[(x,y)]
            for xa in range(max(0,x-1),min(x+1,xsize-1)+1):
                for ya in range(max(0,y-1),min(y+1,ysize-1)+1):
                    if xa == x or ya == y:
                        jumps[cur_loc].append(mapper[(xa,ya)])
    jump_arr = np.zeros([len(jumps),5],dtype=np.int32)
    for i,jump in enumerate(jumps):
        for j in range(5):
            jump_idx = 0 if len(jump) <= j else jump[j]
            jump_arr[i][j] = jump_idx
    return jump_arr

def bfs_gri(arr_jumps,blocked_idxs,start_idx,end_idx):
    visited = np.zeros(len(blocked_idxs),dtype=np.bool)
    cur_ids = np.array([start_idx],dtype=np.int32)
    MAX_DIST = 1000000
    dists = np.ones(len(blocked_idxs),dtype=np.int32)*MAX_DIST
    cur_dist = 0
    while len(cur_ids):
        dists[cur_ids] = cur_dist
        visited[cur_ids] = True
        all_next_ids = arr_jumps[cur_ids].flatten()
        next_ids_work = ~blocked_idxs[all_next_ids] & ~visited[all_next_ids]

        next_ids = all_next_ids[next_ids_work]
        cur_ids = next_ids
        cur_dist += 1

    if dists[end_idx] == MAX_DIST:
        return None
    else:
        shortest_path = [end_idx]
        cur_idx = end_idx
        cur_dist = dists[end_idx]
        for _ in range(cur_dist):
            jump_idxs = arr_jumps[cur_idx]
            jump_idxs = jump_idxs[jump_idxs != 0]
            cur_idx = jump_idxs[np.argmin(dists[jump_idxs])]
            shortest_path.append(cur_idx)
        shortest_path.reverse()
        return shortest_path

class IdxGrid:
    def __init__(self, blocking_map):
        blocking_map = blocking_map.astype(np.bool)
        self.ysize,self.xsize = self.shape = blocking_map.shape
        self.mapper = coord_mapper(self.xsize,self.ysize)
        self.rev_mapper = {v:k for k,v in self.mapper.items()}
        #self.coord_arr = np.concatenate([np.arange(self.ysize) + x * self.ysize + 1 for x in range(self.xsize)])
        self.jumps = grid_to_jumps(self.xsize,self.ysize,self.mapper)
        self.blocking_idx = self.transform(blocking_map)

    def transform(self,arr2d):
        assert self.shape == arr2d.shape
        return np.concatenate([np.zeros([1],dtype=arr2d.dtype),arr2d.flatten()])

    def update_blocks(self,blocking):
        self.blocking_idx = self.transform(blocking_map)

    def set_block(self,x,y,val=True):
        self.blocking_idx[self.mapper[(x,y)]] = val

    def find_path(self,start_coord,end_coord):
        start_idx = self.mapper[start_coord]
        end_idx = self.mapper[end_coord]
        return bfs_gri(self.jumps,self.blocking_idx,start_idx,end_idx)

    def to_coord(self,idx):
        return self.rev_mapper[idx]

def test_main():
    grid = np.array([
        [0,0,0,1,0],
        [0,0,0,1,0],
        [0,1,0,1,0],
        [0,1,0,0,0],
    ],dtype=np.int64)
    bfs = IdxGrid(grid)

    path = (bfs.find_path((0,3),(4,0)))
    coords = [bfs.to_coord(p) for p in path]
    print(coords)

if __name__ == "__main__":
    test_main()
