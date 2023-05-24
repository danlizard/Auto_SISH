import cv2
import numpy as np

def flash_cnt(wrap, loc, lvl, base=None):
    cnt = wrap.cnt_at_lvl(loc,lvl,local=True)
    bounds = wrap._bounds_at_lvl(loc,lvl)
    w,h = bounds['w'],bounds['h']
    flash = base
    if not base:
        flash = np.zeros((h,w,3))
    flash = cv2.drawContours(flash, [cnt], 0, (255,255,255), 2).astype(np.uint8)
    return flash

def match_samples(wrap1, wrap2):
    sorting = dict()
    for opt1 in wrap1.region_tree:
        cnt1 = wrap1.region_tree[opt1]['abs_contour']
        sorting[opt1] = dict()
        for opt2 in wrap2.region_tree:
            cnt2 = wrap2.region_tree[opt2]['abs_contour']
            dist = cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I1, None)
            sorting[opt1][opt2] = dist
    excl1 = []
    excl2 = []
    pairs = []
    while True:
        options = dict()
        for i in sorting:
            if i in excl1:
                continue
            for j in sorting[i]:
                if j in excl2:
                    continue
                options[sorting[i][j]] = (i,j)
        if not options:
            return pairs
        i, j = options[min(options)]
        excl1.append(i)
        excl2.append(j)
        pairs.append((i,j))

def get_warp(src, target, homographic=True, **kwargs):
    if homographic: warp_mode = cv2.MOTION_HOMOGRAPHY; warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:           warp_mode = cv2.MOTION_AFFINE;     warp_matrix = np.eye(2, 3, dtype=np.float32)                          
    if 'base' in kwargs: warp_matrix = kwargs['base']

    num_iters = 1024
    if 'iters' in kwargs: num_iters = kwargs['iters']
    termination_eps = 1e-8
    if 'eps' in kwargs: num_iters = kwargs['eps']
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iters,  termination_eps)

    (cc, warp_matrix) = cv2.findTransformECC(src, target, warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)
    return (cc, warp_matrix)