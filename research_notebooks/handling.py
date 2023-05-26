import numpy as np
import cv2

try:
    import openslide as oslide
except:
    path = "D:/Programs/bin/openslide/bin" # POINT AT BIN WITH OPENSLIDE
    import os
    os.add_dll_directory(path)
    import openslide as oslide

default = {
    'low_edge_thrs':32,
    'high_edge_thrs':48,
    'min_area':2000,
    'base_level':7
}

select_y = [[0,1]]
select_x = [[1,0]]

def alpha_fix(arr):
    if not arr[3]:
        return np.array([255,255,255,255], dtype=np.uint8)
    return arr

class mrxs_wrapper:    
    def __init__(self, img_path:str, settings:dict = default):
        self.image = oslide.open_slide(img_path)
        self.settings = settings
        self.region_tree = dict()
        self.__locate_mains()
    
    def __locate_mains(self):
        approx = np.array(self.image.read_region((0,0),self.settings['base_level'],
                    self.image.level_dimensions[self.settings['base_level']]))
        approx = np.apply_along_axis(alpha_fix, 2, approx) # whites out "hole du corruption" and other alpha-0 regions
        edges = cv2.Canny(approx, threshold1=self.settings['low_edge_thrs'], threshold2=self.settings['high_edge_thrs'])
        edges = cv2.dilate(edges, np.full((3,3),1))
        contours,hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        accepted = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > self.settings['min_area']:
                bounds = self._convert_scale_from_to(cv2.boundingRect(cnt), self.settings['base_level'], 0)
                lb = cv2.boundingRect(cnt)
                region = approx[lb[1]:lb[1]+lb[3],lb[0]:lb[0]+lb[2],:]
                self.region_tree[accepted] = dict()
                self.region_tree[accepted]['children'] = None
                self.region_tree[accepted]['abs_bounds'] = {'x':bounds[0],'y':bounds[1],'w':bounds[2],'h':bounds[3]}
                self.region_tree[accepted]['abs_contour'] = self._convert_scale_from_to(cnt, self.settings['base_level'], 0)
                accepted += 1
    
    def _convert_scale_from_to(self, coords:tuple|list|np.ndarray, lvl1:int, lvl2:int) -> np.ndarray:
        # this is ~0.1% inaccurate for the smallest res image as the conversion rate there is ~0.499 instead of 0.5
        if not isinstance(coords, np.ndarray):
            coords = np.array(coords)
        coeff = 0.5**(lvl2-lvl1)
        out = coords*coeff
        return out.astype(np.int32)
    
    def _bounds_at_lvl(self, loc:list, lvl:int, origin_rescale:bool=False) -> dict:
        major = loc[0]
        view = self.region_tree[major]
        for index in loc[1:]:
            view = view['children'][index]
            if view == None:
                raise Exception(f"view at {loc} does not exist")
        out = dict()
        if origin_rescale:
            out['x'], out['y'] = self._convert_scale_from_to((view['abs_bounds']['x'], view['abs_bounds']['y']), 0, lvl)
        else:
            out['x'], out['y'] = view['abs_bounds']['x'], view['abs_bounds']['y']
        out['h'], out['w'] = self._convert_scale_from_to((view['abs_bounds']['h'], view['abs_bounds']['w']), 0, lvl)
        return out
    
    def obj_at_lvl(self, loc:list, lvl:int, purge:bool=False) -> np.ndarray:
        bounds = self._bounds_at_lvl(loc, lvl)
        out = np.array(self.image.read_region((bounds['x'],bounds['y']), lvl, (bounds['w'],bounds['h'])))
        if purge:
            cnt = self.cnt_at_lvl(loc, lvl, local=True)
            for i in range(out.shape[0]):
                for j in range(out.shape[1]):
                    within_cnt = 1 + cv2.pointPolygonTest(cnt,(j,i),False)
                    if not within_cnt:
                        out[i,j,:] = np.full_like(out[i,j,:], 255)
        return out

    def cnt_at_lvl(self, loc:list, lvl:int, local:bool=False) -> np.ndarray:
        major = loc[0]
        view = self.region_tree[major]
        for index in loc[1:]:
            view = view['children'][index]
            if view == None:
                raise Exception(f"view at {loc} does not exist")
        cnt = self._convert_scale_from_to(view['abs_contour'], 0, lvl)
        if local:
            bounds = self._bounds_at_lvl(loc, lvl, origin_rescale=True)
            mask_x = np.stack([select_x*cnt.shape[0]], axis=1)
            mask_y = np.stack([select_y*cnt.shape[0]], axis=1)
            cnt = np.where(mask_x, cnt-bounds['x'], cnt)
            cnt = np.where(mask_y, cnt-bounds['y'], cnt)
        return cnt