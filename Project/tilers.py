import math
import warnings
from abc import ABC, abstractmethod

import heapq
from collections import namedtuple

import geopandas as gpd
import h3.api.numpy_int as h3
import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import cascaded_union

from skmob.utils import constants, utils


class TessellationTilers:
    def __init__(self):
        self._tilers = {}

    def register_tiler(self, key, tiler):
        self._tilers[key] = tiler

    def create(self, key, **kwargs):
        tiler = self._tilers.get(key)

        if not tiler:
            raise ValueError(key)

        return tiler(**kwargs)

    def get(self, service_id, **kwargs):
        return self.create(service_id, **kwargs)


tiler = TessellationTilers()


class TessellationTiler(ABC):
    @abstractmethod
    def __call__(self, **kwargs):
        pass

    @abstractmethod
    def _build(self, **kwargs):
        pass


class VoronoiTessellationTiler(TessellationTiler):

    def __init__(self):

        super().__init__()
        self._instance = None

    def __call__(
        self, 
        points, 
        base_shape, 
        meters=None, #not needed, kept for compatibility reasons
        which_osm_result=-1,
        crs=constants.DEFAULT_CRS):

        if not self._instance:

            if isinstance(points, gpd.GeoDataFrame):
                
                points.set_crs(constants.DEFAULT_CRS, allow_override = True, inplace = True)
                
                if not all(isinstance(x, Point) for x in points.geometry):
                    
                    raise ValueError("Not valid points object. Accepted type is: GeoDataFrame with a valid geometry column.")
                    
                    
            elif isinstance(points, (list, np.ndarray)):
                
                is_not_correct_type = not all(isinstance(x, (tuple, list, np.ndarray, Point)) for x in points)
                wrong_pairs = not all(len(x) == 2 for x in points if type(x) != Point)
                
                if is_not_correct_type and wrong_pairs:
                    
                    raise ValueError("Not valid array object. Accepted types are shapely.geometry.Point or (lon, lat) pair of coordinates")
                    
                else:
                    
                    points = gpd.GeoDataFrame(geometry = [Point(xy) for xy in points], crs = crs)
                
            base_shape_geometry = self._create_geometry_if_does_not_exists(base_shape, which_osm_result)
            base_shape_geometry_merged = self._merge_all_polygons(base_shape_geometry)
    
        return self._build(points, base_shape_geometry_merged, crs)
    
    def _create_geometry_if_does_not_exists(self, base_shape, which_osm_result):

        if isinstance(base_shape, str):
            base_shape = self._str_to_geometry(base_shape, which_osm_result)

        elif self._isinstance_geodataframe_or_geoseries(base_shape):
            if all(isinstance(x, Point) for x in base_shape.geometry):
                base_shape = utils.bbox_from_points(base_shape, base_shape.crs)
        else:
            raise ValueError("Not valid base_shape object." " Accepted types are str, GeoDataFrame or GeoSeries.")
            
        return base_shape

    def _isinstance_geodataframe_or_geoseries(self, base_shape):
        return True if (isinstance(base_shape, gpd.GeoDataFrame) or isinstance(base_shape, gpd.GeoSeries)) else False

    def _str_to_geometry(self, base_shape, which_osm_result):
        base_shapes = utils.bbox_from_name(base_shape, which_osm_result=which_osm_result)
        polygon_shape = self._find_first_polygon(base_shapes)
        return polygon_shape

    def _find_first_polygon(self, base_shapes):
        return_shape = base_shapes.iloc[[0]]
        for i, current_shape in enumerate(base_shapes["geometry"].values):
            if self._isinstance_poly_or_multipolygon(current_shape):
                return_shape = base_shapes.iloc[[i]]
                break
        return return_shape

    def _isinstance_poly_or_multipolygon(self, shape):
        return True if (isinstance(shape, Polygon) or isinstance(shape, MultiPolygon)) else False

    def _merge_all_polygons(self, base_shape):
        polygons = base_shape.geometry.values
        base_shape = gpd.GeoSeries(cascaded_union(polygons), crs=base_shape.crs)
        return base_shape
    
    @staticmethod
    def convex_hull(verteces):    
        """ 
        Graham Scan algorithm: O(n*log(n)).
        Without it the output of the Voronoi Tessellation would be unordered verteces.
        Computing the convex hull allows to produce the correct polygons of the tiles
        since the tiles are conveniently convex and the vertex order doesn't matter.
        """

        # Find the point with the minimum "first y" coordinates
        min_ref_pt = min(verteces, key=lambda p: (p[1], p[0]))

        # Compute the polar angle of p2 wrt p1
        angle_between = lambda p0, p1: math.atan2(p1[1] - p0[1], p1[0] - p0[0])

        # Sort points by their polar angles with respect to the min ref point
        pts = sorted(verteces, key=lambda p: angle_between(min_ref_pt, p))

        # Initialize a stack with the first three points
        stack = [pts[0], pts[1], pts[2]]

        # Compute the cross product of the vectors [p0, p1] and [p0, p2]
        cross_product = lambda p0, p1, p2: (((p1[0] - p0[0]) * (p2[1] - p0[1])) -
                                            ((p1[1] - p0[1]) * (p2[0] - p0[0])))

        # Processing the rest of the points
        for i in range(3, len(pts)):
            # Remove points from the stack until the next point is on the convex hull
            while len(stack) > 1 and cross_product(stack[-2], stack[-1], pts[i]) <= 0:
                stack.pop()
            stack.append(pts[i])

        return stack

    def compute_voronoi(self, points, crs = constants.DEFAULT_CRS):
        
        class Queue():
            """ 
            I implemented this queue with a minheap as I would always need the
            minimum half edge as the first element and it's reasonably fast.
            """

            def __init__(self):
                self.heap = []

            def is_empty(self):
                return not self.heap

            def insert(self, he, site, dist):
                # the heap is sorted according to the he.y intersection with the sweep_line
                he.vector_point = site
                he.sweep_line = site.y + dist # it is the site.y distance from intersection
                heapq.heappush(self.heap, (he.sweep_line, he.vector_point.x, he))

            def delete(self, he):
                if he.vector_point:
                    self.heap.remove((he.sweep_line, he.vector_point.x, he))
                    heapq.heapify(self.heap)

            def get_min(self):
                sweep_line, x, he = self.heap[0]
                return Point(x, sweep_line)

            def pop_min_he(self):
                sweep_line, x, he = heapq.heappop(self.heap)
                return he

            def __iter__(self):
                return iter(self.heap)

        class HalfEdgesLinkedList():
            """
            A linked list to store the position of half-edges wrt each other.
            
            There is also a linear search method to get a half-edge by
            an input point.
            
            It works by moving closer to the point, checking if the point 
            is to the right of the half-edge during the iterations, moving 
            from either direction to arrive to desired half-edge (left to the point).
            """

            def __init__(self):
                # creating empty half-edges for the ends of the list
                self.leftend = HalfEdge()
                self.rightend = HalfEdge()
                
                # now linking the 2 ends of the list
                self.leftend.right = self.rightend
                self.rightend.left = self.leftend

            def insert(self, first, second):
                # position the second half-edge as the first half-edge to the right of the first
                
                # insert the first half-edge to the left of the second
                second.left = first
                # putting the right of the first half-edge to the right of the second
                second.right = first.right
                
                # go to the right of the first edge and set its left new half-edge
                first.right.left = second
                # then point also the the right of the first to the second
                first.right = second

            def delete(self, he):
                # connecting the left and right of the input hedge
                he.left.right = he.right
                he.right.left = he.left
                # deleting the half-edge marking it as empty set
                he.edge = {} 

            def linear_search(self, pt):
                he = self.leftend
                
                # it will return the left half edge near the pt
                # if it doesn't find anything it will just set to the linked list to the right end
                if he is self.leftend or (he is not self.rightend and HalfEdge.check_pt_right(pt, he)):
                    he = he.right
                    while he is not self.rightend and HalfEdge.check_pt_right(pt, he):
                        he = he.right
                    he = he.left
                else:
                    he = he.left
                    while he is not self.leftend and not HalfEdge.check_pt_right(pt, he):
                        he = he.left
                return he

        class HalfEdge(object):
            """
            Main object to manipulate to get to vector events. 
            
            The vector_point is the point on which two half edges meet, it is 
            initialized as None and it changes when a vecctor event is set in stone.
            If we consider an Edge the parent of two HalfEdges, then we have the 
            Left and Right halves (LH, RH) which are  distinguishable by their 
            "orientation", if they are going from left to right it's a "left half"
            indicated with 0 otherwise it's a "right half" expressed with 1. 
            """
            
            LH, RH = 0, 1

            def __init__(self, edge=None, oriented = LH):
                self.left, self.right = None, None # create a link between halfedges (used in linked list)
                self.oriented = oriented # set the orientation of the half-edge (LH or RH)
                self.vector_point = None  # Point(x, y) == intersection between half-edges
                self.sweep_line = np.inf # y-coord of sweepline associated with site event
                self.edge = edge

            def get_region(self, region, default):
                # region 0 is the left, 1 is the right
                region = 1 if region == 'left' else 0
                other_region = 1 - region
                
                if not self.edge: 
                    return default
                elif self.oriented == 1: # if oriented is left
                    return self.edge.region[region]
                else:
                    return self.edge.region[other_region]

            def __lt__(self, other):
                # It may rarely happen that while pushing the he into the queue two he are
                # compared because the sweepline and intersection point x-axis were equal.
                #
                # In this case it tries to tell which is the smaller but it doesn't really
                # matter as they are identical in the end.
                # The purpose is guaranteing a "first-come-first-served" policy.
                return True
                

            @staticmethod
            def check_pt_right(pt, he):
                # returns True if pt is to right of halfedge
                e = he.edge
                is_pt_right = pt.x > e.region[HalfEdge.RH].x

                # if half edge orientation is left and point is to the right 
                # of the rightmost he region, then the point is to its right
                if is_pt_right and he.oriented == HalfEdge.LH: 
                    return True # RIGHT

                # if half edge orientation is right and point is to the left 
                # of the rightmost he region, then the point is to its left
                if not is_pt_right and he.oriented == HalfEdge.RH:
                    return False # LEFT
                

                # if dx was greater : a was the coefficient, b the slope
                if e.a == 1.0: 
                    positive_slope = e.b >= 0.0
                    dxp = pt.x - e.region[HalfEdge.RH].x
                    dyp = pt.y - e.region[HalfEdge.RH].y
                    no_need_check = False
                    # either they are both True or both False
                    if not (is_pt_right or positive_slope) or (is_pt_right and positive_slope):
                        above = dyp >= e.b * dxp
                        no_need_check = above
                    else:
                        above = pt.x + (pt.y * e.b) > e.c
                        if not positive_slope:
                            above = not above
                        if not above:
                            no_need_check = True
                    if not no_need_check:
                        dxs = e.region[HalfEdge.RH].x - e.region[HalfEdge.LH].x
                        above = e.b * (dxp*dxp - dyp*dyp) < dxs*dyp * (1.0 + 2.0*dxp/dxs + e.b*e.b)
                        if not positive_slope:
                            above = not above
                            
                # if dy was greater : b was the coefficient, a the slope
                else:
                    yl = e.c - e.a * pt.x
                    t1 = np.square(pt.y - yl)
                    t2 = np.square(pt.x - e.region[HalfEdge.RH].x)
                    t3 = np.square(yl - e.region[HalfEdge.RH].y)
                    above = t1 > t2 + t3
                    
                if he.oriented == HalfEdge.LH:
                    return above
                else:
                    return not above
                
        class EventHandler():
            """
            A helper class that manages the events happening during the construction
            of the VoronoiTessellation. It manages to bisect points with an edge and 
            to intersect half-edges during the handling of the two main events of the
            Fortune's algorithm.
            
            Indeed the two core functions are:
            1. The Handling of Site Events when the sweep line crosses a Point while
            going bottom up across coordinates
            2. The Handling of Circle Events when the sweep line crosses the circumcenter 
            of a triangle of three points.
            
            """
            def __init__(self, points, linked_list, queue):
                self.points = iter(points)
                self.linked_list = linked_list
                self.queue = queue

                # a polygon for each point (original + 4 to limit the tessellation)
                # only the ids will be stored here
                self.polygons = {k : [] for k in range(len(points))}
                self.vertices = list() # a list to store the actual vertices

                self.first_point = next(self.points, None)
                self.next_site = next(self.points, None)
                self.minpoint = Point(-np.inf,-np.inf)

                self.sitenum = 0
                
            @staticmethod
            def bisect_points(p0, p1):
                endpoints = [None, None] # no endpoints on the bisector it goes to infinity
                regions = [p0, p1] # storing the original points that are going to be bisected
                
                # it is faster to work with arrays
                p0, p1 = np.array([p0.coords]), np.array([p1.coords])
                
                # get the axis-wise distance of the two points (x from x and y from y)
                dist = (p1 - p0).reshape((2, 1)) # get the flattened array
                
                # is x-axis dist greater? (unpacking array)
                dx_greater = np.greater(*np.abs(dist)) 
                
                # getting the slope with the largest axis as the denominator
                denom, numer = dist if dx_greater else dist[::-1]
                slope = numer / denom
                
                # set the slope of the line and set the coefficient to 1
                a, b = (1.0, slope) if dx_greater else (slope, 1.0)
                # mid-value to compute the slope later
                c = float(p0.dot(dist) + np.sum(np.square(dist))*0.5) / denom
                
                newedge = Edge(a, b, c, endpoints, regions)
                
                return newedge

            @staticmethod
            def intersect_halfedges(he1, he2):
                e1 = he1.edge
                e2 = he2.edge
                
                # Three cases where the intersection cannot be performed:
                # 1. Both edges must exist
                if (e1 is None) or (e2 is None):
                    return

                # 2. The edges shouldn't intersect on the same region:
                if (e1.region[HalfEdge.RH] is e2.region[HalfEdge.RH]):
                    return
                
                # 3. The absolute distance between coefficient and slope
                #    shouldn't be zero nor too close to it:
                
                d = e1.a * e2.b - e1.b * e2.a
                if (abs(d) < 1e-10) or (abs(d) < (1e-10 * abs(d))):
                    return None
                
                # computing coordinates of intersection (equation: ax + by + c = 0)
                intersect_x = (e1.c*e2.b - e2.c*e1.b) / d
                intersect_y = (e2.c*e1.a - e1.c*e2.a) / d
                
                # checking region with bottommost site event
                first_region_btm = _pt_lt_other_(e1.region[HalfEdge.RH], e2.region[HalfEdge.RH])
                
                # selecting the edge and half-edge with a site underneath the other
                he, e = (he1, e1) if first_region_btm else (he2, e2)

                # checking the intersection orientation wrt bottommost edge right region
                intersection_is_right = intersect_x >= e.region[HalfEdge.RH].x
                
                if ((intersection_is_right and he.oriented == HalfEdge.LH) or
                    (not intersection_is_right and he.oriented == HalfEdge.RH)):
                    return None

                # create a new site at the point of the region intersection
                return Point(intersect_x, intersect_y) # vector_event
            
            @staticmethod
            def _add_polygon_ids(edge, polygons, vertices, pmap):
                
                # simple function to extract the original indexes
                idx = lambda p: pmap[(p.x, p.y)]
                
                # find the original points idx associated with the edge end_points
                # for both ends of edges if it is not an infinite edge, otherwise add None
                left_vert, right_vert = [vertices[idx(p)] if p else None for p in edge.end_point]
                
                # looking at the index of the original points to set the region where
                # to add the verteces
                lmap_reg, rmap_reg = [idx(p) for p in edge.region]
                
                # adding the vertices to the corresponding list in the original regions
                polygons[lmap_reg].extend((left_vert, right_vert))
                polygons[rmap_reg].extend((left_vert, right_vert))

            def is_site_event(self):
                return self.next_site and (self.queue.is_empty() or _pt_lt_other_(self.next_site, self.minpoint))

            def handle_site_event(self):
                # finding the left and right HalfEdges corresponding to the next site
                # if it doesn't find any it just create an empty HalfEdge
                left_he = self.linked_list.linear_search(self.next_site)
                right_he = left_he.right

                # regions are reppresented by the original points:
                
                # get the right region of the first left HalfEdge
                # if not found it is the first_point of the list
                first  = left_he.get_region('right', default = self.first_point)
                
                # create the edge bisecting the point on the right region with the current site
                # during the first site event it creates it bisecting the first and second point
                edge = self.bisect_points(first, self.next_site)

                # creating a HalfEdge from the edge created,orienting it LtR
                # it is then added on the right of the left_he found before
                bisector = HalfEdge(edge, HalfEdge.LH)
                self.linked_list.insert(left_he, bisector)

                # if the bisected he intersects with the left edge
                ip = self.intersect_halfedges(left_he, bisector)
                if ip is not None:
                    # remove the left edge's vertex, if exists, from min-heap-queue
                    self.queue.delete(left_he)
                    # pushing the new vertex into the queue (waiting to)
                    self.queue.insert(left_he, ip, self.next_site.distance(ip))

                # now moving to the right of the bisected edge creating a RtL he
                left_he = bisector
                bisector = HalfEdge(edge, HalfEdge.RH)
                
                # inserting the HalfEdge to the right of the previous bisector
                self.linked_list.insert(left_he, bisector)

                # if this bisector intersects with the original right HalfEdge
                ip = self.intersect_halfedges(bisector, right_he)
                if ip is not None:
                    # push the new intersection into the queue
                    self.queue.insert(bisector, ip, self.next_site.distance(ip))

                # move to the next site event to process
                self.next_site = next(self.points, None)

            def handle_circle_event(self, pmap):
                # pop the minimum vector_event from the queue (min-heap)
                left_he  = self.queue.pop_min_he()
                
                # extracting its half edges neighbours
                leftleft_he = left_he.left
                right_he  = left_he.right
                rightright_he = right_he.right

                # get the vector_point that caused this event
                vector_event_point = left_he.vector_point
                
                # reassigning original index to the current site count
                pmap[vector_event_point.coords[0]] = self.sitenum
                self.sitenum += 1
                
                # append the vertices according to the same site count order
                self.vertices.append((vector_event_point.coords[0]))
                
                # triplet of sites through which a circle goes through
                first = left_he.get_region('left', default = self.first_point)
                second = left_he.get_region('right', default = self.first_point)
                third = right_he.get_region('right', default = self.first_point)

                # set the endpoint of the left and right HalfEdge to be this vector
                # add also the polygons found to the dictionary 
                left_he.edge.end_point[left_he.oriented] = vector_event_point
                if left_he.edge.end_point[1 - left_he.oriented] is not None:
                    self._add_polygon_ids(left_he.edge, self.polygons, self.vertices, pmap)

                right_he.edge.end_point[right_he.oriented] = vector_event_point
                if right_he.edge.end_point[1 - right_he.oriented] is not None:
                    self._add_polygon_ids(right_he.edge, self.polygons, self.vertices, pmap)

                # delete the bottommost HE from linked list (already popped from queue)
                self.linked_list.delete(left_he)
                
                # remove also information regarding the right half edge
                self.queue.delete(right_he)
                self.linked_list.delete(right_he)

                # if the site to the left of the event is on top to the Site to the right
                # swap them and set the orientation from RtL (Right to Left)
                
                oriented = HalfEdge.LH
                if first.y > third.y:
                    first, third = third, first
                    oriented = HalfEdge.RH

                # create an edge that is between the left and right sites
                edge = self.bisect_points(first, third)
                # set the bisected half edge according to the previously defined orientation
                bisector = HalfEdge(edge, oriented)

                # insert the new bisector to the right of the left half edge (to the left of the first)
                self.linked_list.insert(leftleft_he, bisector) 
                
                # set only one endpoint to the new edge to be the vector point:
                # if the site to the left of this bisector is on top of the right
                # then this endpoint is set on the left, otherwise on the right
                invert_or = HalfEdge.RH - oriented
                edge.end_point[invert_or] = vector_event_point
                if edge.end_point[1 - invert_or] is not None:
                    self._add_polygon_ids(edge, self.polygons, self.vertices, pmap)

                # if left HE and the new bisector don't intersect
                ip = self.intersect_halfedges(leftleft_he, bisector)
                if ip is not None:
                    self.queue.delete(leftleft_he) # remove from the current queue position
                    self.queue.insert(leftleft_he, ip, first.distance(ip)) # reinsert with new

                # if right HE and the new bisector don't intersect, then push it to the queue
                ip = self.intersect_halfedges(bisector, rightright_he)
                if ip is not None:
                    self.queue.insert(bisector, ip, first.distance(ip))

            def add_remaining_edges(self, pmap):
                # setting the linked list to the right of its end
                he = self.linked_list.leftend.right
                
                # until the list goes to its right most element
                while he is not self.linked_list.rightend:
                    # add polygons to the polygon dictionary
                    self._add_polygon_ids(he.edge, self.polygons, self.vertices, pmap)
                    # go to the right half edge
                    he = he.right
        
        # I also need a simple structure to reference edges 
        Edge = namedtuple('Edge', ['a', 'b', 'c', 'end_point', 'region'],
                          defaults = [0.0, 0.0, 0.0, [None]*2, [None]*2])
        
        
        def _pt_lt_other_(p0, p1): # needed to compare points by smaller "y"
            # if y coords are smaller or they are equal but x is smaller
            return p0.y < p1.y or (p0.y == p1.y and p0.x < p1.x)
            
    
        # putting a external bound to the tessallation by creating 4 fake points
        # just a larger than earth box: 1e4 is out of the map's bounds
        tessellation_limit = [(-1e4, 0),(0, 1e4),(1e4, 0),(0, -1e4)]
        fake_points = [Point(*lim) for lim in tessellation_limit]

        # redefining the points as only unique points + fake_points
        points = gpd.GeoSeries(points.geometry.unique().tolist() +
                               fake_points, crs = crs)

        # mapping index of the unsorted points
        pmap = {pt.coords[0] : i for i, pt in enumerate(points)}

        # getting the points sorted by lowest 'y' coordinate
        bottom_up_sorted_pts = sorted(points, key = lambda pt: (pt.y, pt.x))
        
        event_handler = EventHandler(bottom_up_sorted_pts, 
                                     HalfEdgesLinkedList(), 
                                     Queue())

        
        # while the queue is not empty and there are still points to process
        while True:
            
            # if the queue is not empty get the first value and set it as min
            if not event_handler.queue.is_empty():
                event_handler.minpoint = event_handler.queue.get_min()

            # if the next_site examined is the smallest or the queue is empty
            if event_handler.is_site_event(): # SITE EVENT
                event_handler.handle_site_event()

            # if the min value is the smallest we
            elif not event_handler.queue.is_empty(): # CIRCLE EVENT
                event_handler.handle_circle_event(pmap)

            else:
                break

        # now adding the remaining edges in the linked list
        event_handler.add_remaining_edges(pmap)
        
        # now adding the tiles to the final output
        voronoi_tess = list()

        for polygon in event_handler.polygons.values():
            # filtering the placeholder for vertex pointing to infinity
            valid_polygon = list(filter(lambda vertex: vertex is not None, polygon))
            tile_coords = VoronoiTessellationTiler.convex_hull(valid_polygon)
            
            # if the convex_hull has enough vertex to form a polygon
            if len(tile_coords) > 2: 
                voronoi_tess.append(Polygon(tile_coords))

        return voronoi_tess


    def _build(self, points, base_shape, crs=constants.DEFAULT_CRS):
        if base_shape.crs != constants.DEFAULT_CRS:
            base_shape = base_shape.to_crs(constants.DEFAULT_CRS)
        
        base_shape_gdf = gpd.GeoDataFrame(geometry=base_shape, crs=crs)
        
        # computing the voronoi tessellation
        voronoi = self.compute_voronoi(points, crs=crs)
        # cutting out the areas out of the base shape (as the voronoi should be "infinite")
        gdf = gpd.GeoDataFrame(geometry=voronoi, crs=crs).overlay(base_shape_gdf)
        
        # gdf = gpd.GeoDataFrame(points.copy(), crs=crs)
        gdf.loc[:, constants.TILE_ID] = list(np.arange(0, len(gdf)))

        # Convert TILE_ID to have str type
        gdf[constants.TILE_ID] = gdf[constants.TILE_ID].astype("str")

        return gdf[[constants.TILE_ID, "geometry"]]


# Register the builder
tiler.register_tiler("voronoi", VoronoiTessellationTiler())
