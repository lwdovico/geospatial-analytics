This reposityory contains the exercises of the Geospatial Analytics course I attended, with the final project being the implementation of an algorithm to build the Voronoi tessellation.

# Voronoi Tessellation Project

For this project I implemented the Voronoi Tessellation tiler for the scikit-mobility library.

The implementation was done without using external libraries (at least not part of the basic python package).
In particular the library used which were not already imported in the *tilers.py* file were **heapq** and **collections** (only for namedtuples).

To understand the basic concepts to final implementation I read Chapter 7 of this [Computational Geometry](https://link.springer.com/book/10.1007/978-3-540-77974-2) book and I took inspiration from Steven Fortune's code and various adaptations of it.

Going to the details of the implementation of the sweepline algorithm (bottom to top), the entire implementation can be summarized as it follows:

1. Checking the input array and points validity
2. Managing the base shape as with the rest of the tilers
3. A static method to compute the **convex hull** on the resulting verteces (using the **Graham Scan algorithm**), used in last step of the following part
4. Inside the main "**compute_voronoi**" the following classes and features were implemented:
    * A **min-heap** to build a queue.
    * A **linked list** to manage the halfedges pointers.
    * The **HalfEdge** objects to manage connections to site, regions, edges and other halfedges.
    * A class to handle the **site and circle events** with some other necessary operations like bisecting the plane and getting vector events.
    * I used a namedtuple to implement the Edges as I needed just some attributes for them
    * Limiting the tessellation far from earth max and min coordinates (I selected a box  of sides 10.000 in lon and lat)
    * Computing the **convex hull** of the unsorted verteces</br></br>
    
5. Finally the "large" tessellation is "intersected" (overlayed) with the base_shape in order to get the correct subdivision of the geometry.

I finally also used pytest to make some general tests on the implemented code, I got mainly warnings related to the cascade_union deprecation. 
