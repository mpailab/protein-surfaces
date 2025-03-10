import math
import numpy as np
from numpy import linalg as la
from scipy.optimize import linprog
from multiprocessing import Pool
from typing import Callable, Dict, List, FrozenSet, Tuple
import surface


class Bound (object):

    def __init__(self, normal, bias):
        self.normal = normal
        self.bias = bias

    def valid(self, x):
        return self.normal.dot(x) <= self.bias
    
    def neg(self):
        return Bound(-self.normal, -self.bias)


class Surface (object):

    def __init__(self, center, radius, external):
        self.center = center
        self.radius = radius
        self.bounds = []
        self.feasible = True
        self.external = external

    def add_bound(self, bound : Bound):
        self.bounds.append(bound)

    def resolve(self):
        self.feasible = self.feasible and self.bounds and \
            linprog([0,0,0],
                    A_ub=[x.normal for x in self.bounds],
                    b_ub=[x.bias for x in self.bounds]).success

    def valid(self, x):
        return all (bound.valid(x) for bound in self.bounds)
        
    def filter(self, points, normals):
        d = 1 if self.external else -1
        return [(p, d * n) for p, n in zip(points, normals) if self.valid(p)]
        

class Sphere (Surface):

    def generate_points(self, point_area):
        n = 1 + int(np.sqrt(4 * np.pi * (self.radius ** 2) / point_area - 1))

        u = np.linspace(0, 1, n)  
        v = np.linspace(0, 1, n)  

        u_grid, v_grid = np.meshgrid(u, v)

        u_spher = u_grid * 2 * np.pi  
        v_spher = np.arcsin(2 * v_grid - 1)

        normals = np.array([np.cos(v_spher) * np.cos(u_spher),
                            np.cos(v_spher) * np.sin(u_spher),
                            np.sin(v_spher)])
        points = self.center + self.radius * normals

        return self.filter(points, normals)


class Atom (Sphere):

    def __init__(self, center, radius):
        super().__init__(center, radius, True)
        self.dist = la.norm(self.center) ** 2
        self.bias = self.dist - self.radius ** 2
        

class Probe (Sphere):

    def __init__(self, center, radius):
        super().__init__(center, radius, False)


class Torus (Surface):

    def __init__(self, center, normal, R, r):
        super().__init__(center, R, False)
        self.normal = normal
        self.r = r

        x = - self.normal[1] / np.sqrt(self.normal[0] ** 2 + self.normal[1] ** 2)
        y = self.normal[0] / np.sqrt(self.normal[0] ** 2 + self.normal[1] ** 2)
        t = np.arccos(self.normal[2] / la.norm(self.normal))

        self.rotate_matrix = \
            (1 - np.cos(t)) * np.array([[x ** 2, x * y,  0],
                                        [x * y,  y ** 2, 0],
                                        [0,      0,      0]]) + \
            np.sin(t) * np.array([[ 0, 0,  y],
                                  [ 0, 0, -x],
                                  [-y, x,  0]]) + \
            np.cos(t) * np.identity(3)


    def generate_points(self, point_area):
        n = 1 + int(np.sqrt(2 * (np.pi ** 2) * self.radius * self.r / point_area - 1))

        u = np.linspace(0, 1, n)  
        v = np.linspace(-1, 1, n)  
        u, v = np.meshgrid(u, v)

        u_spher = u * 2 * np.pi
        v_spher = np.arcsin(v - np.sign(v)) + np.pi + np.sign(v) * np.pi / 2

        normals = self.rotate_matrix.dot([np.cos(v_spher) * np.cos(u_spher),
                                          np.cos(v_spher) * np.sin(u_spher),
                                          np.sin(v_spher)])
        points = self.center + self.r * normals + \
            self.radius * self.rotate_matrix.dot([np.cos(u_spher), np.sin(u_spher), 0])

        return self.filter(points, normals)


def resolve(arr, surf, dead = None):
    if dead is None:
        for s in arr:
            s.resolve()
            if s.feasible:
                np.append(surf, s)
    else:
        for i, s in enumerate(arr):
            s.resolve()
            if s.feasible:
                np.append(surf, s)
            else:
                np.append(dead, i)


def find_toroidal_fragments(
        atoms : List[Atom], 
        probe_radius : float) \
            -> Dict[int, Dict[int, Torus]]:
    
    torus_map = {}
    for i, a in enumerate(atoms):
        if i not in torus_map: torus_map[i] = {}
        torus_a = torus_map[i]

        for j, b in enumerate(atoms[i+1:]):
            normal = a.center - b.center
            normal_size = la.norm(normal)
            if normal_size >= a.radius + b.radius + 2 * probe_radius:
                continue

            alpha = (1 - (a.radius - b.radius) \
                         * (a.radius + b.radius + 2 * probe_radius) \
                         / normal_size ** 2) / 2
            center = b.center - alpha * normal
            radius = (b.radius + probe_radius) ** 2 - (alpha * normal_size) ** 2

            a_point = [probe_radius, a.radius] / (probe_radius + a.radius)
            up_bias = la.multi_dot([a_point, [a.center, center], normal])
            
            b_point = [probe_radius, b.radius] / (probe_radius + b.radius)
            down_bias = la.multi_dot([b_point, [b.center, center], normal])

            t = Torus(center, normal, radius, probe_radius)

            bound = Bound(normal, up_bias)
            t.add_bound(bound)
            a.add_bound(bound.neg())

            bound = Bound(normal, down_bias)
            t.add_bound(bound.neg())
            b.add_bound(bound)
                
            if up_bias <= down_bias:
                continue

            torus_a[j + i + 1] = t
    
    return torus_map


def add_atom_torus_bound(a : Atom, t : Torus, p : Probe):
    with np.cross(t.normal, p.center - t.center) as normal:
        bound = Bound(normal, normal.dot(p.center))
        if bound.valid(a):
            p.add_bound(bound)
            t.add_bound(bound.neg())
        else:
            p.add_bound(bound.neg())
            t.add_bound(bound)


def find_probe_fragments(
        atoms : List[Atom], 
        torus_map : Dict[int, Dict[int, Torus]]) \
            -> Dict[int, Dict[int, Torus]]:
    
    probe_map = {}
    for i in torus_map:
        a = atoms[i]
        for j in torus_map[i]:
            b = atoms[j]
            t_ab = torus_map[i][j]
            for k in torus_map[j]:
                if k not in torus_map[i]:
                    continue

                c = atoms[k]
                t_ac = torus_map[i][k]
                t_bc = torus_map[j][k]

                M = np.array([t_ab.normal, t_ac.normal, t_bc.normal])
                if la.matrix_rank(M) != 3:
                    continue

                with la.solve(M, np.array(t_ab.bias, t_ac.bias, t_bc.bias)) as x:
                    if x in probe_map:
                        p = probe_map[x]
                    else:
                        p = Probe(x)
                        probe_map[x] = p

                add_atom_torus_bound(a, t_bc, p)
                add_atom_torus_bound(b, t_ac, p)
                add_atom_torus_bound(c, t_ab, p)

    return probe_map


class Resolver:
    def __init__(self, res, addon):
        self.res = res
        self.addon = addon

    def resolve_task(self, args):
        i, x = args
        resolve(x, self.res[i], self.addon)

    def resolve_task2(self, args):
        i, x = args
        for f in x:
            self.res[i].extend(f.generate_points(self.addon))


def find_ses_fragments(
        atoms : List[Atom], 
        probe_radius : float,
        jobs_num : int) -> List[Surface]:
    
    res = np.array([ np.array([]) for _ in range(jobs_num) ])

    torus_map = find_toroidal_fragments(atoms, probe_radius)
    
    dead_atoms = []
    resolver = Resolver(res, dead_atoms)
    with Pool(jobs_num) as p:
        p.map(resolver.resolve_task,
              enumerate(np.array_split(atoms, jobs_num)), chunksize=1)
        
    tori = []
    for i in dead_atoms: del torus_map[i]
    for i in torus_map:
        for j in dead_atoms: torus_map[i].pop(j, None)
        tori.extend(torus_map[i].values())

    probe_map = find_probe_fragments(atoms, torus_map)
    resolver = Resolver(res, None)
    
    if tori:
        with Pool(jobs_num) as p:
            p.map(resolver.resolve_task, 
                enumerate(np.array_split(tori, jobs_num)), chunksize=1)
    
    if probe_map:
        with Pool(jobs_num) as p:
            p.map(resolver.resolve_task, 
                enumerate(np.array_split(probe_map.values(), jobs_num)), chunksize=1)

    return np.concatenate(res)


def generate_ses_points(coords, radii, probe_radius, point_area, jobs_num):
    atoms = [ Atom(c,r) for c,r in zip(coords, radii) ]
    fragments = find_ses_fragments(atoms, probe_radius, jobs_num)

    res = np.array([ np.array([]) for _ in range(jobs_num) ])
    resolver = Resolver(res, point_area)
    with Pool(jobs_num) as p:
        p.map(resolver.resolve_task2, 
              enumerate(np.split(fragments, jobs_num)), chunksize=1)
        
    return np.concatenate(res)


if __name__ == '__main__':

    RADIUS = {}
    with open("atomtype.txt") as f:
        for line in f.readlines()[1:]:
            atom = line.split()[0]
            RADIUS[atom] = line.split()[-2]

    name_x = "1xvr_DE"
    molecules_npydir = "/auto/datasets/npi/raw/01-benchmark_surfaces_npy"
    coords = np.load(f"{molecules_npydir}/{name_x}_atomxyz.npy")
    types = np.load(f"{molecules_npydir}/{name_x}_atomtypes.npy")
    radii = np.array(list(map(lambda x: RADIUS[x], types)), dtype = float)
    points, atoms_ids = surface.points_with_atomsid(coords, radii, additional_rad = 10)

    probe_radius = 1.4  
    point_area = 1  
    jobs_num = 1

    ses_points = generate_ses_points(coords, radii, probe_radius, point_area, jobs_num)

    print(ses_points.shape)  
    print(ses_points[:5])  
