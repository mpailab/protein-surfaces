import math
import numpy as np
from numpy import linalg as la
from scipy.optimize import linprog
from multiprocessing import Pool
from typing import Callable, Dict, List, FrozenSet, Tuple
import surface


class Bound (object):

    def __init__(self, normal : np.array, bias):
        self.normal = normal
        self.bias = bias

    def valid(self, x):
        return self.normal.dot(x) <= self.bias
    
    def neg(self):
        return Bound(-self.normal, -self.bias)


class Surface (object):

    def __init__(self, center, radius, external):
        self.center = np.array(center)
        self.radius = radius
        self.bounds = []
        self.feasible = True
        self.external = external

    def add_bound(self, bound : Bound):
        self.bounds.append(bound)

    def resolve(self):
        if not self.feasible or not self.bounds:
            return
        
        self.feasible = linprog([0,0,0],
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
        u_grid = u_grid.flatten()
        v_grid = v_grid.flatten()

        u_spher = u_grid * 2 * np.pi  
        v_spher = np.arcsin(2 * v_grid - 1)

        normals = np.array([
            [np.cos(x) * np.cos(y), np.sin(x) * np.cos(y), np.sin(y)] 
            for x, y in zip(u_spher, v_spher) 
        ])
        points = self.radius * normals + self.center

        return self.filter(points, normals)


class Atom (Sphere):

    def __init__(self, id, center, radius):
        super().__init__(center, radius, True)
        self.id = id
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
        self.bias = normal.dot(center)

        x = self.normal[1] / np.sqrt(self.normal[0] ** 2 + self.normal[1] ** 2)
        y = - self.normal[0] / np.sqrt(self.normal[0] ** 2 + self.normal[1] ** 2)
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
        u = u.flatten()
        v = v.flatten()

        u_spher = u * 2 * np.pi
        v_spher = np.arcsin(v) + np.pi

        A = np.array([
            [np.cos(x) * np.cos(y), np.sin(x) * np.cos(y), np.sin(y)] 
            for x, y in zip(u_spher, v_spher) 
        ])
        B = np.array([[np.cos(x), np.sin(x), 0] for x in u_spher])
        
        normals = np.dot(A, self.rotate_matrix)
        points = self.center + self.r * normals + \
            self.radius * np.dot(B, self.rotate_matrix)

        return self.filter(points, normals)


def find_toroidal_fragments(
        atoms : List[Atom], 
        probe_radius : float) \
            -> Dict[int, Dict[int, Torus]]:
    
    torus_map = {}
    for a in atoms:
        if a.id not in torus_map: torus_map[a.id] = {}
        torus_a = torus_map[a.id]

        for b in atoms[a.id+1:]:
            normal = a.center - b.center
            normal_size = la.norm(normal)
            if normal_size >= a.radius + b.radius + 2 * probe_radius:
                continue

            alpha = (1 - (a.radius - b.radius) \
                         * (a.radius + b.radius + 2 * probe_radius) \
                         / normal_size ** 2) / 2
            center = a.center - alpha * normal
            radius = np.sqrt((a.radius + probe_radius) ** 2 - (alpha * normal_size) ** 2)

            a_point = np.array([probe_radius, a.radius]) / (probe_radius + a.radius)
            up_bias = la.multi_dot([a_point, [a.center, center], normal])
            
            b_point = np.array([probe_radius, b.radius]) / (probe_radius + b.radius)
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

            torus_a[b.id] = t
    
    return torus_map


def add_atom_torus_bound(a : Atom, t : Torus, p : Probe):
    normal = np.cross(t.normal, p.center - t.center)
    bound = Bound(normal, normal.dot(p.center))
    if bound.valid(a.center):
        p.add_bound(bound)
        t.add_bound(bound.neg())
    else:
        p.add_bound(bound.neg())
        t.add_bound(bound)


def find_probe_fragments(
        atoms : List[Atom], 
        torus_map : Dict[int, Dict[int, Torus]], 
        probe_radius : float) \
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

                x = np.cross(t_ab.normal, t_ac.normal)
                y = la.solve([t_ab.normal, t_ac.normal, x], [t_ab.bias, t_ac.bias, 0])
                z = y - t_ab.center
                roots = np.roots([x.dot(x), 2 * x.dot(z), z.dot(z) - t_ab.radius ** 2])
                rroots = roots.real[np.abs(roots.imag) < 1e-5]
                froots = [s for s in rroots 
                            if np.abs(t_bc.normal.dot(s * x + y) - t_bc.bias) < 0.00001]
                if not froots:
                    continue

                for s in froots:
                    v = s * x + y
                    key = tuple(np.true_divide(np.fix(v * 100000), 100000))
                    if key in probe_map:
                        p = probe_map[key]
                    else:
                        p = Probe(v, probe_radius)
                        probe_map[key] = p

                    add_atom_torus_bound(a, t_bc, p)
                    add_atom_torus_bound(b, t_ac, p)
                    add_atom_torus_bound(c, t_ab, p)
                    

    return probe_map


def resolve(elem):
    elem.resolve()


def find_ses_fragments(
        atoms : List[Atom], 
        probe_radius : float,
        jobs_num : int) -> List[Surface]:
    
    torus_map = find_toroidal_fragments(atoms, probe_radius)
    
    with Pool(jobs_num) as p:
        p.map(resolve, atoms, chunksize=math.ceil(len(atoms)/jobs_num))

    torus_map = {
        i : { j : t for j, t in m.items() if atoms[j].feasible}
        for i, m in torus_map.items() if atoms[i].feasible
    }
    probe_map = find_probe_fragments(atoms, torus_map, probe_radius)

    tori = [t for _, m in torus_map.items() for t in m.values()]
    with Pool(jobs_num) as p:
        p.map(resolve, tori, chunksize=math.ceil(len(tori)/jobs_num))

    probes = list(probe_map.values())
    with Pool(jobs_num) as p:
        p.map(resolve, probes, chunksize=math.ceil(len(probes)/jobs_num))

    return ([a for a in atoms if a.feasible],
            [t for t in tori if t.feasible],
            [p for p in probes if p.feasible])


class Generator:
    def __init__(self, point_area):
        self.point_area = point_area

    def call(self, elem):
        return elem.generate_points(self.point_area)


def generate_ses_points(coords, radii, probe_radius, point_area, jobs_num=1, split=False):
    atoms = [ Atom(i,c,r) for i,(c,r) in enumerate(zip(coords, radii)) ]
    fragments = find_ses_fragments(atoms, probe_radius, jobs_num)
    generator = Generator(point_area)
    
    if split:
        res = []
        for lf in fragments:
            with Pool(jobs_num) as p:
                results = p.map(generator.call, lf, chunksize=math.ceil(len(lf)/jobs_num))
                results = [x for x in results if x]
                res.append(np.concatenate(results) if results else results)
        return res
    
    else:
        fragments = np.concatenate(fragments)
        with Pool(jobs_num) as p:
            results = p.map(generator.call, fragments, 
                            chunksize=math.ceil(len(fragments)/jobs_num))
            results = [x for x in results if x]
            return np.concatenate(results) if results else results


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

    # coords = [(0,0,0), (1,0,0)]
    # radii = [1, 1]

    probe_radius = 1.4  
    point_area = 1  
    jobs_num = 1

    ses_points = generate_ses_points(coords, radii, probe_radius, point_area, jobs_num)

    print(ses_points.shape)  
    print(ses_points[:5])  
