import torch

import math
import numpy as np
from numpy import linalg as la
from scipy.optimize import linprog
from multiprocessing import Pool
from itertools import product
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

class Atom:
    def __init__(self,
                 coords : torch.Tensor, 
                 radii : torch.Tensor, 
                 probe_radius : float) -> None:
        self.center = coords.clone()
        self.radius = radii.clone()
        self.ext_radius = radii + probe_radius
        self.num = self.center.size(dim=0)

    def apply_mask(self, mask : torch.Tensor) -> None:
        self.center = self.center[mask]
        self.radius = self.radius[mask]
        self.ext_radius = self.ext_radius[mask]
        self.num = self.center.size(dim=0)


class Torus:
    def __init__(self) -> None:
        self.center = None
        self.R = None
        self.r = None
        self.normal = None
        self.phi = None
        self.psi = None
        self.rotation = None
        self.num = 0

    def apply_mask(self, mask : torch.Tensor) -> None:
        self.center = self.center[mask]
        self.R = self.R[mask]
        self.r = self.r[mask]
        self.normal = self.normal[mask]
        self.bias = self.bias[mask]
        self.up_bias = self.up_bias[mask]
        self.down_bias = self.down_bias[mask]
        self.num = self.center.size(dim=0)


def generate_sphere_points(self, center, radius, num):
        u = torch.linspace(0, 1, num)  
        v = torch.linspace(0, 1, num)  

        u_grid, v_grid = torch.meshgrid(u, v)

        u_spher = u_grid.flatten() * 2 * torch.pi  
        v_spher = torch.arcsin(2 * v_grid.flatten() - 1)

        normals = torch.stack((torch.cos(u_spher) * torch.cos(v_spher),
                               torch.sin(u_spher) * torch.cos(v_spher),
                               torch.sin(v_spher)))
        
        points = center + radius * normals

        return self.filter(points, normals)


def get_rotation_matrix(
        vector : torch.Tensor,
        vector_size : torch.Tensor) -> torch.Tensor:
    
    h = vector[:,:2]
    x,y = torch.tensor([[0,1],[-1,0]]).dot(h.swapaxes(0,1) / h.norm(dim=1))[:2]
    t = vector[:,2] / vector_size
    c = [1-t, torch.sqrt(1-t.pow(2)), t]

    r11 = x.pow(2) * c[0] + c[2]
    r22 = y.pow(2) * c[0] + c[2]
    r33 = c[2]
    r12 = x * y * c[0]
    r13 = y * c[1]
    r32 = x * c[1]

    return torch.stack((r11,r12,r13,r12,r22,-r32,-r13,r32,r33), dim=2).squeeze()


def bound(x : torch.Tensor,
          u : torch.Tensor,
          v : torch.Tensor) -> torch.Tensor:
    return torch.arccos((x * u).sum(1) / x.pow(2).sum(1).pow(0.5)) / torch.pi + \
           torch.sign((x * v).sum(1)) / 2


class SES:
    def __init__(self,
                 coords : torch.Tensor, 
                 radii : torch.Tensor, 
                 probe_radius : float, 
                 point_area : float) -> None:
        self.atom = Atom(coords, radii, probe_radius)
        self.torus = Torus()
        self.probe_radius = probe_radius
        self.point_area = point_area

    def find_toroidal_fragments(self) -> None:

        # Identity matrix
        I = torch.eye(self.atom.num)

        # All combinations of pairs of atom indices
        combs = torch.combinations(torch.arange(self.atom.num))
        m = combs.size(dim=0)
        a,b = combs.hsplit(2)
        a.squeeze_(1)
        b.squeeze_(1)

        # Find mask of all pairs of atoms whose distance is less than twice probe's radius
        self.torus.normal = self.atom.center[a] - self.atom.center[b]
        normal_scale = self.torus.normal.pow(2).sum(1)
        normal_size = torch.sqrt(normal_scale)
        mask = (self.atom.radius[a] + self.atom.radius[b] - normal_size).ge(1e-5 - 2 * self.probe_radius)

        # Reduce tensors with respect to mask
        a = a[mask]
        b = b[mask]
        self.torus.normal = self.torus.normal[mask]
        normal_scale = normal_scale[mask]
        normal_size = normal_size[mask]

        # Tensors of atoms centers
        a_center = self.atom.center[a]
        b_center = self.atom.center[b]

        # Tensors of atoms radii
        a_radius = self.atom.radius[a]
        b_radius = self.atom.radius[b]

        # Tensors of atoms extended radii (atom radius + probe radius)
        a_ext_radius = self.atom.ext_radius[a]
        b_ext_radius = self.atom.ext_radius[b]

        # Find the parameters of a torus generated by a probe sweeping around 
        # the intersection of two atoms in the pair.
        alpha = (1 - (a_ext_radius.pow(2) - b_ext_radius.pow(2)) / normal_scale) / 2
        self.torus.center = a_center - alpha * self.torus.normal
        self.torus.R = torch.sqrt(a_ext_radius.pow(2) - (alpha * normal_size).pow(2))
        self.torus.r = torch.tensor(self.probe_radius).repeat(m)
        bias = (self.torus.normal * self.torus.center).sum(axis = 1)
        up_bias = torch.linalg.multi_dot([
            torch.stack((self.torus.r, a_radius)).div(a_ext_radius), 
            torch.stack((a_center, self.torus.center)), 
            self.torus.normal])
        down_bias = torch.linalg.multi_dot([
            torch.stack((self.torus.r, b_radius)).div(b_ext_radius), 
            torch.stack((b_center, self.torus.center)), 
            self.torus.normal])
        self.torus.phi = torch.tensor((0,1)).repeat(m)
        self.torus.psi = torch.stack(((down_bias - bias) / normal_size,
                                      (up_bias - bias) / normal_size), dim=1)
        self.torus.rotation = get_rotation_matrix(self.torus.normal, normal_size)

        # Find atoms for
        A = torch.cat((torch.cat((-self.torus.normal, I[a]), dim=1), 
                       torch.cat((self.torus.normal, I[b]), dim=1))).numpy()
        B = torch.cat((-self.torus.up_bias, self.torus.down_bias)).numpy()
        C = 3 * [0] + self.atom.num * [1]
        bounds = 3 * [(None,None)] + self.atom.num * [(0,None)]
        solution = torch.from_numpy(linprog(C, A_ub=A, b_ub=B, bounds=bounds).x[3:])
        feasible_atoms_mask = solution[3:].le(1e-5)
        self.atom.apply_mask(feasible_atoms_mask)

        feasible_tori_mask = (self.torus.up_bias - self.torus.down_bias).ge(1e-5) \
            * (self.torus.R - self.torus.r).ge(1e-5) \
            * feasible_atoms_mask[a] * feasible_atoms_mask[b]
        self.torus.apply_mask(feasible_tori_mask)

    def find_probe_fragments(self) -> None:
        combs = torch.tensor(list(product(range(self.atom.num), range(self.torus.num))))
        m = combs.size(dim=0)
        a,t = combs.hsplit(2)
        a.squeeze_(1)
        t.squeeze_(1)

        normal = self.atom.center[a] - self.torus.center[t]
        normal_scale = normal.pow(2).sum(1)
        normal_size = torch.sqrt(normal_scale)

        mask = (self.atom.radius[a] + self.torus.R[t] + self.torus.r[t] - normal_size).ge(1e-5)
        a = a[mask]
        t = t[mask]
        normal = normal[mask]
        normal_scale = normal_scale[mask]
        normal_size = normal_size[mask]

        a_center = self.atom.center[a]
        a_radius = self.atom.radius[a]
        a_ext_radius = self.atom.ext_radius[a]

        t_center = self.torus.center[t]
        t_R = self.torus.R[t]
        t_r = self.torus.r[t]
        t_normal = self.torus.normal[t]
        t_bias = self.torus.bias[t]
        t_up_bias = self.torus.up_bias[t]
        t_down_bias = self.torus.down_bias[t]
        t_rotation = self.torus.rotation[t]

        alpha = (1 - (a_ext_radius.pow(2) - t_R.pow(2)) / normal_scale) / 2
        center = a_center - alpha * normal
        R = torch.sqrt(a_ext_radius.pow(2) - (alpha * normal_size).pow(2))
        h = torch.nn.functional.normalize(torch.linalg.cross(normal, self.torus.normal))
        x = torch.round(center + R * h, decimals=5)
        y = torch.round(center - R * h, decimals=5)

        u = torch.nn.functional.normalize(t_rotation.dot(torch.tensor([1,0,0])))
        v = torch.nn.functional.normalize(t_rotation.dot(torch.tensor([0,1,0])))
        bound_x = bound(x - t_center, u, v)
        bound_y = bound(y - t_center, u, v)
        bound_c = bound(center - t_center, u, v)

        self.torus.R = torch.sqrt(a_ext_radius.pow(2) - (alpha * normal_size).pow(2))
        self.torus.r = torch.tensor(self.probe_radius).repeat(m)
        self.torus.bias = (self.torus.normal * self.torus.center).sum(axis = 1)

def generate_points(
        coords : torch.Tensor, 
        radii : torch.Tensor, 
        probe_radius : float, 
        point_area : float) -> torch.Tensor:
    
    # n = coords.size(dim=0)
    # I = torch.eye(n)
    # combs = torch.combinations(torch.arange(n))
    # m = combs.size(dim=0)
    # a,b = combs.hsplit(2)

    # a.squeeze_(1)
    # a_center = coords[a]
    # a_radius = radii[a]
    # a_ext_radius = a_radius + probe_radius

    # b.squeeze_(1)
    # b_center = coords[b]
    # b_radius = radii[b]
    # b_ext_radius = b_radius + probe_radius

    # torus_normal = a_center - b_center
    # torus_normal_scale = torus_normal.pow(2).sum(1)
    # torus_normal_size = torch.sqrt(torus_normal_scale)
    # alpha = (1 - (a_radius - b_radius) * (a_ext_radius + b_ext_radius) / torus_normal_scale) / 2
    # torus_center = a_center - alpha * torus_normal
    # torus_R = torch.sqrt(a_ext_radius.pow(2) - (alpha * torus_normal_size).pow(2))
    # torus_r = torch.tensor(probe_radius).repeat(m)
    # torus_bias = (torus_normal * torus_center).sum(axis = 1)
    # torus_up_bias = torch.linalg.multi_dot([
    #     torch.stack((torus_r, a_radius)).div(a_ext_radius), 
    #     torch.stack((a_center, torus_center)), 
    #     torus_normal])
    # torus_down_bias = torch.linalg.multi_dot([
    #     torch.stack((torus_r, b_radius)).div(b_ext_radius), 
    #     torch.stack((b_center, torus_center)), 
    #     torus_normal])
    # torus_rotation_matrix = get_rotation_matrix(torus_normal, torus_normal_size)

    # A = torch.cat((torch.cat((-torus_normal, -torus_up_bias, I[a]), dim=1), 
    #                torch.cat((torus_normal, torus_down_bias, I[b]), dim=1))).numpy()
    # B = torch.cat((-torus_up_bias, torus_down_bias)).numpy()
    # C = 3 * [0] + n * [1]
    # bounds = 3 * [(None,None)] + n * [(0,None)]
    # solution = torch.from_numpy(linprog(C, A_ub=A, b_ub=B, bounds=bounds).x[3:])
    # feasible_atoms_mask = solution.le(1e-5)

    # mask = (torus_normal_size - a_radius - b_radius).ge(2 * probe_radius) \
    #      * (torus_up_bias - torus_down_bias).ge(0) \
    #      * feasible_atoms_mask[a] * feasible_atoms_mask[b]
    
    # torus_center = torus_center[mask]
    # torus_R = torus_R[mask]
    # torus_r = torus_r[mask]
    # torus_normal = torus_normal[mask]
    # torus_bias = torus_bias[mask]
    # torus_up_bias = torus_up_bias[mask]
    # torus_down_bias = torus_down_bias[mask]
    
    


    # atoms = torch.cat((coords,                       # center
    #                    radii,                        # radius
    #                    torch.arange(n).unsqueeze(1), # id
    #                    coords.pow(2).sum(1),         # dist
    #                    dist - radii.pow(2)),         # bias
    #                   dim=1)
    

    # a,b = torch.combinations(atoms).hsplit(2)
    # a_center, a_radius, _ = a.hsplit([3,4])
    # b_center, b_radius, _ = a.hsplit([3,4])

    # normal = a_center - b_center
    # normal_scale = normal.pow(2).sum(1)
    # normal_size = torch.sqrt(normal_scale)

    # mask = (normal_size - a_radius - b_radius).ge(2 * probe_radius)
    # a = a[mask]
    # b = b[mask]
    # a_center, a_radius, a_id, a_dist, a_bias = a.hsplit([3,4,5,6])
    # b_center, b_radius, b_id, b_dist, b_bias = a.hsplit([3,4,5,6])
    # normal = normal[mask]
    # normal_scale = normal_scale[mask]
    # normal_size = normal_size[mask]
    # m = a.size(dim=0)
    # probe_radius_tensor = torch.tensor(probe_radius).repeat(m)

    # a_ext_radius = a_radius + probe_radius
    # b_ext_radius = b_radius + probe_radius

    # alpha = (1 - (a_radius - b_radius) * (a_ext_radius + b_ext_radius) / normal_scale) / 2
    # center = a_center - alpha * normal
    # radius = torch.sqrt(a_ext_radius.pow(2) - (alpha * normal_size).pow(2))

    # rotation_matrix = get_rotation_matrix(normal, normal_size)

    # tori = torch.cat((center,                          # center
    #                   radius,                          # big radius
    #                   probe_radius_tensor,             # small radius
    #                   normal,                          # normal
    #                   (normal * center).sum(axis = 1), # bias
    #                   rotation_matrix),                  # rotate matrix
    #                  dim=1)

    # a_point = torch.stack((probe_radius_tensor, a_radius)).div(a_ext_radius)
    # up_bias = torch.linalg.multi_dot([a_point, torch.stack((a_center, center)), normal])
    
    # b_point = torch.stack((probe_radius_tensor, b_radius)).div(b_ext_radius)
    # down_bias = torch.linalg.multi_dot([b_point, torch.stack((b_center, center)), normal])

    # up_bound = torch.cat((normal, up_bias), dim=1)
    # up_bound_neg = -up_bound

    # bound = Bound(normal, up_bias)
    # t.add_bound(bound)
    # a.add_bound(bound.neg())

    # bound = Bound(normal, down_bias)
    # t.add_bound(bound.neg())
    # b.add_bound(bound)
        
    # if up_bias <= down_bias:
    #     continue

    # torus_a[b.id] = t
    
    # dist = coords.pow(2).sum(1)
    # bias = dist - radii.pow(2)
    # bounds = torch.empty((0, 4), dtype=torch.double)
    
    # atoms = torch.stack((coords,                              # center
    #                      radii,                               # radius
    #                      torch.zeros(n, dtype=torch.int),     # id
    #                      torch.empty((0, 25), dtype=torch.double), # bounds
    #                      torch.ones(n, dtype=torch.bool),     # feasible
    #                      torch.zeros(n, dtype=torch.double),  # dist
    #                      torch.zeros(n, dtype=torch.double)), # bias
    #                     dim=1)

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
