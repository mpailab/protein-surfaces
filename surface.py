import numpy as np
from scipy.optimize import linprog

def local_points(atom, R, neighbours, rs):
    """
    atom: np.array(1 x dimension)
    R: float
    neighbours: np.array(n x dimension)
    rs: np.array(n)
    """
    dimension = atom.size
    # Перейти в координаты с центром в atom
    neighbours = neighbours - atom
    # Ограничение искомой точки: кубическая окрестность atom
    A_external_cub = np.concatenate((np.eye(dimension), -np.eye(dimension)))
    b_external_cub = np.array([R] * 2 * dimension)
    # Основное неравенство
    A_main = neighbours
    sqr_neighbours = np.array(list(map(lambda x: np.dot(x, x), neighbours))) # Квадратат расстояния до atom
    sqr_rs = rs ** 2
    b_main = (sqr_neighbours - sqr_rs + R ** 2) / 2
    # Поиск центра масс
    # sum_rs = rs.sum()
    # weighted_neighbours = neighbours * rs.reshape(rs.size, 1)
    # M = weighted_neighbours.sum(0) / sum_rs
    # is_null = all([x == 0 for x in M])
    res = []
    for i in range(R * dimension):
        A_eq = A_external_cub[i:i+1, :]
        b_eq = np.array([R])
        A_ub = np.concatenate((A_external_cub[:i, :], A_external_cub[i+1:, :], A_main))
        b_ub = np.concatenate(([R] * (2 * dimension - 1), b_main))
        c = np.zeros(dimension)# if is_null else M
        #c = np.ones(dimension)
        res_i = linprog(c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, bounds = (None, None))
        if res_i.status == 0:
            x = res_i.x
            ro = np.dot(x, x) ** (1 / 2)
            x = x * (R / ro)
            res.append(x + atom)
    return res

def neighbours_with_rs(atom_id, coords, rs):
    ns = []
    ns_rs = []
    atom = coords[atom_id]
    R = rs[atom_id]
    coords = np.delete(coords, atom_id, 0)
    rs = np.delete(rs, atom_id)
    for xyz, r in zip(coords, rs):
        v = xyz - atom
        if np.dot(v, v) < r + R:
            ns.append(xyz)
            ns_rs.append(r)
    return np.array(ns), np.array(ns_rs)

def points_with_atomsid(coords, rs, additional_rad):
    surface_points = []
    atoms_ids = []
    for atom_id, atom in enumerate(coords):
        ns, ns_rs = neighbours_with_rs(atom_id, coords, rs)
        lps = local_points(atom, rs[atom_id], ns, ns_rs)
        surface_points += lps
        atoms_ids += [atom_id] * len(lps)
    return np.array(surface_points), np.array(atoms_ids)