import numpy as np
from scipy.optimize import linprog
import timeit
import faiss

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
    for i in range(2 * dimension):
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

# Медленно: к удалению
def neighbours_mask(atom_id, coords, rs):
    atom = coords[atom_id]
    R = rs[atom_id]
    mask = np.zeros(coords.shape[0], dtype=bool)
    #coords = np.delete(coords, atom_id, 0)
    #rs = np.delete(rs, atom_id)
    for index, (xyz, r) in enumerate(zip(coords, rs)):
        if index == atom_id:
            continue
        v = xyz - atom
        if np.dot(v, v) < r + R:
            mask[index] = True
    return mask

def points_with_atomsid(coords, rs, additional_rad):
    surface_points = []
    atoms_ids = []
    rs = rs + additional_rad

    start_time = timeit.default_timer()
    R = 2 * max(rs)
    d = coords.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(coords)
    lims, D, I = index.range_search(coords, R)

    search_neighbours_time = timeit.default_timer() - start_time
    search_points_time = 0
    for atom_id, atom in enumerate(coords[:3]):
        start_time = timeit.default_timer()

        mask = I[lims[atom_id]:lims[atom_id+1]]
        neighbours = coords[mask]
        neighbours_rs = rs[mask]

        #Удаление atom в neighbours
        mask = (neighbours != atom).all(axis = 1)
        neighbours = neighbours[mask]
        neighbours_rs = neighbours_rs[mask]

        mez_time = timeit.default_timer()
        search_neighbours_time += mez_time - start_time
        lps = local_points(atom, rs[atom_id], neighbours, neighbours_rs)
        search_points_time += timeit.default_timer() - mez_time
        surface_points += lps
        atoms_ids += [atom_id] * len(lps)
    summ = search_neighbours_time + search_points_time
    print("search_neighbours_time: ", search_neighbours_time, search_neighbours_time / summ * 100, "%")
    print("search_points_time: ", search_points_time, search_points_time / summ * 100, "%")
    return np.array(surface_points), np.array(atoms_ids)