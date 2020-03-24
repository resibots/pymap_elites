import map_elites.cvt as cvt_map_elites


# a small test
if __name__ == "__main__":
    def rastrigin(xx):
        x = xx * 10.0 - 5.0
        f = 10 * x.shape[0]
        for i in range(0, x.shape[0]):
            f += x[i] * x[i] - 10 * math.cos(2 * math.pi * x[i])
        return -f, np.array([xx[0], xx[1]])

    archive = cvt_map_elites.compute(2, 6, rastrigin, n_niches=5000, n_gen=2500)
