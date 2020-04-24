import numpy
import matplotlib.pyplot as plt
import seaborn as sns
def to_np_array(l):
    return numpy.array([numpy.array(x) for x in l])
def read_from_archive(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    archive = []
    for l in lines:
        line_s = l.split(' ')
        fitness = float(line_s[0])
        centroid = to_np_array([float(i) for i in line_s[1:-19]])
        x = to_np_array([float(i) for i in line_s[-19:-1]])
        # print(len(centroid), len(x))
        # print(x, centroid)
        archive.append({'x': x, "fitness" : fitness, "centroid" : centroid})
    file.close()
    return archive
def convert_exponent(s):
    if s[0] == "−":
        res = 10.**(-float(s[1:]))
    else:
        res = 10.**(float(s))
    return res
if __name__ == "__main__":
    archive = read_from_archive("archive_30296.dat")
    data = to_np_array([-a["fitness"] for a in archive if a['fitness'] > -float('inf')])
    fig, ax = plt.subplots(figsize=(20,10))
    sns.distplot(numpy.log10(data))
    fig.canvas.draw()
    locs, labels = plt.xticks()
    ax.set(xticklabels=["{:.2E}".format(convert_exponent(i.get_text())) for i in labels])
    fig.tight_layout()
    fig.savefig('fitness_distribution.pdf')
    fig.savefig('fitness_distribution.svg')