def read_data(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()
    title_line  = lines[0]
    data = [] 
    for i in range(1, len(lines), 2):
        line = lines[i].split(';')
        data.append({'noise_level' : float(line[0]), 'fitness_mean': float(line[1]), 'fitness_var': float(line[2][:-1])})
    return data