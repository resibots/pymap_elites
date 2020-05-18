def read_data(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()
    title_line  = lines[0]
    data = [] 
    for i in range(1, len(lines), 2):
        line1 = lines[i].split(';')
        line2 = lines[i+1][1:-2].split(',')
        fitness_values = [float(s) for s in line2]
        filtered_fitness_values = [v for v in fitness_values if v> -float('inf')]
        data.append({'noise_level' : float(line1[0]), 'fitness_mean': float(line1[1]), 'fitness_var': float(line1[2][:-1]), 'min': min(filtered_fitness_values), "max": max(filtered_fitness_values)})
    return data