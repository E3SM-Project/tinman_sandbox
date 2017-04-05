
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def parse_fname(fname):
    """
    Parses the filename from the format %s.level_%d.exec_%d.elem_%d.threads_%d.vectors_%d
    Returns the name, number of levels, number of executions,
    and number of elements used to construct the file
    """
    parts = [int(p.split('_')[1]) for p in fname.split('.')[1:]]
    return [fname.split('.')[0]] + parts

def read_data(fname):
    f_info = parse_fname(fname)
    data = np.zeros(f_info[2])
    i = 0
    f = open(fname, 'r')
    for line in f:
        data[i] = float(line.split(' ')[2])
        i += 1
    if i < f_info[2]:
        data = np.array([])
    return f_info + [data]

def aggregate_data(path):
    files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    aggregated = {}
    for fname in files:
        title, n_lev, n_exec, n_elem, n_thread, n_vector, data = read_data(fname)
        if ((2 ** min_vectors) > n_vector or n_vector > (2 ** max_vectors)
            or min_threads > n_thread or n_thread > max_threads
            or len(data) == 0 or title == '/home/mdeakin/Documents/pref_eval/performance/pointers_only'):
            if len(data) == 0:
                os.unlink(os.path.join(path, fname))
            continue
        if title not in aggregated:
            aggregated[title] = {}
        if n_lev not in aggregated[title]:
            aggregated[title][n_lev] = {}
        if n_elem not in aggregated[title][n_lev]:
            aggregated[title][n_lev][n_elem] = {}
        if n_thread not in aggregated[title][n_lev][n_elem]:
            aggregated[title][n_lev][n_elem][n_thread] = {}
        assert(n_vector not in aggregated[title][n_lev][n_elem][n_thread])
        aggregated[title][n_lev][n_elem][n_thread][n_vector] = data
    return aggregated

def line_area(m, c, min_x, max_x):
    return m * (max_x ** 2 - min_x ** 2) + c * (max_x - min_x)

def plot_vs_levels(aggregate):
    fmt_str = '({} threads, {} vectors, {} elements) least squares fit'

    best_slope_line = None
    best_intercept_line = None
    best_area_line = None

    for title in aggregate:
        plt.figure(figsize=(24, 16))
        plt.title('kokkos-scratch time vs n_lev')
        n_elems = set([])
        n_threads = set([])
        n_vectors = set([])
        for n_lev in aggregate[title]:
            elems = set(aggregate[title][n_lev].keys())
            n_elems |= elems
            for e in elems:
                threads = set(aggregate[title][n_lev][e].keys())
                n_threads |= threads
                for t in threads:
                    vectors = set(aggregate[title][n_lev][e][t].keys())
                    n_vectors |= vectors
        for n_elem in sorted(list(n_elems)):
            for thread in sorted(list(n_threads)):
                for vector in sorted(list(n_vectors)):
                    x_pos = np.array([])
                    y_pos = np.array([])
                    min_n_lev = aggregate[title].keys()[0]
                    max_n_lev = aggregate[title].keys()[0]
                    for n_lev in aggregate[title]:
                        if n_lev > max_n_lev:
                            max_n_lev = n_lev
                        if n_lev < min_n_lev:
                            min_n_lev = n_lev
                        if (n_elem in aggregate[title][n_lev]
                            and thread in aggregate[title][n_lev][n_elem]
                            and vector in aggregate[title][n_lev][n_elem][thread]):
                            y_vals = aggregate[title][n_lev][n_elem][thread][vector]
                            x_pos = np.append(x_pos, np.repeat(n_lev, len(y_vals)))
                            y_pos = np.append(y_pos, y_vals)
                    clr = cm.get_cmap('winter')((n_elem - min_elems + 1) / range_elems
                                                * (thread - min_threads + 1) / range_threads
                                                * (np.log2(vector) - min_vectors + 1) / range_vectors)
                    plt.scatter(x_pos, y_pos, color=clr)
                    # Find the best fitting line with quadratic residual to the timing data
                    m, c = np.linalg.lstsq(np.vstack([x_pos, np.ones(len(x_pos))]).T, y_pos)[0]
                    plt.plot(x_pos, m * x_pos + c, color=clr,
                             label=fmt_str.format(thread, vector, n_elem))
                    area = line_area(m, c, min_n_lev, max_n_lev)
                    if best_slope_line is None or m < best_slope_line[0]:
                        best_slope_line = (m, c, area, n_elem, thread, vector)
                    if best_intercept_line is None or c < best_intercept_line[1]:
                        best_intercept_line = (m, c, area, n_elem, thread, vector)
                    if best_area_line is None or area < best_area_line[2]:
                        best_area_line = (m, c, area, n_elem, thread, vector)

        plt.legend(loc='upper left')
        plt.xlim([0, 600])
        plt.ylim([0, 0.25])
        plt.savefig('kokkos_scratch_time_vs_nlev_{}-{}_threads_vectors.png'.format(min_threads, max_threads))
    return (best_slope_line, best_intercept_line, best_area_line)

def plot_vs_elems(aggregate):
    return
    for title in aggregate:
        plt.figure()
        plt.title(title + ' time vs n_elem')
        for n_lev in sorted(aggregate[title].keys()):
            x_pos = np.array([])
            y_pos = np.array([])
            for n_elem in aggregate[title][n_lev]:
                y_vals = aggregate[title][n_lev][n_elem]
                x_pos = np.append(x_pos, np.repeat(n_elem, len(y_vals)))
                y_pos = np.append(y_pos, y_vals)
            plt.scatter(x_pos, y_pos, color=cm.get_cmap('winter')(n_lev / 1152.0))
            # Find the best fitting line with quadratic residual to the timing data
            m, c = np.linalg.lstsq(np.vstack([x_pos, np.ones(len(x_pos))]).T, y_pos)[0]
            plt.plot(x_pos, m * x_pos + c, color=cm.get_cmap('winter')(n_lev / 1152.0),
                     label=str(n_lev) + ' levels least squares')
        l = plt.legend(loc='upper left')
        plt.xlim([0, 1100])
        plt.ylim([0, 3.0])

min_vectors = 0
max_vectors = 4
range_vectors = max_vectors - min_vectors + 1.0

min_elems = 500
max_elems = 600
range_elems = max_elems - min_elems + 1.0

best_slope = None
best_intercept = None
best_area = None

for i in range(16):
    min_threads = 2 * i + 1
    max_threads = 2 * (i + 1)
    range_threads = max_threads - min_threads + 1.0

    aggregate = aggregate_data('/home/mdeakin/Documents/pref_eval/performance')
    best_lines = plot_vs_levels(aggregate)
    print("Best LS Lines")
    for l in best_lines:
        print(l)
    if best_slope is None or best_lines[0][0] < best_slope[0]:
        best_slope = best_lines[0]
    if best_intercept is None or best_lines[1][1] < best_intercept[1]:
        best_intercept = best_lines[1]
    if best_area is None or best_lines[2][2] < best_area[2]:
        best_area = best_lines[2]
    plot_vs_elems(aggregate)
#plt.show()
print("Best Slope:", best_slope)
print("Best Intercept:", best_intercept)
print("Best Area:", best_area)
