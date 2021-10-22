import glob

import numpy as np
from bs4 import BeautifulSoup
from shared.QiskitMaxcut import Maxcut

provider = None

def _load_provider():
    global provider
    if provider is None:
        import qiskit
        provider = qiskit.IBMQ.load_account()

def print_parameters():
    _load_provider()
    import ibm_parameters
    ibm_parameters.print_parameters()

def display_graph_with_solution_bitstring():
    _load_provider()
    import ibm_parameters
    graph = ibm_parameters.load_graph()
    maxcut, bitstring = get_local_solver_result(graph.name)
    print(graph.name)
    print(f"MaxCut: {maxcut}")
    print(f"BitString: {bitstring}")
    Maxcut(graph).draw(bitstring)


def get_local_solver_result(graphname):
    f = open("../classical/main_local_solver.html")
    soup = BeautifulSoup(f, 'html.parser')
    pres = soup.find_all('pre')
    pre = ""
    for p in pres:
        if p.text.startswith("--"):
            pre = p.text
    p_lines = pre.split('\n')
    found_counter = 0
    maxcut = 0
    bitstring = np.array([])
    for line in p_lines:
        if graphname == line.strip():
            found_counter += 1
        if found_counter > 0 and found_counter < 3:
            if line.startswith("MaxCut"):
                maxcut = int(line.strip().split(': ')[-1])
                found_counter += 1
            if line.startswith("Bitstring"):
                bitstring_str_list = list(line.strip().split(': ')[-1])
                bitstring = np.array([int(x) for x in bitstring_str_list if x.isnumeric()])
                found_counter += 1

    # print(f"maxcut: {maxcut}")
    # print(f"bitstring: {bitstring}")

    return maxcut, bitstring









def get_name(filename):
    fs = filename.split('_')
    fs.pop(0)
    fs.pop(-1)
    if not fs:
        fs = ["Normal"]
    return " ".join(fs)


def parse_value(line):
    return float(line.split(' ')[-1])

def parse_val_porb(line):
    pp = line.strip().replace(' %', '').replace('\n', '').replace('  ', ' ').split(' ')[-1]
    pv = line.strip().replace(' %', '').replace('\n', '').replace('  ', ' ').split(' ')[-3]
    return (float(pv), float(pp))

def parse_probability(line, probs_dict):
    pp = line.split(' ')[-1].replace('%', '').replace('\n', '')
    pv = line.split(' ')[2].replace(',', '').replace('\n', '')
    pv_int = int(float(pv))
    if pv_int in probs_dict.keys():
        probs_dict[pv_int] += float(pp)
    else:
        probs_dict[pv_int] = float(pp)

    return probs_dict


def parse_params(p):
    pr = p.replace('<pre>array([', '').replace(')</pre>', '').replace(']', '').replace('\n', '').replace(' ', '')
    params = list(map(float, pr.split(',')))
    middle_index = len(params) // 2
    gamma = params[:middle_index]
    beta = params[middle_index:]
    return gamma, beta


def get_attributes(soup):
    mean = 0
    gamma = []
    beta = []

    pre = soup.find_all('pre')
    for p in pre:
        p = str(p)
        if p.startswith('<pre>array'):
            gamma, beta = parse_params(p)

        if p.startswith('<pre>optimal'):
            ppp = p.replace('<pre>', '').replace('</pre>', '')
            for l in ppp.splitlines(False):
                if "Expectation Value" in l:
                    mean = parse_value(l)
                if "Highest Exp.Value" in l:
                    max_v = parse_val_porb(l)
                if "Lowest Exp.Value" in l:
                    min_v = parse_val_porb(l)
                if "Highest Probability" in l:
                    max_p = parse_val_porb(l)
                if "Ratio" in l:
                    ratio = parse_value(l)

    reps = len(gamma)


    return reps, gamma, beta, mean, max_v, min_v, max_p, ratio


def get_attr_dict(soup):
    reps, gamma, beta, mean, max_v, min_v, max_p, ratio = get_attributes(soup)
    return {
        'reps': reps,
        'gamma': gamma,
        'beta': beta,
        'mean': mean,
        'max_v': max_v,
        'min_v': min_v,
        'max_p': max_p,
        'ratio': ratio
    }


def html_file_to_attr_dict(filename):
    f = open(filename)
    soup = BeautifulSoup(f, 'html.parser')
    attr_dict = get_attr_dict(soup)
    attr_dict['img'] = soup.find_all('img')[-1]
    attr_dict['name'] = get_name(filename)
    return attr_dict


def parse_html_files(dir_name):
    parsed_files = []
    files = glob.glob(dir_name + "/*.html")
    for file in files:
        attr_dict = html_file_to_attr_dict(file)
        parsed_files.append(attr_dict.copy())

    return parsed_files


def parse_results():
    mitiq_files = parse_html_files("mitiq/results")
    warmstart_files = parse_html_files("warmstart/results")
    normal_files = parse_html_files("normal/results")
    # parsed_files = parse_html_files('/Users/lachermeier/PycharmProjects/master_thesis_qaoa/ibm/normal')
    parsed_files = mitiq_files + normal_files + warmstart_files

    parsed_files_noise_simulator = []
    for file in parsed_files:
        if 'oise' in file['name']:
            parsed_files_noise_simulator.append(file.copy())

    parsed_files_simulator = []
    for file in parsed_files:
        if 'oise' not in file['name']:
            parsed_files_simulator.append(file.copy())

    parsed_files_noise_simulator = sorted(parsed_files_noise_simulator, key=lambda s: s['mean'], reverse=False)
    parsed_files_simulator = sorted(parsed_files_simulator, key=lambda s: s['mean'], reverse=False)

    # for noise_f in parsed_files_noise_simulator:
    #     print(f"{noise_f['name']} - {noise_f['mean']}")
    #
    # for sim in parsed_files_simulator:
    #     print(f"{sim['name']} - {sim['mean']}")

    return parsed_files_simulator, parsed_files_noise_simulator

def convert_parsed_files_to_html(parsed_files):
    content_string = """
        <div style="width: 100%;font-size:14px">
                <div style="width: 50%; float:left;;">
                <br><br>
                <b>{name}</b><br>
                <br>
                Repetitions: {reps} <br>
                Expectation Value: {mean} <br>
                Highest Exp. Value: {max_v[0]} with {max_v[1]} % <br>
                Lowest  Exp. Value: {min_v[0]} with {min_v[1]} % <br>
                Highest Probability: {max_p[0]} with {max_p[1]} % <br>
                Ratio r: {ratio}<br>
                <span style="font-size:8px">
                    Gamma: {gamma} <br>
                    Beta: {beta} <br>
                </span>
                <br><br>
                </div>
                <div style="margin-left: 50%;"> {img} </div>
        </div>
        <br>
        """

    html = ""
    for file in parsed_files:
        html += content_string.format(name=file['name'],
                                                reps=file['reps'],
                                                gamma=file['gamma'],
                                                beta=file['beta'],
                                                mean=round(file['mean'], 3),
                                                max_v=file['max_v'],
                                                min_v=file['min_v'],
                                                max_p=file['max_p'],
                                                ratio=round(file['ratio'], 3),
                                                img=file['img'],
                                                )

    return html


def parse_results_from_html():
    parsed_files_simulator, parsed_files_noise_simulator = parse_results()
    noise_html = convert_parsed_files_to_html(parsed_files_noise_simulator)
    without_noise_html = convert_parsed_files_to_html(parsed_files_simulator)

    return without_noise_html, noise_html



# if __name__ == '__main__':
#     display_graph_with_solution_bitstring()