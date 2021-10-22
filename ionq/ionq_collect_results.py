import glob
from bs4 import BeautifulSoup
from ionq_parameters import *

def print_parameters():
    print(f"Optimizers: {list(optimizers.keys())} with MaxIter of {MAX_ITER}")
    print(f"Number of shots: {SHOTS}")
    print(f"Repitions: [ {REPS_MIN}; {REPS_MAX} ]")
    print(f"Gamma value interval: [ {GAMMA_MIN}; {GAMMA_MAX} ]")
    print(f"Beta value interval: [ {BETA_MAX};{BETA_MIN} ]")
    print(f"Number of Optuna Trials: {N_TRIALS}")

def get_name(filename):
    fs = filename.split('_')
    fs.pop(0)
    fs.pop(-1)
    if not fs:
        fs = ["Normal"]
    return " ".join(fs)


def parse_mean(line):
    return line.split(' ')[-1]


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
    probs_dict = {}

    pre = soup.find_all('pre')
    for p in pre:
        p = str(p)
        if p.startswith('<pre>array'):
            gamma, beta = parse_params(p)

        if p.startswith('<pre>optimal'):
            ppp = p.replace('<pre>', '').replace('</pre>', '')
            for l in ppp.splitlines(False):
                if "Mean: -" in l:
                    mean = parse_mean(l)
                if ", probability" in l:
                    probs_dict = parse_probability(l, probs_dict)

    reps = len(gamma)
    max_v = (max(probs_dict.keys()), f"{probs_dict[max(probs_dict.keys())]:.5}")
    min_v = (min(probs_dict.keys()), f"{probs_dict[min(probs_dict.keys())]:.5}")
    max_p = (max(probs_dict.keys(), key=(lambda key: probs_dict[key])),
             f"{probs_dict[max(probs_dict.keys(), key=(lambda key: probs_dict[key]))]:.5}")


    return reps, gamma, beta, mean, max_v, min_v, max_p, probs_dict


def get_attr_dict(soup):
    reps, gamma, beta, mean, max_v, min_v, max_p, probs_dict = get_attributes(soup)
    return {
        'reps': reps,
        'gamma': gamma,
        'beta': beta,
        'mean': mean,
        'max_v': max_v,
        'min_v': min_v,
        'max_p': max_p,
        'probs_dict': probs_dict
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
    parsed_files = parse_html_files("results")
    parsed_files = sorted(parsed_files, key=lambda s: s['mean'], reverse=True)
    return parsed_files

def convert_parsed_files_to_html(parsed_files):
    content_string = """
        <div style="width: 100%;font-size:14px">
                <div style="width: 50%; float:left;;">
                <br><br>
                <b>{name}</b><br>
                <br>
                Repetitions: {reps} <br>
                Mean: {mean} <br>
                Highest Value: {max_v[0]} with {max_v[1]}% <br>
                Lowest  Value: {min_v[0]} with  {min_v[1]} % <br>
                Highest Probability:{max_p[0]} with {max_p[1]} % <br>
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
                                                mean=round(float(file['mean']), 3),
                                                max_v=file['max_v'],
                                                min_v=file['min_v'],
                                                max_p=file['max_p'],
                                                img=file['img'],
                                                )

    return html


def parse_results_from_html():
    parsed_results = parse_results()
    results_html = convert_parsed_files_to_html(parsed_results)

    return results_html


