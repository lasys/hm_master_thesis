import ast
dict_filename = "/home/hm-tlacherm/qlm_notebooks/notebooks_1.2.1/notebooks/master_thesis_qaoa/ibm/landscape/simulator/comparison_3_reg/data/max_cuts_dict_3_reg.dict"
def get_dict():
    f = open(dict_filename, 'r')
    contents = f.read()
    dictionary = ast.literal_eval(contents)
    f.close()
    return dictionary

def get_max_cut(graph):
    dic = get_dict()
    return dic[graph]