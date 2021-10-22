from matplotlib import pyplot as plt

# evaluation plots 

def plot_exp_evaluation_results_matplotlib(results, name):
    fig1, ax1 = plt.subplots()
    ax1.set_title(name)
    ax1.boxplot(results, meanline=True, showmeans=True)
    ax1.set_ylabel("expectation value")
    ax1.set_xlabel("p")
    fig1.show()
    
def plot_ratio_results_matplotlib(results, name):
    fig1, ax1 = plt.subplots()
    ax1.set_title(name)
    ax1.boxplot(results, meanline=True, showmeans=True)
    ax1.set_ylabel("ratio")
    ax1.set_xlabel("p")
    fig1.show()
    
def plot_approx_ratio_results_matplotlib(results, name):
    fig1, ax1 = plt.subplots()
    ax1.set_title(name)
    ax1.boxplot(results, meanline=True, showmeans=True)
    ax1.set_ylabel("approximation ratio")
    ax1.set_xlabel("p")
    fig1.show()
    

def display_boxplots_results(means, ratios, approx_ratios, prefix=''):
    #plot_exp_evaluation_results(means, f'{prefix}QAOA: Expectation Value')
    plot_exp_evaluation_results_matplotlib(means, f'{prefix}QAOA: Expectation Value')
    plot_ratio_results_matplotlib(ratios, f'{prefix}QAOA: Ratio')
    plot_approx_ratio_results_matplotlib(approx_ratios, f'{prefix}QAOA: Approximation Ratio')
    
    
    
    
