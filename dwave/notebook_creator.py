def get_file_content(i):
    content = """{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from DWave_runner import *\\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "outputs": [],
   "source": [
    "run_all(nodes=[$NODES])"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%"
    }
   }
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3.8"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}"""
    return content.replace('$NODES', str(i))

## Überschreibt alles!

for i in range (5,10):
    file = open(f'notebooks/DWave_MaxCut_0{i}_Nodes.ipynb', 'w+')
    file.write(get_file_content(i))
    file.close()

for i in range (10,33):
    file = open(f'notebooks/DWave_MaxCut_{i}_Nodes.ipynb', 'w+')
    file.write(get_file_content(i))
    file.close()
