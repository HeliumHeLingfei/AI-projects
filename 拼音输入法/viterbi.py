# -*- coding: UTF-8 -*-
import numpy as np
import json
import codecs
import sys

# 读入模型
pi = np.load("./data/pi")
a = np.load("./data/transitions")
with open("./data/emissions.json", 'r') as f:
	b = json.load(f)
with open("./data/characters.json", 'r') as f:
	characters = json.load(f)
with open("./data/py2ch.json", 'r') as f:
	py2ch = json.load(f)


# nue->nve lue->lve
def viterbi(seq):
	if len(seq) == 0:
		return
	delta = []
	fai = []
	_delta = {}
	_fai = {}
	for t in range(len(seq)):
		if seq[t] == 'nue':
			seq[t] = 'nve'
		elif seq[t] == 'lue':
			seq[t] = 'lve'

	for i in py2ch[seq[0]]:
		_delta[i] = pi[characters.index(i)] + b[i][seq[0]]
		_fai[i] = ''
	delta.append(_delta)
	fai.append(_fai)
	for t in range(1, len(seq)):
		_delta = {}
		_fai = {}
		for i in py2ch[seq[t]]:
			maxd = list(delta[-1].values())[0] + a[characters.index(list(delta[-1].keys())[0])][characters.index(i)]
			maxj = list(delta[-1].keys())[0]
			for j in delta[-1]:
				_maxd = delta[-1][j] + a[characters.index(j)][characters.index(i)]
				if _maxd > maxd:
					maxd = _maxd
					maxj = j
			_delta[i] = maxd + b[i][seq[t]]
			_fai[i] = maxj
		delta.append(_delta)
		fai.append(_fai)
	P = list(delta[-1].values())[0]
	index = list(delta[-1].keys())[0]
	for i in delta[-1]:
		if delta[-1][i] > P:
			P = delta[-1][i]
			index = i
	result = [index]
	for t in reversed(range(1, len(seq))):
		index = fai[t][index]
		result.append(index)
	answer = ''
	for t in reversed(range(len(result))):
		answer += result[t]
	return answer


if __name__ == '__main__':
	if len(sys.argv) == 2 and sys.argv[1] == 'type':
		print("开始测试：")
		while True:
			s = input()
			if s != 'quit':
				print(viterbi(s.split()))
			else:
				break
	else:
		inputfile = "./data/input.txt"
		outputfile = "./data/output.txt"
		if len(sys.argv) > 2:
			inputfile = str(sys.argv[1])
			outputfile = str(sys.argv[2])
		with open(inputfile, 'r') as f:
			lines = f.readlines()
			ans = []
			for line in lines:
				ans.append(viterbi(line.split()))
		for i in ans:
			print(i)
		with codecs.open(outputfile, 'w', 'utf-8') as f:
			for line in ans:
				f.write(line + '\n')
